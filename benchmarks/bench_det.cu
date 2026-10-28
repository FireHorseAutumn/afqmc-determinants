// Rung-2 benchmark (DESIGN.md 5.1): fused gather + batched LU vs the vendor
// baseline — standalone gather into contiguous panels, cublasDgetrfBatched,
// and log|det|/sign extraction from its factors — each segment event-timed
// and summed into the vendor end-to-end. Our path is the fused determinant
// kernel (warp-per-det at N <= 32, block-per-det with shared panels at
// N <= 112, global panels above) and, separately, the per-walker signed
// log-sum-exp reduction (the two output modes of DESIGN.md 2.2).
//
// Kill-1 reference: a large cublasDgemm is timed as the part's practical
// FP64 rate; achieved GFLOP/s of both factorization paths (counting
// (2/3)N^3 flops per determinant) are reported against it.
//
// Correctness precondition per case: 64 seeded panels re-checked against
// the host reference (Eigen PartialPivLU) — signs must match exactly,
// |dlog|det|| <= 1e-9 * (1 + |logabs|); a mismatch aborts. The rigorous
// gate is tests/validate.py against msdet_gpu; this is the smoke check.
//
// Sweep: K = 5N, D = 100N, W chosen per N so the vendor's two panel buffers
// stay within the 8 GB part. N = 256 does not fit the vendor path at any
// W (2 * 25600 * 256^2 * 8 B = 27 GB) and is omitted — recorded boundary,
// not silently dropped. Medians with observed spreads; raw per-rep timings
// and machine state in the JSON.

#include "msd/overlap_gpu.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cublas_check(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + ": cublas status " +
                             std::to_string((int)st));
}

using msd::gpu::cuda_check;

struct CaseGen {
  int N, K, D, W;
  std::vector<double> G;        // W*K*N row-major
  std::vector<std::int32_t> occ; // D*N
  std::vector<double> coeff;     // D
};

CaseGen make_case(int N, int W, std::uint64_t seed) {
  CaseGen c;
  c.N = N;
  c.K = 5 * N;
  c.D = 100 * N;
  c.W = W;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss;
  c.G.resize((size_t)W * c.K * N);
  for (auto& v : c.G) v = gauss(rng);
  c.coeff.resize(c.D);
  for (auto& v : c.coeff) v = gauss(rng);
  c.occ.resize((size_t)c.D * N);
  std::vector<std::int32_t> rows(c.K);
  std::iota(rows.begin(), rows.end(), 0);
  for (int d = 0; d < c.D; ++d) {
    std::shuffle(rows.begin(), rows.end(), rng);
    std::copy_n(rows.begin(), N, c.occ.begin() + (size_t)d * N);
  }
  return c;
}

struct Stats {
  std::vector<double> reps;
  double median = 0, min = 0, max = 0;
  void finish() {
    std::sort(reps.begin(), reps.end());
    median = reps[reps.size() / 2];
    min = reps.front();
    max = reps.back();
  }
};

// Event-timed reps of f. Each rep may run several kernels; the events
// bracket the whole rep.
template <typename F>
Stats time_reps(F&& f, int reps) {
  Stats r;
  cudaEvent_t ev0, ev1;
  cuda_check(cudaEventCreate(&ev0), "event create");
  cuda_check(cudaEventCreate(&ev1), "event create");
  for (int i = 0; i < reps; ++i) {
    cuda_check(cudaDeviceSynchronize(), "pre-rep");
    cuda_check(cudaEventRecord(ev0), "ev0");
    f();
    cuda_check(cudaEventRecord(ev1), "ev1");
    cuda_check(cudaDeviceSynchronize(), "post-rep");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
    r.reps.push_back((double)ms * 1e-3);
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  r.finish();
  return r;
}

std::string timestamp_utc() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

void json_stats(std::ofstream& f, const Stats& r, int indent) {
  const std::string pad(indent, ' ');
  f << pad << "{\n"
    << pad << "  \"median_s\": " << r.median << ",\n"
    << pad << "  \"min_s\": " << r.min << ",\n"
    << pad << "  \"max_s\": " << r.max << ",\n"
    << pad << "  \"reps_s\": [";
  for (size_t i = 0; i < r.reps.size(); ++i)
    f << (i ? ", " : "") << r.reps[i];
  f << "]\n" << pad << "}";
}

// Host log|det|/sign by textbook partial-pivoting LU (first-max pivot rule,
// same as the mpmath reference). The bench precondition's reference; the
// rigorous gate is validate.py against msdet_gpu.
std::pair<double, int> logdet_lu_plain(const double* G,
                                       const std::int32_t* occ, int N) {
  std::vector<double> a((size_t)N * N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      a[(size_t)i * N + j] = G[(size_t)occ[i] * N + j];
  double logabs = 0.0;
  int sign = 1;
  for (int k = 0; k < N; ++k) {
    int p = k;
    for (int i = k + 1; i < N; ++i)
      if (std::fabs(a[(size_t)i * N + k]) > std::fabs(a[(size_t)p * N + k]))
        p = i;
    if (a[(size_t)p * N + k] == 0.0) return {-INFINITY, 0};
    if (p != k) {
      for (int j = 0; j < N; ++j)
        std::swap(a[(size_t)k * N + j], a[(size_t)p * N + j]);
      sign = -sign;
    }
    const double u = a[(size_t)k * N + k];
    logabs += std::log(std::fabs(u));
    if (u < 0.0) sign = -sign;
    for (int i = k + 1; i < N; ++i) {
      const double f = (a[(size_t)i * N + k] /= u);
      for (int j = k + 1; j < N; ++j)
        a[(size_t)i * N + j] -= f * a[(size_t)k * N + j];
    }
  }
  return {logabs, sign};
}

// Sampled host-reference precondition on the GPU determinant table.
void precondition(const CaseGen& c, const std::vector<double>& det_la,
                  const std::vector<int>& det_sg) {
  std::mt19937_64 rng(0x9a7c);
  for (int s = 0; s < 64; ++s) {
    const std::int64_t idx = rng() % ((std::int64_t)c.W * c.D);
    const int w = (int)(idx / c.D), d = (int)(idx % c.D);
    const auto [logabs, sign] =
        logdet_lu_plain(c.G.data() + (size_t)w * c.K * c.N,
                        c.occ.data() + (size_t)d * c.N, c.N);
    const int gsign = det_sg[idx];
    if (sign != gsign)
      throw std::runtime_error("precondition: sign mismatch at idx " +
                               std::to_string(idx));
    if (sign == 0) continue; // singular agrees by sign
    const double gla = det_la[idx];
    if (!std::isfinite(gla))
      throw std::runtime_error("precondition: non-finite logabs");
    if (std::fabs(gla - logabs) > 1e-9 * (1.0 + std::fabs(logabs)))
      throw std::runtime_error("precondition: logabs mismatch");
  }
}

struct CaseResult {
  int N, K, D, W;
  std::int64_t batch;
  const char* kernel;
  Stats ours_det, ours_reduce, vendor_gather, vendor_getrf, vendor_extract;
  double roofline_gflops;
};

} // namespace

int main(int argc, char** argv) {
  int reps = 11;
  std::uint64_t seed = 0xAF9C4ULL;
  std::string out_path = std::string(MSD_DATA_DIR) + "/afqmc_det_bench.json";
  for (int i = 1; i < argc; ++i) {
    try {
      if (std::strncmp(argv[i], "--reps=", 7) == 0)
        reps = std::stoi(argv[i] + 7);
      else if (std::strncmp(argv[i], "--seed=", 7) == 0)
        seed = std::stoull(argv[i] + 7, nullptr, 0);
      else if (std::strncmp(argv[i], "--out=", 6) == 0)
        out_path = argv[i] + 6;
      else
        throw std::invalid_argument(std::string("unknown argument: ") +
                                    argv[i]);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "bench_det: %s\n", e.what());
      return 1;
    }
  }

  try {
    int dev = 0;
    cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, dev), "props");
    int driver_v = 0, runtime_v = 0;
    cuda_check(cudaDriverGetVersion(&driver_v), "driver");
    cuda_check(cudaRuntimeGetVersion(&runtime_v), "runtime");
    std::printf("gpu: %s sm_%d%d, %d SMs, %zu MiB, driver %d, runtime %d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem >> 20, driver_v, runtime_v);

    cublasHandle_t cublas = nullptr;
    cublas_check(cublasCreate(&cublas), "cublasCreate");

    // Practical FP64 rate: one large square DGEMM, event-timed.
    double roofline_gflops = 0.0;
    {
      const int R = 4096;
      double *da, *db, *dc;
      cuda_check(cudaMalloc(&da, (size_t)R * R * 8), "dgemm a");
      cuda_check(cudaMalloc(&db, (size_t)R * R * 8), "dgemm b");
      cuda_check(cudaMalloc(&dc, (size_t)R * R * 8), "dgemm c");
      cuda_check(cudaMemset(da, 1, (size_t)R * R * 8), "memset a");
      cuda_check(cudaMemset(db, 2, (size_t)R * R * 8), "memset b");
      const double one = 1.0, zero = 0.0;
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      cublas_check(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, R, R, R,
                               &one, da, R, db, R, &zero, dc, R),
                   "dgemm warmup");
      cuda_check(cudaDeviceSynchronize(), "dgemm warmup sync");
      cudaEventRecord(e0);
      cublas_check(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, R, R, R,
                               &one, da, R, db, R, &zero, dc, R),
                   "dgemm");
      cudaEventRecord(e1);
      cuda_check(cudaDeviceSynchronize(), "dgemm sync");
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, e0, e1);
      roofline_gflops = 2.0 * R * R * (double)R / ((double)ms * 1e-3) / 1e9;
      std::printf("measured FP64 DGEMM rate: %.1f GFLOP/s\n\n",
                  roofline_gflops);
      cudaEventDestroy(e0);
      cudaEventDestroy(e1);
      cudaFree(da);
      cudaFree(db);
      cudaFree(dc);
    }

    // (N, W) grid; K = 5N, D = 100N. N=256 omitted: the vendor path alone
    // needs 27 GB of panel buffers at W=1 on an 8 GB part.
    const struct {
      int N;
      std::vector<int> Ws;
    } grid[] = {
        {16, {1, 8, 64, 512}},
        {32, {1, 8, 64}},
        {64, {1, 8}},
        {128, {1}},
    };

    std::printf("%4s %6s %8s %10s | %10s %10s | %10s %10s %10s %10s\n", "N",
                "W", "D", "batch", "ours ms", "ours+red", "v gather",
                "v getrf", "v extract", "v total");

    std::vector<CaseResult> results;
    for (const auto& g : grid)
      for (const int W : g.Ws) {
        const CaseGen cg = make_case(g.N, W, seed + g.N * 7919 + W);
        const std::int64_t batch = (std::int64_t)cg.W * cg.D;
        msd::gpu::Case dc(cg.G.data(), cg.occ.data(), cg.coeff.data(), cg.K,
                          cg.N, cg.D, cg.W, true);

        CaseResult cr;
        cr.N = cg.N;
        cr.K = cg.K;
        cr.D = cg.D;
        cr.W = cg.W;
        cr.batch = batch;
        cr.kernel = cg.N <= 32   ? "warp"
                    : cg.N <= msd::gpu::kMaxSharedN ? "block_shared"
                                                    : "block_global";
        cr.roofline_gflops = roofline_gflops;

        // Warmup + precondition + info check.
        dc.run_det();
        dc.run_reduce();
        {
          std::vector<double> dla;
          std::vector<double> wla;
          std::vector<int> dsg, wsg;
          dc.download(dla, dsg, wla, wsg);
          precondition(cg, dla, dsg);
        }

        // Vendor warmup: gather into the LU buffer, factor, extract.
        const auto vendor_all = [&]() {
          msd::gpu::gather_panels<<<(unsigned)batch, msd::gpu::kBlockThreads>>>(
              dc.d_G, cg.K, cg.N, cg.D, dc.d_occ, dc.d_lu);
          cublas_check(cublasDgetrfBatched(
                           cublas, cg.N, dc.d_ptrs, cg.N, dc.d_pivots,
                           dc.d_info, (int)batch),
                       "getrfBatched");
          msd::gpu::extract_logdet<<<(unsigned)((batch +
                                                 msd::gpu::kWarpsPerBlock - 1) /
                                                msd::gpu::kWarpsPerBlock),
                                     msd::gpu::kBlockThreads>>>(
              dc.d_lu, dc.d_pivots, dc.d_info, cg.N, batch, dc.d_det_logabs,
              dc.d_det_sign);
        };
        vendor_all();
        cuda_check(cudaDeviceSynchronize(), "vendor warmup");
        {
          std::vector<int> h_info(batch);
          cuda_check(cudaMemcpy(h_info.data(), dc.d_info,
                                batch * sizeof(int), cudaMemcpyDeviceToHost),
                     "info down");
          for (std::int64_t i = 0; i < batch; ++i)
            if (h_info[i] != 0)
              throw std::runtime_error("getrfBatched info != 0 at panel " +
                                       std::to_string(i));
        }

        cr.ours_det = time_reps([&]() { dc.run_det(); }, reps);
        cr.ours_reduce = time_reps([&]() { dc.run_reduce(); }, reps);
        cr.vendor_gather = time_reps(
            [&]() {
              msd::gpu::gather_panels<<<(unsigned)batch,
                                        msd::gpu::kBlockThreads>>>(
                  dc.d_G, cg.K, cg.N, cg.D, dc.d_occ, dc.d_lu);
            },
            reps);
        cr.vendor_getrf = time_reps(
            [&]() {
              cublas_check(cublasDgetrfBatched(cublas, cg.N, dc.d_ptrs, cg.N,
                                               dc.d_pivots, dc.d_info,
                                               (int)batch),
                           "getrfBatched rep");
            },
            reps);
        cr.vendor_extract = time_reps(
            [&]() {
              msd::gpu::extract_logdet<<<(unsigned)(
                                             (batch + msd::gpu::kWarpsPerBlock -
                                              1) /
                                             msd::gpu::kWarpsPerBlock),
                                         msd::gpu::kBlockThreads>>>(
                  dc.d_lu, dc.d_pivots, dc.d_info, cg.N, batch,
                  dc.d_det_logabs, dc.d_det_sign);
            },
            reps);

        results.push_back(cr);
        const double vtot = cr.vendor_gather.median + cr.vendor_getrf.median +
                            cr.vendor_extract.median;
        std::printf("%4d %6d %8d %10" PRId64 " | %10.3f %10.3f | %10.3f "
                    "%10.3f %10.3f %10.3f\n",
                    cr.N, cr.W, cr.D, batch, cr.ours_det.median * 1e3,
                    (cr.ours_det.median + cr.ours_reduce.median) * 1e3,
                    cr.vendor_gather.median * 1e3,
                    cr.vendor_getrf.median * 1e3,
                    cr.vendor_extract.median * 1e3, vtot * 1e3);
        const double flops = (2.0 / 3.0) * cg.N * cg.N * (double)cg.N * batch;
        std::printf("      det/s: ours %.3g  vendor %.3g | GFLOP/s: ours "
                    "%.1f (%.1f%% of DGEMM)  vendor %.1f (%.1f%%)\n",
                    batch / cr.ours_det.median, batch / vtot,
                    flops / cr.ours_det.median / 1e9,
                    100.0 * flops / cr.ours_det.median / 1e9 /
                        roofline_gflops,
                    flops / vtot / 1e9,
                    100.0 * flops / vtot / 1e9 / roofline_gflops);
        std::printf("      spread ms [min,max]: ours [%.3f,%.3f] getrf "
                    "[%.3f,%.3f]\n",
                    cr.ours_det.min * 1e3, cr.ours_det.max * 1e3,
                    cr.vendor_getrf.min * 1e3, cr.vendor_getrf.max * 1e3);
      }

    cublasDestroy(cublas);

    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("cannot open " + out_path);
    f.precision(17);
    f << "{\n"
      << "  \"timestamp_utc\": \"" << timestamp_utc() << "\",\n"
      << "  \"seed\": \"" << seed << "\",\n"
      << "  \"gpu\": \"" << prop.name << "\",\n"
      << "  \"compute_capability\": \"" << prop.major << "." << prop.minor
      << "\",\n"
      << "  \"sm_count\": " << prop.multiProcessorCount << ",\n"
      << "  \"total_global_mem\": " << prop.totalGlobalMem << ",\n"
      << "  \"driver_version\": " << driver_v << ",\n"
      << "  \"runtime_version\": " << runtime_v << ",\n"
      << "  \"nvcc\": \"" << __CUDACC_VER_MAJOR__ << "."
      << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << "\",\n"
      << "  \"build_type\": \"" << MSD_BUILD_TYPE << "\",\n"
      << "  \"n_reps\": " << reps << ",\n"
      << "  \"fp64_dgemm_gflops\": " << roofline_gflops << ",\n"
      << "  \"note\": \"K=5N, D=100N; vendor = gather + cublasDgetrfBatched "
         "+ extract, each event-timed. flops counted as (2/3)N^3 per "
         "determinant. N=256 omitted: vendor panel buffers need 27 GB at "
         "W=1.\",\n"
      << "  \"cases\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const CaseResult& c = results[i];
      const double flops =
          (2.0 / 3.0) * c.N * c.N * (double)c.N * c.batch;
      const double vtot = c.vendor_gather.median + c.vendor_getrf.median +
                          c.vendor_extract.median;
      f << "    {\"N\": " << c.N << ", \"K\": " << c.K << ", \"D\": " << c.D
        << ", \"W\": " << c.W << ", \"batch\": " << c.batch
        << ", \"kernel\": \"" << c.kernel << "\",\n"
        << "     \"speedup_vs_vendor_e2e\": " << vtot / c.ours_det.median
        << ",\n"
        << "     \"ours_gflops\": " << flops / c.ours_det.median / 1e9
        << ", \"vendor_gflops\": " << flops / vtot / 1e9 << ",\n"
        << "     \"ours_det\": ";
      json_stats(f, c.ours_det, 5);
      f << ",\n     \"ours_reduce\": ";
      json_stats(f, c.ours_reduce, 5);
      f << ",\n     \"vendor_gather\": ";
      json_stats(f, c.vendor_gather, 5);
      f << ",\n     \"vendor_getrf\": ";
      json_stats(f, c.vendor_getrf, 5);
      f << ",\n     \"vendor_extract\": ";
      json_stats(f, c.vendor_extract, 5);
      f << "}" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    if (!f) throw std::runtime_error("write failed: " + out_path);
    std::printf("\nJSON -> %s\n", out_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_det failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
