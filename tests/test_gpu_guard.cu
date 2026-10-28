// Tail-guard checks for the warp-per-item kernels. Both gather_lu_warp and
// extract_logdet launch on a warp-granular grid (kWarpsPerBlock warps per
// block), so a batch that is not a multiple of kWarpsPerBlock leaves tail
// warps with idx >= batch; those warps must not touch any output. The
// output arrays are padded to the launched warp count and filled with a
// sentinel; a tail write clobbers it.
//
// The block-per-item kernels (gather_lu_block, gather_panels,
// reduce_walker) launch one block per work item and have no tail.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "msd/overlap_gpu.cuh"

using namespace msd::gpu;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

constexpr std::int64_t kBatch = 5;  // not a multiple of kWarpsPerBlock (4)
constexpr std::int64_t kPadded = 8;  // warps launched: 2 blocks x 4
constexpr double kSentinel = -777.0;
constexpr int kSentinelInt = -9;

}  // namespace

int main() {
  const int N = 4;
  const int blocks = (int)((kBatch + kWarpsPerBlock - 1) / kWarpsPerBlock);

  // extract_logdet: fabricated vendor output — identity panels, identity
  // pivots, info 0 — so each panel gives log|det| = 0, sign = +1.
  {
    std::vector<double> h_panels(kPadded * N * N, 0.0);
    for (std::int64_t p = 0; p < kPadded; ++p)
      for (int i = 0; i < N; ++i) h_panels[p * N * N + i * N + i] = 1.0;
    std::vector<int> h_pivots(kPadded * N);
    for (std::int64_t p = 0; p < kPadded; ++p)
      for (int i = 0; i < N; ++i) h_pivots[p * N + i] = i + 1;
    std::vector<int> h_info(kPadded, 0);
    std::vector<double> h_la(kPadded, kSentinel);
    std::vector<int> h_sg(kPadded, kSentinelInt);

    double* d_panels = nullptr;
    int *d_pivots = nullptr, *d_info = nullptr, *d_sg = nullptr;
    double* d_la = nullptr;
    cuda_check(cudaMalloc(&d_panels, h_panels.size() * sizeof(double)), "p");
    cuda_check(cudaMalloc(&d_pivots, h_pivots.size() * sizeof(int)), "pv");
    cuda_check(cudaMalloc(&d_info, h_info.size() * sizeof(int)), "info");
    cuda_check(cudaMalloc(&d_la, h_la.size() * sizeof(double)), "la");
    cuda_check(cudaMalloc(&d_sg, h_sg.size() * sizeof(int)), "sg");
    cuda_check(cudaMemcpy(d_panels, h_panels.data(),
                          h_panels.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "p up");
    cuda_check(cudaMemcpy(d_pivots, h_pivots.data(),
                          h_pivots.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "pv up");
    cuda_check(cudaMemcpy(d_info, h_info.data(), h_info.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "info up");
    cuda_check(cudaMemcpy(d_la, h_la.data(), h_la.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "la up");
    cuda_check(cudaMemcpy(d_sg, h_sg.data(), h_sg.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "sg up");

    extract_logdet<<<blocks, kBlockThreads>>>(d_panels, d_pivots, d_info, N,
                                              kBatch, d_la, d_sg);
    cuda_check(cudaGetLastError(), "extract launch");
    cuda_check(cudaDeviceSynchronize(), "extract");

    cuda_check(cudaMemcpy(h_la.data(), d_la, h_la.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "la down");
    cuda_check(cudaMemcpy(h_sg.data(), d_sg, h_sg.size() * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "sg down");
    for (std::int64_t i = 0; i < kBatch; ++i) {
      check(h_la[i] == 0.0 && h_sg[i] == 1, "extract: in-batch value");
    }
    for (std::int64_t i = kBatch; i < kPadded; ++i) {
      check(h_la[i] == kSentinel && h_sg[i] == kSentinelInt,
            "extract_logdet: tail warp wrote past batch");
    }
    cudaFree(d_panels);
    cudaFree(d_pivots);
    cudaFree(d_info);
    cudaFree(d_la);
    cudaFree(d_sg);
  }

  // gather_lu_warp (the guarded control): W = 5 walkers, D = 1, K = N = 4,
  // G_w the identity, occ = {0,1,2,3} — each panel is the identity, so
  // log|det| = 0, sign = +1.
  {
    const int K = N, D = 1;
    std::vector<double> h_G(kBatch * K * N, 0.0);
    for (std::int64_t w = 0; w < kBatch; ++w)
      for (int i = 0; i < N; ++i) h_G[w * K * N + i * N + i] = 1.0;
    std::vector<std::int32_t> h_occ(D * N);
    for (int i = 0; i < N; ++i) h_occ[i] = i;
    std::vector<double> h_la(kPadded, kSentinel);
    std::vector<int> h_sg(kPadded, kSentinelInt);

    double* d_G = nullptr;
    std::int32_t* d_occ = nullptr;
    double* d_la = nullptr;
    int* d_sg = nullptr;
    cuda_check(cudaMalloc(&d_G, h_G.size() * sizeof(double)), "G");
    cuda_check(cudaMalloc(&d_occ, h_occ.size() * sizeof(std::int32_t)), "occ");
    cuda_check(cudaMalloc(&d_la, h_la.size() * sizeof(double)), "la");
    cuda_check(cudaMalloc(&d_sg, h_sg.size() * sizeof(int)), "sg");
    cuda_check(cudaMemcpy(d_G, h_G.data(), h_G.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "G up");
    cuda_check(cudaMemcpy(d_occ, h_occ.data(),
                          h_occ.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               "occ up");
    cuda_check(cudaMemcpy(d_la, h_la.data(), h_la.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "la up");
    cuda_check(cudaMemcpy(d_sg, h_sg.data(), h_sg.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "sg up");

    const size_t smem = (size_t)kWarpsPerBlock * N * N * sizeof(double);
    gather_lu_warp<<<blocks, kBlockThreads, smem>>>(d_G, K, N, D, d_occ,
                                                    kBatch, d_la, d_sg);
    cuda_check(cudaGetLastError(), "warp launch");
    cuda_check(cudaDeviceSynchronize(), "warp");

    cuda_check(cudaMemcpy(h_la.data(), d_la, h_la.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "la down");
    cuda_check(cudaMemcpy(h_sg.data(), d_sg, h_sg.size() * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "sg down");
    for (std::int64_t i = 0; i < kBatch; ++i) {
      check(h_la[i] == 0.0 && h_sg[i] == 1, "gather_lu_warp: in-batch value");
    }
    for (std::int64_t i = kBatch; i < kPadded; ++i) {
      check(h_la[i] == kSentinel && h_sg[i] == kSentinelInt,
            "gather_lu_warp: tail warp wrote past batch");
    }
    cudaFree(d_G);
    cudaFree(d_occ);
    cudaFree(d_la);
    cudaFree(d_sg);
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d gpu_guard checks failed\n", failures);
    return 1;
  }
  std::puts("gpu_guard: all checks passed");
  return 0;
}
