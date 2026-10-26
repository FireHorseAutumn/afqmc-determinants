// GPU driver: same case/results interface as msdet_ref, so the mpmath gate
// harness (tests/validate.py) runs against it unchanged. The per-walker
// G = Phi^T W product is the vendor-GEMM part of the pipeline (DESIGN.md 2)
// and is computed on the host via Eigen here — the kernels under gate are
// the fused gather + LU and the walker reduction.
//
// usage: msdet_gpu [--force=block] <case> <results>
//   default kernel choice: warp-per-determinant at N <= 32, block-per-
//   determinant otherwise. --force=block forces the block kernel, so the
//   gate covers both kernels.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "msd/overlap_gpu.cuh"

namespace {

template <typename T>
void read_exact(std::ifstream& in, T* dst, std::size_t count) {
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) {
    std::fprintf(stderr, "msdet_gpu: truncated case file\n");
    std::exit(2);
  }
}

template <typename T>
void write_exact(std::ofstream& out, const T* src, std::size_t count) {
  out.write(reinterpret_cast<const char*>(src),
            static_cast<std::streamsize>(count * sizeof(T)));
  if (!out) {
    std::fprintf(stderr, "msdet_gpu: failed writing results\n");
    std::exit(2);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool force_block = false;
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--force=block")
      force_block = true;
    else
      args.push_back(a);
  }
  if (args.size() != 2) {
    std::fprintf(stderr, "usage: msdet_gpu [--force=block] <case> <results>\n");
    return 2;
  }
  std::ifstream in(args[0], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "msdet_gpu: cannot open %s\n", args[0].c_str());
    return 2;
  }
  char magic[4];
  read_exact(in, magic, 4);
  if (std::memcmp(magic, "MSD1", 4) != 0) {
    std::fprintf(stderr, "msdet_gpu: bad magic\n");
    return 2;
  }
  std::int64_t dims[5];
  read_exact(in, dims, 5);
  const std::int64_t M = dims[0], K = dims[1], N = dims[2], D = dims[3], W = dims[4];
  if (M <= 0 || K <= 0 || N <= 0 || D <= 0 || W <= 0 || K > M || N > K) {
    std::fprintf(stderr, "msdet_gpu: bad dims\n");
    return 2;
  }

  std::vector<double> Phi(M * K);
  read_exact(in, Phi.data(), static_cast<std::size_t>(M * K));
  std::vector<double> coeff(D);
  read_exact(in, coeff.data(), static_cast<std::size_t>(D));
  std::vector<std::int32_t> occ(D * N);
  read_exact(in, occ.data(), static_cast<std::size_t>(D * N));
  std::vector<double> walkers(W * M * N);
  read_exact(in, walkers.data(), static_cast<std::size_t>(W * M * N));

  // Host-side per-walker G = Phi^T W_w (K x N row-major each), the
  // vendor-GEMM stage of the pipeline; plain loop, gate sizes are small.
  std::vector<double> h_G(static_cast<std::size_t>(W) * K * N, 0.0);
  for (std::int64_t w = 0; w < W; ++w) {
    const double* Ww = walkers.data() + w * M * N;
    double* Gw = h_G.data() + w * K * N;
    for (std::int64_t m = 0; m < M; ++m)
      for (std::int64_t k = 0; k < K; ++k) {
        const double p = Phi[m * K + k];
        if (p == 0.0) continue;
        for (std::int64_t j = 0; j < N; ++j)
          Gw[k * N + j] += p * Ww[m * N + j];
      }
  }

  msd::gpu::Case c(h_G.data(), occ.data(), coeff.data(), (int)K, (int)N,
                   (int)D, (int)W, false);
  c.run_det(force_block);
  c.run_reduce();

  std::vector<double> det_logabs, w_logabs;
  std::vector<int> det_sign, w_sign;
  c.download(det_logabs, det_sign, w_logabs, w_sign);

  std::ofstream out(args[1], std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "msdet_gpu: cannot write %s\n", args[1].c_str());
    return 2;
  }
  write_exact(out, "MSR1", 4);
  const std::int64_t wd[2] = {W, D};
  write_exact(out, wd, 2);
  write_exact(out, det_logabs.data(), det_logabs.size());
  write_exact(out, det_sign.data(), det_sign.size());
  write_exact(out, w_logabs.data(), w_logabs.size());
  write_exact(out, w_sign.data(), w_sign.size());
  return 0;
}
