// Host reference driver: reads a case file, evaluates every (walker,
// determinant) overlap through the pool path (G = Phi^T W, row gather, LU
// log/sign accumulator), reduces per-walker weights, writes results.
//
// Case file (little-endian, row-major arrays):
//   char[4] "MSD1", i64 M, K, N, D, W
//   f64 Phi[M*K]      pool orbitals, M x K
//   f64 coeff[D]      trial coefficients c_d
//   i32 occ[D*N]      occupation lists I_d, indices into [0, K)
//   f64 walkers[W*M*N]
// Results file:
//   char[4] "MSR1", i64 W, D
//   f64 det_logabs[W*D], i32 det_sign[W*D]
//   f64 w_logabs[W], i32 w_sign[W]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include "msd/overlap.hpp"

namespace {

template <typename T>
void read_exact(std::ifstream& in, T* dst, std::size_t count) {
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) {
    std::fprintf(stderr, "msdet_ref: truncated case file\n");
    std::exit(2);
  }
}

template <typename T>
void write_exact(std::ofstream& out, const T* src, std::size_t count) {
  out.write(reinterpret_cast<const char*>(src),
            static_cast<std::streamsize>(count * sizeof(T)));
  if (!out) {
    std::fprintf(stderr, "msdet_ref: failed writing results\n");
    std::exit(2);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: msdet_ref <case> <results>\n");
    return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "msdet_ref: cannot open %s\n", argv[1]);
    return 2;
  }
  char magic[4];
  read_exact(in, magic, 4);
  if (std::memcmp(magic, "MSD1", 4) != 0) {
    std::fprintf(stderr, "msdet_ref: bad magic\n");
    return 2;
  }
  std::int64_t dims[5];
  read_exact(in, dims, 5);
  const std::int64_t M = dims[0], K = dims[1], N = dims[2], D = dims[3], W = dims[4];
  if (M <= 0 || K <= 0 || N <= 0 || D <= 0 || W <= 0 || K > M || N > K) {
    std::fprintf(stderr, "msdet_ref: bad dims M=%lld K=%lld N=%lld D=%lld W=%lld\n",
                 (long long)M, (long long)K, (long long)N, (long long)D, (long long)W);
    return 2;
  }

  msd::RowMat Phi(M, K);
  read_exact(in, Phi.data(), static_cast<std::size_t>(M * K));
  std::vector<double> coeff(D);
  read_exact(in, coeff.data(), static_cast<std::size_t>(D));
  std::vector<std::int32_t> occ(D * N);
  read_exact(in, occ.data(), static_cast<std::size_t>(D * N));
  std::vector<double> walkers(W * M * N);
  read_exact(in, walkers.data(), static_cast<std::size_t>(W * M * N));

  std::vector<double> det_logabs(W * D);
  std::vector<std::int32_t> det_sign(W * D);
  std::vector<double> w_logabs(W);
  std::vector<std::int32_t> w_sign(W);

  for (std::int64_t w = 0; w < W; ++w) {
    Eigen::Map<const msd::RowMat> Ww(walkers.data() + w * M * N, M, N);
    const msd::RowMat G = msd::walker_overlap_matrix(Phi, Ww);
    std::vector<msd::Overlap> ov(D);
    for (std::int64_t d = 0; d < D; ++d) {
      ov[d] = msd::gather_logdet(G, occ.data() + d * N, static_cast<int>(N));
      det_logabs[w * D + d] = ov[d].logabs;
      det_sign[w * D + d] = ov[d].sign;
    }
    const msd::Overlap weight = msd::reduce_walker(coeff.data(), ov.data(), static_cast<int>(D));
    w_logabs[w] = weight.logabs;
    w_sign[w] = weight.sign;
  }

  std::ofstream out(argv[2], std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "msdet_ref: cannot write %s\n", argv[2]);
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
