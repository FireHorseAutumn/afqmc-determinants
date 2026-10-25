#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "msd/overlap.hpp"

static int failures = 0;

static void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

static int parity_of(const std::vector<int>& p) {
  int inv = 0;
  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t j = i + 1; j < p.size(); ++j)
      if (p[i] > p[j]) ++inv;
  return (inv % 2 == 0) ? 1 : -1;
}

int main() {
  // Permutation matrices: pivots are exactly +/-1, so logabs == 0.0 exactly
  // and sign is the permutation parity.
  {
    std::mt19937_64 rng(1234);
    for (int n = 1; n <= 40; ++n) {
      std::vector<int> p(n);
      for (int i = 0; i < n; ++i) p[i] = i;
      std::shuffle(p.begin(), p.end(), rng);
      msd::ColMat A = msd::ColMat::Zero(n, n);
      for (int i = 0; i < n; ++i) A(i, p[i]) = 1.0;
      const msd::Overlap ov = msd::logdet_lu(A);
      check(ov.logabs == 0.0, "permutation logabs exact zero");
      check(ov.sign == parity_of(p), "permutation sign == parity");
    }
  }

  // Diagonal with a product that underflows FP64: the sum-of-logs stays
  // finite where the naive product is zero.
  {
    msd::ColMat A = msd::ColMat::Zero(4, 4);
    A(0, 0) = 1e-120;
    A(1, 1) = -3.0;
    A(2, 2) = 2e-150;
    A(3, 3) = -1e-100;
    const double naive = 1e-120 * -3.0 * 2e-150 * -1e-100;
    check(naive == 0.0, "naive product underflows to zero");
    const msd::Overlap ov = msd::logdet_lu(A);
    const double want = std::log(1e-120) + std::log(3.0) + std::log(2e-150) + std::log(1e-100);
    check(std::isfinite(ov.logabs), "logabs finite past product underflow");
    check(std::fabs(ov.logabs - want) <= 1e-13 * std::fabs(want), "logabs matches sum of logs");
    check(ov.sign == 1, "diagonal sign");
  }

  // Gather matches manual row assembly, bitwise.
  {
    std::mt19937_64 rng(99);
    std::normal_distribution<double> gauss;
    const int K = 120, N = 35;
    msd::RowMat G(K, N);
    for (auto& v : G.reshaped()) v = gauss(rng);
    std::uniform_int_distribution<std::int32_t> pick(0, K - 1);
    for (int rep = 0; rep < 20; ++rep) {
      std::vector<std::int32_t> occ(N);
      for (auto& o : occ) o = pick(rng);
      const msd::Overlap via_gather = msd::gather_logdet(G, occ.data(), N);
      msd::ColMat manual(N, N);
      for (int i = 0; i < N; ++i) manual.row(i) = G.row(occ[i]);
      const msd::Overlap via_manual = msd::logdet_lu(manual);
      check(via_gather.logabs == via_manual.logabs && via_gather.sign == via_manual.sign,
            "gather bitwise vs manual assembly");
    }
  }

  // Cross-check against Eigen's determinant on well-conditioned matrices.
  {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> gauss;
    for (int rep = 0; rep < 50; ++rep) {
      const int n = 1 + rep % 60;
      msd::ColMat A(n, n);
      for (auto& v : A.reshaped()) v = gauss(rng);
      A += msd::ColMat::Identity(n, n) * n;
      const msd::Overlap ov = msd::logdet_lu(A);
      const double det = A.determinant();
      check(ov.sign == (det > 0.0 ? 1 : -1), "sign vs Eigen determinant");
      check(std::fabs(ov.logabs - std::log(std::fabs(det))) <= 1e-12 * std::fabs(ov.logabs),
            "logabs vs Eigen determinant");
    }
  }

  // Singular inputs report zero weight, never NaN.
  {
    std::mt19937_64 rng(5);
    std::normal_distribution<double> gauss;
    for (int rep = 0; rep < 10; ++rep) {
      msd::ColMat A(16, 16);
      for (auto& v : A.reshaped()) v = gauss(rng);
      msd::ColMat dup = A;
      dup.row(3) = A.row(7);
      const msd::Overlap ov = msd::logdet_lu(dup);
      check(ov.sign == 0 && ov.logabs == msd::kNegInf, "duplicate row is zero weight");
      msd::ColMat zrow = A;
      zrow.row(11).setZero();
      const msd::Overlap ov2 = msd::logdet_lu(zrow);
      check(ov2.sign == 0 && ov2.logabs == msd::kNegInf, "zero row is zero weight");
    }
  }

  // Walker reduction: mixed signs vs long double, and exact cancellation.
  {
    std::vector<double> c = {2.0, -3.0, 1.5};
    std::vector<msd::Overlap> ov = {{std::log(4.0), 1}, {std::log(5.0), 1}, {std::log(1.0), -1}};
    // w = {8, -15, -1.5}
    const msd::Overlap w = msd::reduce_walker(c.data(), ov.data(), 3);
    const long double want = 8.0L - 15.0L - 1.5L;
    check(w.sign == -1, "walker weight sign");
    check(std::fabs(w.logabs - std::log(static_cast<double>(-want))) <= 1e-14,
          "walker weight magnitude");

    std::vector<double> c2 = {1.0, -1.0};
    std::vector<msd::Overlap> ov2 = {{std::log(7.0), 1}, {std::log(7.0), 1}};
    const msd::Overlap w2 = msd::reduce_walker(c2.data(), ov2.data(), 2);
    check(w2.sign == 0 && w2.logabs == msd::kNegInf, "exact cancellation is zero weight");
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d unit checks failed\n", failures);
    return 1;
  }
  std::puts("unit: all checks passed");
  return 0;
}
