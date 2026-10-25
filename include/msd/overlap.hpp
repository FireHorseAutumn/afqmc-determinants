// Multi-Slater overlap-determinant evaluation, host reference (DESIGN.md §8
// rung 1). Real FP64. The magnitude-separation discipline is reimplemented
// here (log|det| as a sum of logs, sign carried separately, products never
// formed); no code is imported from the DQMC LDR header.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <Eigen/Dense>

namespace msd {

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColMat = Eigen::MatrixXd;

struct Overlap {
  double logabs;  // natural log of |det|; -inf iff singular
  int sign;       // +1, -1, or 0 (zero weight)
};

inline constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// log|det| = sum_i log|U_ii| and sign = parity(P) * prod_i sign(U_ii) from
// P*A = L*U with partial pivoting. An exactly zero pivot is a zero weight.
inline Overlap logdet_lu(const Eigen::Ref<const ColMat>& A) {
  Eigen::PartialPivLU<ColMat> lu(A);
  const ColMat& lu_factors = lu.matrixLU();
  const int n = static_cast<int>(A.rows());
  double logabs = 0.0;
  int sign = 1;
  for (int i = 0; i < n; ++i) {
    const double u = lu_factors(i, i);
    if (u == 0.0) return {kNegInf, 0};
    logabs += std::log(std::fabs(u));
    if (u < 0.0) sign = -sign;
  }
  sign *= lu.permutationP().determinant();
  return {logabs, sign};
}

// G = Phi^T * W, K x N, row-major so the per-determinant row gather copies
// contiguous rows.
inline RowMat walker_overlap_matrix(const Eigen::Ref<const RowMat>& Phi,
                                    const Eigen::Ref<const RowMat>& W) {
  return RowMat(Phi.transpose() * W);
}

// O = G[occ, :] then the LU accumulator.
inline Overlap gather_logdet(const Eigen::Ref<const RowMat>& G,
                             const std::int32_t* occ, int n) {
  ColMat panel(n, n);
  for (int i = 0; i < n; ++i) panel.row(i) = G.row(occ[i]);
  return logdet_lu(panel);
}

// Signed log-sum-exp of w_d = c_d * det_d given per-determinant overlaps:
// log|w_d| = log|c_d| + logabs_d, s_d = sign(c_d) * sign_d. With m the max
// log|w_d|, S = sum_d s_d exp(log|w_d| - m); exact cancellation (S == 0) is
// a zero weight.
inline Overlap reduce_walker(const double* coeff, const Overlap* ov, int d_count) {
  double max_log = kNegInf;
  for (int d = 0; d < d_count; ++d) {
    if (ov[d].sign == 0 || coeff[d] == 0.0) continue;
    const double lw = std::log(std::fabs(coeff[d])) + ov[d].logabs;
    if (lw > max_log) max_log = lw;
  }
  if (max_log == kNegInf) return {kNegInf, 0};
  double sum = 0.0;
  for (int d = 0; d < d_count; ++d) {
    if (ov[d].sign == 0 || coeff[d] == 0.0) continue;
    const int s = (coeff[d] > 0.0 ? 1 : -1) * ov[d].sign;
    const double lw = std::log(std::fabs(coeff[d])) + ov[d].logabs;
    sum += s * std::exp(lw - max_log);
  }
  if (sum == 0.0) return {kNegInf, 0};
  return {max_log + std::log(std::fabs(sum)), sum > 0.0 ? 1 : -1};
}

}  // namespace msd
