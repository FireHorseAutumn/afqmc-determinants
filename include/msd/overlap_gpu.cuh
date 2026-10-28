#pragma once

// overlap_gpu.cuh — GPU fused gather + batched LU for the multi-Slater
// overlap determinants (DESIGN.md section 8, rung 2). Real FP64; the
// magnitude-separation discipline of the host reference applies: log|det|
// as a sum of log|U_ii|, sign = pivot parity * prod sign(U_ii), products
// never formed.
//
// Kernels:
//   gather_lu_block<InShared> — one 128-thread block per (walker, det):
//     fused row gather off G_w, then block-cooperative LU with partial
//     pivoting. Panel in dynamic shared memory when InShared (this part
//     fits N <= 112); the global-panel instantiation covers larger N at
//     latency cost.
//   gather_lu_warp — one warp per (walker, det), N <= 32, panel in a
//     per-warp shared slice; better occupancy where panels are tiny.
//   reduce_walker — block per walker, signed log-sum-exp over the
//     determinant table (DESIGN.md 2.2's second output mode).
//   gather_panels / extract_logdet — the vendor-baseline pieces: a
//     standalone gather producing contiguous panels for
//     cublasDgetrfBatched, and the log|det|/sign extraction from its LU
//     factors and pivot vector.
//
// Pivot search keeps the first maximum on ties (strict >, lower index
// wins), matching the mpmath reference's rule. Both GPU paths are
// deterministic run to run (fixed geometry, fixed reduction shapes); they
// are not bitwise identical to the host reference (tree reductions,
// device libm), and the gate is criterion-1 tolerance per DESIGN.md 5.2.

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace msd {
namespace gpu {

constexpr int kBlockThreads = 128;
constexpr int kWarpsPerBlock = kBlockThreads / 32;
constexpr int kMaxSharedN = 112; // 112^2 * 8 B = 98 KB dynamic shared

inline void cuda_check(cudaError_t st, const char* what) {
  if (st != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(st));
}

// Same -inf marker as msd::kNegInf; spelled with INFINITY because the
// constexpr host initializer in overlap.hpp is not usable in device code
// (nvcc warning #20015-D).
constexpr double kNegInfG = -INFINITY;

struct ShCtx {
  double red[32];
  int ired[32];
  double pval;
  int pidx;
  int singular;
};

// Block maxloc over per-thread candidates (v, i); empty candidates pass
// v = -inf. All threads must call. Result broadcast via sh.pval/sh.pidx.
__device__ inline void block_maxloc(double v, int i, ShCtx& sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1) {
    const double ov = __shfl_down_sync(0xffffffffu, v, o);
    const int oi = __shfl_down_sync(0xffffffffu, i, o);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
  if (lane == 0) { sh.red[warp] = v; sh.ired[warp] = i; }
  __syncthreads();
  if (warp == 0) {
    const int nwarp = (blockDim.x + 31) >> 5;
    v = (lane < nwarp) ? sh.red[lane] : -INFINITY;
    i = (lane < nwarp) ? sh.ired[lane] : 0x7fffffff;
    for (int o = 16; o; o >>= 1) {
      const double ov = __shfl_down_sync(0xffffffffu, v, o);
      const int oi = __shfl_down_sync(0xffffffffu, i, o);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if (lane == 0) { sh.pval = v; sh.pidx = i; }
  }
  __syncthreads();
}

// One determinant per block: gather G_w[occ, :] into the panel, then LU
// with partial pivoting. Writes det_logabs/det_sign for (w, d).
template <bool InShared>
__global__ void gather_lu_block(const double* __restrict__ G, int K, int N,
                                int D, const std::int32_t* __restrict__ occ,
                                double* __restrict__ panel_glob,
                                double* __restrict__ det_logabs,
                                int* __restrict__ det_sign) {
  extern __shared__ double panel_shared[];
  __shared__ ShCtx sh;
  const std::int64_t idx = blockIdx.x;
  const int w = (int)(idx / D), d = (int)(idx % D);
  double* panel =
      InShared ? panel_shared : panel_glob + idx * (size_t)N * N;
  const double* Gw = G + (size_t)w * K * N;
  const std::int32_t* o = occ + (size_t)d * N;
  const int tid = threadIdx.x, B = blockDim.x;

  for (int t = tid; t < N * N; t += B) {
    const int i = t / N, j = t % N;
    panel[(size_t)i * N + j] = Gw[(size_t)o[i] * N + j];
  }
  __syncthreads();

  double logabs = 0.0;
  int sign = 1;
  for (int k = 0; k < N; ++k) {
    double mine = -INFINITY;
    int mini = 0x7fffffff;
    for (int i = k + tid; i < N; i += B) {
      const double v = fabs(panel[(size_t)i * N + k]);
      if (v > mine) { mine = v; mini = i; }
    }
    block_maxloc(mine, mini, sh);
    const int p = sh.pidx;
    const double pv = sh.pval;
    if (p != k) {
      for (int j = tid; j < N; j += B) {
        const double t = panel[(size_t)k * N + j];
        panel[(size_t)k * N + j] = panel[(size_t)p * N + j];
        panel[(size_t)p * N + j] = t;
      }
    }
    if (tid == 0) sh.singular = (pv == 0.0) ? 1 : 0;
    __syncthreads();
    if (sh.singular) {
      if (tid == 0) {
        det_logabs[idx] = kNegInfG;
        det_sign[idx] = 0;
      }
      return;
    }
    const double u = panel[(size_t)k * N + k];
    if (tid == 0) {
      logabs += log(fabs(u));
      if (u < 0.0) sign = -sign;
      if (p != k) sign = -sign;
    }
    // Fused scale + update: each trailing row is owned by one thread, which
    // divides its own pivot-column entry and immediately updates its row —
    // no separate scale pass, one sync per column step total.
    for (int i = k + 1 + tid; i < N; i += B) {
      double* irow = panel + (size_t)i * N;
      const double f = (irow[k] /= u);
      const double* arow = panel + (size_t)k * N;
      for (int j = k + 1; j < N; ++j) irow[j] -= f * arow[j];
    }
    __syncthreads();
  }
  if (tid == 0) {
    det_logabs[idx] = logabs;
    det_sign[idx] = sign;
  }
}

// One determinant per warp, N <= 32, panel in a per-warp shared slice.
// warp_panels: kWarpsPerBlock * N * N doubles of dynamic shared per block.
// batch = W*D; warps with idx >= batch exit (idx is warp-uniform).
__global__ void gather_lu_warp(const double* __restrict__ G, int K, int N,
                               int D, const std::int32_t* __restrict__ occ,
                               std::int64_t batch,
                               double* __restrict__ det_logabs,
                               int* __restrict__ det_sign) {
  extern __shared__ double warp_panels[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const std::int64_t idx =
      (std::int64_t)blockIdx.x * kWarpsPerBlock + warp;
  if (idx >= batch)
    return;
  double* panel = warp_panels + (size_t)warp * N * N;

  const int w = (int)(idx / D), d = (int)(idx % D);
  const double* Gw = G + (size_t)w * K * N;
  const std::int32_t* o = occ + (size_t)d * N;

  for (int t = lane; t < N * N; t += 32) {
    const int i = t / N, j = t % N;
    panel[(size_t)i * N + j] = Gw[(size_t)o[i] * N + j];
  }
  __syncwarp();

  double logabs = 0.0;
  int sign = 1;
  bool singular = false;
  for (int k = 0; k < N; ++k) {
    double mine = -INFINITY;
    int mini = 0x7fffffff;
    for (int i = k + lane; i < N; i += 32) {
      const double v = fabs(panel[(size_t)i * N + k]);
      if (v > mine) { mine = v; mini = i; }
    }
    for (int off = 16; off; off >>= 1) {
      const double ov = __shfl_down_sync(0xffffffffu, mine, off);
      const int oi = __shfl_down_sync(0xffffffffu, mini, off);
      if (ov > mine || (ov == mine && oi < mini)) { mine = ov; mini = oi; }
    }
    const int p = __shfl_sync(0xffffffffu, mini, 0);
    const double pv = __shfl_sync(0xffffffffu, mine, 0);
    if (pv == 0.0) {
      singular = true;
      break;
    }
    if (p != k) {
      for (int j = lane; j < N; j += 32) {
        const double t = panel[(size_t)k * N + j];
        panel[(size_t)k * N + j] = panel[(size_t)p * N + j];
        panel[(size_t)p * N + j] = t;
      }
    }
    __syncwarp();
    const double u = panel[(size_t)k * N + k];
    if (lane == 0) {
      logabs += log(fabs(u));
      if (u < 0.0) sign = -sign;
      if (p != k) sign = -sign;
    }
    // Fused scale + update, one warp-sync per column step (see the block
    // kernel).
    for (int i = k + 1 + lane; i < N; i += 32) {
      double* irow = panel + (size_t)i * N;
      const double f = (irow[k] /= u);
      const double* arow = panel + (size_t)k * N;
      for (int j = k + 1; j < N; ++j) irow[j] -= f * arow[j];
    }
    __syncwarp();
  }
  if (lane == 0) {
    if (singular) {
      det_logabs[idx] = kNegInfG;
      det_sign[idx] = 0;
    } else {
      det_logabs[idx] = logabs;
      det_sign[idx] = sign;
    }
  }
}

// Block-wide sum, all threads call, identity 0.
__device__ inline double block_sum(double v, ShCtx& sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (lane == 0) sh.red[warp] = v;
  __syncthreads();
  if (warp == 0) {
    const int nwarp = (blockDim.x + 31) >> 5;
    v = (lane < nwarp) ? sh.red[lane] : 0.0;
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) sh.red[0] = v;
  }
  __syncthreads();
  const double r = sh.red[0];
  __syncthreads();
  return r;
}

// Block-wide max, all threads call, identity -inf.
__device__ inline double block_max(double v, ShCtx& sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1)
    v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
  if (lane == 0) sh.red[warp] = v;
  __syncthreads();
  if (warp == 0) {
    const int nwarp = (blockDim.x + 31) >> 5;
    v = (lane < nwarp) ? sh.red[lane] : -INFINITY;
    for (int o = 16; o; o >>= 1)
      v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0) sh.red[0] = v;
  }
  __syncthreads();
  const double r = sh.red[0];
  __syncthreads();
  return r;
}

// Signed log-sum-exp over each walker's determinant row (DESIGN.md 2.2):
// w_d = c_d * det_d, log|w_d| = log|c_d| + logabs_d, s_d = sign(c_d)*sign_d.
// A zero-sign determinant (or zero coefficient) contributes nothing; exact
// cancellation (sum == 0) is a zero weight. Block per walker.
__global__ void reduce_walker(const double* __restrict__ coeff,
                              const double* __restrict__ det_logabs,
                              const int* __restrict__ det_sign, int D,
                              double* __restrict__ w_logabs,
                              int* __restrict__ w_sign) {
  __shared__ ShCtx sh;
  const int w = blockIdx.x;
  const double* la = det_logabs + (size_t)w * D;
  const int* sg = det_sign + (size_t)w * D;
  const int tid = threadIdx.x, B = blockDim.x;

  double mymax = -INFINITY;
  for (int d = tid; d < D; d += B) {
    if (sg[d] == 0 || coeff[d] == 0.0) continue;
    mymax = fmax(mymax, log(fabs(coeff[d])) + la[d]);
  }
  const double mx = block_max(mymax, sh);
  if (tid == 0) sh.singular = (mx == -INFINITY) ? 1 : 0;
  __syncthreads();
  if (sh.singular) {
    if (tid == 0) { w_logabs[w] = kNegInfG; w_sign[w] = 0; }
    return;
  }
  double mysum = 0.0;
  for (int d = tid; d < D; d += B) {
    if (sg[d] == 0 || coeff[d] == 0.0) continue;
    const int s = (coeff[d] > 0.0 ? 1 : -1) * sg[d];
    mysum += s * exp(log(fabs(coeff[d])) + la[d] - mx);
  }
  const double sum = block_sum(mysum, sh);
  if (tid == 0) {
    if (sum == 0.0) {
      w_logabs[w] = kNegInfG;
      w_sign[w] = 0;
    } else {
      w_logabs[w] = mx + log(fabs(sum));
      w_sign[w] = sum > 0.0 ? 1 : -1;
    }
  }
}

// Vendor baseline, part 1: standalone gather into contiguous column-major
// panels for cublasDgetrfBatched. One block per panel.
__global__ void gather_panels(const double* __restrict__ G, int K, int N,
                              int D, const std::int32_t* __restrict__ occ,
                              double* __restrict__ panels) {
  const std::int64_t idx = blockIdx.x;
  const int w = (int)(idx / D), d = (int)(idx % D);
  const double* Gw = G + (size_t)w * K * N;
  const std::int32_t* o = occ + (size_t)d * N;
  double* P = panels + idx * (size_t)N * N;
  for (int t = threadIdx.x; t < N * N; t += blockDim.x) {
    const int i = t / N, j = t % N;
    P[(size_t)j * N + i] = Gw[(size_t)o[i] * N + j]; // column-major
  }
}

// Vendor baseline, part 2: log|det|/sign extraction from getrfBatched's
// in-place LU factors (column-major) and 1-based pivot vector.
// One warp per panel; warps with idx >= batch exit (idx is warp-uniform).
__global__ void extract_logdet(const double* __restrict__ panels,
                               const int* __restrict__ pivots,
                               const int* __restrict__ info, int N,
                               std::int64_t batch,
                               double* __restrict__ det_logabs,
                               int* __restrict__ det_sign) {
  const std::int64_t idx = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (idx >= batch)
    return;
  const double* P = panels + idx * (size_t)N * N;
  const int* pv = pivots + idx * N;
  double part = 0.0;
  int mysign = 1;
  for (int i = lane; i < N; i += 32) {
    const double u = P[(size_t)i * N + i];
    if (u == 0.0) mysign = 0;
    else {
      part += log(fabs(u));
      if (u < 0.0) mysign = -mysign;
    }
    if (pv[i] != i + 1) mysign = -mysign;
  }
  // combine: sum of logs, product of signs (0 if any singular)
  for (int o = 16; o; o >>= 1) {
    part += __shfl_down_sync(0xffffffffu, part, o);
    const int os = __shfl_down_sync(0xffffffffu, mysign, o);
    mysign *= os;
  }
  if (lane == 0) {
    if (info[idx] != 0 || mysign == 0) {
      det_logabs[idx] = kNegInfG;
      det_sign[idx] = 0;
    } else {
      det_logabs[idx] = part;
      det_sign[idx] = mysign;
    }
  }
}

// Device-resident case: per-walker G (K x N, row-major, walker-invariant
// occupation lists and coefficients). Upload once, evaluate many times.
struct Case {
  int K = 0, N = 0, D = 0, W = 0;
  double* d_G = nullptr;      // W*K*N
  std::int32_t* d_occ = nullptr; // D*N
  double* d_coeff = nullptr;  // D
  double* d_det_logabs = nullptr; // W*D
  int* d_det_sign = nullptr;      // W*D
  double* d_w_logabs = nullptr;   // W
  int* d_w_sign = nullptr;        // W
  double* d_panels = nullptr;     // W*D*N*N (block kernel, global-panel N;
                                  // vendor baseline all N)
  double* d_lu = nullptr;         // vendor LU in-place
  int* d_pivots = nullptr;        // vendor W*D*N
  int* d_info = nullptr;          // vendor W*D
  double** d_ptrs = nullptr;      // vendor pointer array

  Case(const double* h_G, const std::int32_t* h_occ, const double* h_coeff,
       int K_, int N_, int D_, int W_, bool vendor_buffers)
      : K(K_), N(N_), D(D_), W(W_) {
    const std::int64_t batch = (std::int64_t)W * D;
    cuda_check(cudaMalloc(&d_G, (size_t)W * K * N * sizeof(double)), "G");
    cuda_check(cudaMalloc(&d_occ, (size_t)D * N * sizeof(std::int32_t)),
               "occ");
    cuda_check(cudaMalloc(&d_coeff, D * sizeof(double)), "coeff");
    cuda_check(cudaMalloc(&d_det_logabs, batch * sizeof(double)), "det la");
    cuda_check(cudaMalloc(&d_det_sign, batch * sizeof(int)), "det sign");
    cuda_check(cudaMalloc(&d_w_logabs, W * sizeof(double)), "w la");
    cuda_check(cudaMalloc(&d_w_sign, W * sizeof(int)), "w sign");
    cuda_check(cudaMemcpy(d_G, h_G, (size_t)W * K * N * sizeof(double),
                          cudaMemcpyHostToDevice),
               "G up");
    cuda_check(cudaMemcpy(d_occ, h_occ, (size_t)D * N * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               "occ up");
    cuda_check(cudaMemcpy(d_coeff, h_coeff, D * sizeof(double),
                          cudaMemcpyHostToDevice),
               "coeff up");
    const bool need_panels = vendor_buffers || N > kMaxSharedN;
    if (need_panels) {
      cuda_check(cudaMalloc(&d_panels, (size_t)batch * N * N * sizeof(double)),
                 "panels");
    }
    if (vendor_buffers) {
      cuda_check(cudaMalloc(&d_lu, (size_t)batch * N * N * sizeof(double)),
                 "lu");
      cuda_check(cudaMalloc(&d_pivots, (size_t)batch * N * sizeof(int)),
                 "pivots");
      cuda_check(cudaMalloc(&d_info, batch * sizeof(int)), "info");
      std::vector<double*> h_ptrs(batch);
      for (std::int64_t i = 0; i < batch; ++i)
        h_ptrs[i] = d_lu + (size_t)i * N * N;
      cuda_check(cudaMalloc(&d_ptrs, batch * sizeof(double*)), "ptrs");
      cuda_check(cudaMemcpy(d_ptrs, h_ptrs.data(), batch * sizeof(double*),
                            cudaMemcpyHostToDevice),
                 "ptrs up");
    }
  }
  Case(const Case&) = delete;
  Case& operator=(const Case&) = delete;
  ~Case() {
    cudaFree(d_G);
    cudaFree(d_occ);
    cudaFree(d_coeff);
    cudaFree(d_det_logabs);
    cudaFree(d_det_sign);
    cudaFree(d_w_logabs);
    cudaFree(d_w_sign);
    cudaFree(d_panels);
    cudaFree(d_lu);
    cudaFree(d_pivots);
    cudaFree(d_info);
    cudaFree(d_ptrs);
  }

  bool use_warp_path() const { return N <= 32; }
  bool use_shared_panel() const { return N <= kMaxSharedN; }

  // Fused gather + LU determinant table. force_block forces the block
  // kernel at N <= 32 (gate coverage for both kernels).
  void run_det(bool force_block = false) {
    const std::int64_t batch = (std::int64_t)W * D;
    if (!force_block && use_warp_path()) {
      const int blocks = (int)((batch + kWarpsPerBlock - 1) / kWarpsPerBlock);
      const size_t smem = (size_t)kWarpsPerBlock * N * N * sizeof(double);
      gather_lu_warp<<<blocks, kBlockThreads, smem>>>(
          d_G, K, N, D, d_occ, batch, d_det_logabs, d_det_sign);
    } else if (use_shared_panel()) {
      const size_t smem = (size_t)N * N * sizeof(double);
      static bool configured = false;
      if (!configured) {
        cuda_check(cudaFuncSetAttribute(
                       gather_lu_block<true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)((size_t)kMaxSharedN * kMaxSharedN * 8)),
                   "smem attr");
        configured = true;
      }
      gather_lu_block<true><<<(unsigned)batch, kBlockThreads, smem>>>(
          d_G, K, N, D, d_occ, nullptr, d_det_logabs, d_det_sign);
    } else {
      gather_lu_block<false><<<(unsigned)batch, kBlockThreads>>>(
          d_G, K, N, D, d_occ, d_panels, d_det_logabs, d_det_sign);
    }
    cuda_check(cudaGetLastError(), "gather_lu launch");
    cuda_check(cudaDeviceSynchronize(), "gather_lu");
  }

  void run_reduce() {
    reduce_walker<<<W, kBlockThreads>>>(d_coeff, d_det_logabs, d_det_sign, D,
                                        d_w_logabs, d_w_sign);
    cuda_check(cudaGetLastError(), "reduce launch");
    cuda_check(cudaDeviceSynchronize(), "reduce");
  }

  void download(std::vector<double>& det_la, std::vector<int>& det_sg,
                std::vector<double>& w_la, std::vector<int>& w_sg) const {
    const std::int64_t batch = (std::int64_t)W * D;
    det_la.resize(batch);
    det_sg.resize(batch);
    w_la.resize(W);
    w_sg.resize(W);
    cuda_check(cudaMemcpy(det_la.data(), d_det_logabs,
                          batch * sizeof(double), cudaMemcpyDeviceToHost),
               "det la down");
    cuda_check(cudaMemcpy(det_sg.data(), d_det_sign, batch * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "det sign down");
    cuda_check(cudaMemcpy(w_la.data(), d_w_logabs, W * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "w la down");
    cuda_check(cudaMemcpy(w_sg.data(), d_w_sign, W * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "w sign down");
  }
};

} // namespace gpu
} // namespace msd
