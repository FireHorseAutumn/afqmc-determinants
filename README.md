# afqmc-determinants

Batched evaluation of many small determinants of overlapping submatrices
across multi-Slater AFQMC walkers, exploiting shared occupation structure.

## Vendor gap

Determinant evaluation with combinatorial indexing is not dense linear
algebra; no vendor determinant kernel exists.

## Physics payload

Strongly correlated pairing observables; the multi-Slater AFQMC pattern used
for unconventional-superconductivity calculations. Adjacent to the
determinant engine in
[attractive-hubbard-dqmc](https://github.com/FireHorseAutumn/attractive-hubbard-dqmc).

## LDR reuse

The determinant/LDR machinery in the DQMC repo
(`include/dqmc/ldr.hpp`) is still inside its own DQMC-3 correctness gates.
Reuse becomes available after those gates clear, not before; rung 1
reimplements the magnitude-separation discipline in place.

## Design note

[`DESIGN.md`](DESIGN.md): the batching/overlap structure with the index
math and reuse factor, the audit of what transfers from the LDR code, the
prototype measurement plan, and the pre-written gates and kill criteria.

## Status

Rung 1 (CPU reference + gate harness, `DESIGN.md` §8) landed 2026-10-25:
`include/msd/overlap.hpp` implements the pool GEMM, row gather, LU
log|det|/sign accumulator, and signed log-sum-exp walker reduction;
`src/msdet_ref.cpp` is the case-file driver; `tests/validate.py` is the
100-digit mpmath gate harness. The §5.2 gate passes at every tested point:
210 determinant checks across the κ = 1e0..1e12 sweep, the underflow
ensembles (ln|det| = -1000 flat and -800 spread, below the FP64 product
range), exact-singularity detection, and the end-to-end pool path, with
observed max C = 32 against the C ≤ 100 bound, zero sign mismatches, zero
NaN/Inf, bitwise-identical reruns.

Rung 2 (GPU) landed the same day: `include/msd/overlap_gpu.cuh` — fused
gather + batched LU, warp-per-determinant at N ≤ 32, block-per-determinant
with shared panels at N ≤ 112 (this part's ~98 KB dynamic-shared ceiling;
the design note's N ≤ 128 assumed A100/H100-class shared memory),
global-panel block path above; plus the signed log-sum-exp walker-reduction
kernel and the vendor-baseline pieces (standalone gather,
`cublasDgetrfBatched`, log/sign extraction). `src/msdet_gpu.cu` speaks the
rung-1 case-file protocol, so the mpmath gate runs against the GPU path
unchanged — it passes on the default kernel selection and with the block
kernel forced, at the identical observed max C = 32.000. CTest 4/4, zero
build warnings.

Measured (`benchmarks/bench_det.cu`, `data/afqmc_det_bench.json`; RTX 5060
Laptop, K = 5N, D = 100N, median of 11 event-timed reps, raw reps in the
JSON; governor-pinned idle-machine rerun 2026-10-30, measured DGEMM FP64
rate 252 GFLOP/s): the fused kernel loses to the vendor baseline
(gather + `cublasDgetrfBatched` + extract) end-to-end at every measured
point — vendor/ours ratios 0.26–0.48x across N = 16–128, W·D =
1.6e3–8.2e5, worst 3.8x behind at N = 128 on the global-panel path. Kill 1
did NOT trigger, but the margin narrowed: pinned, the vendor reaches
14–61% of the measured DGEMM FP64 rate (the unpinned evening runs showed
8–33%), so headroom exists in principle but is smaller than the drifted
numbers suggested. The straightforward per-panel LU does not capture it;
the bottleneck is per-column-step block synchronization latency, and
closing it means a blocked batched LU (the vendor's own algorithm class;
DESIGN.md §9 records the rung-3 decision: fall back to vendor, keep the
fused kernel as the gated reference). The pinned rerun also resolved the
one apparent parity cell: the unpinned runs showed vendor/ours = 1.05x at
N = 32, W·D = 204800, but under pinning that cell is 0.48x — the parity
was a thermal artifact (the vendor's getrf throttled ~4–5x under soak
while the sync-latency-bound fused kernel barely moved; that asymmetry is
itself evidence for the latency diagnosis). N = 256 is unmeasured: the
vendor path alone needs 27 GB of panel buffers at W = 1 on this 8 GB part.

Max pass (2026-10-28): `extract_logdet` (the vendor-baseline extraction
kernel) had no batch-tail guard — its warp-per-panel grid
(ceil(batch / 4) blocks of kWarpsPerBlock = 4 warps) left tail warps with
idx ≥ batch reading panels/pivots/info and writing det_logabs/det_sign
out of bounds whenever batch was not a multiple of 4. Latent in-repo: the
benchmark grid has D = 100N, so batch is always a multiple of 4. Fixed
with the same `idx >= batch` guard `gather_lu_warp` already carries, and
pinned by `tests/test_gpu_guard.cu` (sentinel-padded outputs for both
warp-per-item kernels, red before the fix). CTest 4/4 → 5/5.
