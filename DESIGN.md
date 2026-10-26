# DESIGN.md — afqmc-determinants

Design note for a batched multi-Slater overlap-determinant kernel. Rung 1
(the section 8 CPU reference and gate harness) is built; this note remains
the design record. It decides the batching structure, audits the local LDR
code for reuse, defines what a GPU prototype would measure, and pre-writes
the gates and kill criteria.

## 1. Problem

Trial wavefunction: |Ψ_T> = Σ_{d=1..D} c_d |D_d>, where each |D_d> is a
Slater determinant of N orbitals drawn from a common pool of K orbitals in
an M-dimensional single-particle basis (M ≥ K ≥ N) — the shared-pool
multi-Slater pattern behind recent GPU AFQMC implementations. Walker w
carries its own M×N orbital matrix W_w. The overlap is

```
<D_d|w> = det(O_{w,d}),   O_{w,d} = Φ_{I_d}^† W_w,
```

where Φ is the M×K pool matrix and I_d ⊂ {1..K}, |I_d| = N, is the
occupation list of determinant d. The QMC weight of walker w is
Σ_d c_d det(O_{w,d}), per spin sector; sectors factor, so one sector is
treated below.

Sizes this note designs for (estimates from multi-Slater AFQMC usage, to
be pinned by the physics campaign before any GPU work):

- M (basis): 10²–10⁴
- K (pool): 128–4096
- N (matrix order = electrons per sector): 16–256
- D (determinants): 10–3·10⁴
- W (walkers): 10²–10⁴

## 2. Sharing structure

Because every determinant draws from the same pool,

```
O_{w,d} = G_w[I_d, :],   G_w = Φ^† W_w   (K×N),
```

i.e. each overlap submatrix is a row gather of a per-walker matrix that is
computed once. The work splits into:

- Per walker, once: G_w = Φ^† W_w, a (K×M)·(M×N) product, 2·K·M·N flops.
  Across walkers this is the single tall GEMM Φ^†·[W_1 … W_W], which runs
  at full vendor GEMM efficiency and is not the kernel problem.
- Per (w,d): gather N rows of G_w (N² elements) and factor the N×N panel.

### 2.1 Where the reuse factor comes from

Without pooling, determinant d carries its own M×N orbital matrix and the
overlap costs 2·M·N² flops to construct before any factorization. Pooling
divides the construction cost per (w,d) by

```
(2·M·N²·D) / (2·K·M·N) = D·N/K.
```

The same number falls out in row terms: the batch touches D·N row slots of
G_w, but only K distinct rows exist, so each pool row is reused D·N/K
times on average. That is a counting fact, not an estimate. Representative
points:

|      D |   N |    K | D·N/K |
|-------:|----:|-----:|------:|
|    10² |  64 |  256 |    25 |
|    10³ | 100 |  500 |   200 |
|    10⁴ | 100 |  500 |  2000 |
|  3·10⁴ | 200 | 2000 |  3000 |

The reuse materializes as cache residency: distinct gather-source data per
walker is 8·K·N bytes (400 KB at K=500, N=100, FP64), which fits in L2 on
current parts, so each row is served from L2 ~10²–10³ times instead of
from HBM once per determinant.

Two further sharings, both exact:

- The occupation data {I_d} and coefficients {c_d} are walker-invariant:
  D·N indices (4 MB at D=10⁴, N=100, int32) are read once for the whole
  population, and the gather pattern is identical for every walker.
- Across spin sectors nothing is shared; sectors are two independent
  batches.

What is not shared: the determinant values. O_{w,d} differs for every
(w,d) by construction, so the (2/3)N³ factorization work per pair is
irreducible within this algorithm. At W=10³, D=10⁴, N=100 that is
(2/3)·10⁶·10⁷ ≈ 7·10¹² FP64 flops per propagation step (estimate; the
regime where Tier 2 in §3 matters).

### 2.2 The output is a reduction, not a table

The walker weight needs log|Σ_d c_d det(O_{w,d})| and its phase, not the
W×D determinant table. A fused kernel accumulates the weighted sum with a
running max-shift (log-sum-exp form) and writes W log-weights instead of
W·D determinants — at the largest §2.1 point that avoids 10⁷ × 16 B ≈
160 MB of determinant writes per step. Per-determinant values are still
needed downstream for local-energy and force-bias ratios, so the design
keeps both output modes.

## 3. Two algorithm tiers

Tier 1 — gather + batched factorization. Assemble O_{w,d} by row gather,
factor by LU with partial pivoting, take det = (−1)^p Π U_ii with the
log/sign accumulation of §5.2. One thread block per (w,d) while the FP64
panel fits in shared memory (80 KB at N=100 fits on A100/H100-class parts;
512 KB at N=256 does not, so large N needs a tiled/global-memory path),
warp-per-determinant at small N. This is the tier that races
cublasDgetrfBatched.

Tier 2 — excitation-tree ratios, only if Tier 1's N³ work dominates the
campaign. Organize the CI expansion as excitations off a reference
determinant I_0. Per walker, one factorization of O_{w,0} plus one product
P_w = G_w·O_{w,0}^{-1} (a K×N by N×N GEMM, O(K·N²)) turns every single
excitation ratio into O(1) table entries of P_w and every double into a
2×2 determinant — the standard table method. Per-(w,d) work drops from
(2/3)N³ to O(e²) at excitation level e, a ~10²–10³× reduction at the §2
sizes (estimate). Costs: the reference must be re-chosen per walker
(det(O_{w,0}) ≈ 0 poisons every ratio), errors compound relative to the
reference, and each walker carries inverse state. Tier 2 is outside the
first rung; it is recorded here because the §5.2 stability gate was
written to admit it.

## 4. Reuse audit: ldr.hpp in attractive-hubbard-dqmc

File under audit: `include/dqmc/ldr.hpp` in
[attractive-hubbard-dqmc](https://github.com/FireHorseAutumn/attractive-hubbard-dqmc)
(64 lines, header-only, Eigen, real FP64). Read in full for this note.

What it implements. `Ldr` is a factorization A = L·diag(d)·R with L
orthogonal, d positive (the absolute diagonal of R from a full-pivoting
Householder QR), and R upper triangular with unit-magnitude diagonal;
|det A| = Π d_i falls out for free. `prepend_ldr` re-factorizes
b·(L·diag(d)·R) after each new slice factor b — the incremental form used
to keep long imaginary-time products B_l···B_1 permanently decomposed,
with a fresh recompute each slice (the solver's recompute-each-slice
path). `inverse_identity_plus` and `inverse_inverse_plus` build (I+A)^{-1}
and (U^{-1}+V^{-1})^{-1} by splitting d into d_min = min(d,1) and
d_max = max(d,1) so that only matrices with O(1)-bounded entries are ever
inverted (via `pivoted_inverse`, a full-pivot LU inverse) — the standard
stable Green-function construction, here in three Eigen expressions.

Transfers:

1. The magnitude-separation discipline: a log-magnitude carried as a sum
   of logs of positive numbers, unit-magnitude structure carried
   separately, products never formed. The log|det| accumulator in §5.2 is
   exactly this, and the d_min/d_max split is the same move applied to
   inversion. The idea transfers; the code is three lines and gets
   reimplemented in place, not imported.
2. The re-factorization rule: long chains get re-decomposed on a schedule
   (per slice, in DQMC) instead of being updated forever. Tier 2's ratio
   tree needs exactly this — periodic re-factorization of each walker's
   reference overlap — and the design adopts the rule.
3. Sign accountability: LDR pushes all sign into det(L)·det(R) and the
   column permutation and never extracts it; extracting it from the
   factors is O(N). The kernel's sign/phase accumulator (§5.2) is the
   same extraction done on LU factors instead.

Does not transfer:

- The problem ldr.hpp solves — stabilizing products of ~10²–10³ slice
  matrices and the Green-function inversions built from them — has no
  analog here: each (w,d) overlap is a single N×N matrix with no product
  structure. `prepend_ldr`, `inverse_identity_plus`, and
  `inverse_inverse_plus` have no role in this kernel.
- No code is lifted as-is: it is real FP64 only (weights here need sign at
  minimum, phase for complex Hamiltonians), host-side, one matrix at a
  time, with no batching and no gather indexing.

Gate marker: this code sits inside the DQMC repo's DQMC-3 correctness
gates (checkpoint/restart, production data path, finite-size/statistical/
Trotter analysis). Any reuse of the actual code
— as opposed to the ideas above — waits for those gates to clear. Reuse is
recorded as LATER, not now.

## 5. What a GPU prototype measures

### 5.1 Measurements

- Baseline: cublasDgetrfBatched on pre-gathered panels, with the gather
  kernel timed separately and included in the end-to-end number; MAGMA
  batched getrf as a second reference where available.
- Metrics: determinants/sec and full-panel step time versus baseline, over
  batch W·D ∈ {10² … 10⁶} and N ∈ {16, 32, 64, 128, 256}, at fixed
  K/N ≈ 5 and D/N ≈ 100 unless stated. Medians with observed spreads, raw
  per-repetition timings kept, machine state recorded with the run.
- Roofline: FP64 flops ((2/3)N³ per determinant) against the part's
  measured FP64 rate; gather bytes against measured L2 bandwidth. FP64
  rates differ by orders of magnitude across parts, so every conclusion
  names the part it was measured on.
- Output modes: the full determinant table and the fused per-walker
  log-weight reduction (§2.2), timed separately.

### 5.2 Correctness and the FP64 stability gate (pre-written)

Outputs per (w,d): log|det|, and sign (real) or phase (complex). From LU
with partial pivoting, det = (−1)^p Π U_ii, so

```
log|det| = Σ log|U_ii|,   sign = (−1)^p · Π sign(U_ii),
```

and for complex inputs the phase is carried as a running unit-modulus
product Π (U_ii/|U_ii|) rather than a sum of atan2 values, so branch cuts
never accumulate. log|det| comes from the sum, never from the log of a
product: at N=256 with |U_ii| spanning 10⁻³–10³ the product underflows
FP64 while the sum stays O(10²). This is the §4 magnitude-separation
discipline applied to LU factors.

Reference: CPU Eigen PartialPivLU in FP64, plus an arbitrary-precision
check (mpmath, 100 digits) at N ≤ 64 to separate reference error from
kernel error.

Ensembles per N:

- (a) random dense with prescribed singular spectra, condition number κ
  swept 10⁰–10¹²;
- (b) AFQMC-shaped near-orthogonality: walker/det pairs with |det| down to
  ~10⁻²⁸⁰ (log|det| ≈ −645) — the product underflows, the log must not;
- (c) exactly singular pairs, which must report a zero weight, never
  NaN/Inf.

Pass criteria, fixed in advance:

1. |Δ log|det|| ≤ C·N·ε·κ(A) with ε = 2⁻⁵³ and one constant C ≤ 10²
   covering the whole κ sweep and all ensembles; if no such C exists, the
   gate fails.
2. Sign exact on every nonsingular input. Phase error within criterion 1
   in radians, enforced only where |det| exceeds a stated floor relative
   to the panel maximum — below it the weight is negligible and the phase
   is meaningless; saying so is part of the gate.
3. log|det| finite and within criterion 1 for every nonsingular input,
   including the underflow ensemble (b).
4. Singular inputs detected as zero weight; no NaN/Inf anywhere.
5. Bit-stability: fixed seeds and fixed reduction orders give
   bitwise-identical outputs across runs, and CPU-reference reruns are
   bitwise; GPU-vs-CPU agreement is tolerance-based under criterion 1, not
   bitwise.

This gate is where the idea lives or dies: a kernel that is fast but fails
criteria 1–4 at N ≥ 128 produces wrong QMC weights, and no timing number
is reported for any configuration that has not passed the gate.

## 6. Scope

This is a batched-determinant kernel with an AFQMC-shaped workload,
labeled as such. It is not an AFQMC framework: no Hubbard–Stratonovich
propagation, no population control, no importance-sampling dynamics, no
constraint. Trial coefficients and occupation lists are inputs; there is
no trial-wavefunction optimization. Validation is against CPU linear
algebra; there are no physics results.

## 7. Gates and kill criteria

- Gate 0: CPU-reference equality per §5.2 (bitwise where reduction order
  is fixed, criterion-1 tolerance otherwise) at small sizes before any
  timing run exists.
- Gate 1: the §5.2 stability suite passes at every (N, batch) point before
  any speedup is quoted at that point.
- Kill 1: gather + cublasDgetrfBatched within 20% of the part's measured
  FP64 roofline at the AFQMC-relevant sizes → there is nothing to win;
  record the numbers and close.
- Kill 2: the custom kernel meets the stability gate only at vendor-like
  speed (e.g. by reverting to the same pivoting and blocking) → no delta,
  close.
- Kill 3: at the campaign's real (W, D, N) the total factorization work is
  small enough that the existing host path already meets the step-time
  budget → a GPU kernel is not justified; close. Expected to be false for
  D ≥ 10³, N ≥ 64 (estimate from §2.1); checked with real campaign numbers
  before any GPU work starts.

## 8. First rung, if code starts

Rung 1 (CPU; ~1 week, estimate): occupation-list and pool data structures;
the G_w GEMM via Eigen; per-(w,d) gather plus PartialPivLU with the §5.2
log/sign accumulator; the mpmath validation harness at N ≤ 64; the full
κ-sweep / near-orthogonality / singularity suite. Deliverable: a validated
host reference plus the gate harness every later kernel must pass. No LDR
code import — the magnitude-separation discipline is reimplemented in
place; DQMC-3 remains the blocker for any import.

Rung 2 (GPU; ~1–2 weeks, estimate): fused gather + batched LU with
pivot-parity sign and log accumulation; block-per-determinant with
shared-memory panels at N ≤ 128, warp-per-determinant at N ≤ 32, a tiled
path above; benchmark per §5.1.

Rung 3 (decision; days): roofline comparison against Gate 1 and the kill
criteria → adopt, fall back to vendor, or close.

## 9. Rung-2 outcome (2026-10-25, RTX 5060 Laptop)

Built as designed, with two measured corrections to the plan:

- The "shared panels at N ≤ 128" line assumed A100/H100-class shared
  memory (228 KB); this consumer part allows ~98 KB of dynamic shared per
  block, so the shared-panel path covers N ≤ 112 and N = 128 runs the
  global-panel block path.
- Fusing the scale pass into the trailing-row update (one block sync per
  column step instead of two) is the only fast-path deviation; the gate
  was re-run after it.

Gate: the mpmath harness passes on the GPU driver unchanged (the driver
speaks the rung-1 case-file protocol) — 210 checks, observed max C = 32.000
identical to the host path, on both the default kernel selection and with
the block kernel forced. Kill 1 verdict: NOT triggered — the vendor
baseline reaches only ~8–33% of this part's measured DGEMM FP64 rate, so
headroom over `cublasDgetrfBatched` exists in principle. But the fused
kernel as designed does not capture it: vendor/ours end-to-end ratios run
0.2–1.05x across the sweep (one narrow parity cell at N = 32,
W·D = 204800, where the vendor's separate gather/extract overhead
dominates at small N; worst ~4.5x behind at N = 128 on the global-panel
path). The bottleneck is the per-column-step block synchronization latency
of the straightforward per-panel LU; closing the gap means a MAGMA-style
blocked batched LU, which is Kill 2 territory (vendor's own algorithm
class — no delta). Rung-3 decision: fall back to vendor for production;
the fused kernel stays as the gated reference and the gather/extract
kernels remain useful either way. Any future attempt starts from a blocked
design, not from tuning this one. Machine-state caveat: three same-evening
runs drifted up to ~2x in absolute time (thermally soaked laptop; the
measured DGEMM rate moved 254 -> 119 GFLOP/s across runs) and the
pre/post-fusion comparison is confounded by it; ratios are stabler than
absolutes, and a pinned rerun is scheduled with the portfolio measurement
day.

Dependencies: Rung 1 has none beyond Eigen and mpmath. Any LDR code reuse
depends on DQMC-3 clearing. Complex-valued support depends on the physics
target; real arithmetic suffices for the Hubbard cases adjacent to the
DQMC repo.

Open questions recorded here:

- Whether the campaign's CI expansions pool as cleanly as assumed
  (K ≪ D·N). The §2.1 reuse factor is conditional on that; expansions
  built over one orbital set satisfy it by construction, others may not.
- Whether phase is needed, not just sign: real Hamiltonians give real
  overlaps; flux or frustrated targets do not.
- Actual cublasDgetrfBatched efficiency at FP64 for N ∈ [16,128] on the
  measurement part. Published batched-LU results suggest headroom at small
  N, but whether that holds is Kill 1's measurement, not an assumption.
