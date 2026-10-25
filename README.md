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
NaN/Inf, bitwise-identical reruns. Rung 2 (GPU fused gather + batched LU)
has not started.
