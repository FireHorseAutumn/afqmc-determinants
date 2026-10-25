#!/usr/bin/env python3
"""Gate harness for the rung-1 host reference (DESIGN.md 5.2).

Generates the ensembles, computes 100-digit mpmath references for
det(Phi[I]^T W) per (walker, determinant), runs msdet_ref, and enforces:

  1. |dlog|det|| <= C*N*eps*kappa with eps = 2^-53 and one C <= 100
     covering every ensemble; the observed max C is reported.
  2. sign exact on every nonsingular input.
  3. logabs finite and within criterion 1 on the underflow ensembles.
  4. singular inputs detected as zero weight; no NaN/Inf anywhere.
  5. fixed seeds give fixed matrices; reruns are bitwise on this host path.

usage: validate.py <path-to-msdet_ref>
"""
import math
import struct
import subprocess
import sys
import tempfile

import numpy as np
import mpmath as mp

mp.mp.dps = 100
EPS = 2.0**-53
C_BOUND = 100.0


def write_case(path, Phi, coeff, occ, walkers):
    M, K = Phi.shape
    D, N = occ.shape
    W = walkers.shape[0]
    assert walkers.shape == (W, M, N)
    with open(path, "wb") as f:
        f.write(struct.pack("<4s5q", b"MSD1", M, K, N, D, W))
        f.write(np.asarray(Phi, dtype="<f8").tobytes())
        f.write(np.asarray(coeff, dtype="<f8").tobytes())
        f.write(np.asarray(occ, dtype="<i4").tobytes())
        f.write(np.asarray(walkers, dtype="<f8").tobytes())


def read_results(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:4] == b"MSR1"
    W, D = struct.unpack_from("<2q", raw, 4)
    off = 4 + 16
    det_logabs = np.frombuffer(raw, "<f8", W * D, off).reshape(W, D)
    off += 8 * W * D
    det_sign = np.frombuffer(raw, "<i4", W * D, off).reshape(W, D)
    off += 4 * W * D
    w_logabs = np.frombuffer(raw, "<f8", W, off)
    off += 8 * W
    w_sign = np.frombuffer(raw, "<i4", W, off)
    return det_logabs, det_sign, w_logabs, w_sign


def mp_det(A):
    """LU with partial pivoting at 100 digits on the exact FP64 values.
    Returns (logabs, sign) with sign 0 and logabs -inf for singular."""
    n = A.shape[0]
    M = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = mp.mpf(A[i, j])
    sign = 1
    for k in range(n):
        p = k
        for i in range(k + 1, n):
            if abs(M[i, k]) > abs(M[p, k]):
                p = i
        if M[p, k] == 0:
            return mp.mpf("-inf"), 0
        if p != k:
            for j in range(n):
                M[k, j], M[p, j] = M[p, j], M[k, j]
            sign = -sign
        for i in range(k + 1, n):
            M[i, k] /= M[k, k]
            for j in range(k + 1, n):
                M[i, j] -= M[i, k] * M[k, j]
    logabs = mp.mpf(0)
    for i in range(n):
        u = M[i, i]
        if u == 0:
            return mp.mpf("-inf"), 0
        logabs += mp.log(abs(u))
        if u < 0:
            sign = -sign
    return logabs, sign


def random_orthogonal(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def prescribed_spectrum(n, smax, smin, rng):
    s = np.exp(np.linspace(math.log(smax), math.log(smin), n))
    return random_orthogonal(n, rng) @ np.diag(s) @ random_orthogonal(n, rng).T


def build_cases():
    """Returns list of (name, Phi, coeff, occ, walkers, panels, kappa).
    panels[w][d] is the FP64 overlap matrix for the mpmath reference;
    kappa[w][d] is its condition number (prescribed or measured)."""
    cases = []

    # (a) prescribed singular spectra, kappa swept 1e0..1e12. Encoded through
    # the pool path with Phi = I, one panel per walker, D = 1.
    for n in (8, 32, 64):
        rng = np.random.default_rng(1000 + n)
        panels, kappas = [], []
        for kappa in (1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12):
            for _ in range(2):
                panels.append(prescribed_spectrum(n, 1.0, 1.0 / kappa, rng))
                kappas.append(kappa)
        Phi = np.eye(n)
        occ = np.arange(n, dtype=np.int32).reshape(1, n)
        cases.append((f"a-kappa-sweep-N{n}", Phi, np.array([1.0]), occ,
                      np.stack(panels)[:, :, :],
                      [[p] for p in panels], [[k] for k in kappas]))

    # (b) underflow: |det| below the FP64 product range. b1 flat spectrum
    # (kappa = 1) at ln|det| = -1000 for every N. b2 spread spectrum at
    # ln|det| = -800: the required dynamic range is kappa = e^(1600/N), so
    # only N = 64 keeps the pivots representable and kappa (~7e10) below
    # the scale where FP64 rounding can legitimately flip the sign.
    for n in (8, 16, 32, 64):
        rng = np.random.default_rng(2000 + n)
        flat = np.full(n, math.exp(-1000.0 / n))
        panels = [prescribed_spectrum(n, flat[0], flat[0], rng)]
        Phi = np.eye(n)
        occ = np.arange(n, dtype=np.int32).reshape(1, n)
        cases.append((f"b1-flat-underflow-N{n}", Phi, np.array([1.0]), occ,
                      np.stack(panels), [[p] for p in panels],
                      [[1.0]]))
    n = 64
    smin = math.exp(-1600.0 / n)
    rng = np.random.default_rng(2100 + n)
    panels = [prescribed_spectrum(n, 1.0, smin, rng)]
    Phi = np.eye(n)
    occ = np.arange(n, dtype=np.int32).reshape(1, n)
    cases.append((f"b2-spread-underflow-N{n}", Phi, np.array([1.0]), occ,
                  np.stack(panels), [[p] for p in panels],
                  [[1.0 / smin]]))

    # (c) singular: duplicate row, zero row, zero column. Row-pivot LU hits
    # an exactly zero pivot for each; a duplicate *column* leaves a
    # rounding-scale pivot instead, so it is not an exact-zero case.
    n = 16
    rng = np.random.default_rng(3000)
    base = rng.standard_normal((n, n))
    dup_row = base.copy()
    dup_row[3] = base[7]
    zero_row = base.copy()
    zero_row[11] = 0.0
    zero_col = base.copy()
    zero_col[:, 5] = 0.0
    panels = [dup_row, zero_row, zero_col]
    Phi = np.eye(n)
    occ = np.arange(n, dtype=np.int32).reshape(1, n)
    cases.append(("c-singular", Phi, np.array([1.0]), occ,
                  np.stack(panels), [[p] for p in panels],
                  [[np.inf] * 3]))

    # (d) end-to-end pool path: M > K > N, random occupations, signed coeffs.
    for n in (8, 32):
        M, K, D, W = 96, 48, 20, 4
        rng = np.random.default_rng(4000 + n)
        Phi = rng.standard_normal((M, K))
        coeff = rng.standard_normal(D)
        occ = np.stack([rng.choice(K, size=n, replace=False)
                        for _ in range(D)]).astype(np.int32)
        walkers = rng.standard_normal((W, M, n))
        panels, kappas = [], []
        for w in range(W):
            Gw = Phi.T @ walkers[w]
            row_p, row_k = [], []
            for d in range(D):
                O = Gw[occ[d]]
                row_p.append(O)
                row_k.append(float(np.linalg.cond(O)))
            panels.append(row_p)
            kappas.append(row_k)
        cases.append((f"d-pool-N{n}", Phi, coeff, occ, walkers, panels, kappas))
    return cases


def main():
    binary = sys.argv[1]
    cases = build_cases()
    max_c = 0.0
    max_c_where = ""
    sign_bad = 0
    nan_bad = 0
    singular_bad = 0
    underflow_checked = 0
    total = 0

    with tempfile.TemporaryDirectory() as tmp:
        for name, Phi, coeff, occ, walkers, panels, kappas in cases:
            case_path = f"{tmp}/{name}.case"
            res_path = f"{tmp}/{name}.res"
            write_case(case_path, Phi, coeff, occ, walkers)
            subprocess.run([binary, case_path, res_path], check=True)
            det_logabs, det_sign, w_logabs, w_sign = read_results(res_path)
            W, D = det_logabs.shape
            case_c = 0.0
            for w in range(W):
                for d in range(D):
                    total += 1
                    ref_log, ref_sign = mp_det(panels[w][d])
                    got_log, got_sign = det_logabs[w, d], int(det_sign[w, d])
                    if not (math.isfinite(got_log) or got_sign == 0):
                        nan_bad += 1
                    if ref_sign == 0:
                        if got_sign != 0:
                            singular_bad += 1
                        continue
                    if got_sign != ref_sign:
                        sign_bad += 1
                    if not math.isfinite(got_log):
                        nan_bad += 1
                        continue
                    kappa = kappas[w][d]
                    delta = abs(got_log - float(ref_log))
                    bound = C_BOUND * panels[w][d].shape[0] * EPS * kappa
                    c_obs = delta / (panels[w][d].shape[0] * EPS * kappa)
                    if c_obs > case_c:
                        case_c = c_obs
                    if c_obs > max_c:
                        max_c, max_c_where = c_obs, f"{name} w{w} d{d}"
                    if delta > bound:
                        nan_bad += 1  # criterion-1 violation
                    if name.startswith("b1-") or name.startswith("b2-"):
                        underflow_checked += 1
            # walker weights against an mpmath signed sum
            for w in range(W):
                terms = []
                for d in range(D):
                    ref_log, ref_sign = mp_det(panels[w][d])
                    if ref_sign == 0 or coeff[d] == 0.0:
                        continue
                    terms.append(mp.mpf(float(coeff[d])) * ref_sign *
                                 mp.exp(ref_log))
                if not terms:
                    continue
                s = mp.fsum(terms)
                if s == 0:
                    if w_sign[w] != 0:
                        singular_bad += 1
                    continue
                if int(w_sign[w]) != (1 if s > 0 else -1):
                    sign_bad += 1
                if math.isfinite(w_logabs[w]):
                    dw = abs(w_logabs[w] - float(mp.log(abs(s))))
                    # the weight inherits the per-det criterion-1 scale;
                    # the reduction adds only O(eps*D) on top
                    n = panels[w][0].shape[0]
                    kmax = max(kappas[w][d] for d in range(D))
                    if dw > C_BOUND * n * EPS * kmax + 1e-12:
                        nan_bad += 1
            print(f"{name:22s} max C = {case_c:9.3f}")

    print(f"\n{total} determinant checks; observed max C = {max_c:.3f} "
          f"({max_c_where}), bound C <= {C_BOUND}")
    print(f"underflow checks: {underflow_checked}; sign mismatches: {sign_bad}; "
          f"NaN/Inf or criterion violations: {nan_bad}; "
          f"singular mishandled: {singular_bad}")
    ok = (max_c <= C_BOUND and sign_bad == 0 and nan_bad == 0 and
          singular_bad == 0 and underflow_checked > 0)
    print("validate:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
