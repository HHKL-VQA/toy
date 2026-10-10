"""Target evolution V = exp(-i H t) by Chebyshev expansion (PLAN.md §4.3).

    H_s = (H - c) / s,
    exp(-iHt) = exp(-ict) [ J_0(st) + 2 sum_{k>=1} (-i)^k J_k(st) T_k(H_s) ]

with c, s from the Lanczos spectrum ends and a 2 % margin on s.  V^dagger flips
(-i)^k -> (+i)^k and exp(-ict) -> exp(+ict).

The Lanczos routine here is also the one certify.py uses.
"""

import math

import numpy as np
import torch
from scipy.linalg import eigvalsh_tridiagonal
from scipy.special import jv

from . import sv


def lanczos(matvec, v0, max_iter=300, tol=1e-4, window=10, on_iter=None):
    """Extreme eigenvalues of a Hermitian operator: 3-vector Lanczos, no reorthogonalisation.

    Ritz values approach the true extremes from inside.  Stops when both
    extreme Ritz values moved by less than tol * (hi - lo) over the last
    `window` iterations, on an invariant subspace, or at max_iter.  Measuring
    the change against the spread (not against |lo|, |hi|) keeps the test
    equally strict wherever the spectrum is centred.
    `matvec(v)` must return a new tensor.  `on_iter(k, lo, hi)` is called
    after each step.  v0 is normalised in place and consumed: pass a fresh
    vector and keep no other reference, so that at most three Lanczos vectors
    are alive (v_prev, v, and w = matvec(v)).

    Returns dict(lam_min, lam_max, iters, converged, history=[[lo, hi], ...]).
    """
    v = v0.div_(sv.norm(v0))
    del v0
    v_prev, beta = None, 0.0
    alphas, betas, hist = [], [], []
    converged = False
    for k in range(max_iter):
        w = matvec(v)
        a = sv.vdot(v, w).real
        w.sub_(v, alpha=a)
        if v_prev is not None:
            w.sub_(v_prev, alpha=beta)
        b = sv.norm(w)
        alphas.append(a)
        if k == 0:
            lo = hi = a
        else:
            ritz = eigvalsh_tridiagonal(np.array(alphas), np.array(betas))
            lo, hi = float(ritz[0]), float(ritz[-1])
        hist.append([lo, hi])
        if on_iter is not None:
            on_iter(k + 1, lo, hi)
        if k >= window:
            scale = max(hi - lo, 1e-300)
            lo0, hi0 = hist[k - window]
            if abs(lo - lo0) < tol * scale and abs(hi - hi0) < tol * scale:
                converged = True
                break
        if b <= 1e-14 * max(abs(lo), abs(hi), 1e-300):
            converged = True          # invariant subspace: Ritz values are exact
            break
        betas.append(b)
        w.div_(b)
        v_prev, v, beta = v, w, b
    return dict(lam_min=hist[-1][0], lam_max=hist[-1][1], iters=len(hist),
                converged=converged, history=hist)


def _H_matvec(ops):
    def mv(v):
        return sv.apply_H(v, ops, torch.zeros_like(v))
    return mv


def spectral_range(ops, seed=0, max_iter=200, tol=1e-6):
    """(E_min, E_max) of the block H by Lanczos from a seeded Haar vector."""
    r = lanczos(_H_matvec(ops), sv.haar_state(ops.n, 10_000 + seed, ops.device),
                max_iter=max_iter, tol=tol)
    return dict(emin=r["lam_min"], emax=r["lam_max"], iters=r["iters"],
                converged=r["converged"])


def cheb_params(emin, emax, margin=1.02):
    """Centre c and half-width s (with margin) of the spectrum."""
    return 0.5 * (emax + emin), 0.5 * (emax - emin) * margin


def cheb_terms(tau, tol=1e-16):
    """Last index K kept: the first k > tau with 2|J_k(tau)| < tol is dropped."""
    k = 1
    while k <= tau or 2.0 * abs(jv(k, tau)) >= tol:
        k += 1
    return k - 1


def evolve(v, ops, t, c, s, sign=-1, K=None, info=None):
    """exp(sign * i H t) v as a new tensor; v is left unchanged.

    sign=-1 gives V v, sign=+1 gives V^dagger v.  Clenshaw summation
    (b_k = a_k v + 2 H_s b_{k+1} - b_{k+2}, result a_0 v + H_s b_1 - b_2) needs
    only two work vectors besides v; the result reuses one of them.  If `info`
    is a dict it receives K and | ||result|| / ||v|| - 1 |, which stays at
    rounding level unless the spectrum leaks far outside [c - s, c + s].
    """
    tau = s * t
    if K is None:
        K = cheb_terms(tau)
    phase = complex(math.cos(c * t), sign * math.sin(c * t))     # e^{sign i c t}
    unit = complex(0.0, float(sign))                              # (sign i)
    a = [jv(0, tau)] + [2.0 * unit ** k * jv(k, tau) for k in range(1, K + 1)]

    if K == 0:
        res = v * a[0]
    else:
        p = v * a[K]                       # b_K
        q = torch.zeros_like(v)            # b_{K+1}
        for k in range(K - 1, 0, -1):
            # b_k = a_k v + 2 H_s b_{k+1} - b_{k+2}, written over b_{k+2}
            q.neg_()
            q.add_(v, alpha=a[k])
            sv.apply_H(p, ops, q, scale=2.0 / s)
            q.sub_(p, alpha=2.0 * c / s)
            p, q = q, p
        # a_0 v + H_s b_1 - b_2, written over b_2
        q.neg_()
        q.add_(v, alpha=a[0])
        sv.apply_H(p, ops, q, scale=1.0 / s)
        q.sub_(p, alpha=c / s)
        del p
        res = q
    res.mul_(phase)
    if info is not None:
        info.update(K=K, norm_dev=abs(sv.norm(res) / sv.norm(v) - 1.0))
    return res
