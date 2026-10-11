"""Spectral-norm certification eps = min_phi ||U - e^{i phi} V|| (PLAN.md §4.6).

    W = e^{i phi} U^dag V,   S = (W - W^dag) / 2i,   eig(S) = sin(delta_j)

delta_j are the eigenphases of W.  Lanczos gives the two extreme eigenvalues
of S, and with w = arcsin(lam_max) - arcsin(lam_min) the arc width,

    eps_spec = 2 sin(w / 4)          (not 2 sin(w / 2): PLAN.md §9.2)

w does not depend on phi as long as every |delta_j| < pi/2, so phi only needs
to be roughly right.  Sign convention: phi = arg <V psi|U psi> as in the loss,
so that U ~ e^{i phi} V and W ~ 1 (PLAN.md §4.6, corrected 2026-10-10: the
earlier e^{-i phi} centred W at e^{-2i phi} and folds arcsin once 2|phi| nears
pi/2).  The Lanczos stopping test is relative to the arc width, so an offset
short of the fold costs no accuracy.

Ritz values sit inside the spectrum, so a truncated run underestimates eps.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor

from . import ansatz, sv, target


def _sv_matvec(ops_list, seq, c, t, cc, ss, phase):
    """v -> S v.  One device: U v and V^dag U v first, then V v and U^dag V v,
    so at most five vectors live at once (v_prev and v of Lanczos, one finished
    half, two Chebyshev work vectors).  Two devices: device 0 does U^dag V v
    while device 1 does V^dag U v (at most four and three vectors).
    """
    ep = complex(math.cos(phase), math.sin(phase))
    pool = ThreadPoolExecutor(max_workers=2) if len(ops_list) > 1 else None

    def udv(v, ops):                     # U^dag V v
        a = target.evolve(v, ops, t, cc, ss, sign=-1)
        return ansatz.apply_circuit(a, ops, seq, c, t, inverse=True)

    def vdu(v, ops):                     # V^dag U v
        b = ansatz.apply_circuit(v.to(ops.device, copy=True), ops, seq, c, t)
        out = target.evolve(b, ops, t, cc, ss, sign=+1)
        del b
        return out

    def matvec(v):
        if pool is None:
            b = vdu(v, ops_list[0])
            a = udv(v, ops_list[0])
        else:
            fb = pool.submit(vdu, v, ops_list[1])
            fa = pool.submit(udv, v, ops_list[0])
            a = fa.result()
            b = fb.result().to(v.device)
        a.mul_(ep)
        a.sub_(b, alpha=ep.conjugate())
        del b
        return a.mul_(-0.5j)             # / (2i)

    return matvec, pool


def certify(ops_list, seq, c, t, cc, ss, phase=0.0, seed=1, max_iter=300,
            tol=1e-4, window=10, log=None):
    """Lanczos certification of one circuit against V = exp(-iHt).

    ops_list: [Ops] on one device or [Ops, Ops] on two; cc, ss: Chebyshev
    centre and half-width; seed: start vector.
    """
    t0 = time.time()
    if not isinstance(ops_list, (list, tuple)):
        ops_list = [ops_list]
    ops = ops_list[0]
    matvec, pool = _sv_matvec(ops_list, seq, c, t, cc, ss, phase)

    def on_iter(k, lo, hi):
        if log is not None and (k % 10 == 0 or k == 1):
            log(f"  certify it {k:4d}  lam_min {lo:+.10e}  lam_max {hi:+.10e}"
                f"  ({time.time() - t0:.0f}s)")

    try:
        r = target.lanczos(matvec, sv.haar_state(ops.n, 20_000 + seed, ops.device),
                           max_iter=max_iter, tol=tol, window=window, on_iter=on_iter)
    finally:
        if pool is not None:
            pool.shutdown()
    lo = max(-1.0, min(1.0, r["lam_min"]))
    hi = max(-1.0, min(1.0, r["lam_max"]))
    width = math.asin(hi) - math.asin(lo)
    # arcsin is only faithful for |delta| < pi/2; with phi from the loss the
    # arc is centred near 0, so this should be ~eps, never near 1.57
    max_delta = max(abs(math.asin(lo)), abs(math.asin(hi)))
    if max_delta > 1.0 and log is not None:
        log(f"  WARNING certify: |delta| reaches {max_delta:.3f} rad; phase estimate is off")
    return dict(eps_spec=2.0 * math.sin(width / 4.0), lam_max=r["lam_max"],
                lam_min=r["lam_min"], width=width, max_abs_delta=max_delta,
                iters=r["iters"],
                converged=r["converged"], phase=phase, seed=seed,
                ritz_history=r["history"], wall_s=time.time() - t0)


def certify_two(ops_list, seq, c, t, cc, ss, phase=0.0, seeds=(1, 2), agree_tol=1e-3,
                **kw):
    """One run per start vector; eps_spec is the largest (Ritz values underestimate).

    start_vectors_agree is None with a single seed.
    """
    runs = [certify(ops_list, seq, c, t, cc, ss, phase, seed=s, **kw) for s in seeds]
    e = [r["eps_spec"] for r in runs]
    agree = (max(e) - min(e) <= agree_tol * max(e)) if len(e) > 1 else None
    return dict(eps_spec=max(e), runs=runs, start_vectors_agree=agree)
