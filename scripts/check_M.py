"""Is M training states enough?  (PLAN.md §2.1 item 2, §11)

    python scripts/check_M.py --patch 3x3 --L 4 --M 1 2 4 8 16 --sets 3
    python scripts/check_M.py --patch 4x5 --L 4 --M 2 4 8 --sets 2 --devices cuda   # desktop

For each M and each of --sets independent seed sets, trains P1 at t and scores
the trained circuit with quantities that do not depend on the training states:

    J_heldout   mean loss over --heldout fresh Haar states
    J_haar      exact Haar average 1 - |tr(V^dag U)|/d   (dense, n <= 12 only)
    eps_spec    Lanczos certification (independent of the training states)

and reports the per-state spread of J, whose relative size falls as 2^{-n/2}.
M is enough when J_heldout and eps_spec stop changing with M.  Certification
never touches the training states: it only needs U (the trained angles) and V.
Writes results/check_M_{patch}_L{L}.json; the last line is PASS or FAIL.
"""

import argparse
import functools
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hhkl_toy import adjoint, ansatz, certify, sv, target, train
from hhkl_toy.lattice import Patch, dense_H


_V = {}


def exact_haar(p, seq, c, t):
    """1 - |tr(V^dag U)|/d with dense U and V (small n; V is built once).

    Evaluated as ||U - e^{i phi} V||_F^2 / 2d with phi = arg tr(V^dag U), the
    same residual form as the loss: 1 - |tr|/d itself cancels to 1e-16 steps.
    """
    if (p.name, t) not in _V:
        w, Q = np.linalg.eigh(dense_H(p))
        _V[p.name, t] = (Q * np.exp(-1j * w * t)) @ Q.conj().T
    V = _V[p.name, t]
    U = torch.eye(1 << p.n, dtype=sv.CDTYPE)
    U = ansatz.apply_circuit(U, sv.Ops(p), seq, c, t).numpy()
    tr = np.vdot(V, U)
    return float(np.linalg.norm(U - (tr / abs(tr)) * V) ** 2 / (2 << p.n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", default="3x3")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--t", type=float, default=3.41e-3)
    ap.add_argument("--M", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--sets", type=int, default=3, help="independent seed sets per M")
    ap.add_argument("--heldout", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--devices", nargs="+", default=["auto"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    dev = train.pick_devices(a.devices)[0]
    p = Patch.from_name(a.patch, 0)
    ops = sv.Ops(p, dev, diag=True)
    rr = target.spectral_range(ops)
    cc, ss = target.cheb_params(rr["emin"], rr["emax"])
    seq = ansatz.slots(a.L)
    c0 = ansatz.trotter_params(seq, a.L, ansatz.default_init(a.L))
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    dense = p.n <= 12

    def score(c):
        """Held-out per-state losses (states rebuilt one at a time to save memory)."""
        Js = []
        for sd in range(5000, 5000 + a.heldout):
            (mk, vpsi), = train.make_states(ops, [sd], a.t, cc, ss)
            Js.append(adjoint.state_loss_grad(ops, seq, c, a.t, mk, vpsi, False)[0])
            del mk, vpsi
        cert = certify.certify(ops, seq, c, a.t, cc, ss)
        out = dict(J_heldout=statistics.fmean(Js),
                   J_rel_spread=statistics.stdev(Js) / statistics.fmean(Js),
                   eps_spec=cert["eps_spec"])
        if dense:
            out["J_haar"] = exact_haar(p, seq, c, a.t)
        return out

    log(f"check_M {p.name} (n={p.n}) L={a.L} t={a.t:g} on {dev}")
    init = score(c0)
    log(f"  Trotter init: {init}")
    rows = []
    for M in a.M:
        for k in range(a.sets):
            seeds = list(range(100 * (k + 1) + 1000 * M, 100 * (k + 1) + 1000 * M + M))
            st = train.make_states(ops, seeds, a.t, cc, ss)
            f = train.objective([ops], seq, a.t, [st])
            J0 = f(c0, need_grad=False)[0]
            B = train.preconditioner(train.fd_hessian(f, c0, J0, log=lambda m: None))[0]
            tr = train.lbfgs(f, c0, J0, B=B, maxiter=a.maxiter, log=lambda m: None)
            del st, f
            sc = score(tr["c_final"])
            row = dict(M=M, set=k, J_train=tr["J_final"], iters=tr["iters"],
                       stop=tr["stop_reason"], **sc)
            rows.append(row)
            log(f"  M={M:2d} set {k}: J_train {tr['J_final']:.4e}  J_heldout {sc['J_heldout']:.4e}"
                + (f"  J_haar {sc['J_haar']:.4e}" if dense else "")
                + f"  eps_spec {sc['eps_spec']:.4e}  ({tr['iters']} its)")

    # summary per M: mean and spread over seed sets, relative to the largest M
    key = "J_haar" if dense else "J_heldout"
    ref_M = max(a.M)
    ref_J = statistics.fmean(r[key] for r in rows if r["M"] == ref_M)
    ref_e = statistics.fmean(r["eps_spec"] for r in rows if r["M"] == ref_M)
    print(f"\n| M | {key} (mean over sets) | vs M={ref_M} | eps_spec | vs M={ref_M} "
          f"| train-heldout gap |")
    print("|---|---|---|---|---|---|")
    summary = {}
    for M in a.M:
        rs = [r for r in rows if r["M"] == M]
        J = statistics.fmean(r[key] for r in rs)
        e = statistics.fmean(r["eps_spec"] for r in rs)
        gap = statistics.fmean((r["J_train"] - r["J_heldout"]) / r["J_heldout"] for r in rs)
        summary[M] = dict(J=J, J_vs_ref=J / ref_J - 1, eps=e, eps_vs_ref=e / ref_e - 1, gap=gap)
        print(f"| {M} | {J:.4e} | {100 * (J / ref_J - 1):+.2f} % | {e:.4e} "
              f"| {100 * (e / ref_e - 1):+.2f} % | {100 * gap:+.2f} % |")
    spread = statistics.fmean(r["J_rel_spread"] for r in rows)
    print(f"\nper-state relative spread of J at n={p.n}: {spread:.3e} "
          f"(~2^(-n/2) scaling -> {spread * 2 ** ((p.n - 30) / 2):.1e} at n=30)")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"check_M_{p.name}_L{a.L}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(patch=p.name, n=p.n, L=a.L, t=a.t, init=init, rows=rows,
                       summary=summary, per_state_rel_spread=spread), fh, indent=1)
    ok = 4 in summary and abs(summary[4]["J_vs_ref"]) < 0.02 and abs(summary[4]["eps_vs_ref"]) < 0.02
    print(f"{'PASS' if ok else 'FAIL'} check_M {p.name} L={a.L}: M=4 within "
          f"{100 * summary.get(4, {}).get('J_vs_ref', float('nan')):+.2f} % (loss) / "
          f"{100 * summary.get(4, {}).get('eps_vs_ref', float('nan')):+.2f} % (eps) of M={ref_M}")


if __name__ == "__main__":
    main()
