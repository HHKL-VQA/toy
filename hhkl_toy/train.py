"""L-BFGS training of P1 on Haar states, split over one or two devices (PLAN.md §4.5).

scipy's L-BFGS-B runs on the host on J/J0 (J is ~1e-18 at the operating point,
J0 = J at the Trotter angles); only loss and gradient touch the devices.

At small t the Hessian of J/J0 at the Trotter point is tiered (per-group total
times ~1e10, then ~1e5, ~1e1, gauge ~0), so plain L-BFGS crawls.  lbfgs()
therefore works in c = c0 + B y with B = |H|^{-1/2} from a finite-difference
Hessian (fd_hessian, preconditioner).  Stops on whichever comes first:

    plateau   J_k > (1 - 1e-3) J_{k-W}   (W = 30 preconditioned, 100 plain)
    maxiter   500 iterations in total (across restarts)
    scipy     its own convergence test (ftol, gtol, line-search failure)

Every iteration appends {iter, J, wall, c} to a JSON-lines checkpoint; a
restart resumes from the last c with a fresh L-BFGS (curvature history is lost).
The returned c is the best one seen in any evaluation.
"""

import functools
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.optimize import minimize

from . import adjoint, sv, target


def pick_devices(spec=None):
    """'auto' -> up to two CUDA devices, else CPU.  Otherwise a list of names."""
    if spec in (None, "auto", ["auto"]):
        k = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(min(k, 2))] if k else ["cpu"]
    if isinstance(spec, str):
        spec = [spec]
    out = []
    for s in spec:
        out.append("cuda:0" if s == "cuda" else s)
    return out


def split(seeds, k):
    """Round-robin assignment of state seeds to k devices."""
    return [list(seeds[i::k]) for i in range(k)]


def make_states(ops, seeds, t, cc, ss):
    """[(make_psi, V psi)] on ops.device; psi is regenerated, never stored."""
    out = []
    for sd in seeds:
        mk = functools.partial(sv.haar_state, ops.n, sd, ops.device)
        psi = mk()
        vpsi = target.evolve(psi, ops, t, cc, ss)
        del psi
        out.append((mk, vpsi))
    return out


def objective(ops_list, seq, t, per_device):
    """f(c, need_grad) -> (mean J, mean grad, per-state infos) over all devices."""
    pool = ThreadPoolExecutor(max_workers=len(ops_list)) if len(ops_list) > 1 else None
    M = sum(len(s) for s in per_device)

    def f(c, need_grad=True):
        jobs = [(ops, st) for ops, st in zip(ops_list, per_device) if st]
        if pool is None:
            res = [adjoint.loss_grad(ops, seq, c, t, st, need_grad) for ops, st in jobs]
        else:
            futs = [pool.submit(adjoint.loss_grad, ops, seq, c, t, st, need_grad)
                    for ops, st in jobs]
            res = [fu.result() for fu in futs]
        J = sum(r[0] * len(st) for r, (_, st) in zip(res, jobs)) / M
        grad = None
        if need_grad:
            grad = [sum(r[1][i] * len(st) for r, (_, st) in zip(res, jobs)) / M
                    for i in range(len(c))]
        infos = [i for r in res for i in r[2]]
        return J, grad, infos

    return f


def read_checkpoint(path):
    """(header, [iteration records]) from a JSON-lines checkpoint, or (None, [])."""
    if not path or not os.path.exists(path):
        return None, []
    header, its = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "header":
                header = rec
            else:
                its.append(rec)
    return header, its


def _append(path, rec):
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def fd_hessian(f, c0, J0, h=1e-4, central=False, cache=None, log=print):
    """Hessian of J/J0 at c0 by finite differences of the adjoint gradient.

    Forward differences take P+1 gradient evaluations, central ones 2P; on the
    test patches both give the same preconditioner.  Finished columns are kept
    in `cache` (JSON) so an interrupted run resumes where it stopped.
    """
    c0 = np.asarray(c0, dtype=float)
    P = len(c0)
    rec = dict(h=h, central=central, c0=list(c0), g0=None, cols={})
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            old = json.load(fh)
        if (old.get("h"), old.get("central")) == (h, central) and np.array_equal(old["c0"], c0):
            rec = old
            rec["cols"] = {int(k): v for k, v in rec["cols"].items()}
            log(f"  Hessian: {len(rec['cols'])}/{P} columns from cache")

    def save():
        if cache:
            with open(cache + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            os.replace(cache + ".tmp", cache)

    t0 = time.time()
    if not central and rec["g0"] is None:
        rec["g0"] = list(f(list(c0))[1])
        save()
    for i in range(P):
        if i in rec["cols"]:
            continue
        e = np.zeros(P)
        e[i] = h
        gp = np.asarray(f(list(c0 + e))[1])
        if central:
            col = (gp - np.asarray(f(list(c0 - e))[1])) / (2.0 * h * J0)
        else:
            col = (gp - np.asarray(rec["g0"])) / (h * J0)
        rec["cols"][i] = list(col)
        save()
        log(f"  Hessian column {i + 1}/{P} ({time.time() - t0:.0f}s)")
    H = np.array([rec["cols"][i] for i in range(P)]).T
    return 0.5 * (H + H.T)


def preconditioner(H, floor=1e-9):
    """B = V |w|^{-1/2} from H = V w V^T, so B^T H B ~ 1 on every resolved direction.

    Eigenvalues below floor * max|w| (gauge directions, finite-difference noise)
    are raised to that floor.  Returns (B, sorted eigenvalues of H).
    """
    w, V = np.linalg.eigh(H)
    return V / np.sqrt(np.maximum(np.abs(w), floor * np.abs(w).max())), np.sort(w)


def lbfgs(f, c0, J0, B=None, maxiter=500, window=30, min_rel=1e-3, ftol=1e-15,
          gtol=1e-12, maxcor=50, ckpt=None, resume=False, log=print):
    """Minimise J(c)/J0 from c0 in the coordinates c = c0 + B y (B = 1 if None).

    window <= 0 disables the plateau rule.  Returns a dict (stored as 'train' in
    the result JSON).
    """
    t_start = time.time()
    hist, it0, wall0 = [], 0, 0.0
    if resume:
        header, its = read_checkpoint(ckpt)
        if header is not None:
            J0 = header["J0"]
            if its:
                c0 = its[-1]["c"]
                it0, wall0 = its[-1]["iter"], its[-1]["wall"]
                hist = [[0, J0, 0.0]] + [[r["iter"], r["J"], r["wall"]] for r in its]
                log(f"  resume from iteration {it0}, J = {its[-1]['J']:.6e}")
    if not hist:
        if ckpt and os.path.exists(ckpt):
            os.remove(ckpt)
        _append(ckpt, dict(type="header", J0=J0, c0=list(c0), preconditioned=B is not None))
        hist = [[0, J0, 0.0]]

    c0 = np.asarray(c0, dtype=float)
    B = np.eye(len(c0)) if B is None else np.asarray(B, dtype=float)
    best = dict(J=math.inf, c=list(c0))
    stats = dict(evals=0, drift_max=0.0)
    state = dict(stop=None)

    def fun(y):
        c = c0 + B @ y
        J, g, infos = f(list(map(float, c)))
        stats["evals"] += 1
        stats["drift_max"] = max([stats["drift_max"]] + [i["drift"] for i in infos])
        if J < best["J"]:
            best.update(J=J, c=list(map(float, c)))
        return J / J0, B.T @ (np.asarray(g) / J0)

    def callback(intermediate_result):
        k = hist[-1][0] + 1
        J = float(intermediate_result.fun) * J0
        wall = wall0 + time.time() - t_start
        hist.append([k, J, wall])
        c = c0 + B @ intermediate_result.x
        _append(ckpt, dict(type="iter", iter=k, J=J, wall=wall, c=list(map(float, c))))
        log(f"  it {k:4d}  J {J:.6e}  J/J0 {J / J0:.4e}  ({wall:.0f}s)")
        if window > 0 and len(hist) > window and J > (1.0 - min_rel) * hist[-1 - window][1]:
            state["stop"] = f"plateau: <{min_rel:g} relative gain over {window} iterations"
            raise StopIteration
        if k >= maxiter:
            state["stop"] = f"maxiter {maxiter}"
            raise StopIteration

    remaining = maxiter - it0
    if remaining <= 0:
        state["stop"] = f"maxiter {maxiter}"
        J, _, _ = f(list(c0))
        best.update(J=J, c=list(c0))
        res_msg = "no iterations left"
    else:
        res = minimize(fun, np.zeros(len(c0)), jac=True, method="L-BFGS-B",
                       callback=callback,
                       options=dict(maxiter=remaining, maxcor=maxcor, ftol=ftol,
                                    gtol=gtol, maxfun=20 * maxiter))
        res_msg = str(res.message)
        if state["stop"] is None:
            state["stop"] = f"scipy: {res_msg}"
    return dict(J0=J0, J_final=best["J"], c_final=best["c"], iters=hist[-1][0],
                evals=stats["evals"], stop_reason=state["stop"], scipy_message=res_msg,
                history=hist, drift_max=stats["drift_max"],
                wall_s=wall0 + time.time() - t_start)
