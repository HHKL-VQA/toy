"""L-BFGS training of P1 on Haar states, split over one or two devices (PLAN.md §4.5).

scipy's L-BFGS-B runs on the host on J/J0 (J is ~1e-18 at the operating point,
J0 = J at the Trotter angles); only loss and gradient touch the devices.  Stops
on whichever comes first:

    plateau   J_k > (1 - 1e-3) J_{k-20}
    maxiter   500 iterations in total (across restarts)
    scipy     its own convergence test (ftol, gtol)

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
    with open(path) as fh:
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
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def lbfgs(f, c0, J0, maxiter=500, window=20, min_rel=1e-3, ftol=1e-15, gtol=1e-12,
          maxcor=50, ckpt=None, resume=False, log=print):
    """Minimise J(c)/J0 from c0.  window <= 0 disables the plateau rule.

    Returns a dict (stored as 'train' in the result JSON).
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
        _append(ckpt, dict(type="header", J0=J0, c0=list(c0)))
        hist = [[0, J0, 0.0]]

    best = dict(J=math.inf, c=list(c0))
    stats = dict(evals=0, drift_max=0.0)
    state = dict(stop=None)

    def fun(x):
        J, g, infos = f(list(map(float, x)))
        stats["evals"] += 1
        stats["drift_max"] = max([stats["drift_max"]] + [i["drift"] for i in infos])
        if J < best["J"]:
            best.update(J=J, c=list(map(float, x)))
        return J / J0, np.asarray(g) / J0

    def callback(intermediate_result):
        k = hist[-1][0] + 1
        J = float(intermediate_result.fun) * J0
        wall = wall0 + time.time() - t_start
        hist.append([k, J, wall])
        _append(ckpt, dict(type="iter", iter=k, J=J, wall=wall,
                           c=list(map(float, intermediate_result.x))))
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
        res = minimize(fun, np.asarray(c0, dtype=float), jac=True, method="L-BFGS-B",
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
