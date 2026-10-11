"""Time and memory of the kernels at a given size (PLAN.md §7 Stage 9).

    python scripts/bench.py --n 26 28 30 --devices cuda:0 cuda:1
    python scripts/bench.py --n 12 --devices cpu                 # smoke test

For each size: bond gate, colour and field slots, H application, full P1
forward (L layers), one adjoint loss+gradient, Chebyshev V psi, one
certification matvec, and the peak device memory of each.  Extrapolates the
time of one training iteration and one certification.  Writes
results/bench_{patch}_{device}.json; the last line is PASS or FAIL.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hhkl_toy import adjoint, ansatz, block, certify, sv, target, train
from hhkl_toy.lattice import Patch

PATCH_BY_N = {12: "3x4", 16: "4x4", 20: "4x5", 24: "4x6", 25: "5x5", 26: "2x13",
              28: "4x7", 30: "5x6"}
GB = 1e9


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def reset(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def peak(dev):
    return torch.cuda.max_memory_allocated(dev) / GB if dev.type == "cuda" else None


def timed(fn, dev, reps=1):
    """Mean time of fn over reps calls (perf_counter: immune to wall-clock jumps).

    Repeated (cheap) kernels get one untimed warm-up call first, so CUDA
    start-up does not land in the first sample.
    """
    if reps > 1:
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync(dev)
    return (time.perf_counter() - t0) / reps, out


def bench(patch_name, devices, L, t, M, reps, log):
    p = Patch.from_name(patch_name, 0)
    dev = torch.device(devices[0])
    R = dict(patch=p.name, n=p.n, L=L, t=t, devices=devices,
             vector_GB=(1 << p.n) * 16 / GB)
    reset(dev)
    R["ops_s"], ops = timed(lambda: sv.Ops(p, dev, diag=True), dev)
    ops_list = [ops] + [sv.Ops(p, d, diag=True) for d in devices[1:]]
    R["ops_GB"] = peak(dev)

    psi = sv.haar_state(p.n, 1, dev)
    i, j = p.colors["A"][0]
    R["bond_s"], _ = timed(lambda: sv.apply_bond(psi, p.n, i, j, 0.1, 0.2), dev, reps)
    R["slot_A_s"], _ = timed(lambda: sv.apply_color(psi, ops, "A", 0.1, 0.2), dev, reps)
    R["slot_F_s"], _ = timed(lambda: sv.apply_field(psi, ops, 0.1), dev, reps)
    out = torch.zeros_like(psi)
    R["apply_H_s"], _ = timed(lambda: sv.apply_H(psi, ops, out), dev, reps)
    del out
    log(f"  {p.name}: bond {R['bond_s']:.3g}s  slot A {R['slot_A_s']:.3g}s  "
        f"slot F {R['slot_F_s']:.3g}s  H {R['apply_H_s']:.3g}s")

    seq = ansatz.slots(L)
    c = ansatz.trotter_params(seq, L, ansatz.default_init(L))
    psi0 = psi.clone()
    R["forward_s"], _ = timed(lambda: ansatz.apply_circuit(psi, ops, seq, c, t), dev)
    ansatz.apply_circuit(psi, ops, seq, c, t, inverse=True)
    R["roundtrip_err"] = float((psi - psi0).abs().max())
    del psi0

    reset(dev)
    R["range_s"], rr = timed(lambda: target.spectral_range(ops, max_iter=200), dev)
    R["range_iters"] = rr["iters"]
    cc, ss = target.cheb_params(rr["emin"], rr["emax"])
    R["spectral_range"] = [rr["emin"], rr["emax"]]
    R["cheb_terms"] = target.cheb_terms(ss * t)
    reset(dev)
    R["vpsi_s"], vpsi = timed(lambda: target.evolve(psi, ops, t, cc, ss), dev)
    R["vpsi_GB"] = peak(dev)
    log(f"  {p.name}: forward L={L} {R['forward_s']:.3g}s  Lanczos range "
        f"{R['range_s']:.3g}s ({rr['iters']} its)  V psi {R['vpsi_s']:.3g}s (K={R['cheb_terms']})")

    del psi
    reset(dev)
    mk = lambda out=None: sv.haar_state(p.n, 1, dev, out=out)
    R["adjoint_s"], (J, g, info) = timed(
        lambda: adjoint.state_loss_grad(ops, seq, c, t, mk, vpsi), dev)
    R["adjoint_GB_one_state"] = peak(dev)
    R["drift"] = info["drift"]
    del vpsi
    log(f"  {p.name}: adjoint loss+grad {R['adjoint_s']:.3g}s, J {J:.3e}, drift {info['drift']:.1e}")

    reset(dev)
    matvec, pool = certify._sv_matvec(ops_list, seq, c, t, cc, ss, 0.0)
    v = sv.haar_state(p.n, 2, dev)
    R["cert_matvec_s"], w = timed(lambda: matvec(v), dev)
    R["cert_GB"] = peak(dev)
    if pool is not None:
        pool.shutdown()
    del v, w
    log(f"  {p.name}: certification matvec {R['cert_matvec_s']:.3g}s on {len(devices)} device(s)")

    per_dev = math.ceil(M / len(devices))
    R["estimate"] = dict(
        train_iter_s=1.1 * per_dev * R["adjoint_s"],
        train_GB_per_device=(None if R["adjoint_GB_one_state"] is None else
                             R["adjoint_GB_one_state"] + (per_dev - 1) * R["vector_GB"]),
        cert_300_iters_h=300 * R["cert_matvec_s"] / 3600,
        vpsi_all_states_s=(M + 2) / len(devices) * R["vpsi_s"])
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="*", default=[])
    ap.add_argument("--patch", nargs="*", default=[])
    ap.add_argument("--devices", nargs="+", default=["auto"])
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--t", type=float, default=3.41e-3)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    patches = a.patch + [PATCH_BY_N[n] for n in a.n]
    if not patches:
        ap.error("give --n or --patch")
    devices = train.pick_devices(a.devices)
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    log(f"bench {patches} on {devices}, torch {torch.__version__}")
    for d in devices:
        if d.startswith("cuda"):
            pr = torch.cuda.get_device_properties(torch.device(d))
            log(f"  {d}: {pr.name}, {pr.total_memory / GB:.0f} GB")
    summary, ok = [], True
    for name in patches:
        try:
            R = bench(name, devices, a.L, a.t, a.M, a.reps, log)
            R["env"] = block.env_info(devices)
            good = R["roundtrip_err"] < 1e-12 and R["drift"] < 1e-12
            ok &= good
            e = R["estimate"]
            summary.append(f"{name}: train it ~{e['train_iter_s']:.3g}s, "
                           f"cert 300 its ~{e['cert_300_iters_h']:.3g}h"
                           + ("" if good else " (roundtrip/drift too large)"))
        except Exception:
            traceback.print_exc()
            R = dict(patch=name, error=traceback.format_exc().splitlines()[-1])
            ok = False
            summary.append(f"{name}: {R['error']}")
        path = os.path.join(a.out, f"bench_{name}_{devices[0].replace(':', '')}"
                                   f"{'x2' if len(devices) > 1 else ''}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(R, fh, indent=1)
        log(f"  wrote {path}")
    print(f"{'PASS' if ok else 'FAIL'} bench {devices}: " + "; ".join(summary))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
