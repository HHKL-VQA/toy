"""Stage 7 validation in one command (PLAN.md §7).

    python scripts/validate.py --device cuda          # desktop GPU, full
    python scripts/validate.py --device cpu --quick   # cloud / laptop

(a) training never ends above its Trotter start (loss and certified eps),
    2x3 and 3x3 at t* = 3.41e-3, L = 2 and 4
(b) 2x3, t = 0.05, L = 10 from S4: S4 certifies at ~7.6e-6 and the trained
    circuit is ~2x better (PLAN.md §1.3)
(c) held-out vs training loss differ by <= 1 % at 16-20 qubits with M = 4
    (if not, rerun with --M 8 and use M = 8 for the main experiment)
(d) (GPU only) CPU and device agree on circuit, Chebyshev and gradient at n = 12

Writes results/validate_{device}.json and the per-run JSON under
results/validate_{device}/.  The last line is PASS or FAIL.
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hhkl_toy import adjoint, ansatz, block, sv, target, train
from hhkl_toy.lattice import Patch

T_STAR = 3.41e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="cpu, cuda, cuda:0 or auto")
    ap.add_argument("--quick", action="store_true",
                    help="(c) on 4x4 only (16 qubits); otherwise 4x4 and 4x5")
    ap.add_argument("--heldout-patches", nargs="+", default=None)
    ap.add_argument("--heldout-L", type=int, default=2)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    device = train.pick_devices(a.device)[0]
    dtag = device.replace(":", "")
    run_dir = os.path.join(a.out, f"validate_{dtag}")
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()
    checks = []

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def record(name, ok, **vals):
        checks.append(dict(name=name, passed=bool(ok), **vals))
        log(f"{'PASS' if ok else 'FAIL'} {name}  {vals}")

    quiet = lambda msg: None if " it " in msg else log(msg)

    def run(patch, t, L, init=None, cert=True, M=4):
        setup = block.Setup(patch, 0, t, [device], log=quiet)
        return block.run(setup, L, init=init, M=M, out_dir=run_dir, force=True,
                         cert_seeds=[1, 2] if cert else [], log=quiet)

    try:
        # (a) ------------------------------------------------------------------
        for patch in ("2x3", "3x3"):
            for L in (2, 4):
                R = run(patch, T_STAR, L)
                r = R["result"]
                ok = (R["train"]["J_final"] <= R["train"]["J0"]
                      and r["eps_spec_trained"] <= r["eps_spec_init"])
                record(f"(a) {patch} L={L} trained <= Trotter", ok,
                       J0=R["train"]["J0"], J_final=R["train"]["J_final"],
                       eps_init=r["eps_spec_init"], eps_trained=r["eps_spec_trained"])

        # (b) ------------------------------------------------------------------
        R = run("2x3", 0.05, 10, init="S4")
        r = R["result"]
        ok = abs(r["eps_spec_init"] / 7.6e-6 - 1) < 0.05 and 1.8 <= r["improvement"] <= 2.6
        record("(b) 2x3 t=0.05 L=10: S4 ~7.6e-6, HVA ~2x better", ok,
               eps_S4=r["eps_spec_init"], eps_HVA=r["eps_spec_trained"],
               improvement=r["improvement"])

        # (c) ------------------------------------------------------------------
        patches = a.heldout_patches or (["4x4"] if a.quick else ["4x4", "4x5"])
        for patch in patches:
            R = run(patch, T_STAR, a.heldout_L, cert=False, M=a.M)
            d = R["heldout"]["rel_diff_final"]
            record(f"(c) {patch} L={a.heldout_L} M={a.M} held-out vs train", d <= 0.01,
                   J_train=R["train"]["J_final"], J_heldout=R["heldout"]["J_final"],
                   rel_diff=d, n=R["n"])

        # (d) ------------------------------------------------------------------
        if device != "cpu":
            name, ok, diffs = device_consistency(device)
            record(name, ok, **diffs)
    except Exception:
        traceback.print_exc()
        record("exception", False, error=traceback.format_exc().splitlines()[-1])

    out = dict(device=device, checks=checks, wall_s=time.time() - t_start,
               env=block.env_info([device]))
    path = os.path.join(a.out, f"validate_{dtag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    n_ok = sum(c["passed"] for c in checks)
    failed = [c["name"] for c in checks if not c["passed"]]
    log(f"results in {path}")
    print(f"{'PASS' if not failed else 'FAIL'} validate {device}: {n_ok}/{len(checks)} checks"
          + (f"; failed: {failed}" if failed else ""))
    sys.exit(0 if not failed else 1)


def device_consistency(device):
    """Same circuit, Chebyshev and gradient on CPU and on the device (3x4, n = 12)."""
    p = Patch(3, 4, 0)
    ops_c, ops_d = sv.Ops(p, "cpu", diag=True), sv.Ops(p, device, diag=True)
    rr = target.spectral_range(ops_c)
    cc, ss = target.cheb_params(rr["emin"], rr["emax"])
    seq = ansatz.slots(4)
    c = list(np.random.default_rng(0).uniform(-1, 1, ansatz.n_params(seq)))
    psi = sv.haar_state(p.n, 5)
    diffs = {}
    u_c = ansatz.apply_circuit(psi.clone(), ops_c, seq, c, 0.05)
    u_d = ansatz.apply_circuit(psi.to(device), ops_d, seq, c, 0.05)
    diffs["circuit"] = float((u_c - u_d.cpu()).abs().max())
    v_c = target.evolve(psi, ops_c, 0.05, cc, ss)
    v_d = target.evolve(psi.to(device), ops_d, 0.05, cc, ss)
    diffs["chebyshev"] = float((v_c - v_d.cpu()).abs().max())
    grads = []
    for ops, v in ((ops_c, v_c), (ops_d, v_d)):
        mk = lambda out=None, ops=ops: (psi.to(ops.device, copy=True) if out is None
                                         else out.copy_(psi))
        _, g, _ = adjoint.state_loss_grad(ops, seq, c, 0.05, mk, v)
        grads.append(np.array(g))
    diffs["gradient_rel"] = float(np.linalg.norm(grads[0] - grads[1]) / np.linalg.norm(grads[0]))
    ok = diffs["circuit"] < 1e-13 and diffs["chebyshev"] < 1e-13 and diffs["gradient_rel"] < 1e-10
    return f"(d) cpu vs {device} agree (n=12, complex128)", ok, diffs


if __name__ == "__main__":
    main()
