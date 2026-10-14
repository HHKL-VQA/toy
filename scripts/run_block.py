"""Train and certify P1 on one block for each L (PLAN.md §7 Stages 8 and 10).

    python scripts/run_block.py --patch 5x6 --seed 0 --t 3.41e-3 --L 2 4 6 10
    python scripts/run_block.py --patch 2x3 --t 0.05 --L 10 --devices cpu   # smoke test

Writes results/{patch}_s{seed}_t{t}_L{L}.json (rewritten after every phase),
results/ckpt/*.jsonl (per-iteration training checkpoint) and
results/{patch}_s{seed}_t{t}.log.  --resume continues from both.
The last line is PASS or FAIL with the certified errors.
"""

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hhkl_toy import block, train


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", default="5x6", help="sx x sy, e.g. 5x6 (5 columns, 6 rows)")
    ap.add_argument("--seed", type=int, default=0, help="disorder seed")
    ap.add_argument("--t", type=float, default=3.41e-3)
    ap.add_argument("--L", type=int, nargs="+", default=[2, 4, 6, 10])
    ap.add_argument("--init", default=None,
                    help="S2^r, Y4 or S4 (default: S4 for L=10m, Y4 for L=6m, else S2^(L/2))")
    ap.add_argument("--M", type=int, default=4, help="training Haar states")
    ap.add_argument("--M-heldout", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--precond", default="own",
                    help="Hessian preconditioner: own (2P extra gradients on the block), "
                         "a smaller patch such as 3x4, or none (PLAN.md §4.5)")
    ap.add_argument("--plateau-window", type=int, default=None,
                    help="stop when J gained < --plateau-rel over this many iterations "
                         "(default 30 preconditioned, 100 with --precond none; 0 disables)")
    ap.add_argument("--plateau-rel", type=float, default=1e-3)
    ap.add_argument("--cert-seeds", type=int, nargs="*", default=[1, 2],
                    help="Lanczos start vectors per certification (none = skip)")
    ap.add_argument("--cert-maxiter", type=int, default=300)
    ap.add_argument("--cert-tol", type=float, default=1e-4)
    ap.add_argument("--eps-blk", type=float, default=None,
                    help="block budget (default from PLAN.md for t = 3.41e-3, 1.35e-3)")
    ap.add_argument("--devices", nargs="+", default=["auto"],
                    help="auto (up to two GPUs, else cpu), cpu, cuda, or cuda:0 cuda:1")
    ap.add_argument("--no-diag", action="store_true",
                    help="do not precompute the float64 diagonal of H (saves 8.6 GB at n=30)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo combinations already complete")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    logpath = os.path.join(a.out, f"{a.patch}_s{a.seed}_t{a.t:g}.log")
    logfh = open(logpath, "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logfh.write(line + "\n")
        logfh.flush()

    devices = train.pick_devices(a.devices)
    log(f"run_block {' '.join(sys.argv[1:])}  devices {devices}")
    summary, ok = [], True
    try:
        setup = block.Setup(a.patch, a.seed, a.t, devices, diag=not a.no_diag, log=log)
        for L in a.L:
            R = block.run(setup, L, init=a.init, M=a.M, M_heldout=a.M_heldout,
                          maxiter=a.maxiter, precond=a.precond,
                          plateau_window=a.plateau_window,
                          plateau_rel=a.plateau_rel, cert_seeds=a.cert_seeds,
                          cert_maxiter=a.cert_maxiter, cert_tol=a.cert_tol,
                          out_dir=a.out, resume=a.resume, force=a.force,
                          eps_blk=a.eps_blk, log=log)
            res = R.get("result")
            if res is None:
                summary.append(f"L={L} J {R['train']['J0']:.3e}->{R['train']['J_final']:.3e} "
                               "(not certified)")
                continue
            blk = ""
            if R["eps_blk"]:
                e5 = block.to_target(R["patch"], res["eps_spec_reported"])
                blk = (f" ({e5 / R['eps_blk']:.3g} eps_blk)" if R["patch"] == block.TARGET
                       else f" (-> 5x6 est. {e5:.2e} = {e5 / R['eps_blk']:.3g} eps_blk)")
            summary.append(f"L={L} eps {res['eps_spec_init']:.3e}->"
                           f"{res['eps_spec_reported']:.3e}{blk}")
            agree = [R["certify"][w]["start_vectors_agree"] for w in ("init", "trained")]
            ok &= R["train"]["drift_max"] < 1e-10 and False not in agree
    except Exception:
        log(traceback.format_exc())
        log(f"FAIL run_block {a.patch} t={a.t:g}: exception (see above)")
        sys.exit(1)
    log(f"{'PASS' if ok else 'FAIL'} run_block {a.patch} s{a.seed} t={a.t:g}: "
        + "; ".join(summary))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
