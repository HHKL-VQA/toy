"""One (patch, seed, t, L) combination end to end: train -> held-out -> certify -> JSON.

Shared by scripts/run_block.py and scripts/validate.py.  The result file is
rewritten after every phase, so --resume skips whatever is already in it.
Memory order matters at 30 qubits: the cached V psi of the training states
are freed before held-out evaluation and certification.
"""

import cmath
import gc
import json
import os
import platform
import socket
import subprocess
import time

import torch

from . import adjoint, ansatz, certify, sv, target, train
from .lattice import Patch

# block error budgets from PLAN.md §1.2
EPS_BLK = {3.41e-3: 1.71e-9, 1.35e-3: 6.75e-10}

TRAIN_SEED0, HELDOUT_SEED0 = 101, 201


def tag(patch, seed, t, L):
    return f"{patch}_s{seed}_t{t:g}_L{L}"


def env_info(devices):
    gpus = [torch.cuda.get_device_name(torch.device(d)) for d in devices
            if d.startswith("cuda")]
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=os.path.dirname(__file__)).stdout.strip()
    except OSError:
        git = ""
    return dict(host=socket.gethostname(), gpu=gpus or "cpu", devices=devices,
                torch=torch.__version__, python=platform.python_version(), git=git)


def _release():
    """Return freed device memory after the caller dropped its references."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_phase(infos):
    return cmath.phase(sum(cmath.exp(1j * i["phase"]) for i in infos))


class Setup:
    """Devices, Ops and Chebyshev parameters shared by every L of one run."""

    def __init__(self, patch, seed, t, devices, diag=True, log=print):
        self.patch = Patch.from_name(patch, seed)
        self.t = t
        self.devices = devices
        self.ops = [sv.Ops(self.patch, d, diag=diag) for d in devices]
        t0 = time.time()
        rr = target.spectral_range(self.ops[0])
        self.emin, self.emax = rr["emin"], rr["emax"]
        self.cc, self.ss = target.cheb_params(self.emin, self.emax)
        self.K = target.cheb_terms(self.ss * t)
        log(f"[{self.patch.name} seed {seed}] spectrum [{self.emin:.6f}, {self.emax:.6f}] "
            f"({rr['iters']} Lanczos its, {time.time() - t0:.1f}s), "
            f"Chebyshev s*t = {self.ss * t:.4f}, K = {self.K}")

    def states(self, seeds):
        per = train.split(seeds, len(self.ops))
        return [train.make_states(o, s, self.t, self.cc, self.ss) for o, s in zip(self.ops, per)]


def run(setup, L, init=None, M=4, M_heldout=2, maxiter=500, plateau_window=20,
        plateau_rel=1e-3, cert_seeds=(1, 2),
        cert_maxiter=300, cert_tol=1e-4, out_dir="results", resume=False, force=False,
        eps_blk=None, log=print):
    """Train, evaluate and certify one L.  Returns the result dict (also saved)."""
    p, t = setup.patch, setup.t
    init = init or ansatz.default_init(L)
    name = tag(p.name, p.seed, t, L)
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    path = os.path.join(out_dir, name + ".json")
    ckpt = os.path.join(out_dir, "ckpt", name + ".jsonl")
    wall0 = time.time()

    R = {}
    if os.path.exists(path) and not force:
        with open(path) as fh:
            R = json.load(fh)
        if R.get("complete") and not resume:
            log(f"[{name}] already complete, skipping (use --force to redo)")
            return R
        if not resume:
            R = {}
    seq = ansatz.slots(L)
    c0 = ansatz.trotter_params(seq, L, init)
    cost = ansatz.costs(p, seq)
    if eps_blk is None:
        eps_blk = EPS_BLK.get(round(t, 12))
    train_seeds = list(range(TRAIN_SEED0, TRAIN_SEED0 + M))
    ho_seeds = list(range(HELDOUT_SEED0, HELDOUT_SEED0 + M_heldout))
    R.update(patch=p.name, n=p.n, seed=p.seed, t=t, eps_blk=eps_blk, L=L, init=init,
             n_params=cost["n_params"], cnot=cost["cnot"], cnot_depth=cost["cnot_depth"],
             dtype="complex128", M_train=M, M_heldout=M_heldout,
             haar_seeds=dict(train=train_seeds, heldout=ho_seeds),
             spectral_range=[setup.emin, setup.emax], cheb_terms=setup.K,
             cheb_center_halfwidth=[setup.cc, setup.ss])
    prior_wall = R.get("env", {}).get("wall_total_s", 0.0)

    def save():
        R["env"] = env_info(setup.devices)
        R["env"]["wall_total_s"] = prior_wall + time.time() - wall0
        with open(path + ".tmp", "w") as fh:
            json.dump(R, fh, indent=1)
        os.replace(path + ".tmp", path)

    # --- training --------------------------------------------------------
    if "train" not in R:
        log(f"[{name}] {cost} init {init}")
        st = setup.states(train_seeds)
        f = train.objective(setup.ops, seq, t, st)
        J0, _, inf0 = f(c0, need_grad=False)
        log(f"[{name}] J0 = {J0:.6e}  (eps_rHS {adjoint.eps_rhs(J0):.3e})")
        tr = train.lbfgs(f, c0, J0, maxiter=maxiter, window=plateau_window,
                         min_rel=plateau_rel, ckpt=ckpt, resume=resume, log=log)
        Jf, _, inff = f(tr["c_final"], need_grad=False)
        tr.update(phase_init=mean_phase(inf0), phase_final=mean_phase(inff),
                  J_final_check=Jf)
        del st, f
        _release()
        R["train"] = tr
        R["eps_rhs"] = dict(init=adjoint.eps_rhs(tr["J0"]),
                            trained=adjoint.eps_rhs(tr["J_final"]))
        save()
        log(f"[{name}] trained: J {tr['J0']:.4e} -> {tr['J_final']:.4e} in {tr['iters']} its "
            f"({tr['stop_reason']}), drift {tr['drift_max']:.1e}")
    tr = R["train"]

    # --- held-out ----------------------------------------------------------
    if "heldout" not in R:
        Ji, Jf = [], []
        for k, sd in enumerate(ho_seeds):
            ops = setup.ops[k % len(setup.ops)]
            (mk, vpsi), = train.make_states(ops, [sd], t, setup.cc, setup.ss)
            Ji.append(adjoint.state_loss_grad(ops, seq, c0, t, mk, vpsi, False)[0])
            Jf.append(adjoint.state_loss_grad(ops, seq, tr["c_final"], t, mk, vpsi, False)[0])
            del mk, vpsi
            _release()
        ho = dict(J_init=sum(Ji) / len(Ji), J_final=sum(Jf) / len(Jf),
                  per_state_init=Ji, per_state_final=Jf)
        ho["rel_diff_final"] = abs(ho["J_final"] - tr["J_final"]) / tr["J_final"]
        R["heldout"] = ho
        save()
        log(f"[{name}] held-out J: {ho['J_init']:.4e} -> {ho['J_final']:.4e} "
            f"(train/held-out differ by {100 * ho['rel_diff_final']:.2f} %)")

    # --- certification -----------------------------------------------------
    R.setdefault("certify", {})
    for which, c, ph in (("init", c0, tr["phase_init"]),
                         ("trained", tr["c_final"], tr["phase_final"])):
        if which in R["certify"] or not cert_seeds:
            continue
        log(f"[{name}] certify {which} (start vectors {list(cert_seeds)})")
        res = certify.certify_two(setup.ops, seq, c, t, setup.cc, setup.ss, phase=ph,
                                  seeds=cert_seeds, max_iter=cert_maxiter, tol=cert_tol,
                                  log=log)
        R["certify"][which] = res
        save()
        log(f"[{name}] eps_spec({which}) = {res['eps_spec']:.6e}  "
            f"(its {[r['iters'] for r in res['runs']]}, "
            f"agree {res['start_vectors_agree']})")

    cert = R["certify"]
    if "init" in cert and "trained" in cert:
        e_i, e_t = cert["init"]["eps_spec"], cert["trained"]["eps_spec"]
        fallback = e_t > e_i
        R["result"] = dict(
            eps_spec_init=e_i, eps_spec_trained=e_t,
            eps_spec_reported=e_i if fallback else e_t, fallback_to_init=fallback,
            improvement=e_i / e_t,
            ratio_to_eps_blk=(min(e_i, e_t) / eps_blk) if eps_blk else None,   # reported
            spectral_over_rhs=e_t / R["eps_rhs"]["trained"])
        if fallback:
            log(f"[{name}] WARNING trained circuit certifies worse than its Trotter start; "
                f"reporting the initial value")
        R["complete"] = True
        save()
    return R
