"""Stage 7 (a)(b): training driver, resume, and the two-device code paths."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hhkl_toy import ansatz, certify, sv, target, train
from hhkl_toy.lattice import Patch


def problem(patch, L, t, init=None, n_dev=1, seeds=(101, 102, 103, 104)):
    p = Patch.from_name(patch, 0)
    ops_list = [sv.Ops(p) for _ in range(n_dev)]
    rr = target.spectral_range(ops_list[0])
    cc, ss = target.cheb_params(rr["emin"], rr["emax"])
    seq = ansatz.slots(L)
    c0 = ansatz.trotter_params(seq, L, init or ansatz.default_init(L))
    per = [train.make_states(o, s, t, cc, ss)
           for o, s in zip(ops_list, train.split(list(seeds), n_dev))]
    return ops_list, seq, c0, cc, ss, train.objective(ops_list, seq, t, per)


def test_7a_training_never_worse_than_trotter():
    t = 3.41e-3
    ops_list, seq, c0, cc, ss, f = problem("2x3", 2, t)
    J0 = f(c0, need_grad=False)[0]
    r = train.lbfgs(f, c0, J0, log=lambda m: None)
    assert r["J_final"] <= J0
    e0 = certify.certify(ops_list, seq, c0, t, cc, ss)["eps_spec"]
    e1 = certify.certify(ops_list, seq, r["c_final"], t, cc, ss)["eps_spec"]
    assert e1 <= e0, (e0, e1)
    assert r["drift_max"] < 1e-13


def test_7b_hva_twice_better_than_s4():
    """PLAN.md §1.3: S4 ~7.6e-6 and the trained circuit ~2x better.  Plain
    L-BFGS reproduces the previous session's ~2x; preconditioned goes further."""
    t = 0.05
    ops_list, seq, c0, cc, ss, f = problem("2x3", 10, t, init="S4")
    J0 = f(c0, need_grad=False)[0]
    e0 = certify.certify(ops_list, seq, c0, t, cc, ss)["eps_spec"]
    assert abs(e0 / 7.574e-6 - 1) < 1e-3, e0           # PLAN.md: ~7.6e-6
    plain = train.lbfgs(f, c0, J0, window=20, log=lambda m: None)
    e_plain = certify.certify(ops_list, seq, plain["c_final"], t, cc, ss)["eps_spec"]
    assert 1.8 <= e0 / e_plain <= 2.6, (e0, e_plain)     # PLAN.md: ~2.1x
    B = train.preconditioner(train.fd_hessian(f, c0, J0, log=lambda m: None))[0]
    pre = train.lbfgs(f, c0, J0, B=B, maxiter=200, log=lambda m: None)
    e_pre = certify.certify(ops_list, seq, pre["c_final"], t, cc, ss)["eps_spec"]
    assert pre["J_final"] < plain["J_final"] and e_pre < e_plain, (e_pre, e_plain)


def test_checkpoint_resume():
    ops_list, seq, c0, cc, ss, f = problem("2x3", 4, 3.41e-3)
    J0 = f(c0, need_grad=False)[0]
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "ck.jsonl")
        a = train.lbfgs(f, c0, J0, maxiter=8, window=0, ckpt=ck, log=lambda m: None)
        assert a["iters"] == 8 and a["stop_reason"].startswith("maxiter")
        header, its = train.read_checkpoint(ck)
        assert header["J0"] == J0 and [r["iter"] for r in its] == list(range(1, 9))
        b = train.lbfgs(f, c0, J0, maxiter=15, window=0, ckpt=ck, resume=True,
                        log=lambda m: None)
        assert [h[0] for h in b["history"]] == list(range(16))
        assert b["J_final"] <= its[-1]["J"]
        # the restart begins from the last checkpointed c
        assert np.allclose(b["history"][8][1], its[-1]["J"])


def test_plateau_rule():
    ops_list, seq, c0, cc, ss, f = problem("2x3", 2, 3.41e-3)
    J0 = f(c0, need_grad=False)[0]
    r = train.lbfgs(f, c0, J0, maxiter=500, window=20, log=lambda m: None)
    h = r["history"]
    if r["stop_reason"].startswith("plateau"):
        assert h[-1][1] > (1 - 1e-3) * h[-21][1]
        assert all(h[k][1] <= (1 - 1e-3) * h[k - 20][1] for k in range(20, len(h) - 1))


def test_fd_hessian_and_preconditioner_on_a_stiff_quadratic():
    """J = c^T A c / 2 with eigenvalues 1e10, 1e5, 1 (the tiers seen at t*)."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A = Q @ np.diag([1e10, 1e5, 1.0]) @ Q.T
    calls = []

    def f(c, need_grad=True):
        calls.append(1)
        c = np.asarray(c)
        return 0.5 * c @ A @ c, list(A @ c), [dict(drift=0.0)]

    c0 = [0.3, -0.2, 0.1]
    for central in (False, True):
        H = train.fd_hessian(f, c0, 1.0, central=central, log=lambda m: None)
        assert np.abs(H - A).max() < 1e-6 * 1e10
        B, eig = train.preconditioner(H, floor=1e-12)
        assert np.allclose(B.T @ A @ B, np.eye(3), atol=1e-2)
        # the default floor (1e-9 of the largest) lifts the 1-tier to 10
        B, eig = train.preconditioner(H)
        assert np.allclose(np.linalg.eigvalsh(B.T @ A @ B), [0.1, 1, 1], atol=1e-2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "h.json")
        H1 = train.fd_hessian(f, c0, 1.0, cache=path, log=lambda m: None)
        calls.clear()
        H2 = train.fd_hessian(f, c0, 1.0, cache=path, log=lambda m: None)
        assert not calls and np.array_equal(H1, H2)        # all columns from cache
    # preconditioned L-BFGS solves the stiff quadratic in a few iterations
    B = train.preconditioner(A)[0]
    J0 = f(c0)[0]
    r = train.lbfgs(f, c0, J0, B=B, window=0, log=lambda m: None)
    assert r["J_final"] / J0 < 1e-20 and r["iters"] <= 5


def test_preconditioning_removes_the_slow_phase():
    """3x3, L=4, t*: plain L-BFGS sits at J/J0 ~ 0.71 for ~150 iterations and
    only reaches ~0.199 after ~300; with the Hessian preconditioner it gets
    there in a few tens (PLAN.md §4.5)."""
    t = 3.41e-3
    ops_list, seq, c0, cc, ss, f = problem("3x3", 4, t)
    J0 = f(c0, need_grad=False)[0]
    plain = train.lbfgs(f, c0, J0, maxiter=40, window=0, log=lambda m: None)
    B = train.preconditioner(train.fd_hessian(f, c0, J0, log=lambda m: None))[0]
    pre = train.lbfgs(f, c0, J0, B=B, maxiter=40, window=0, log=lambda m: None)
    assert plain["J_final"] / J0 > 0.7
    assert pre["J_final"] / J0 < 0.2, pre["J_final"] / J0


def test_two_device_split_matches_one():
    """Two Ops on the CPU exercise the threaded split used for two GPUs."""
    t = 0.05
    one = problem("3x3", 4, t, n_dev=1)
    two = problem("3x3", 4, t, n_dev=2)
    c = list(np.array(one[2]) + 0.01)
    J1, g1, _ = one[5](c)
    J2, g2, _ = two[5](c)
    assert abs(J1 / J2 - 1) < 1e-12
    assert np.allclose(g1, g2, rtol=1e-10, atol=0)
    ops_list, seq, _, cc, ss, _ = two
    e1 = certify.certify(ops_list[:1], seq, c, t, cc, ss, tol=1e-10)["eps_spec"]
    e2 = certify.certify(ops_list, seq, c, t, cc, ss, tol=1e-10)["eps_spec"]
    assert abs(e1 / e2 - 1) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("PASS test_train")
