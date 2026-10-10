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
    t = 0.05
    ops_list, seq, c0, cc, ss, f = problem("2x3", 10, t, init="S4")
    J0 = f(c0, need_grad=False)[0]
    r = train.lbfgs(f, c0, J0, log=lambda m: None)
    e0 = certify.certify(ops_list, seq, c0, t, cc, ss)["eps_spec"]
    e1 = certify.certify(ops_list, seq, r["c_final"], t, cc, ss)["eps_spec"]
    assert abs(e0 / 7.574e-6 - 1) < 1e-3, e0           # PLAN.md: ~7.6e-6
    assert 1.8 <= e0 / e1 <= 2.6, (e0, e1)               # PLAN.md: ~2.1x


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
