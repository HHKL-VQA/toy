"""Stage 5: residual loss and adjoint gradient against central differences."""

import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hhkl_toy import adjoint, ansatz, sv, target
from hhkl_toy.lattice import Patch


def problem(sx, sy, L, t, seeds=(1, 2)):
    p = Patch(sx, sy, 0)
    ops = sv.Ops(p)
    rr = target.spectral_range(ops)
    cc, ss = target.cheb_params(rr["emin"], rr["emax"])
    seq = ansatz.slots(L)
    c0 = ansatz.trotter_params(seq, L, ansatz.default_init(L))
    states = []
    for sd in seeds:
        mk = functools.partial(sv.haar_state, p.n, sd, "cpu")
        states.append((mk, target.evolve(mk(), ops, t, cc, ss)))
    return ops, seq, c0, states


def fd_grad(ops, seq, c, t, states, h):
    g = []
    for i in range(len(c)):
        cp, cm = list(c), list(c)
        cp[i] += h
        cm[i] -= h
        Jp = adjoint.loss_grad(ops, seq, cp, t, states, need_grad=False)[0]
        Jm = adjoint.loss_grad(ops, seq, cm, t, states, need_grad=False)[0]
        g.append((Jp - Jm) / (2 * h))
    return np.array(g)


def check(ops, seq, c, t, states, h):
    J, g, infos = adjoint.loss_grad(ops, seq, c, t, states)
    g = np.array(g)
    gf = fd_grad(ops, seq, c, t, states, h)
    rel = np.linalg.norm(g - gf) / np.linalg.norm(gf)
    drift = max(i["drift"] for i in infos)
    return J, rel, drift


def test_grad_random_c():
    rng = np.random.default_rng(0)
    for sx, sy, L, t in [(3, 3, 2, 0.05), (3, 3, 4, 0.3), (3, 4, 2, 0.05)]:
        ops, seq, c0, states = problem(sx, sy, L, t)
        c = list(rng.uniform(-3, 3, size=len(c0)))
        J, rel, drift = check(ops, seq, c, t, states, h=1e-5)
        assert rel <= 1e-6, (sx, sy, L, t, J, rel)
        assert drift <= 1e-13, drift


def test_grad_trotter_c():
    for sx, sy, L, t in [(3, 3, 4, 0.05), (3, 3, 2, 3.41e-3), (2, 3, 10, 0.05),
                         (3, 4, 6, 3.41e-3)]:
        ops, seq, c0, states = problem(sx, sy, L, t)
        J, rel, drift = check(ops, seq, c0, t, states, h=1e-4)
        assert rel <= 1e-6, (sx, sy, L, t, J, rel)
        assert drift <= 1e-13, drift


def test_empty_colour_has_zero_gradient():
    ops, seq, c0, states = problem(2, 3, 2, 0.05)          # sx = 2: colour B is empty
    _, g, _ = adjoint.loss_grad(ops, seq, c0, 0.05, states)
    for s in seq:
        if s["g"] == "B":
            assert all(g[i] == 0.0 for i in s["idx"])


def test_loss_is_one_minus_overlap():
    ops, seq, c0, states = problem(3, 3, 2, 0.3)
    for mk, vpsi in states:
        J, _, info = adjoint.state_loss_grad(ops, seq, c0, 0.3, mk, vpsi, need_grad=False)
        u = ansatz.apply_circuit(mk(), ops, seq, c0, 0.3)
        ov = abs(sv.vdot(vpsi, u))
        assert J > 1e-6
        assert abs(J - (1 - ov)) < 1e-14
        assert abs(J - (1 - ov ** 2) / (1 + ov)) < 1e-14


def test_no_cancellation_at_1e_minus_20():
    """V psi = e^{i g}(U psi cos e + w sin e), w orthogonal to U psi: J = 2 sin^2(e/2)."""
    p = Patch(3, 3, 0)
    ops = sv.Ops(p)
    seq = ansatz.slots(2)
    c = ansatz.trotter_params(seq, 2, "S2^1")
    mk = functools.partial(sv.haar_state, p.n, 7, "cpu")
    u = ansatz.apply_circuit(mk(), ops, seq, c, 0.05)
    w = sv.haar_state(p.n, 8)
    w -= u * sv.vdot(u, w)
    w /= sv.norm(w)
    for eps in (1e-6, 1e-8, 1e-10):
        vpsi = np.exp(0.7j) * (u * np.cos(eps) + w * np.sin(eps))
        J, _, _ = adjoint.state_loss_grad(ops, seq, c, 0.05, mk, vpsi, need_grad=False)
        exact = 2 * np.sin(eps / 2) ** 2
        assert abs(J / exact - 1) < 1e-6, (eps, J, exact)
        # the overlap form loses everything at this level
        ov = abs(sv.vdot(vpsi, u))
        naive = 1 - ov ** 2
        if eps <= 1e-8:
            assert abs(naive / (2 * exact) - 1) > 1e-2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("PASS test_adjoint")
