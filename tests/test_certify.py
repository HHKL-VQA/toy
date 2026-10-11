"""Stage 6: Lanczos spectral-norm certification against dense references."""

import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.optimize as so
import torch

from hhkl_toy import ansatz, certify, sv, target
from hhkl_toy.lattice import Patch, dense_H


@functools.lru_cache(maxsize=None)
def _eigh(sx, sy):
    return np.linalg.eigh(dense_H(Patch(sx, sy, 0)))


def dense_V(p, t):
    w, Q = _eigh(p.sx, p.sy)
    return (Q * np.exp(-1j * w * t)) @ Q.conj().T


def dense_U(p, seq, c, t):
    U = torch.eye(1 << p.n, dtype=sv.CDTYPE)
    return ansatz.apply_circuit(U, sv.Ops(p), seq, c, t).numpy()


def exact_eps(U, V, phase=0.0):
    W = np.exp(1j * phase) * (U.conj().T @ V)
    lam = np.linalg.eigvalsh((W - W.conj().T) / 2j)
    w = np.arcsin(lam[-1]) - np.arcsin(lam[0])
    return 2 * np.sin(w / 4)


def grid_eps(U, V):
    """min_phi ||U - e^{i phi} V||_2 by a phase grid plus bounded Brent."""
    f = lambda ph: np.linalg.norm(U - np.exp(1j * ph) * V, 2)
    grid = np.linspace(-0.05, 0.05, 201)
    k = int(np.argmin([f(g) for g in grid]))
    d = grid[1] - grid[0]
    r = so.minimize_scalar(f, bounds=(grid[k] - d, grid[k] + d), method="bounded",
                           options=dict(xatol=1e-13))
    return r.fun


def setup(sx, sy, L, t, perturb=0.0):
    p = Patch(sx, sy, 0)
    ops = sv.Ops(p)
    rr = target.spectral_range(ops)
    cc, ss = target.cheb_params(rr["emin"], rr["emax"])
    seq = ansatz.slots(L)
    c = np.array(ansatz.trotter_params(seq, L, ansatz.default_init(L)))
    c = list(c + perturb * np.random.default_rng(3).normal(size=len(c)))
    return p, ops, seq, c, cc, ss


CASES = [(2, 3, 2, 0.05, 0.0), (2, 3, 10, 0.05, 0.0), (2, 3, 2, 3.41e-3, 0.0),
         (3, 3, 2, 0.05, 0.0), (3, 3, 4, 3.41e-3, 0.0), (3, 3, 6, 0.05, 0.05)]


def test_lanczos_matches_exact_and_grid():
    for sx, sy, L, t, pert in CASES:
        p, ops, seq, c, cc, ss = setup(sx, sy, L, t, pert)
        U, V = dense_U(p, seq, c, t), dense_V(p, t)
        ex = exact_eps(U, V)
        gr = grid_eps(U, V)
        assert abs(gr / ex - 1) < 1e-6, (sx, sy, L, t, ex, gr)
        res = [certify.certify(ops, seq, c, t, cc, ss, seed=s, tol=1e-10)
               for s in (1, 2)]
        for r in res:
            assert abs(r["eps_spec"] / ex - 1) <= 1e-6, (sx, sy, L, t, ex, r["eps_spec"])
        assert abs(res[0]["eps_spec"] / res[1]["eps_spec"] - 1) <= 1e-6


def test_lanczos_3x4():
    p, ops, seq, c, cc, ss = setup(3, 4, 2, 0.05)
    ex = exact_eps(dense_U(p, seq, c, 0.05), dense_V(p, 0.05))
    two = certify.certify_two(ops, seq, c, 0.05, cc, ss, tol=1e-10)
    for r in two["runs"]:
        assert abs(r["eps_spec"] / ex - 1) <= 1e-6, (ex, r["eps_spec"], r["iters"])
    assert two["start_vectors_agree"]


def test_phase_independence_and_wrong_formula():
    p, ops, seq, c, cc, ss = setup(3, 3, 2, 0.05)
    U, V = dense_U(p, seq, c, 0.05), dense_V(p, 0.05)
    ex = exact_eps(U, V)
    for phase in (0.0, 0.2, -0.4):
        r = certify.certify(ops, seq, c, 0.05, cc, ss, phase=phase, tol=1e-10)
        assert abs(r["eps_spec"] / ex - 1) <= 1e-6, phase
        assert abs(exact_eps(U, V, phase) / ex - 1) < 1e-9
    # the old 2 sin(w/2) is about twice the true spectral distance
    w = 4 * np.arcsin(ex / 2)
    assert abs(2 * np.sin(w / 2) / ex - 2) < 1e-3


def test_phase_sign_convention():
    """PLAN.md §4.6 sign: W = e^{+i phi} U^dag V with phi = arg <V psi|U psi>.

    For U' = e^{i g} U the training phase is phi ~ g, and certify(U', phi) is
    certify(U, 0).  The old e^{-i phi} would be certify(U, -2g): W centred at
    e^{-2ig}.  At g = pi/4 the arc sits on the arcsin fold and eps is wrong.
    Offsets short of the fold are harmless because the Lanczos stopping test
    is relative to the arc width, not to |lambda|.
    """
    t = 3.41e-3
    p, ops, seq, c, cc, ss = setup(3, 3, 4, t)
    ex = exact_eps(dense_U(p, seq, c, t), dense_V(p, t))
    good = certify.certify(ops, seq, c, t, cc, ss, phase=0.0)["eps_spec"]
    assert abs(good / ex - 1) < 1e-6, (good, ex)
    # the training phase of U' = e^{ig} U recovers g
    g = 0.3
    mk = functools.partial(sv.haar_state, p.n, 3, "cpu")
    u = ansatz.apply_circuit(mk(), ops, seq, c, t) * np.exp(1j * g)
    v = target.evolve(mk(), ops, t, cc, ss)
    assert abs(np.angle(sv.vdot(v, u)) - g) < 1e-6
    # old sign with g = pi/4: W centred at -i, arcsin folds, eps is wrong
    folded = certify.certify(ops, seq, c, t, cc, ss, phase=-np.pi / 2)["eps_spec"]
    assert abs(folded / ex - 1) > 0.1, folded
    # offsets well short of the fold (here 1e4 x the arc width) do not matter
    w = 4 * np.arcsin(ex / 2)
    for ph in (-100 * w, -1e4 * w):
        off = certify.certify(ops, seq, c, t, cc, ss, phase=ph)["eps_spec"]
        assert abs(off / ex - 1) < 1e-5, (ph, off, ex)


def test_default_tolerance_is_close_and_below():
    """With the production stopping rule the estimate sits just under the truth."""
    p, ops, seq, c, cc, ss = setup(3, 3, 4, 3.41e-3)
    ex = exact_eps(dense_U(p, seq, c, 3.41e-3), dense_V(p, 3.41e-3))
    r = certify.certify(ops, seq, c, 3.41e-3, cc, ss)
    assert r["eps_spec"] <= ex * (1 + 1e-9)
    assert r["eps_spec"] >= ex * (1 - 1e-2), (r["eps_spec"], ex, r["iters"])
    assert len(r["ritz_history"]) == r["iters"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("PASS test_certify")
