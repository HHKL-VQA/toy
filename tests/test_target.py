"""Stage 3: spectrum ends and Chebyshev evolution against eigh / expm."""

import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.linalg as sla
import torch

from hhkl_toy import sv, target
from hhkl_toy.lattice import Patch, dense_H


def rand_state(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return v / np.linalg.norm(v)


@functools.lru_cache(maxsize=None)
def _exact(sx, sy, seed):
    return np.linalg.eigh(dense_H(Patch(sx, sy, seed)))


def exact(p):
    return _exact(p.sx, p.sy, p.seed)


def test_spectral_range_vs_eigh():
    for sx, sy in [(2, 3), (3, 3), (3, 4)]:
        p = Patch(sx, sy, seed=0)
        w, _ = exact(p)
        ops = sv.Ops(p)
        # converged Lanczos reproduces both ends
        r = target.lanczos(target._H_matvec(ops), sv.haar_state(p.n, 1),
                           max_iter=300, tol=1e-13)
        assert abs(r["lam_min"] - w[0]) < 1e-9 * abs(w[0]), (sx, sy, r["lam_min"], w[0])
        assert abs(r["lam_max"] - w[-1]) < 1e-9 * abs(w[-1]), (sx, sy, r["lam_max"], w[-1])
        # default run: Ritz values inside the spectrum, and c +- s covers it
        rr = target.spectral_range(ops)
        assert w[0] - 1e-9 <= rr["emin"] and rr["emax"] <= w[-1] + 1e-9
        assert abs(rr["emin"] - w[0]) < 1e-4 * abs(w[0])
        assert abs(rr["emax"] - w[-1]) < 1e-4 * abs(w[-1])
        c, s = target.cheb_params(rr["emin"], rr["emax"])
        assert c - s < w[0] and w[-1] < c + s


def test_cheb_vs_eigh_and_expm():
    for sx, sy in [(3, 3), (3, 4)]:
        p = Patch(sx, sy, seed=0)
        w, Q = exact(p)
        ops = sv.Ops(p)
        rr = target.spectral_range(ops)
        c, s = target.cheb_params(rr["emin"], rr["emax"])
        v0 = rand_state(p.n, 3)
        for t in (3.41e-3, 0.05, 1.0):
            for sign in (-1, +1):
                ref = Q @ (np.exp(sign * 1j * w * t) * (Q.conj().T @ v0))
                info = {}
                out = target.evolve(torch.tensor(v0), ops, t, c, s, sign=sign, info=info)
                assert np.abs(out.numpy() - ref).max() < 1e-13, (sx, sy, t, sign)
                assert info["norm_dev"] <= 1e-13
        if p.n <= 9:
            H = dense_H(p)
            for t in (3.41e-3, 0.05):
                out = target.evolve(torch.tensor(v0), ops, t, c, s)
                ref = sla.expm(-1j * H * t) @ v0
                assert np.abs(out.numpy() - ref).max() < 1e-13, (sx, sy, t)


def test_K_and_K_plus_5():
    p = Patch(3, 4, seed=0)
    ops = sv.Ops(p)
    rr = target.spectral_range(ops)
    c, s = target.cheb_params(rr["emin"], rr["emax"])
    v0 = torch.tensor(rand_state(p.n, 4))
    for t in (3.41e-3, 0.05):
        K = target.cheb_terms(s * t)
        a = target.evolve(v0, ops, t, c, s, K=K)
        b = target.evolve(v0, ops, t, c, s, K=K + 5)
        assert float((a - b).abs().max()) <= 1e-15, t


def test_cheb_terms():
    # st ~ 0.22 at the 30-qubit operating point: about ten terms
    assert 8 <= target.cheb_terms(0.22) <= 12
    for tau in (1e-3, 0.22, 1.0, 5.0):
        K = target.cheb_terms(tau)
        assert 2 * abs(target.jv(K + 1, tau)) < 1e-16
        assert 2 * abs(target.jv(K, tau)) >= 1e-16


def test_diag_option_and_roundtrip():
    p = Patch(3, 4, seed=1)
    ops, ops_d = sv.Ops(p), sv.Ops(p, diag=True)
    rr = target.spectral_range(ops)
    c, s = target.cheb_params(rr["emin"], rr["emax"])
    v0 = torch.tensor(rand_state(p.n, 5))
    a = target.evolve(v0, ops, 0.05, c, s)
    b = target.evolve(v0, ops_d, 0.05, c, s)
    assert float((a - b).abs().max()) < 1e-15
    back = target.evolve(a, ops, 0.05, c, s, sign=+1)
    assert float((back - v0).abs().max()) < 1e-14
    assert torch.equal(v0, torch.tensor(rand_state(p.n, 5)))   # input untouched


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("PASS test_target")
