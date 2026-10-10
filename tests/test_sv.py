"""Stage 2: statevector kernels against dense references."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.linalg as sla
import torch

from hhkl_toy import sv
from hhkl_toy.lattice import COLORS, Patch, dense_H

TOL = 1e-14


# --- dense single-bond operators built from bit arithmetic (independent of sv)

def dense_xy(n, i, j):
    d = 1 << n
    x = np.arange(d)
    M = np.zeros((d, d), dtype=np.complex128)
    differ = ((x >> i) & 1) != ((x >> j) & 1)
    M[x[differ] ^ ((1 << i) | (1 << j)), x[differ]] = 2.0
    return M


def dense_zz(n, i, j):
    x = np.arange(1 << n)
    return np.diag((1.0 - 2.0 * ((x >> i) & 1)) * (1.0 - 2.0 * ((x >> j) & 1))).astype(np.complex128)


def dense_field(p):
    x = np.arange(1 << p.n)
    f = sum(p.h[q] * (1.0 - 2.0 * ((x >> q) & 1)) for q in range(p.n))
    return np.diag(f).astype(np.complex128)


def rand_state(n, seed, m=None):
    rng = np.random.default_rng(seed)
    shape = (1 << n,) if m is None else (1 << n, m)
    v = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return v / np.linalg.norm(v, axis=0)


def T(a):
    return torch.tensor(a, dtype=sv.CDTYPE)


def maxdiff(a, b):
    a = a.numpy() if isinstance(a, torch.Tensor) else a
    b = b.numpy() if isinstance(b, torch.Tensor) else b
    return float(np.abs(a - b).max())


# --- tests

def test_bond_closed_form():
    rng = np.random.default_rng(1)
    for _ in range(5):
        a, z = rng.uniform(-2, 2, size=2)
        U = torch.eye(4, dtype=sv.CDTYPE).contiguous()
        sv.apply_bond(U, 2, 0, 1, a, z)
        ref = sla.expm(-1j * (a * dense_xy(2, 0, 1) + z * dense_zz(2, 0, 1)))
        assert maxdiff(U, ref) < TOL
        c, s = np.cos(2 * a), np.sin(2 * a)
        e = np.exp(1j * z)
        closed = np.array([[1 / e, 0, 0, 0], [0, e * c, -1j * e * s, 0],
                           [0, -1j * e * s, e * c, 0], [0, 0, 0, 1 / e]])
        assert maxdiff(U, closed) < TOL


def test_bond_vs_dense_n10():
    n = 10
    psi0 = rand_state(n, 2)
    rng = np.random.default_rng(3)
    for i, j in [(0, 9), (3, 4), (7, 2), (5, 6), (9, 8)]:
        a, z = rng.uniform(-1, 1, size=2)
        psi = T(psi0)
        sv.apply_bond(psi, n, i, j, a, z)
        ref = sla.expm(-1j * (a * dense_xy(n, i, j) + z * dense_zz(n, i, j))) @ psi0
        assert maxdiff(psi, ref) < TOL, (i, j)


def test_slots_vs_dense():
    for sx, sy in [(3, 3), (2, 5)]:
        p = Patch(sx, sy, seed=0)
        ops = sv.Ops(p)
        n = p.n
        psi0 = rand_state(n, 4)
        rng = np.random.default_rng(5)
        for g in COLORS:
            a, z = rng.uniform(-1, 1, size=2)
            XY = sum((dense_xy(n, i, j) for i, j in p.colors[g]), np.zeros((1 << n,) * 2))
            ZZ = sum((dense_zz(n, i, j) for i, j in p.colors[g]), np.zeros((1 << n,) * 2))
            psi = T(psi0)
            sv.apply_color(psi, ops, g, a, z)
            ref = sla.expm(-1j * (a * XY + z * ZZ)) @ psi0
            assert maxdiff(psi, ref) < TOL, (sx, sy, g)
        phi = 0.7
        psi = T(psi0)
        sv.apply_field(psi, ops, phi)
        ref = sla.expm(-1j * phi * dense_field(p)) @ psi0
        assert maxdiff(psi, ref) < TOL


def test_inverse_slots_and_norm_n12():
    p = Patch(3, 4, seed=0)
    ops = sv.Ops(p)
    psi0 = T(rand_state(p.n, 6))
    psi = psi0.clone()
    rng = np.random.default_rng(7)
    angles = [(g, *rng.uniform(-1, 1, size=2)) for g in COLORS + COLORS]
    for g, a, z in angles:
        sv.apply_color(psi, ops, g, a, z)
        sv.apply_field(psi, ops, a)
    assert abs(sv.norm(psi) - 1.0) < TOL
    for g, a, z in reversed(angles):
        sv.apply_field(psi, ops, -a)
        sv.apply_color(psi, ops, g, -a, -z)
    assert maxdiff(psi, psi0) < TOL


def test_little_endian():
    p = Patch(2, 3, seed=0)
    ops = sv.Ops(p)
    n = p.n
    for q in range(n):
        psi = torch.zeros(1 << n, dtype=sv.CDTYPE)
        psi[1 << q] = 1.0
        phi = 0.3
        sv.apply_field(psi, ops, phi)
        expect = np.exp(-1j * phi * (p.h.sum() - 2 * p.h[q]))
        assert abs(complex(psi[1 << q]) - expect) < TOL
    # a full swap (alpha = pi/4) moves the excitation from qubit i to qubit j
    i, j = 1, 4
    psi = torch.zeros(1 << n, dtype=sv.CDTYPE)
    psi[1 << i] = 1.0
    sv.apply_bond(psi, n, i, j, np.pi / 4, 0.0)
    assert abs(complex(psi[1 << j]) - (-1j)) < TOL
    assert abs(complex(psi[1 << i])) < TOL


def test_apply_H_vs_dense_n12():
    p = Patch(3, 4, seed=0)
    H = dense_H(p)
    v0 = rand_state(p.n, 8)
    o0 = rand_state(p.n, 9)
    for diag in (False, True):
        ops = sv.Ops(p, diag=diag)
        out = T(o0)
        sv.apply_H(T(v0), ops, out, scale=0.37)
        assert maxdiff(out, o0 + 0.37 * (H @ v0)) < TOL, diag
    # batch of states
    V0 = rand_state(p.n, 10, m=3)
    for diag in (False, True):
        out = torch.zeros(V0.shape, dtype=sv.CDTYPE)
        sv.apply_H(T(V0), sv.Ops(p, diag=diag), out)
        assert maxdiff(out, H @ V0) < TOL, diag


def test_diag_D():
    p = Patch(3, 3, seed=2)
    ops = sv.Ops(p, diag=True)
    assert maxdiff(ops.D.numpy(), np.real(np.diag(dense_H(p)))) < TOL


def test_overlaps_vs_dense():
    p = Patch(3, 3, seed=0)
    ops = sv.Ops(p)
    n = p.n
    lam, phi = rand_state(n, 11), rand_state(n, 12)
    for g in COLORS:
        XY = sum((dense_xy(n, i, j) for i, j in p.colors[g]), np.zeros((1 << n,) * 2))
        ZZ = sum((dense_zz(n, i, j) for i, j in p.colors[g]), np.zeros((1 << n,) * 2))
        xy, zz = sv.color_overlaps(T(lam), T(phi), ops, g)
        assert abs(complex(xy) - np.vdot(lam, XY @ phi)) < TOL, g
        assert abs(complex(zz) - np.vdot(lam, ZZ @ phi)) < TOL, g
    f = sv.field_overlap(T(lam), T(phi), ops)
    assert abs(complex(f) - np.vdot(lam, dense_field(p) @ phi)) < TOL


def test_chunked_kernels_agree():
    p = Patch(3, 4, seed=0)
    ops = sv.Ops(p)
    psi0, lam0 = T(rand_state(p.n, 13)), T(rand_state(p.n, 14))
    results = []
    old = sv.CHUNK
    try:
        for chunk in (old, 8):
            sv.CHUNK = chunk
            psi = psi0.clone()
            for g in COLORS:
                sv.apply_color(psi, ops, g, 0.3, -0.2)
            ov = [complex(x) for g in COLORS for x in sv.color_overlaps(lam0, psi, ops, g)]
            ov.append(complex(sv.field_overlap(lam0, psi, ops)))
            results.append((psi, np.array(ov)))
    finally:
        sv.CHUNK = old
    assert maxdiff(results[0][0], results[1][0]) < TOL
    assert np.abs(results[0][1] - results[1][1]).max() < TOL


def test_batch_matches_columns():
    p = Patch(3, 3, seed=0)
    ops = sv.Ops(p)
    B = T(rand_state(p.n, 15, m=4))
    cols = [B[:, k].clone() for k in range(4)]
    sv.apply_color(B, ops, "C", 0.4, 0.1)
    sv.apply_field(B, ops, 0.9)
    for k, c in enumerate(cols):
        sv.apply_color(c, ops, "C", 0.4, 0.1)
        sv.apply_field(c, ops, 0.9)
        assert maxdiff(B[:, k], c) < TOL


def test_haar_state():
    a = sv.haar_state(10, 123)
    b = sv.haar_state(10, 123)
    c = sv.haar_state(10, 124)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert abs(sv.norm(a) - 1.0) < TOL
    buf = torch.empty(1 << 10, dtype=sv.CDTYPE)
    sv.haar_state(10, 123, out=buf)
    assert torch.equal(buf, a)
    # complex Gaussian: real and imaginary parts carry equal weight
    a = sv.haar_state(16, 5)
    assert abs(float((a.real ** 2).sum()) - 0.5) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("PASS test_sv")
