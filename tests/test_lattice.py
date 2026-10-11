"""Stage 1: lattice, colouring, fields, dense H."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hhkl_toy.lattice import COLORS, Patch, color_bonds, dense_H, fields

I2 = np.eye(2, dtype=np.complex128)
PAULI = {"X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
         "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
         "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128)}


def pauli_string(n, ops):
    """Kronecker product with qubit 0 as the least significant factor."""
    M = np.ones((1, 1), dtype=np.complex128)
    for q in reversed(range(n)):
        M = np.kron(M, PAULI[ops[q]] if q in ops else I2)
    return M


def pauli_sum_H(sx, sy, h):
    """Independent construction: nearest-neighbour bonds from geometry."""
    n = sx * sy
    H = np.zeros((1 << n, 1 << n), dtype=np.complex128)
    for r in range(sy):
        for c in range(sx):
            for dr, dc in ((0, 1), (1, 0)):
                if r + dr < sy and c + dc < sx:
                    i, j = r * sx + c, (r + dr) * sx + (c + dc)
                    for P in "XYZ":
                        H += pauli_string(n, {i: P, j: P})
    for j in range(n):
        H += h[j] * pauli_string(n, {j: "Z"})
    return H


def test_counts_5x6():
    p = Patch(5, 6)
    assert p.n == 30
    assert [len(p.colors[g]) for g in COLORS] == [12, 12, 15, 10]
    assert len(p.bonds) == 49


def test_colours_are_matchings_and_cover_all_bonds():
    for sx, sy in [(5, 6), (6, 5), (2, 3), (3, 2), (3, 3), (3, 4), (4, 5), (1, 4)]:
        cb = color_bonds(sx, sy)
        for g in COLORS:
            used = [q for b in cb[g] for q in b]
            assert len(used) == len(set(used)), (sx, sy, g)
        all_bonds = sorted(b for g in COLORS for b in cb[g])
        expect = sorted([(r * sx + c, r * sx + c + 1)
                         for r in range(sy) for c in range(sx - 1)]
                        + [(r * sx + c, (r + 1) * sx + c)
                           for r in range(sy - 1) for c in range(sx)])
        assert all_bonds == expect, (sx, sy)
        assert len(all_bonds) == sx * (sy - 1) + sy * (sx - 1)


def test_colour_definitions_5x6():
    sx = 5
    cb = color_bonds(5, 6)
    for i, j in cb["A"] + cb["B"]:
        assert j == i + 1
    for i, j in cb["C"] + cb["D"]:
        assert j == i + sx
    assert all((i % sx) % 2 == 0 for i, _ in cb["A"])
    assert all((i % sx) % 2 == 1 for i, _ in cb["B"])
    assert all((i // sx) % 2 == 0 for i, _ in cb["C"])
    assert all((i // sx) % 2 == 1 for i, _ in cb["D"])


def test_fields_reproducible():
    h = fields(30, 0)
    assert np.array_equal(h, np.random.default_rng(0).uniform(-1.0, 1.0, size=30))
    assert np.array_equal(h, Patch(5, 6, seed=0).h)
    assert not np.array_equal(h, fields(30, 1))
    assert np.all(np.abs(h) <= 1.0)
    # pinned so that a numpy RNG change cannot silently move the disorder
    np.testing.assert_allclose(h[:3], [0.27392337, -0.46042657, -0.91805295],
                               atol=1e-8)


def test_dense_H_matches_pauli_sum():
    for sx, sy in [(2, 3), (3, 2), (3, 3)]:
        p = Patch(sx, sy, seed=0)
        H = dense_H(p)
        ref = pauli_sum_H(sx, sy, p.h)
        assert np.abs(H - ref).max() < 1e-14, (sx, sy)
        assert np.abs(H - H.conj().T).max() == 0.0


def test_little_endian_and_u1():
    p = Patch(2, 3, seed=0)
    H = dense_H(p)
    n = p.n
    # basis state 1 has only qubit 0 flipped: Z_0 = -1, every other Z = +1
    zz_bonds = sum(1 if 0 not in b else -1 for b in p.bonds)
    assert abs(H[1, 1] - (zz_bonds + p.h.sum() - 2 * p.h[0])) < 1e-14
    # H conserves total Z
    x = np.arange(1 << n)
    pop = np.array([bin(v).count("1") for v in x])
    rows, cols = np.nonzero(np.abs(H) > 0)
    assert np.all(pop[rows] == pop[cols])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("PASS test_lattice")
