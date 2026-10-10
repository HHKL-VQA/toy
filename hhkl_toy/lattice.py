"""Rectangular open patch of the disordered 2D Heisenberg model (PLAN.md §3).

    H = sum_<ij> (XX + YY + ZZ)_ij + sum_j h_j Z_j ,   h_j ~ U[-1, 1]

Qubit i = r * sx + c (row r in [0, sy), column c in [0, sx)), and qubit i is bit i
of the statevector index (little-endian).  Z_i = +1 when that bit is 0.

The bonds are split into four colours, each a set of disjoint bonds:

    A  horizontal (r,c)-(r,c+1), c even        B  horizontal, c odd
    C  vertical   (r,c)-(r+1,c), r even        D  vertical,   r odd

and the field group is F = sum_j h_j Z_j.
"""

import numpy as np

COLORS = ("A", "B", "C", "D")


def parse_patch(name):
    """'5x6' -> (sx, sy) = (5 columns, 6 rows)."""
    sx, sy = (int(v) for v in name.lower().split("x"))
    return sx, sy


def color_bonds(sx, sy):
    """{colour: [(i, j), ...]} with i < j, in row-major order."""
    out = {g: [] for g in COLORS}
    for r in range(sy):
        for c in range(sx):
            i = r * sx + c
            if c + 1 < sx:
                out["A" if c % 2 == 0 else "B"].append((i, i + 1))
            if r + 1 < sy:
                out["C" if r % 2 == 0 else "D"].append((i, i + sx))
    return out


def fields(n, seed):
    """Disorder h_j, drawn exactly as in vqa/scripts/13 and 14."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=n)


class Patch:
    """An sx-by-sy open patch with its colouring and seeded fields."""

    def __init__(self, sx, sy, seed=0):
        self.sx, self.sy, self.seed = sx, sy, seed
        self.n = sx * sy
        self.name = f"{sx}x{sy}"
        self.h = fields(self.n, seed)
        self.colors = color_bonds(sx, sy)
        self.bonds = [b for g in COLORS for b in self.colors[g]]

    @classmethod
    def from_name(cls, name, seed=0):
        return cls(*parse_patch(name), seed=seed)

    def pauli_one_norm(self):
        """3 * bonds + sum |h_j|: a safe upper bound on ||H||."""
        return 3.0 * len(self.bonds) + float(np.abs(self.h).sum())


# ---------------------------------------------------------------------------
# dense references (small n only)

def _z(n):
    """z[i][x] = +-1, the Z_i eigenvalue of basis state x."""
    x = np.arange(1 << n)
    return [1.0 - 2.0 * ((x >> i) & 1) for i in range(n)]


def dense_group(patch, g):
    """Dense matrix of one group: a colour's sum of (XX+YY+ZZ), or 'F'."""
    n = patch.n
    d = 1 << n
    x = np.arange(d)
    z = _z(n)
    H = np.zeros((d, d), dtype=np.complex128)
    if g == "F":
        H[x, x] = sum(patch.h[j] * z[j] for j in range(n))
        return H
    diag = np.zeros(d)
    for i, j in patch.colors[g]:
        diag += z[i] * z[j]
        # XX + YY = 2 (|01><10| + |10><01|): flips both bits when they differ
        m = (1 << i) | (1 << j)
        differ = ((x >> i) & 1) != ((x >> j) & 1)
        H[x[differ] ^ m, x[differ]] += 2.0
    H[x, x] += diag
    return H


def dense_H(patch):
    """Dense block Hamiltonian, the sum of the five groups."""
    return sum(dense_group(patch, g) for g in COLORS + ("F",))
