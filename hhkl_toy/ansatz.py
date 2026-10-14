"""P1: palindromic HVA with one angle pair per colour slot (PLAN.md §4.1).

Layer l (0-based) sweeps A->B->C->D->F when l is even and F->D->C->B->A when l
is odd.  At a forward->backward boundary the two F slots merge, at a
backward->forward boundary the two A slots merge (saving 36 CNOTs on 5x6), so
an L-layer circuit has 4L+1 slots, each with its own parameters:

    colour slot g:  exp(-i t [c_a (XX+YY) + c_z ZZ]) on every bond of g
    field slot F:   exp(-i t c_f sum_j h_j Z_j)

The first slot in the list acts first: U = G_P ... G_2 G_1.
"""

import math

from . import sv
from .lattice import COLORS

FORWARD = COLORS + ("F",)
BACKWARD = tuple(reversed(FORWARD))
S4_U = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
Y4_W = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))     # Yoshida / Forest-Ruth 3-stage 4th order


def slots(L):
    """[{'g': group, 'layers': [l, ...], 'idx': (param indices)}] in time order."""
    seq = []
    for l in range(L):
        for g in (FORWARD if l % 2 == 0 else BACKWARD):
            if seq and seq[-1]["g"] == g:          # F,F or A,A at a layer boundary
                seq[-1]["layers"].append(l)
            else:
                seq.append({"g": g, "layers": [l]})
    p = 0
    for s in seq:
        width = 1 if s["g"] == "F" else 2
        s["idx"] = tuple(range(p, p + width))
        p += width
    return seq


def n_params(seq):
    return sum(len(s["idx"]) for s in seq)


def default_init(L):
    """Highest-order Trotter start that fits L layers (PLAN.md §4.1 table).

    S4 (5-stage 4th order) for L = 10m, Y4 (3-stage 4th order) for L = 6m,
    otherwise S2^(L/2).  At L = 30 both fit; S4^3 has the smaller error.
    """
    if L % 10 == 0:
        return "S4"
    if L % 6 == 0:
        return "Y4"
    return f"S2^{L // 2}"


def layer_weights(L, init):
    """Per-layer coefficient w_l: each slot of layer l carries t * w_l.

    S2(tau) is one forward plus one backward layer with every slot at tau/2.
    'S2^r' (L = 2r) repeats S2(t/r); 'S4' (L = 10m) is m copies of
    S2(ut) S2(ut) S2((1-4u)t) S2(ut) S2(ut) at t/m; 'Y4' (L = 6m) is m copies of
    the 3-stage 4th-order S2(wt) S2((1-2w)t) S2(wt), w = 1/(2 - 2^{1/3}).
    """
    if init.startswith("S2"):
        r = int(init.split("^")[1]) if "^" in init else 1
        if L != 2 * r:
            raise ValueError(f"{init} needs L = {2 * r}, got L = {L}")
        return [1.0 / (2 * r)] * L
    if init == "S4":
        if L % 10:
            raise ValueError(f"S4 needs L divisible by 10, got L = {L}")
        m = L // 10
        s = [S4_U, S4_U, 1.0 - 4.0 * S4_U, S4_U, S4_U] * m
        return [s[l // 2] / (2 * m) for l in range(L)]
    if init == "Y4":
        if L % 6:
            raise ValueError(f"Y4 needs L divisible by 6, got L = {L}")
        m = L // 6
        s = [Y4_W, 1.0 - 2.0 * Y4_W, Y4_W] * m
        return [s[l // 2] / (2 * m) for l in range(L)]
    raise ValueError(f"unknown init {init!r}")


def trotter_params(seq, L, init):
    """Parameter vector c reproducing the Trotter formula; merged slots add."""
    w = layer_weights(L, init)
    c = [0.0] * n_params(seq)
    for s in seq:
        val = sum(w[l] for l in s["layers"])
        for i in s["idx"]:
            c[i] = val
    return c


def apply_slot(psi, ops, s, c, t, inverse=False):
    sign = -1.0 if inverse else 1.0
    if s["g"] == "F":
        sv.apply_field(psi, ops, sign * t * c[s["idx"][0]])
    else:
        ia, iz = s["idx"]
        sv.apply_color(psi, ops, s["g"], sign * t * c[ia], sign * t * c[iz])


def apply_circuit(psi, ops, seq, c, t, inverse=False):
    """psi <- U psi (or U^dagger psi), in place."""
    order = reversed(seq) if inverse else seq
    for s in order:
        apply_slot(psi, ops, s, c, t, inverse)
    return psi


def costs(patch, seq):
    """CNOT count (3 per bond), CNOT depth (3 per non-empty colour slot), params."""
    cnot = sum(3 * len(patch.colors[s["g"]]) for s in seq if s["g"] != "F")
    depth = sum(3 for s in seq if s["g"] != "F" and patch.colors[s["g"]])
    return dict(cnot=cnot, cnot_depth=depth, n_params=n_params(seq),
                n_slots=len(seq))


def costs_formula(L):
    """PLAN.md §4.1 closed forms for the 5x6 patch."""
    k = (L - 1) // 2
    return dict(cnot=147 * L - 36 * k, cnot_depth=12 * L - 3 * k,
                n_params=2 * (4 * L - k) + (L - math.ceil((L - 1) / 2)))
