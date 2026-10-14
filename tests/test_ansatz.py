"""Stage 4: P1 slot sequence, merging, Trotter initial angles, cost table."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.linalg as sla
import torch

from hhkl_toy import ansatz, sv
from hhkl_toy.lattice import Patch, dense_group, dense_H

TOL = 1e-14


def bond_unitary(n, i, j, a, z):
    """exp(-i[a(XX+YY) + z ZZ]) on qubits i, j as a dense matrix (bit arithmetic)."""
    d = 1 << n
    x = np.arange(d)
    same = ((x >> i) & 1) == ((x >> j) & 1)
    diff = ~same
    U = np.zeros((d, d), dtype=np.complex128)
    U[x[same], x[same]] = np.exp(-1j * z)
    U[x[diff], x[diff]] = np.exp(1j * z) * np.cos(2 * a)
    U[x[diff] ^ ((1 << i) | (1 << j)), x[diff]] = -1j * np.exp(1j * z) * np.sin(2 * a)
    return U


def E(p, g, tau):
    """exp(-i tau H_g) exactly: a product of disjoint bond unitaries, or a diagonal.

    (An eigh-based expm of the 512-dim group matrix carries ~1e-15 error per
    factor, which accumulates past 1e-14 over the ~50 factors of S4.)
    """
    if g == "F":
        x = np.arange(1 << p.n)
        f = sum(p.h[q] * (1.0 - 2.0 * ((x >> q) & 1)) for q in range(p.n))
        return np.diag(np.exp(-1j * tau * f))
    M = np.eye(1 << p.n, dtype=np.complex128)
    for i, j in p.colors[g]:
        M = bond_unitary(p.n, i, j, tau, tau) @ M
    return M


def test_reference_factors():
    """The dense group factors equal expm of the dense group Hamiltonians."""
    p = Patch(3, 3, 0)
    for g in ansatz.FORWARD:
        ref = sla.expm(-0.37j * dense_group(p, g))
        assert np.abs(E(p, g, 0.37) - ref).max() < 1e-14, g


def dense_S2(p, tau):
    """Strang splitting built directly from the five group exponentials."""
    M = np.eye(1 << p.n, dtype=np.complex128)
    for g in ansatz.FORWARD + ansatz.BACKWARD:
        M = E(p, g, tau / 2) @ M
    return M


def dense_formula(p, t, init):
    M = np.eye(1 << p.n, dtype=np.complex128)
    if init == "S4":
        u = ansatz.S4_U
        steps = [u * t, u * t, (1 - 4 * u) * t, u * t, u * t]
    elif init == "Y4":
        w = ansatz.Y4_W
        steps = [w * t, (1 - 2 * w) * t, w * t]
    else:
        r = int(init.split("^")[1])
        steps = [t / r] * r
    for tau in steps:
        M = dense_S2(p, tau) @ M
    return M


def circuit_unitary(p, L, init, t):
    ops = sv.Ops(p)
    seq = ansatz.slots(L)
    c = ansatz.trotter_params(seq, L, init)
    U = torch.eye(1 << p.n, dtype=sv.CDTYPE)
    ansatz.apply_circuit(U, ops, seq, c, t)
    return U.numpy()


def test_slot_structure():
    for L in range(1, 13):
        seq = ansatz.slots(L)
        groups = [s["g"] for s in seq]
        assert len(seq) == 4 * L + 1
        assert all(a != b for a, b in zip(groups, groups[1:]))
        if L % 2 == 0:
            assert groups == groups[::-1]                 # palindromic
        merged = [s for s in seq if len(s["layers"]) == 2]
        assert len(merged) == L - 1
        for s in merged:
            l0, l1 = s["layers"]
            assert l1 == l0 + 1
            assert s["g"] == ("F" if l0 % 2 == 0 else "A")
        idx = [i for s in seq for i in s["idx"]]
        assert idx == list(range(ansatz.n_params(seq)))


def test_cost_table_5x6():
    p = Patch(5, 6)
    table = {2: (294, 24, 17), 4: (552, 45, 32), 6: (810, 66, 47), 10: (1326, 108, 77)}
    for L, (cnot, depth, npar) in table.items():
        c = ansatz.costs(p, ansatz.slots(L))
        assert (c["cnot"], c["cnot_depth"], c["n_params"]) == (cnot, depth, npar), L
    for L in range(1, 21):
        c = ansatz.costs(p, ansatz.slots(L))
        f = ansatz.costs_formula(L)
        assert all(c[k] == f[k] for k in f), L


def test_trotter_values():
    L = 10
    seq = ansatz.slots(L)
    c = ansatz.trotter_params(seq, L, "S4")
    u = ansatz.S4_U
    assert abs(u - 0.41449) < 1e-5
    first = seq[0]
    assert first["g"] == "A" and abs(c[first["idx"][0]] - u / 2) < 1e-15
    assert abs(u / 2 - 0.20724) < 1e-5 and abs((1 - 4 * u) / 2 + 0.32899) < 1e-5
    # an unmerged slot of the middle pair (layers 4, 5) carries (1 - 4u)/2
    mid = [s for s in seq if s["layers"] == [4] and s["g"] == "C"][0]
    assert abs(c[mid["idx"][0]] - (1 - 4 * u) / 2) < 1e-15
    # merged slots add the neighbouring coefficients
    for s in seq:
        w = ansatz.layer_weights(L, "S4")
        assert abs(c[s["idx"][0]] - sum(w[l] for l in s["layers"])) < 1e-15
    # total time carried by every group is t
    for g in ansatz.FORWARD:
        tot = sum(c[s["idx"][0]] for s in seq if s["g"] == g)
        assert abs(tot - 1.0) < 1e-14, g
    assert [ansatz.default_init(L) for L in (2, 4, 6, 8, 10, 12, 30)] == \
        ["S2^1", "S2^2", "Y4", "S2^4", "S4", "Y4", "S4"]
    for L in (2, 4, 6):
        seq = ansatz.slots(L)
        c = ansatz.trotter_params(seq, L, f"S2^{L // 2}")
        for s in seq:
            assert np.allclose([c[i] for i in s["idx"]], len(s["layers"]) / L)


def test_circuit_equals_direct_trotter():
    for sx, sy in [(3, 3), (2, 3), (3, 2)]:
        p = Patch(sx, sy, 0)
        for L, init in ((2, "S2^1"), (4, "S2^2"), (6, "S2^3"), (6, "Y4"), (10, "S4")):
            for t in (0.05, 0.3):
                U = circuit_unitary(p, L, init, t)
                ref = dense_formula(p, t, init)
                assert np.abs(U - ref).max() < TOL, (sx, sy, L, init, t)


def test_order_of_accuracy():
    """S2 error ~ t^3 and S4 error ~ t^5: the palindrome is really symmetric."""
    p = Patch(3, 3, 0)
    w, Q = np.linalg.eigh(dense_H(p))

    def err(L, t):
        V = (Q * np.exp(-1j * w * t)) @ Q.conj().T
        return np.linalg.norm(circuit_unitary(p, L, ansatz.default_init(L), t) - V, 2)

    r2 = err(2, 0.1) / err(2, 0.05)          # 7.65
    r4 = err(10, 0.1) / err(10, 0.05)        # 30.7
    assert 7.0 < r2 < 9.0, r2
    assert 28.0 < r4 < 36.0, r4


def test_y4_init():
    """L = 6 at Y4 angles is the 3-stage 4th-order formula (error ~ t^5)."""
    p = Patch(3, 3, 0)
    for t in (0.05, 0.3):
        assert np.abs(circuit_unitary(p, 6, "Y4", t) - dense_formula(p, t, "Y4")).max() < TOL
    w, Q = np.linalg.eigh(dense_H(p))

    def err(t):
        V = (Q * np.exp(-1j * w * t)) @ Q.conj().T
        return np.linalg.norm(circuit_unitary(p, 6, "Y4", t) - V, 2)

    # Y4's sub-steps are large (1 - 2w = -1.70), so it turns asymptotic later than S4
    assert 28.0 < err(0.05) / err(0.025) < 36.0              # 30.7
    c = ansatz.trotter_params(ansatz.slots(6), 6, "Y4")
    for g in ansatz.FORWARD:            # every group still carries total time t
        tot = sum(c[s["idx"][0]] for s in ansatz.slots(6) if s["g"] == g)
        assert abs(tot - 1.0) < 1e-14, g


def test_inverse_circuit():
    p = Patch(3, 4, 0)
    ops = sv.Ops(p)
    seq = ansatz.slots(4)
    c = list(np.random.default_rng(0).uniform(-1, 1, ansatz.n_params(seq)))
    psi0 = sv.haar_state(p.n, 1)
    psi = ansatz.apply_circuit(psi0.clone(), ops, seq, c, 0.3)
    ansatz.apply_circuit(psi, ops, seq, c, 0.3, inverse=True)
    assert float((psi - psi0).abs().max()) < TOL


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("PASS test_ansatz")
