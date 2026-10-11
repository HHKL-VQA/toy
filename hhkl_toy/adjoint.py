"""Residual-vector loss and adjoint gradient (PLAN.md §4.4).

For one Haar state psi with cached V psi:

    r = U psi - e^{i phi} V psi,   e^{i phi} = <V psi|U psi> / |<V psi|U psi>|,
    J = ||r||^2 / 2  (= 1 - |<V psi|U psi>|, but computed without cancellation)

With U = G_P ... G_1 and G_k = exp(-i theta_k A_k), theta = t c:

    dJ/dtheta_k = Im <lam_k| A_k |phi_k>,   lam_k = (G_P..G_{k+1})^dag r,
                                            phi_k = G_k..G_1 psi,

swept backwards from lam = r, phi = U psi by undoing one slot at a time.  phi
is optimal in J, so it is held fixed (envelope theorem).  At the end phi is back
at psi; ||phi - psi|| against a regenerated psi is the drift diagnostic.

Memory: V psi (cached by the caller) plus two working vectors per state.
"""

import cmath

import torch

from . import ansatz, sv


def state_loss_grad(ops, seq, c, t, make_psi, vpsi, need_grad=True):
    """Loss and gradient for one state.

    make_psi(out=None) regenerates psi (into `out` if given); vpsi = V psi.
    Returns (J, grad as a list over c or None, info dict).
    """
    phi = ansatz.apply_circuit(make_psi(), ops, seq, c, t)
    ov = sv.vdot(vpsi, phi)
    ph = ov / abs(ov)
    lam = phi.clone()
    lam.sub_(vpsi, alpha=ph)                       # r = U psi - e^{i phi} V psi
    J = 0.5 * sv.norm(lam) ** 2
    info = dict(J=J, overlap_abs=abs(ov), phase=cmath.phase(ov))
    if not need_grad:
        return J, None, info

    parts = []                                     # (param index, overlap tensor)
    for s in reversed(seq):
        if s["g"] == "F":
            parts.append((s["idx"][0], sv.field_overlap(lam, phi, ops)))
        else:
            xy, zz = sv.color_overlaps(lam, phi, ops, s["g"])
            parts += [(s["idx"][0], xy), (s["idx"][1], zz)]
        ansatz.apply_slot(phi, ops, s, c, t, inverse=True)
        ansatz.apply_slot(lam, ops, s, c, t, inverse=True)

    im = torch.stack([v for _, v in parts]).imag.cpu().tolist()
    grad = [0.0] * len(c)
    for (i, _), g in zip(parts, im):
        grad[i] += t * g                           # theta = t c
    psi = make_psi(out=lam)                        # reuse lam's memory
    info["drift"] = sv.norm(phi.sub_(psi))
    return J, grad, info


def loss_grad(ops, seq, c, t, states, need_grad=True):
    """Mean loss and gradient over states = [(make_psi, vpsi), ...] on one device."""
    Js, grads, infos = [], [], []
    for make_psi, vpsi in states:
        J, g, info = state_loss_grad(ops, seq, c, t, make_psi, vpsi, need_grad)
        Js.append(J)
        grads.append(g)
        infos.append(info)
    M = len(states)
    J = sum(Js) / M
    grad = [sum(col) / M for col in zip(*grads)] if need_grad else None
    return J, grad, infos


def eps_rhs(J):
    """Normalised Hilbert-Schmidt error estimate sqrt(2J) (Haar average)."""
    return (2.0 * J) ** 0.5
