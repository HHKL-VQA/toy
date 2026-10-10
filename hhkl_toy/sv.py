"""Statevector kernels in torch (PLAN.md §4.2-4.4).

A state is a contiguous tensor of shape (2**n,) or (2**n, m); the optional
trailing axis is a batch of m states (used by the dense tests).  Qubit i is bit
i of the row index.  Every gate is applied in place through strided views, so
the same code runs on CPU for small n and on one GPU for 30 qubits.

Bond gate (closed form, basis |00>,|01>,|10>,|11>):

    exp(-i[a(XX+YY) + z ZZ]) = diag(e^{-iz}) on {00, 11}
                               e^{iz} [[cos 2a, -i sin 2a], [-i sin 2a, cos 2a]] on {01, 10}
"""

import cmath
import math

import torch

CDTYPE = torch.complex128
RDTYPE = torch.float64

# Largest temporary (elements) created by a chunked kernel.  At 30 qubits a
# quarter of the state is 2**28 elements (4.3 GB), so the chunked kernels keep
# their scratch at 2**24 elements (268 MB) instead.
CHUNK = 1 << 24


class Ops:
    """Device-side tables for one patch: bonds per colour and field factors.

    sum_j h_j z_j(x) = f_hi[x >> k] + f_lo[x & (2**k - 1)], so the field slot
    and the field part of H are two broadcast multiplies, not n passes.
    """

    def __init__(self, patch, device="cpu", diag=False):
        self.patch = patch
        self.n = patch.n
        self.device = torch.device(device)
        self.colors = patch.colors
        self.bonds = patch.bonds
        self.k = self.n // 2
        h = torch.tensor(patch.h, dtype=RDTYPE, device=self.device)
        self.f_lo = _field_table(h[:self.k], self.device)
        self.f_hi = _field_table(h[self.k:], self.device)
        self.D = diag_D(self) if diag else None


def _field_table(h, device):
    """f[x] = sum_q h_q z_q(x) over len(h) qubits."""
    m = h.numel()
    x = torch.arange(1 << m, device=device)
    f = torch.zeros(1 << m, dtype=RDTYPE, device=device)
    for q in range(m):
        f += h[q] * (1.0 - 2.0 * ((x >> q) & 1).to(RDTYPE))
    return f


def _field_view(psi, ops):
    return psi.view(1 << (ops.n - ops.k), 1 << ops.k, -1)


def _bond_view(psi, n, i, j):
    """5-D view (high, bit j, middle, bit i, low) with i < j."""
    if i > j:
        i, j = j, i
    return psi.view(1 << (n - 1 - j), 2, 1 << (j - 1 - i), 2, -1)


def _blocks(v):
    """The four (bit j, bit i) blocks 00, 01, 10, 11 of a bond view."""
    return v[:, 0, :, 0, :], v[:, 0, :, 1, :], v[:, 1, :, 0, :], v[:, 1, :, 1, :]


def _chunks(*views):
    """Split equally shaped views into aligned pieces of <= CHUNK elements."""
    shape = views[0].shape
    numel = views[0].numel()
    if numel <= CHUNK:
        yield views
        return
    d = max(range(len(shape)), key=lambda k: shape[k])
    step = max(1, (shape[d] * CHUNK) // numel)
    for s in range(0, shape[d], step):
        w = min(step, shape[d] - s)
        yield tuple(v.narrow(d, s, w) for v in views)


def _dot(a, b):
    """sum conj(a) * b over equally shaped (strided) views, as a 0-d tensor."""
    acc = torch.zeros((), dtype=CDTYPE, device=a.device)
    for pa, pb in _chunks(a, b):
        acc += (pa.conj() * pb).sum()
    return acc


def vdot(a, b):
    """<a|b> of two full states, as a Python complex."""
    return complex(torch.vdot(a.reshape(-1), b.reshape(-1)))


def norm(a):
    return float(torch.linalg.vector_norm(a))


# ---------------------------------------------------------------------------
# gates

def apply_bond(psi, n, i, j, alpha, zeta):
    """psi <- exp(-i[alpha (XX+YY) + zeta ZZ])_{ij} psi, in place."""
    b00, b01, b10, b11 = _blocks(_bond_view(psi, n, i, j))
    em = cmath.exp(-1j * zeta)
    b00.mul_(em)
    b11.mul_(em)
    ep = cmath.exp(1j * zeta)
    c = ep * math.cos(2.0 * alpha)
    s = -1j * ep * math.sin(2.0 * alpha)
    for a, b in _chunks(b01, b10):
        ta = a.clone()
        a.mul_(c).add_(b, alpha=s)
        b.mul_(c).add_(ta, alpha=s)


def apply_color(psi, ops, g, alpha, zeta):
    """Colour slot G_g = prod_{b in g} exp(-i[alpha(XX+YY) + zeta ZZ]_b).

    Bonds of one colour are disjoint, so the order does not matter.  The
    inverse slot is apply_color(psi, ops, g, -alpha, -zeta).
    """
    for i, j in ops.colors[g]:
        apply_bond(psi, ops.n, i, j, alpha, zeta)


def apply_field(psi, ops, phi):
    """Field slot G_F = exp(-i phi sum_j h_j Z_j), in place."""
    v = _field_view(psi, ops)
    v.mul_(torch.exp((-1j * phi) * ops.f_hi)[:, None, None])
    v.mul_(torch.exp((-1j * phi) * ops.f_lo)[None, :, None])


# ---------------------------------------------------------------------------
# generator overlaps <lam| A |phi>, for the adjoint gradient

def color_overlaps(lam, phi, ops, g, total=None):
    """(<lam|sum_b (XX+YY)_b|phi>, <lam|sum_b ZZ_b|phi>) over colour g.

    Uses ZZ = 2 P_{00 or 11} - 1, so each bond costs one pass over the four
    blocks.  `total` = <lam|phi> may be passed in to avoid recomputing it.
    Returns 0-d device tensors (no host sync).
    """
    bonds = ops.colors[g]
    zero = torch.zeros((), dtype=CDTYPE, device=lam.device)
    if not bonds:
        return zero, zero
    if total is None:
        total = torch.vdot(lam.reshape(-1), phi.reshape(-1))
    xy, same = zero.clone(), zero.clone()
    for i, j in bonds:
        l00, l01, l10, l11 = _blocks(_bond_view(lam, ops.n, i, j))
        p00, p01, p10, p11 = _blocks(_bond_view(phi, ops.n, i, j))
        xy += 2.0 * (_dot(l01, p10) + _dot(l10, p01))
        same += _dot(l00, p00) + _dot(l11, p11)
    return xy, 2.0 * same - len(bonds) * total


def field_overlap(lam, phi, ops):
    """<lam| sum_j h_j Z_j |phi>, as a 0-d device tensor."""
    L, P = _field_view(lam, ops), _field_view(phi, ops)
    rows = max(1, CHUNK // (L.shape[1] * L.shape[2]))
    acc = torch.zeros((), dtype=CDTYPE, device=lam.device)
    for s in range(0, L.shape[0], rows):
        prod = L[s:s + rows].conj() * P[s:s + rows]
        acc += (prod.sum(dim=(1, 2)) * ops.f_hi[s:s + rows]).sum()
        acc += (prod.sum(dim=(0, 2)) * ops.f_lo).sum()
    return acc


# ---------------------------------------------------------------------------
# Hamiltonian

def diag_D(ops):
    """D(x) = sum_b z_i z_j + sum_j h_j z_j as a float64 vector (8.6 GB at n=30)."""
    n = ops.n
    D = torch.zeros(1 << n, dtype=RDTYPE, device=ops.device)
    for i, j in ops.bonds:
        b00, b01, b10, b11 = _blocks(_bond_view(D, n, i, j))
        b00.add_(1.0)
        b11.add_(1.0)
        b01.sub_(1.0)
        b10.sub_(1.0)
    v = D.view(1 << (n - ops.k), 1 << ops.k)
    v.add_(ops.f_hi[:, None])
    v.add_(ops.f_lo[None, :])
    return D


def apply_H(v, ops, out, scale=1.0):
    """out += scale * H v.  out must not alias v.

    With ops.D the diagonal is one fused pass; without it the ZZ terms are
    added bond by bond and the field by two broadcast passes.
    """
    n = ops.n
    if ops.D is not None:
        D = ops.D if v.dim() == 1 else ops.D[:, None]
        out.addcmul_(v, D, value=scale)
    else:
        for i, j in ops.bonds:
            o00, o01, o10, o11 = _blocks(_bond_view(out, n, i, j))
            v00, v01, v10, v11 = _blocks(_bond_view(v, n, i, j))
            o00.add_(v00, alpha=scale)
            o11.add_(v11, alpha=scale)
            o01.add_(v01, alpha=-scale)
            o10.add_(v10, alpha=-scale)
        ov, vv = _field_view(out, ops), _field_view(v, ops)
        ov.addcmul_(vv, ops.f_hi[:, None, None], value=scale)
        ov.addcmul_(vv, ops.f_lo[None, :, None], value=scale)
    for i, j in ops.bonds:
        _, o01, o10, _ = _blocks(_bond_view(out, n, i, j))
        _, v01, v10, _ = _blocks(_bond_view(v, n, i, j))
        o01.add_(v10, alpha=2.0 * scale)
        o10.add_(v01, alpha=2.0 * scale)
    return out


# ---------------------------------------------------------------------------
# states

def haar_state(n, seed, device="cpu", out=None):
    """Normalised complex Gaussian vector, regenerated from (seed, device).

    The same seed gives the same state on the same device type; CPU and CUDA
    generators differ, which is fine because only the Haar average matters.
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    if out is None:
        out = torch.empty(1 << n, dtype=CDTYPE, device=device)
    torch.randn(out.shape, dtype=CDTYPE, device=device, generator=g, out=out)
    out /= torch.linalg.vector_norm(out)
    return out
