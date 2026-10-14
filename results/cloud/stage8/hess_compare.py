"""How different are the Hessians of J/J0 on different patches, and how well
does a small patch's Hessian precondition a larger one?"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from hhkl_toy import sv, target, ansatz, train
from hhkl_toy.lattice import Patch
t = 3.41e-3
OUT = os.path.dirname(os.path.abspath(__file__))

def hess(patch, L, init):
    p = Patch.from_name(patch, 0); ops = sv.Ops(p, diag=True)
    rr = target.spectral_range(ops); cc, ss = target.cheb_params(rr['emin'], rr['emax'])
    st = train.make_states(ops, [101, 102, 103, 104], t, cc, ss)
    seq = ansatz.slots(L); c0 = ansatz.trotter_params(seq, L, init)
    f = train.objective([ops], seq, t, [st]); J0 = f(c0, need_grad=False)[0]
    return train.fd_hessian(f, c0, J0, log=lambda m: None)

def resolved(w, floor=1e-9):
    a = np.abs(w); return a[a > floor * a.max()]

out = {}
for L, init in ((4, 'S2^2'), (6, 'Y4'), (10, 'S4')):
    Hs = {p: hess(p, L, init) for p in ('2x3', '3x3', '3x4', '4x4')}
    np.savez(f'{OUT}/H_L{L}.npz', **{k.replace('x', '_'): v for k, v in Hs.items()})
    Ht = Hs['4x4']; wt = np.abs(np.linalg.eigvalsh(Ht))
    ng = int((wt < 1e-9 * wt.max()).sum())          # directions any preconditioner floors
    k = len(wt) - ng
    raw = np.sort(wt)[::-1][:k]
    print(f"\nL={L} {init}: target 4x4, P={len(wt)}, resolved {k} (floored {ng}); raw condition {raw[0]/raw[-1]:.1e}", flush=True)
    for src in ('4x4', '3x4', '3x3', '2x3'):
        Hsrc = Hs[src]
        rel = np.linalg.norm(Hsrc - Ht) / np.linalg.norm(Ht)
        B, _ = train.preconditioner(Hsrc)
        m = np.sort(np.abs(np.linalg.eigvalsh(B.T @ Ht @ B)))[::-1][:k]
        ws = np.abs(np.linalg.eigvalsh(Hsrc)); ws = ws[ws > 1e-9 * ws.max()]
        dec = np.floor(np.log10(ws)).astype(int)
        tiers = {int(d): int((dec == d).sum()) for d in sorted(set(dec), reverse=True)}
        print(f"  B from {src}: rel diff {rel:.2f}; preconditioned |eig| {m[0]:.2f} .. {m[-1]:.2e}, "
              f"condition {m[0]/m[-1]:.1e};  H_src eigen-decades {tiers}", flush=True)
        out[f"L{L}_{src}"] = dict(rel=rel, cond_pre=float(m[0] / m[-1]), cond_raw=float(raw[0] / raw[-1]), tiers=tiers)
json.dump(out, open(f'{OUT}/hess_compare.json', 'w'), indent=1)
