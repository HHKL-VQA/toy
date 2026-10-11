"""Small-patch scan at t*: init and trained eps for S2^r, Y4 (L=6), S4 (L=10); optional transfer."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from hhkl_toy import sv, target, ansatz, train, certify
from hhkl_toy.lattice import Patch
t = 3.41e-3
OUT = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [(2, 'S2^1'), (4, 'S2^2'), (6, 'S2^3'), (6, 'Y4'), (10, 'S4')]

def problem(patch):
    p = Patch.from_name(patch, 0); ops = sv.Ops(p, diag=True)
    rr = target.spectral_range(ops); cc, ss = target.cheb_params(rr['emin'], rr['emax'])
    st = train.make_states(ops, [101, 102, 103, 104], t, cc, ss)
    return p, ops, cc, ss, st

def hess_on(patch, L, init):
    p, ops, cc, ss, st = problem(patch)
    seq = ansatz.slots(L); c0 = ansatz.trotter_params(seq, L, init)
    f = train.objective([ops], seq, t, [st]); J0 = f(c0, need_grad=False)[0]
    return train.fd_hessian(f, c0, J0, log=lambda m: None)

def run(patch, transfer_from=()):
    p, ops, cc, ss, st = problem(patch)
    rows = []
    for L, init in CONFIGS:
        seq = ansatz.slots(L); c0 = ansatz.trotter_params(seq, L, init)
        f = train.objective([ops], seq, t, [st]); J0 = f(c0, need_grad=False)[0]
        e0 = certify.certify([ops], seq, c0, t, cc, ss, tol=1e-6)['eps_spec']
        variants = [('own', None)] + [(s, s) for s in transfer_from if L >= 6]
        for name, src in variants:
            t1 = time.perf_counter()
            H = train.fd_hessian(f, c0, J0, log=lambda m: None) if src is None else hess_on(src, L, init)
            th = time.perf_counter() - t1
            B = train.preconditioner(H)[0]
            r = train.lbfgs(f, c0, J0, B=B, log=lambda m: None)
            e1 = certify.certify([ops], seq, r['c_final'], t, cc, ss, tol=1e-6)['eps_spec']
            row = dict(patch=patch, n=p.n, bonds=len(p.bonds), L=L, init=init, precond=name,
                       J0=J0, J_final=r['J_final'], iters=r['iters'], evals=r['evals'],
                       hess_evals=(len(c0) + 1) if src is None else 0, stop=r['stop_reason'],
                       eps_init=e0, eps_trained=e1, hess_s=th)
            rows.append(row)
            print(f"{patch} L={L:2d} {init:5s} [{name:4s}] eps {e0:.3e} -> {e1:.3e} (x{e0/e1:.2f})  "
                  f"J/J0 {r['J_final']/J0:.4f}  its {r['iters']} evals {r['evals']}+{row['hess_evals']}  ({r['stop_reason'][:20]})", flush=True)
        json.dump(rows, open(f'{OUT}/scan_{patch}.json', 'w'))

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2:])
