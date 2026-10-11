"""(a) L=6 Y4 trained without plateau rule; (b) L=8 = Y4 + identity pair, trained."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hhkl_toy import ansatz, train, certify
from scan import problem, t
patch = sys.argv[1]
p, ops, cc, ss, st = problem(patch)
rows = []
for L, label in ((6, 'Y4 long'), (8, 'Y4+id'), (8, 'S2^4')):
    seq = ansatz.slots(L)
    if label == 'Y4+id':
        w = ansatz.layer_weights(6, 'Y4') + [0.0, 0.0]
        c0 = [0.0] * ansatz.n_params(seq)
        for s in seq:
            for i in s['idx']: c0[i] = sum(w[l] for l in s['layers'])
    else:
        c0 = ansatz.trotter_params(seq, L, 'Y4' if L == 6 else 'S2^4')
    f = train.objective([ops], seq, t, [st]); J0 = f(c0, need_grad=False)[0]
    e0 = certify.certify([ops], seq, c0, t, cc, ss, tol=1e-6)['eps_spec']
    B = train.preconditioner(train.fd_hessian(f, c0, J0, log=lambda m: None))[0]
    kw = dict(window=0, maxiter=300) if label == 'Y4 long' else {}
    r = train.lbfgs(f, c0, J0, B=B, log=lambda m: None, **kw)
    e1 = certify.certify([ops], seq, r['c_final'], t, cc, ss, tol=1e-6)['eps_spec']
    cost = ansatz.costs(p, seq)
    rows.append(dict(patch=patch, bonds=len(p.bonds), L=L, label=label, eps_init=e0, eps_trained=e1,
                     J_ratio=r['J_final'] / J0, iters=r['iters'], stop=r['stop_reason'], cnot=cost['cnot']))
    print(f"{patch} L={L} {label:8s}: eps {e0:.3e} -> {e1:.3e} (x{e0/e1:.2f}) J/J0 {r['J_final']/J0:.4f} its {r['iters']} ({r['stop_reason'][:25]})", flush=True)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'y4ext_{patch}.json'), 'w'))
