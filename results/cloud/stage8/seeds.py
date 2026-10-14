"""Disorder-seed spread at t*: init and trained eps_spec for L=6 (Y4) and L=10 (S4).

    python results/cloud/stage8/seeds.py 4x4 0 1 2 3 4 5 6 7 8 9
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from hhkl_toy import sv, target, ansatz, train, certify
from hhkl_toy.lattice import Patch
t = 3.41e-3
OUT = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [(6, 'Y4'), (10, 'S4')]

patch, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
path = f'{OUT}/seeds_{patch}.json'
rows = json.load(open(path)) if os.path.exists(path) else []
done = {(r['seed'], r['L']) for r in rows}
for seed in seeds:
    p = Patch.from_name(patch, seed); ops = sv.Ops(p, diag=True)
    rr = target.spectral_range(ops); cc, ss = target.cheb_params(rr['emin'], rr['emax'])
    st = train.make_states(ops, [101, 102, 103, 104], t, cc, ss)
    for L, init in CONFIGS:
        if (seed, L) in done:
            continue
        t1 = time.perf_counter()
        seq = ansatz.slots(L); c0 = ansatz.trotter_params(seq, L, init)
        f = train.objective([ops], seq, t, [st]); J0 = f(c0, need_grad=False)[0]
        e0 = certify.certify([ops], seq, c0, t, cc, ss, tol=1e-6)['eps_spec']
        B = train.preconditioner(train.fd_hessian(f, c0, J0, log=lambda m: None))[0]
        r = train.lbfgs(f, c0, J0, B=B, log=lambda m: None)
        e1 = certify.certify([ops], seq, r['c_final'], t, cc, ss, tol=1e-6)['eps_spec']
        rows.append(dict(patch=patch, seed=seed, bonds=len(p.bonds), h_absmean=float(abs(p.h).mean()),
                         L=L, init=init, J0=J0, J_final=r['J_final'], iters=r['iters'],
                         stop=r['stop_reason'], eps_init=e0, eps_trained=e1,
                         wall_s=time.perf_counter() - t1))
        print(f"{patch} seed {seed} L={L:2d} {init}: eps {e0:.3e} -> {e1:.3e} (x{e0/e1:.2f})  "
              f"its {r['iters']}  ({time.perf_counter() - t1:.0f}s)", flush=True)
        json.dump(rows, open(path, 'w'), indent=1)
