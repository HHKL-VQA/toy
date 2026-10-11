import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S8 = os.path.dirname(os.path.abspath(__file__))
pts = {}
for p in ("3x3", "3x4", "4x4"):
    for r in json.load(open(f"{S8}/scan_{p}.json")):
        if r["precond"] == "own":
            pts.setdefault((r["L"], r["init"]), []).append((r["bonds"], r["eps_init"], r["eps_trained"]))
for p, b in (("4x5", 31), ("4x6", 38)):
    for L in (2, 4, 6, 10):
        R = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results", f"{p}_s0_t0.00341_L{L}.json")))
        pts.setdefault((L, R["init"]), []).append((b, R["result"]["eps_spec_init"], R["result"]["eps_spec_reported"]))
colors = {(2, "S2^1"): "#9aa5b1", (4, "S2^2"): "#5b8bd0", (6, "S2^3"): "#2a5aa8", (6, "Y4"): "#d9822b", (10, "S4"): "#2f9e5b"}
names = {(2, "S2^1"): "L=2 (S2)", (4, "S2^2"): "L=4 (S2²)", (6, "S2^3"): "L=6 (S2³)", (6, "Y4"): "L=6 (Y4)", (10, "S4"): "L=10 (S4)"}
fig, ax = plt.subplots(figsize=(8.6, 4.6))
xs = np.linspace(10, 52, 50)
for key, rows in pts.items():
    rows = sorted(rows); b = np.array([r[0] for r in rows])
    for idx, ls, mk, alpha in ((1, "--", "o", 0.45), (2, "-", "s", 1.0)):
        e = np.array([r[idx] for r in rows])
        a, c = np.polyfit(b[-3:], e[-3:], 1)
        ax.plot(b, e, mk, color=colors[key], alpha=alpha, ms=5)
        ax.plot(xs, a * xs + c, ls, color=colors[key], alpha=alpha, lw=1.2,
                label=f"{names[key]}, {'Trotter' if idx == 1 else 'trained'}")
        ax.plot([49], [a * 49 + c], mk, color=colors[key], alpha=alpha, ms=7, mfc="white")
ax.axhline(1.71e-9, color="#c0392b", lw=1.2, ls=":")
ax.text(11, 2.2e-9, r"$\epsilon_\mathrm{blk}=1.71\times10^{-9}$", color="#c0392b", fontsize=8)
ax.axvline(49, color="gray", lw=0.8, ls=":")
ax.text(47.5, 3e-11, "5×6 (49 bonds)", rotation=90, fontsize=8, color="gray", va="bottom")
ax.set_yscale("log"); ax.set_xlabel("bonds in block"); ax.set_ylabel(r"certified $\epsilon_\mathrm{spec}$")
ax.set_title(r"$t^\star=3.41\times10^{-3}$, seed 0: 3×3, 3×4, 4×4 (cloud CPU), 4×5, 4×6 (RTX 2060)", fontsize=9)
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=150)
