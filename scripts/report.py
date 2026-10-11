"""Summarise results/*.json into PLAN.md §8 Table 1 (markdown).

    python scripts/report.py                      # all complete runs in results/
    python scripts/report.py --plot fig1.png      # also Figure 1 (needs matplotlib)

Writes results/report.md and prints it.
"""

import argparse
import glob
import json
import os
import sys


def fmt(x, spec=".3e"):
    return "-" if x is None else format(x, spec)


def load(out_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_s*_t*_L*.json"))):
        with open(path, encoding="utf-8") as fh:
            R = json.load(fh)
        if "train" in R:
            R["_path"] = path
            runs.append(R)
    return runs


def table(runs):
    groups = {}
    for R in runs:
        groups.setdefault((R["patch"], R["seed"], R["t"]), []).append(R)
    lines = []
    for (patch, seed, t), rs in sorted(groups.items()):
        eb = rs[0].get("eps_blk")
        lines += [f"### {patch}, seed {seed}, t = {t:g}"
                  + (f", eps_blk = {eb:.3g}" if eb else ""), "",
                  "| L | CNOT | depth | params | eps_spec (Trotter) | eps_spec (trained) "
                  "| improvement | eps_spec / eps_blk | eps_rHS (trained) | spectral / rHS |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for R in sorted(rs, key=lambda r: r["L"]):
            res = R.get("result", {})
            e_i, e_t = res.get("eps_spec_init"), res.get("eps_spec_reported")
            note = " (init kept)" if res.get("fallback_to_init") else ""
            lines.append(
                f"| {R['L']} ({R['init']}) | {R['cnot']} | {R['cnot_depth']} | {R['n_params']} "
                f"| {fmt(e_i)} | {fmt(e_t)}{note} | {fmt(res.get('improvement'), '.2f')} "
                f"| {fmt(res.get('ratio_to_eps_blk'), '.3g')} "
                f"| {fmt(R['eps_rhs']['trained'])} | {fmt(res.get('spectral_over_rhs'), '.2f')} |")
        lines += ["", "| L | J0 | J final | iters | stop | drift | held-out diff | "
                      "cert its (init / trained) | starts agree | wall |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for R in sorted(rs, key=lambda r: r["L"]):
            tr, ho, ce = R["train"], R.get("heldout", {}), R.get("certify", {})
            its = " / ".join(",".join(str(r["iters"]) for r in ce[w]["runs"])
                             if w in ce else "-" for w in ("init", "trained"))
            agree = "/".join(str(ce[w]["start_vectors_agree"]) if w in ce else "-"
                             for w in ("init", "trained"))
            hd = ho.get("rel_diff_final")
            lines.append(
                f"| {R['L']} | {tr['J0']:.3e} | {tr['J_final']:.3e} | {tr['iters']} "
                f"| {tr['stop_reason'].split(':')[0]} | {tr['drift_max']:.1e} "
                f"| {'-' if hd is None else f'{100 * hd:.2f} %'} | {its} | {agree} "
                f"| {R['env']['wall_total_s'] / 3600:.2f} h |")
        lines.append("")
    return "\n".join(lines)


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    groups = {}
    for R in runs:
        if "result" in R:
            groups.setdefault((R["patch"], R["seed"], R["t"]), []).append(R)
    for (patch, seed, t), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["cnot"])
        x = [r["cnot"] for r in rs]
        lab = f"{patch} s{seed} t={t:g}"
        ax.plot(x, [r["result"]["eps_spec_init"] for r in rs], "o--", label=f"Trotter, {lab}")
        ax.plot(x, [r["result"]["eps_spec_reported"] for r in rs], "s-", label=f"trained, {lab}")
        if rs[0].get("eps_blk"):
            ax.axhline(rs[0]["eps_blk"], color="gray", lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("CNOT count")
    ax.set_ylabel(r"$\epsilon_\mathrm{spec}$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot", default=None, help="PNG path for Figure 1")
    a = ap.parse_args()
    runs = load(a.out)
    if not runs:
        print(f"FAIL report: no result JSON in {a.out}/")
        sys.exit(1)
    text = "# Results\n\n" + table(runs)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    if a.plot:
        plot(runs, a.plot)
    n_done = sum("result" in R for R in runs)
    print(f"PASS report: {len(runs)} runs ({n_done} certified) -> {a.out}/report.md")


if __name__ == "__main__":
    main()
