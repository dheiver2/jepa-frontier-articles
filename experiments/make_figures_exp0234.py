"""
Figures for articles 2, 3 and 4 from results_exp02.csv, results_exp03.csv, results_exp04.csv.
Writes ../figs/fig02_world_model.pdf, fig03_causal.pdf, fig04_reasoning.pdf
(3.5 x 2.4 in, serif, TrueType embedded). Error bars: 95% bootstrap CI over seeds.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
W, H = 3.5, 2.4
RNG = np.random.default_rng(0)


def read(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))


def boot(x, n=5000):
    x = np.array([float(v) for v in x if v not in ("", "nan")], float)
    x = x[~np.isnan(x)]
    b = [np.mean(RNG.choice(x, size=len(x))) for _ in range(n)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return x.mean(), x.mean() - lo, hi - x.mean()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
    plt.close(fig)


def fig02():
    rows = read("results_exp02.csv")
    order = [("Random", "Random"), ("Recon", "Recon.\nmodel"), ("JEPA", "JEPA"), ("Known", "Known\nmodel")]
    fig, ax = plt.subplots(figsize=(W, H))
    cols = ["#999999", "#4C78A8", "#E45756", "#333333"]
    for i, (k, lab) in enumerate(order):
        m, lo, hi = boot([r["success"] for r in rows if r["method"] == k])
        ax.bar(i, m, yerr=[[lo], [hi]], color=cols[i], capsize=3, width=0.6)
        ax.text(i, min(m + hi + 0.03, 1.05), f"{m:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([l for _, l in order])
    ax.set_ylabel("Planning success rate")
    ax.set_ylim(0, 1.12)
    save(fig, "fig02_world_model.pdf")


def fig03():
    rows = read("results_exp03.csv")
    methods = ["Reg", "VAE", "JEPA", "Oracle"]
    cols = ["#4C78A8", "#72B7B2", "#E45756", "#333333"]
    groups = [("int_err_conf", "Confounded\nparent"), ("int_err_unconf", "Other\nparents")]
    fig, ax = plt.subplots(figsize=(W, H))
    w = 0.2
    for j, m in enumerate(methods):
        for g, (key, _) in enumerate(groups):
            mu, lo, hi = boot([r[key] for r in rows if r["method"] == m])
            ax.bar(g + (j - 1.5) * w, mu, w, yerr=[[lo], [hi]], color=cols[j], capsize=2,
                   label=m if g == 0 else None)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([l for _, l in groups])
    ax.set_ylabel("Interventional error\n(normalized by target variance)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig03_causal.pdf")


def fig04():
    rows = read("results_exp04.csv")
    fig, ax = plt.subplots(figsize=(W, H))
    spec = [("AR", "AR", "#4C78A8", "o"), ("JEPA", "JEPA + paraphrase", "#E45756", "s"),
            ("JEPA-nopara", "JEPA, no paraphrase", "#F2A33A", "^")]
    depths = sorted({int(r["depth"]) for r in rows})
    for m, lab, c, mk in spec:
        mus, los, his = [], [], []
        for k in depths:
            mu, lo, hi = boot([r["acc_para"] for r in rows if r["method"] == m and int(r["depth"]) == k])
            mus.append(mu); los.append(lo); his.append(hi)
        ax.errorbar(depths, mus, yerr=[los, his], color=c, marker=mk, ms=3.5, capsize=2, label=lab, lw=1)
    ax.axhline(0.1, ls=":", color="gray", lw=1)
    ax.set_xticks(depths)
    ax.set_xlabel("Reasoning depth $k$")
    ax.set_ylabel("Accuracy (paraphrased test)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig04_reasoning.pdf")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["2", "3", "4"]
    for w in which:
        {"2": fig02, "3": fig03, "4": fig04}[w]()
        print("fig", w, "ok")
