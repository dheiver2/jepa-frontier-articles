"""
Figures for articles 5, 6 and 7 from results_exp05/06/07.csv (and fits_exp06.json).
Style as in make_figures.py: serif, 3.5 x 2.4 in, TrueType-embedded PDF. Mean and 95% bootstrap CI over seeds.
"""
import csv
import json
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


def ci(x, n=3000):
    x = np.asarray(x, float)
    b = [np.mean(RNG.choice(x, size=len(x))) for _ in range(n)]
    return float(np.mean(x)), *np.percentile(b, [2.5, 97.5])


def read(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))


def fig05():
    rows = read("results_exp05.csv")
    Ks = sorted({int(r["K"]) for r in rows})
    fig, ax = plt.subplots(figsize=(W, H))
    style = {"H-JEPA": ("o-", "C0"), "H-JEPA-noSec": ("s--", "C2"), "Flat-concat": ("^-", "C1"),
             "Flat-first": ("v-", "C3"), "Raw-mean": ("x:", "0.4")}
    for m, (fmt, c) in style.items():
        mu, lo, hi = [], [], []
        for K in Ks:
            v = [float(r["r2"]) for r in rows if r["method"] == m and int(r["K"]) == K]
            a, l, h = ci(v)
            mu.append(a), lo.append(l), hi.append(h)
        ax.errorbar(Ks, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], fmt=fmt, color=c, ms=3, lw=1,
                    capsize=1.5, label=m)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks)
    ax.set_xlabel("sections per document $K$")
    ax.set_ylabel("slow-state $R^2$")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig05_hierarchical.pdf"), bbox_inches="tight")


def fig06():
    rows = read("results_exp06.csv")
    Cs = sorted({float(r["budget"]) for r in rows})
    widths = sorted({int(r["width"]) for r in rows})
    fig, ax = plt.subplots(figsize=(W, H))
    for m, c in [("JEPA", "C0"), ("MAE", "C1"), ("CL", "C2")]:
        mu, lo, hi = [], [], []
        for C in Cs:
            best = None
            for w in widths:
                v = [float(r["mse"]) for r in rows if r["method"] == m and int(r["width"]) == w and float(r["budget"]) == C]
                if not v:
                    continue
                a = ci(v)
                if best is None or a[0] < best[0]:
                    best = a
            mu.append(best[0]), lo.append(best[1]), hi.append(best[2])
        ax.errorbar(Cs, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], fmt="o-", color=c, ms=3, lw=1,
                    capsize=1.5, label={"CL": "Contrastive"}.get(m, m))
    ax.axhline(0.09, color="0.5", ls=":", lw=0.8)
    ax.text(Cs[0], 0.095, "label-noise floor", fontsize=6, color="0.4", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("compute $C = 6ND$ (FLOPs)")
    ax.set_ylabel("test MSE (best width)")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig06_scaling.pdf"), bbox_inches="tight")


def fig07():
    rows = read("results_exp07.csv")
    heads = ["det", "ens", "gauss", "mix"]
    labels = ["Determ.", "Ensemble", "Gaussian", "Mixture"]
    groups = [("ball", "state"), ("multi", "state"), ("ball", "latent"), ("multi", "latent")]
    gl = ["ball, state", "multi, state", "ball, latent", "multi, latent"]
    fig, ax = plt.subplots(figsize=(W, H))
    bw = 0.2
    for i, h in enumerate(heads):
        mu, lo, hi = [], [], []
        for sy, ar in groups:
            v = [float(r["cov90"]) for r in rows if r["system"] == sy and r["arm"] == ar and r["head"] == h]
            a, l, u = ci(v)
            mu.append(a), lo.append(l), hi.append(u)
        x = np.arange(len(groups)) + (i - 1.5) * bw
        ax.bar(x, mu, bw, yerr=[np.array(mu) - lo, np.array(hi) - mu], capsize=1.2, label=labels[i],
               error_kw={"lw": 0.7})
    ax.axhline(0.90, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(gl, fontsize=6)
    ax.set_ylabel("90% region coverage")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="upper center", columnspacing=0.8, handlelength=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig07_uncertainty.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    import sys
    for n, f in [("05", fig05), ("06", fig06), ("07", fig07)]:
        if len(sys.argv) < 2 or n in sys.argv[1:]:
            f()
            print("fig", n, "ok")
