"""
Builds one main figure per article from the experiment outputs and writes
PDF files to ../figs/ (vector, column width 3.5 in, serif font, TrueType embedded).

Article 1 uses results_exp01.csv (means with 95% normal CI over seeds).
Articles 2-10 use results/results_all.json (means and 95% CI over trials).
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
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
W, H = 3.5, 2.4


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
    plt.close(fig)


def load_all():
    with open(os.path.join(HERE, "results", "results_all.json")) as f:
        return json.load(f)


def series(rows, art, key, metric):
    sel = [r for r in rows if r["article"] == art]
    x = [r["cfg"][key] for r in sel]
    m = [r["result"][metric]["mean"] for r in sel]
    c = [r["result"][metric]["ci95"] for r in sel]
    return np.array(x, float), np.array(m), np.array(c)


def fig01():
    rows = []
    with open(os.path.join(HERE, "results_exp01.csv")) as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(W, H))
    dims = [2, 8, 32]
    width = 0.25
    for i, method in enumerate(["EMA", "VICReg", "SIGReg"]):
        means, errs = [], []
        for d in dims:
            g = np.array([float(r["id_gap"]) for r in rows
                          if r["method"] == method and int(r["d"]) == d])
            means.append(g.mean())
            errs.append(1.96 * g.std(ddof=1) / np.sqrt(len(g)))
        xs = np.arange(len(dims)) + (i - 1) * width
        ax.bar(xs, means, width, yerr=errs, capsize=2, label=method)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([f"$d={d}$" for d in dims])
    ax.set_ylabel(r"Intrinsic-dimension gap $|\hat{d}-d|$")
    ax.legend(frameon=False)
    save(fig, "fig01_collapse.pdf")


def line_fig(rows, art, key, metric, xlabel, ylabel, name, logx=False,
             extra=None):
    fig, ax = plt.subplots(figsize=(W, H))
    x, m, c = series(rows, art, key, metric)
    ax.errorbar(x, m, yerr=c, marker="o", ms=3, capsize=2, label=metric)
    if extra:
        for lab, yvals, style in extra(rows):
            ax.plot(x, yvals, style, label=lab)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if extra:
        ax.legend(frameon=False)
    save(fig, name)


def fig02(rows):
    line_fig(rows, 2, "h", "r2_h", "Rollout horizon $h$", r"$R^2$ of rollout",
             "fig02_world_model.pdf")


def fig03(rows):
    fig, ax = plt.subplots(figsize=(W, H))
    x, m, c = series(rows, 3, "c", "obs_slope")
    ax.plot(x, m, "o-", ms=3, label="observational slope")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="true effect (1.0)")
    ax.set_xlabel("Confounding strength $c$")
    ax.set_ylabel(r"Slope $X_1 \to X_n$")
    ax.legend(frameon=False)
    save(fig, "fig03_causal.pdf")


def fig04(rows):
    line_fig(rows, 4, "k", "acc", "Reasoning depth $k$", "Readout accuracy",
             "fig04_reasoning.pdf",
             extra=lambda r: [("chance (0.1)", np.full(10, 0.1), "k:")])


def fig05(rows):
    fig, ax = plt.subplots(figsize=(W, H))
    x, mh, ch = series(rows, 5, "K", "r2_hier")
    _, mf, cf = series(rows, 5, "K", "r2_flat")
    ax.errorbar(x, mh, yerr=ch, marker="o", ms=3, capsize=2, label="hierarchical")
    ax.errorbar(x, mf, yerr=cf, marker="s", ms=3, capsize=2, label="flat")
    ax.set_xlabel("Number of sections $K$")
    ax.set_ylabel(r"$R^2$ of slow-state recovery")
    ax.legend(frameon=False)
    save(fig, "fig05_hierarchical.pdf")


def fig06(rows):
    line_fig(rows, 6, "w", "test_mse", "Model width $w$", "Test MSE",
             "fig06_scaling.pdf", logx=True)


def fig07(rows):
    fig, ax = plt.subplots(figsize=(W, H))
    x, mu, cu = series(rows, 7, "sigma", "cov_unimodal")
    _, mm, cm = series(rows, 7, "sigma", "cov_mixture")
    ax.errorbar(x, mu, yerr=cu, marker="o", ms=3, capsize=2, label="unimodal")
    ax.errorbar(x, mm, yerr=cm, marker="s", ms=3, capsize=2, label="mixture")
    ax.axhline(0.9, color="k", ls="--", lw=0.8, label="nominal 0.90")
    ax.set_xlabel(r"Noise level $\sigma$")
    ax.set_ylabel("90\\% interval coverage")
    ax.legend(frameon=False)
    save(fig, "fig07_uncertainty.pdf")


def fig08(rows):
    fig, ax = plt.subplots(figsize=(W, H))
    x, mp, cp = series(rows, 8, "frac", "acc_pca")
    _, mr, cr = series(rows, 8, "frac", "acc_raw")
    ax.errorbar(x, mp, yerr=cp, marker="o", ms=3, capsize=2, label="PCA features")
    ax.errorbar(x, mr, yerr=cr, marker="s", ms=3, capsize=2, label="raw signal")
    ax.set_xlabel("Labeled fraction")
    ax.set_ylabel("Test accuracy")
    ax.legend(frameon=False)
    save(fig, "fig08_ecg.pdf")


def fig09(rows):
    fig, ax = plt.subplots(figsize=(W, H))
    for key, lab, mk in [("acc_no_adapt", "no adaptation", "o"),
                         ("acc_replay", "replay adaptation", "s")]:
        x, m, c = series(rows, 9, "s", key)
        ax.errorbar(x, m, yerr=c, marker=mk, ms=3, capsize=2, label=lab)
    ax.set_xlabel("Shift severity $s$")
    ax.set_ylabel("Target accuracy")
    ax.legend(frameon=False)
    save(fig, "fig09_shift.pdf")


def fig10(rows):
    fig, ax = plt.subplots(figsize=(W, H))
    x, ms, cs = series(rows, 10, "m", "selectivity")
    _, mt, ct = series(rows, 10, "m", "acc_true")
    _, mc, cc = series(rows, 10, "m", "acc_control")
    ax.errorbar(x, mt, yerr=ct, marker="o", ms=3, capsize=2, label="true labels")
    ax.errorbar(x, mc, yerr=cc, marker="s", ms=3, capsize=2, label="control")
    ax.set_xscale("log")
    ax.set_xlabel("Probe input width $m$")
    ax.set_ylabel("Probe accuracy")
    ax.legend(frameon=False)
    save(fig, "fig10_probing.pdf")


def main():
    rows = load_all()
    fig01()
    for f in [fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10]:
        f(rows)
    print("figures written to", os.path.abspath(FIGS))


if __name__ == "__main__":
    main()
