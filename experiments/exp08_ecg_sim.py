"""
Experiment 8 (synthetic arm) for artigo_08_ecg.tex.

Synthetic 3-lead ECG beats (sum-of-Gaussians P-Q-R-S-T model) with subtle anomalies
and patient-level variation. NOT real ECG; no clinical claim.

Anomaly (label 1), half of each of two kinds:
  - morphology: ST-segment shift of 0.03-0.06 (R amplitude is 1) and QRS widened by 12%
  - rhythm-like: P wave attenuated to 25% and the QRS complex moved 0.04 earlier

Pretraining on unlabeled beats (same budget for both):
  JEPA  - masked context encoder -> predictor -> embedding of the full beat from an EMA target
          encoder (+ variance term against collapse, weight 1.0)
  MAE   - same encoder, linear decoder reconstructs the masked tokens (masked reconstruction)
Compared against: random frozen encoder (linear probe) and a supervised encoder trained from
scratch on the same labels. Labeled fractions 1%, 10%, 100%. Train and test patients are disjoint.

Usage:  python3 exp08_ecg_sim.py            (runs all seeds, writes csv, summary and figure)
"""
import copy
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")
SEEDS = list(range(1, 11))
FRACS = [0.01, 0.10, 1.0]
METHODS = ["JEPA", "MaskedRecon", "Random", "Supervised"]
T, LEADS, PATCH = 128, 3, 16
NTOK = LEADS * (T // PATCH)         # 24 tokens of 16 samples
DIN = LEADS * T
EMB = 64
PRE_STEPS, BATCH, LR = 1500, 128, 1e-3
N_PRE_PAT, N_LAB_PAT, N_TEST_PAT = 200, 100, 60
BEATS_PRE, BEATS_LAB, BEATS_TEST = 200, 40, 40   # beats per patient
MASK_RATIO = 0.5
EMA_M = 0.99


def gauss(t, c, w, a):
    return a * np.exp(-0.5 * ((t - c) / w) ** 2)


def make_patients(n, rng):
    return [{
        "wscale": rng.uniform(0.85, 1.2),
        "amp": rng.uniform(0.8, 1.25),
        "mix": 1.0 + 0.25 * rng.standard_normal(LEADS),
        "jit": rng.uniform(0.0, 0.03),
        "wander": rng.uniform(0.03, 0.15),
        "noise": rng.uniform(0.03, 0.07),
        "tshift": rng.uniform(-0.03, 0.03),
    } for _ in range(n)]


def beats(pat, n, rng, anomaly):
    """n beats of one patient. anomaly: array of 0/1/2 (0 normal, 1 morphology, 2 rhythm)."""
    t = np.linspace(0, 1, T, endpoint=False)
    X = np.zeros((n, LEADS, T), np.float32)
    for i in range(n):
        j = rng.uniform(-pat["jit"], pat["jit"])
        ws = pat["wscale"]
        kind = anomaly[i]
        qrs_w = 1.12 if kind == 1 else 1.0
        shift = -0.04 if kind == 2 else 0.0
        pa = 0.25 if kind == 2 else 1.0
        base = (gauss(t, 0.18 + j, 0.03 * ws, 0.12 * pa)
                + gauss(t, 0.38 + j + shift, 0.012 * ws * qrs_w, -0.15)
                + gauss(t, 0.42 + j + shift, 0.015 * ws * qrs_w, 1.0)
                + gauss(t, 0.46 + j + shift, 0.012 * ws * qrs_w, -0.25)
                + gauss(t, 0.68 + j + pat["tshift"], 0.05 * ws, 0.3))
        st = np.zeros_like(t)
        if kind == 1:
            d = rng.uniform(0.03, 0.06)
            st = d / (1 + np.exp(-(t - 0.50) / 0.01)) / (1 + np.exp((t - 0.60) / 0.01))
        phase = rng.uniform(0, 2 * np.pi)
        wander = pat["wander"] * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t + phase)
        for l in range(LEADS):
            X[i, l] = pat["amp"] * pat["mix"][l] * (base + st) + wander + pat["noise"] * rng.standard_normal(T)
    return X.reshape(n, DIN)


def make_split(n_pat, per_pat, rng, ab_frac=0.5):
    pats = make_patients(n_pat, rng)
    Xs, ys, ids = [], [], []
    for k, p in enumerate(pats):
        a = rng.choice([0, 1, 2], size=per_pat, p=[1 - ab_frac, ab_frac / 2, ab_frac / 2])
        Xs.append(beats(p, per_pat, rng, a))
        ys.append((a > 0).astype(np.int64))
        ids.append(np.full(per_pat, k))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ids)


def mlp_enc():
    return nn.Sequential(nn.Linear(DIN, 256), nn.GELU(), nn.Linear(256, EMB))


def token_mask(n, gen):
    k = int(NTOK * MASK_RATIO)
    m = torch.zeros(n, NTOK)
    for i in range(n):
        m[i, torch.randperm(NTOK, generator=gen)[:k]] = 1
    return m


def apply_mask(x, m):
    """x: (n, DIN) viewed as (n, LEADS, T//PATCH, PATCH); m: (n, NTOK) 1 = masked."""
    xt = x.view(-1, NTOK, PATCH)
    return (xt * (1 - m.unsqueeze(-1))).reshape(-1, DIN)


def pretrain(kind, Xpre, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = mlp_enc()
    if kind == "JEPA":
        tgt = copy.deepcopy(enc)
        for p in tgt.parameters():
            p.requires_grad_(False)
        head = nn.Sequential(nn.Linear(EMB, 128), nn.GELU(), nn.Linear(128, EMB))
    else:
        head = nn.Linear(EMB, DIN)   # linear decoder
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=LR)
    X = torch.from_numpy(Xpre)
    for _ in range(PRE_STEPS):
        idx = torch.randint(0, X.shape[0], (BATCH,), generator=g)
        x = X[idx]
        m = token_mask(BATCH, g)
        z = enc(apply_mask(x, m))
        if kind == "JEPA":
            with torch.no_grad():
                t = tgt(x)
            loss = F.mse_loss(head(z), t)
            std = torch.sqrt(z.var(dim=0) + 1e-4)
            loss = loss + 1.0 * F.relu(1.0 - std).mean()
        else:
            rec = head(z).view(-1, NTOK, PATCH)
            tok = x.view(-1, NTOK, PATCH)
            w = m.unsqueeze(-1)
            loss = (((rec - tok) ** 2) * w).sum() / (w.sum() * PATCH)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if kind == "JEPA":
            with torch.no_grad():
                for pt, pe in zip(tgt.parameters(), enc.parameters()):
                    pt.mul_(EMA_M).add_(pe.detach(), alpha=1 - EMA_M)
    return enc


@torch.no_grad()
def feats(enc, X):
    return enc(torch.from_numpy(X)).numpy()


def eff_rank(Z):
    Zc = Z - Z.mean(0)
    e = np.maximum(np.linalg.eigvalsh(np.cov(Zc, rowvar=False)), 0)
    return float(e.sum() ** 2 / (np.sum(e ** 2) + 1e-12))


def probe_auc(Ztr, ytr, Zte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Ztr), ytr)
    p = clf.predict_proba(sc.transform(Zte))[:, 1]
    return roc_auc_score(yte, p), float(((p > 0.5) == yte).mean())


def supervised(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    enc = mlp_enc()
    head = nn.Linear(EMB, 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr).float()
    n = X.shape[0]
    for _ in range(300):
        idx = torch.randint(0, n, (min(BATCH, n),), generator=g)
        loss = F.binary_cross_entropy_with_logits(head(enc(X[idx])).squeeze(-1), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(head(enc(torch.from_numpy(Xte))).squeeze(-1)).numpy()
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(yte, p), float(((p > 0.5) == yte).mean())


def run_seed(seed):
    rng = np.random.default_rng(777 * seed)
    Xpre, _, _ = make_split(N_PRE_PAT, BEATS_PRE, rng)
    Xlab, ylab, plab = make_split(N_LAB_PAT, BEATS_LAB, rng)
    Xte, yte, _ = make_split(N_TEST_PAT, BEATS_TEST, rng)
    mu, sd = Xpre.mean(), Xpre.std()
    Xpre, Xlab, Xte = [(a - mu) / sd for a in (Xpre, Xlab, Xte)]
    Xpre, Xlab, Xte = [a.astype(np.float32) for a in (Xpre, Xlab, Xte)]
    encs = {"JEPA": pretrain("JEPA", Xpre, seed), "MaskedRecon": pretrain("MaskedRecon", Xpre, seed)}
    torch.manual_seed(seed)
    encs["Random"] = mlp_enc()
    rows = []
    for fr in FRACS:
        # label fraction chosen by whole patients (patient-level), at least 1 patient
        n_pat = max(1, int(round(fr * N_LAB_PAT)))
        keep = np.isin(plab, rng.permutation(N_LAB_PAT)[:n_pat])
        Xl, yl = Xlab[keep], ylab[keep]
        for meth in METHODS:
            if meth == "Supervised":
                auc, acc = supervised(Xl, yl, Xte, yte, seed)
                er = float("nan")
            else:
                Zl, Zt = feats(encs[meth], Xl), feats(encs[meth], Xte)
                auc, acc = probe_auc(Zl, yl, Zt, yte)
                er = eff_rank(feats(encs[meth], Xte))
            rows.append({"method": meth, "frac": fr, "seed": seed, "n_labeled": int(len(yl)),
                         "auroc": auc, "acc": acc, "eff_rank": er})
            print(f"seed={seed} {meth:11s} frac={fr:4.2f} n={len(yl):5d} auc={auc:.3f} acc={acc:.3f} erank={er:5.1f}", flush=True)
    return rows


def boot(v, rng, n=5000):
    v = np.asarray(v, float)
    b = [np.mean(rng.choice(v, size=len(v))) for _ in range(n)]
    return float(v.mean()), *np.percentile(b, [2.5, 97.5])


def aggregate_and_plot(rows):
    path = os.path.join(HERE, "results_exp08.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    lines = [f"Experiment 8 (synthetic ECG arm), {len(rows)} runs, {len(SEEDS)} seeds. AUROC mean [95% bootstrap CI].", ""]
    stats = {}
    for meth in METHODS:
        for fr in FRACS:
            sel = [r for r in rows if r["method"] == meth and r["frac"] == fr]
            m, lo, hi = boot([r["auroc"] for r in sel], rng)
            ma, _, _ = boot([r["acc"] for r in sel], rng)
            er = np.nanmean([r["eff_rank"] for r in sel])
            n_lab = int(np.mean([r["n_labeled"] for r in sel]))
            stats[(meth, fr)] = (m, lo, hi)
            lines.append(f"{meth:11s} frac={fr:4.2f} n_labeled~{n_lab:5d}  auroc={m:.3f} [{lo:.3f}, {hi:.3f}]  acc={ma:.3f}  eff_rank={er:5.1f}")
    lines.append("")
    lines.append("Paired difference JEPA - MaskedRecon (AUROC, mean [95% CI over seeds]):")
    for fr in FRACS:
        a = {r["seed"]: r["auroc"] for r in rows if r["method"] == "JEPA" and r["frac"] == fr}
        b = {r["seed"]: r["auroc"] for r in rows if r["method"] == "MaskedRecon" and r["frac"] == fr}
        d = [a[s] - b[s] for s in a]
        m, lo, hi = boot(d, rng)
        lines.append(f"  frac={fr:4.2f}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    with open(os.path.join(HERE, "summary_exp08.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for meth, mk in zip(METHODS, ["o", "s", "^", "D"]):
        m = [stats[(meth, fr)][0] for fr in FRACS]
        lo = [stats[(meth, fr)][1] for fr in FRACS]
        hi = [stats[(meth, fr)][2] for fr in FRACS]
        x = [100 * fr for fr in FRACS]
        ax.errorbar(x, m, yerr=[np.array(m) - lo, np.array(hi) - m], marker=mk, ms=3, capsize=2, lw=1, label=meth)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("Labeled patients (%)")
    ax.set_ylabel("Test AUROC")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig08_ecg.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    allrows = []
    for s in SEEDS:
        allrows.extend(run_seed(s))
    aggregate_and_plot(allrows)
