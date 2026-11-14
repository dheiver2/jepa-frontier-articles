"""
Experiment 9 (synthetic arm) for artigo_09_shift.tex.

Source task: x = tanh(u B) M (u in R^8 latent, ambient dim 128), label y = 1[w.u > 0] with a
nonlinear twist, so a linear probe on raw x is imperfect. A JEPA encoder is pretrained without
labels on the source (two noisy views, EMA target + variance term), then a linear head is fit on
source labels (frozen encoder) = the deployed model.

Shift with known severity s in {1..5}: x' = (1-a) x + a Q x + noise, a = 0.15 s (Q a fixed random
orthogonal matrix, i.e. feature mixing) plus Gaussian noise 0.05 s and a gain change (1 - 0.04 s).
The label rule is unchanged.

Adaptation on 200 labeled shifted samples, 300 steps, same budget for all methods:
  NoAdapt   - deployed model as is
  FullFT    - fine-tune all encoder weights and the head on the shifted samples
  Adapter   - frozen encoder + head with low-rank (r=4) residual adapters on both encoder
              layers and a trainable head
  Replay    - full fine-tuning, half of every batch from 200 stored source samples
One model is evaluated on both domains (no domain switch). Forgetting F = Acc_src(before) - Acc_src(after).

Usage:  python3 exp09_shift_sim.py
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
SEVS = [1, 2, 3, 4, 5]
METHODS = ["NoAdapt", "FullFT", "Adapter", "Replay"]
D, AMB, EMB = 8, 128, 64
PRE_STEPS, BATCH, LR = 1500, 256, 1e-3
N_SRC_HEAD, N_TGT_ADAPT, N_REPLAY, N_TEST = 4000, 200, 200, 2000
ADAPT_STEPS = 300
RANK = 4


def make_world(rng):
    B = rng.standard_normal((D, 64)) / np.sqrt(D)
    M = rng.standard_normal((64, AMB)) / 8.0
    w = rng.standard_normal(D)
    Q, _ = np.linalg.qr(rng.standard_normal((AMB, AMB)))
    return B, M, w, Q


def sample(n, W, rng):
    B, M, w, _ = W
    u = rng.standard_normal((n, D))
    x = np.tanh(u @ B) @ M
    y = ((u @ w + 0.5 * u[:, 0] * u[:, 1]) > 0).astype(np.int64)   # mild nonlinearity
    return x.astype(np.float32), y


def shift(x, s, W, rng):
    Q = W[3]
    a = 0.15 * s
    x2 = (1 - a) * x + a * (x @ Q.T.astype(np.float32))
    x2 = (1 - 0.04 * s) * x2 + 0.05 * s * rng.standard_normal(x.shape).astype(np.float32)
    return x2.astype(np.float32)


class LoRALinear(nn.Module):
    def __init__(self, base, r):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(base.in_features, r) * 0.05)
        self.Bm = nn.Parameter(torch.zeros(r, base.out_features))

    def forward(self, x):
        return self.base(x) + x @ self.A @ self.Bm


def mk_enc():
    return nn.Sequential(nn.Linear(AMB, 256), nn.GELU(), nn.Linear(256, EMB))


def pretrain(Xs, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = mk_enc()
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(EMB, 128), nn.GELU(), nn.Linear(128, EMB))
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    X = torch.from_numpy(Xs)
    for _ in range(PRE_STEPS):
        x = X[torch.randint(0, X.shape[0], (BATCH,), generator=g)]
        x1 = x + 0.05 * torch.randn(x.shape, generator=g)
        x2 = x + 0.05 * torch.randn(x.shape, generator=g)
        z = enc(x1)
        with torch.no_grad():
            t = tgt(x2)
        loss = F.mse_loss(pred(z), t)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        loss = loss + F.relu(1.0 - std).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pt, pe in zip(tgt.parameters(), enc.parameters()):
                pt.mul_(0.99).add_(pe.detach(), alpha=0.01)
    return enc


def fit_head(enc, X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    with torch.no_grad():
        Z = enc(torch.from_numpy(X)).numpy()
    sc = StandardScaler().fit(Z)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Z), y)
    head = nn.Linear(EMB, 1)
    # fold the scaler into the head so the model is a single torch module
    W = clf.coef_[0] / sc.scale_
    b = clf.intercept_[0] - np.sum(clf.coef_[0] * sc.mean_ / sc.scale_)
    head.weight.data = torch.tensor(W, dtype=torch.float32).unsqueeze(0)
    head.bias.data = torch.tensor([b], dtype=torch.float32)
    return head


def acc(model, X, y):
    enc, head = model
    with torch.no_grad():
        p = head(enc(torch.from_numpy(X))).squeeze(-1)
    return float(((p > 0).numpy() == y).mean())


def adapt(enc0, head0, method, Xt, yt, Xsrc, ysrc, seed):
    g = torch.Generator().manual_seed(seed)
    enc, head = copy.deepcopy(enc0), copy.deepcopy(head0)
    if method == "Adapter":
        for p in enc.parameters():
            p.requires_grad_(False)
        enc[0], enc[2] = LoRALinear(enc[0], RANK), LoRALinear(enc[2], RANK)
        params = [p for p in enc.parameters() if p.requires_grad] + list(head.parameters())
        lr = 3e-3
    else:
        params = list(enc.parameters()) + list(head.parameters())
        lr = 1e-3
    opt = torch.optim.Adam(params, lr=lr)
    Xt_, yt_ = torch.from_numpy(Xt), torch.from_numpy(yt).float()
    Xs_, ys_ = torch.from_numpy(Xsrc), torch.from_numpy(ysrc).float()
    bs = 64
    for _ in range(ADAPT_STEPS):
        if method == "Replay":
            it = torch.randint(0, Xt_.shape[0], (bs // 2,), generator=g)
            ir = torch.randint(0, Xs_.shape[0], (bs // 2,), generator=g)
            xb, yb = torch.cat([Xt_[it], Xs_[ir]]), torch.cat([yt_[it], ys_[ir]])
        else:
            it = torch.randint(0, Xt_.shape[0], (bs,), generator=g)
            xb, yb = Xt_[it], yt_[it]
        loss = F.binary_cross_entropy_with_logits(head(enc(xb)).squeeze(-1), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc, head


def run_seed(seed):
    rng = np.random.default_rng(31 * seed)
    W = make_world(rng)
    Xpre, _ = sample(20000, W, rng)
    Xh, yh = sample(N_SRC_HEAD, W, rng)
    Xsrc_te, ysrc_te = sample(N_TEST, W, rng)
    Xrep, yrep = sample(N_REPLAY, W, rng)
    enc = pretrain(Xpre, seed)
    head = fit_head(enc, Xh, yh, seed)
    src_before = acc((enc, head), Xsrc_te, ysrc_te)
    rows = []
    for s in SEVS:
        Xt, yt = sample(N_TGT_ADAPT, W, rng)
        Xt = shift(Xt, s, W, rng)
        Xte, yte = sample(N_TEST, W, rng)
        Xte = shift(Xte, s, W, rng)
        for meth in METHODS:
            if meth == "NoAdapt":
                model = (enc, head)
            else:
                model = adapt(enc, head, meth, Xt, yt, Xrep, yrep, seed)
            ta, sa = acc(model, Xte, yte), acc(model, Xsrc_te, ysrc_te)
            rows.append({"method": meth, "severity": s, "seed": seed, "acc_tgt": ta,
                         "acc_src_before": src_before, "acc_src_after": sa,
                         "forgetting": src_before - sa})
            print(f"seed={seed} s={s} {meth:8s} tgt={ta:.3f} src_after={sa:.3f} F={src_before - sa:+.3f}", flush=True)
    return rows


def boot(v, rng, n=5000):
    v = np.asarray(v, float)
    b = [np.mean(rng.choice(v, size=len(v))) for _ in range(n)]
    return float(v.mean()), *np.percentile(b, [2.5, 97.5])


def aggregate_and_plot(rows):
    with open(os.path.join(HERE, "results_exp09.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    lines = [f"Experiment 9 (synthetic shift arm), {len(rows)} runs, {len(SEEDS)} seeds. Mean [95% bootstrap CI].",
             f"Source accuracy before adaptation: {np.mean([r['acc_src_before'] for r in rows]):.3f}", ""]
    st = {}
    for meth in METHODS:
        for s in SEVS:
            sel = [r for r in rows if r["method"] == meth and r["severity"] == s]
            t = boot([r["acc_tgt"] for r in sel], rng)
            fgt = boot([r["forgetting"] for r in sel], rng)
            st[(meth, s)] = (t, fgt)
            lines.append(f"{meth:8s} s={s}  target_acc={t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]  forgetting={fgt[0]:+.3f} [{fgt[1]:+.3f}, {fgt[2]:+.3f}]")
    lines.append("")
    lines.append("Mean over severities:")
    for meth in METHODS:
        ta = np.mean([st[(meth, s)][0][0] for s in SEVS])
        fg = np.mean([st[(meth, s)][1][0] for s in SEVS])
        lines.append(f"  {meth:8s} target_acc={ta:.3f}  forgetting={fg:+.3f}")
    with open(os.path.join(HERE, "summary_exp09.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.4), sharex=True)
    for meth, mk in zip(METHODS, ["o", "s", "^", "D"]):
        for k, ax in enumerate(axs):
            m = np.array([st[(meth, s)][k][0] for s in SEVS])
            lo = np.array([st[(meth, s)][k][1] for s in SEVS])
            hi = np.array([st[(meth, s)][k][2] for s in SEVS])
            ax.errorbar(SEVS, m, yerr=[m - lo, hi - m], marker=mk, ms=2.5, capsize=1.5, lw=0.9,
                        label=meth if k == 0 else None)
    axs[0].set_ylabel("Target accuracy")
    axs[1].set_ylabel("Forgetting (source)")
    for ax in axs:
        ax.set_xlabel("Severity")
        ax.set_xticks(SEVS)
    axs[0].legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig09_shift.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    allrows = []
    for s in SEEDS:
        allrows.extend(run_seed(s))
    aggregate_and_plot(allrows)
