"""
Experiment 7 (simulation arm) for artigo_07_uncertainty.tex.

Transitions with known branching. State s in R^2, next state
    s' = A s + b_k + sigma * eps,   k ~ Categorical(pi(s)),   eps ~ N(0, I)
with A a contraction-rotation, so the true conditional p(s'|s) is a Gaussian mixture in closed form.
  system "ball"  : K = 2 branches (bounce left/right at a bifurcation), pi_1(s) = sigmoid(3 s_x)
  system "multi" : K = 3 branches, pi(s) = softmax(W s)
Two arms:
  state   heads see the state s directly; the oracle NLL is exact (closed form).
  latent  observation o = tanh(M2 tanh(M1 s)) in R^12; a JEPA (encoder + predictor, EMA target,
          VICReg variance/covariance terms) is trained on (o_t, o_t+1); the encoder is then frozen and
          the probabilistic heads are trained on the frozen 4-d latents (the oracle is not available there).

Predictive heads (same MLP trunk, same data, same steps):
  det    deterministic MLP (MSE); predictive Gaussian N(mean, residual covariance of the training set)
  ens    5 deterministic MLPs; Gaussian from the members' mean and spread (+1e-4 I), no residual term
  gauss  single heteroscedastic Gaussian head (diagonal), trained by NLL
  mix    K=3 mixture head (diagonal components), trained by NLL

Metrics on 2000 held-out transitions:
  NLL        mean -log p_hat(s'|s)   (state arm: also the gap to the oracle NLL)
  cov90      fraction of true next states inside the 90% highest-density region of p_hat
  ECE        top-label expected calibration error (10 bins) of the predicted branch probabilities;
             branch probabilities = fraction of 100 predictive samples that an MLP classifier
             (trained on (s_t, s_t+1) -> true branch) assigns to each branch; truth = generating branch

Usage:  python3 exp07_uncertainty_sim.py --shard k n   (writes part_exp07_k.json)
        python3 exp07_uncertainty_sim.py               (aggregates)
"""
import copy
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import chi2

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))

SEEDS = list(range(1, 11))
SYSTEMS = ["ball", "multi"]
ARMS = ["state", "latent"]
HEADS = ["det", "ens", "gauss", "mix"]
SIGMA = 0.15
N_TRAIN, N_TEST = 20000, 2000
STEPS, BATCH, LR = 1500, 512, 2e-3
K_MIX = 3
N_SAMP = 400
DLAT = 4
LOG2PI = math.log(2 * math.pi)

TH = 0.3
A = 0.9 * np.array([[math.cos(TH), -math.sin(TH)], [math.sin(TH), math.cos(TH)]])


def system(name):
    if name == "ball":
        b = np.array([[1.5, 0.8], [-1.5, 0.8]])
        def pi(s):
            p1 = 1 / (1 + np.exp(-3 * s[:, 0]))
            return np.stack([p1, 1 - p1], axis=1)
    else:
        b = 1.8 * np.array([[math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3)] for k in range(3)])
        Wp = np.array([[1.5, 0.0], [-0.75, 1.3], [-0.75, -1.3]])
        def pi(s):
            l = s @ Wp.T
            l -= l.max(axis=1, keepdims=True)
            e = np.exp(l)
            return e / e.sum(axis=1, keepdims=True)
    return b, pi


def simulate(name, n, rng):
    b, pi = system(name)
    s = rng.standard_normal((n, 2))
    p = pi(s)
    u = rng.random(n)[:, None]
    k = (u > np.cumsum(p, axis=1)).sum(axis=1).clip(max=len(b) - 1)
    s2 = s @ A.T + b[k] + SIGMA * rng.standard_normal((n, 2))
    return s, s2, k


def oracle_nll(name, s, s2):
    b, pi = system(name)
    p = pi(s)
    mu = (s @ A.T)[:, None, :] + b[None]
    lp = -0.5 * (((s2[:, None, :] - mu) / SIGMA) ** 2).sum(-1) - np.log(2 * math.pi * SIGMA ** 2)
    return float(-np.mean(np.log((p * np.exp(lp)).sum(1) + 1e-300)))


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


# ---------------------------------------------------------------- JEPA latent arm
def make_obs(rng):
    M1 = rng.standard_normal((8, 2)) * 2.0 / np.sqrt(2)
    M2 = rng.standard_normal((12, 8)) * 2.0 / np.sqrt(8)
    return M1, M2


def observe(s, M, rng):
    M1, M2 = M
    return (np.tanh(np.tanh(s @ M1.T) @ M2.T) + 0.02 * rng.standard_normal((len(s), 12))).astype(np.float32)


def var_cov(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    zc = z - z.mean(0)
    cov = zc.T @ zc / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return F.relu(1 - std).mean() + (off ** 2).sum() / z.shape[1]


def train_jepa(o1, o2, seed):
    torch.manual_seed(seed)
    f, g = mlp(12, DLAT), mlp(DLAT, DLAT)
    ft = copy.deepcopy(f)
    for p in ft.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), lr=LR)
    rs = np.random.default_rng(seed)
    n = len(o1)
    X1, X2 = torch.from_numpy(o1), torch.from_numpy(o2)
    for t in range(3000):
        i = torch.from_numpy(rs.integers(0, n, BATCH))
        z = f(X1[i])
        with torch.no_grad():
            tz = ft(X2[i])
        loss = F.mse_loss(g(z), tz) + var_cov(z)
        opt.zero_grad()
        loss.backward()
        opt.step()
        mom = min(0.99, (1 + t) / (10 + t))
        with torch.no_grad():
            for pt, ps in zip(ft.parameters(), f.parameters()):
                pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
    return f


# ---------------------------------------------------------------- heads
def fit(model, loss_fn, X, Y, seed, steps=STEPS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rs = np.random.default_rng(seed)
    for _ in range(steps):
        i = torch.from_numpy(rs.integers(0, len(X), BATCH))
        loss = loss_fn(model(X[i]), Y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def gauss_lp(y, mu, logsig):
    return (-0.5 * ((y - mu) / logsig.exp()) ** 2 - logsig - 0.5 * LOG2PI).sum(-1)


def nll_gauss(out, y):
    d = y.shape[1]
    return -gauss_lp(y, out[:, :d], out[:, d:].clamp(-4, 3)).mean()


def mix_parts(out, d):
    K = K_MIX
    lg = out[:, :K]
    mu = out[:, K:K + K * d].reshape(-1, K, d)
    ls = out[:, K + K * d:].reshape(-1, K, d).clamp(-4, 3)
    return lg, mu, ls


def mix_lp(y, lg, mu, ls):
    comp = gauss_lp(y[:, None, :], mu, ls)
    return torch.logsumexp(F.log_softmax(lg, -1) + comp, dim=-1)


def nll_mix(out, y):
    return -mix_lp(y, *mix_parts(out, y.shape[1])).mean()


class Predictive:
    """Unified predictive distribution: log_prob(x_next | cond), sample(n), per held-out input."""

    def __init__(self, kind, d):
        self.kind, self.d = kind, d


def fit_heads(Xtr, Ytr, seed):
    d = Ytr.shape[1]
    H = {}
    # deterministic
    det = fit(mlp(d, d), F.mse_loss, Xtr, Ytr, seed)
    with torch.no_grad():
        res = (Ytr - det(Xtr)).numpy()
    Sres = np.cov(res.T) + 1e-6 * np.eye(d)
    H["det"] = ("full", det, Sres)
    ens = [fit(mlp(d, d), F.mse_loss, Xtr, Ytr, seed * 31 + m) for m in range(5)]
    H["ens"] = ("ens", ens, None)
    gh = fit(mlp(d, 2 * d), nll_gauss, Xtr, Ytr, seed + 1)
    H["gauss"] = ("diag", gh, None)
    mh = fit(mlp(d, K_MIX + 2 * K_MIX * d), nll_mix, Xtr, Ytr, seed + 2)
    H["mix"] = ("mix", mh, None)
    return H


@torch.no_grad()
def evaluate_head(kind, obj, extra, X, Y, d, rs):
    """Returns per-point log p, cov90 indicator, and predictive samples (n, N_SAMP, d)."""
    n = len(X)
    thr = chi2.ppf(0.9, d)
    if kind in ("full", "ens"):
        if kind == "full":
            mu = obj(X).numpy()[:, None, :].repeat(1, 1)
            Sig = np.broadcast_to(extra, (n, d, d))
            mu = mu[:, 0, :]
        else:
            P = np.stack([m(X).numpy() for m in obj], 0)           # 5, n, d
            mu = P.mean(0)
            dev = P - mu
            Sig = np.einsum("mni,mnj->nij", dev, dev) / (len(obj) - 1) + 1e-4 * np.eye(d)
        Sinv = np.linalg.inv(Sig)
        diff = Y.numpy() - mu
        m2 = np.einsum("ni,nij,nj->n", diff, Sinv, diff)
        logdet = np.linalg.slogdet(Sig)[1]
        lp = -0.5 * (m2 + logdet + d * LOG2PI)
        cov = m2 <= thr
        L = np.linalg.cholesky(Sig)
        smp = mu[:, None, :] + np.einsum("nij,nsj->nsi", L, rs.standard_normal((n, N_SAMP, d)))
        return lp, cov, smp
    if kind == "diag":
        out = obj(X)
        mu, ls = out[:, :d], out[:, d:].clamp(-4, 3)
        lp = gauss_lp(Y, mu, ls).numpy()
        z2 = (((Y - mu) / ls.exp()) ** 2).sum(-1).numpy()
        smp = mu.numpy()[:, None, :] + ls.exp().numpy()[:, None, :] * rs.standard_normal((n, N_SAMP, d))
        return lp, z2 <= thr, smp
    # mixture
    lg, mu, ls = mix_parts(obj(X), d)
    lp = mix_lp(Y, lg, mu, ls).numpy()
    pw = F.softmax(lg, -1).numpy()
    comp = (rs.random((n, N_SAMP, 1)) > np.cumsum(pw, 1)[:, None, :]).sum(-1).clip(max=K_MIX - 1)
    mu_n, sg_n = mu.numpy(), ls.exp().numpy()
    idx = np.arange(n)[:, None]
    smp = mu_n[idx, comp] + sg_n[idx, comp] * rs.standard_normal((n, N_SAMP, d))
    # HPD threshold: 10% quantile of log p_hat over the predictive samples
    lps = np.empty((n, N_SAMP))
    for c0 in range(0, n, 500):
        sl = slice(c0, c0 + 500)
        ys = torch.from_numpy(smp[sl].astype(np.float32))
        lps[sl] = mix_lp_batch(ys, lg[sl], mu[sl], ls[sl]).numpy()
    tau = np.quantile(lps, 0.10, axis=1)
    return lp, lp >= tau, smp


def mix_lp_batch(ys, lg, mu, ls):
    comp = gauss_lp(ys[:, :, None, :], mu[:, None], ls[:, None])
    return torch.logsumexp(F.log_softmax(lg, -1)[:, None, :] + comp, dim=-1)


def ece_top(prob, correct, bins=10):
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == correct).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b) if a > 0 else (conf >= a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def run_one(job):
    sysname, arm, seed = job
    rng = np.random.default_rng(7000 + seed)
    s, s2, k = simulate(sysname, N_TRAIN, rng)
    st, st2, kt = simulate(sysname, N_TEST, np.random.default_rng(88000 + seed))
    nb = system(sysname)[0].shape[0]
    orc = oracle_nll(sysname, st, st2)
    if arm == "state":
        X, Y, Xt, Yt = s, s2, st, st2
    else:
        M = make_obs(np.random.default_rng(500 + seed))
        o = observe(s, M, rng)
        o2 = observe(s2, M, rng)
        f = train_jepa(o, o2, seed)
        with torch.no_grad():
            Z, Z2 = f(torch.from_numpy(o)).numpy(), f(torch.from_numpy(o2)).numpy()
            ot, ot2 = observe(st, M, rng), observe(st2, M, rng)
            Zt, Zt2 = f(torch.from_numpy(ot)).numpy(), f(torch.from_numpy(ot2)).numpy()
        mu, sd = Z.mean(0), Z.std(0) + 1e-6
        X, Y, Xt, Yt = (Z - mu) / sd, (Z2 - mu) / sd, (Zt - mu) / sd, (Zt2 - mu) / sd
    X, Y, Xt, Yt = [torch.from_numpy(a.astype(np.float32)) for a in (X, Y, Xt, Yt)]
    d = Y.shape[1]
    # branch classifier on (x_t, x_t+1) -> generating branch
    torch.manual_seed(seed)
    clf = mlp(2 * d, nb)
    clf = fit(clf, lambda out, yy: F.cross_entropy(out, yy.long()), torch.cat([X, Y], 1),
              torch.from_numpy(k.astype(np.float32)), seed, steps=1500)
    heads = fit_heads(X, Y, seed)
    rs = np.random.default_rng(123 + seed)
    rows = []
    for name in HEADS:
        kind, obj, extra = heads[name]
        lp, cov, smp = evaluate_head(kind, obj, extra, Xt, Yt, d, rs)
        # predicted branch probabilities from samples
        with torch.no_grad():
            nsub = 100
            Xc = Xt[:, None, :].expand(-1, nsub, -1)
            logits = clf(torch.cat([Xc, torch.from_numpy(smp[:, :nsub].astype(np.float32))], -1).reshape(-1, 2 * d))
            lab = logits.argmax(-1).reshape(N_TEST, nsub).numpy()
        prob = np.stack([(lab == j).mean(1) for j in range(nb)], 1)
        row = {"system": sysname, "arm": arm, "head": name, "seed": seed,
               "nll": float(-lp.mean()), "cov90": float(np.mean(cov)),
               "ece": ece_top(prob, kt), "oracle_nll": orc if arm == "state" else float("nan")}
        rows.append(row)
        print(f"{sysname:5s} {arm:6s} s={seed:2d} {name:5s} nll={row['nll']:7.3f} cov90={row['cov90']:.3f} "
              f"ece={row['ece']:.3f} oracle={orc:.3f}", flush=True)
    return rows


def jobs():
    return [(sy, ar, s) for s in SEEDS for sy in SYSTEMS for ar in ARMS]


def run_shard(k, n):
    path = os.path.join(HERE, f"part_exp07_{k}.jsonl")
    open(path, "w").close()
    for j in jobs()[k::n]:
        for row in run_one(j):
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")


def boot_ci(x, rng, n=5000):
    x = np.asarray(x, float)
    b = [np.mean(rng.choice(x, size=len(x))) for _ in range(n)]
    return float(np.mean(x)), *np.percentile(b, [2.5, 97.5])


def aggregate():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "part_exp07_*.jsonl"))):
        rows.extend(json.loads(l) for l in open(p) if l.strip())
    rows.sort(key=lambda r: (r["system"], r["arm"], HEADS.index(r["head"]), r["seed"]))
    with open(os.path.join(HERE, "results_exp07.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    lines = [f"Experiment 7 (simulation arm), {len(rows)} head evaluations. Mean [95% bootstrap CI] over seeds.", ""]
    for sy in SYSTEMS:
        for ar in ARMS:
            lines.append(f"== system={sy} arm={ar}")
            orc = [r["oracle_nll"] for r in rows if r["system"] == sy and r["arm"] == ar and r["head"] == "det"]
            if ar == "state":
                lines.append(f"   oracle NLL = {np.mean(orc):.3f}")
            for h in HEADS:
                sel = [r for r in rows if r["system"] == sy and r["arm"] == ar and r["head"] == h]
                n_ = boot_ci([r["nll"] for r in sel], rng)
                c_ = boot_ci([r["cov90"] for r in sel], rng)
                e_ = boot_ci([r["ece"] for r in sel], rng)
                gap = ""
                if ar == "state":
                    g = boot_ci([r["nll"] - r["oracle_nll"] for r in sel], rng)
                    gap = f"  gap-to-oracle={g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}]"
                lines.append(f"   {h:5s} n={len(sel)} NLL={n_[0]:7.3f} [{n_[1]:7.3f}, {n_[2]:7.3f}]  "
                             f"cov90={c_[0]:.3f} [{c_[1]:.3f}, {c_[2]:.3f}]  ECE={e_[0]:.3f} [{e_[1]:.3f}, {e_[2]:.3f}]{gap}")
            lines.append("")
    open(os.path.join(HERE, "summary_exp07.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--shard"]:
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    else:
        aggregate()
