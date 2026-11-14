"""
Experiment 5 (simulation arm) for artigo_05_hierarchical.tex.

Simulated documents with two time scales. A document has a slow latent state s
(d_s = 4) shared by all sections and, per section, a fast latent u_k (d_u = 6) plus
observation noise. Section k is   x_k = tanh(W_s s + W_u u_k) + NOISE * eps_k.
The "abstract" is the clean summary   a = tanh(W_s s)   (no fast part, no noise),
so it is a section-shaped vector that depends only on the slow state.

Models (all learned ones: same encoder, same steps, EMA target, VICReg variance term):
  H-JEPA        hierarchical: z_k = f(x_k); top predictor g(mean_k z_k) -> f_bar(a);
                section predictor g2(mean z_<k) -> f_bar(x_k) with weight LAMBDA
  H-JEPA-noSec  same with LAMBDA = 0 (ablation of the section-level loss)
  Flat-first    flat context = first section only (matched to one section of tokens)
  Flat-concat   MLP over the concatenation of all K sections (no pooling prior)
  Raw-mean      no learning: mean of the raw section vectors (reference)

Metrics (held-out documents):
  R2      ridge readout from the frozen document representation to the slow state s
  rec@1/10 retrieval of the document's abstract embedding f_bar(a) among N_RET
          held-out abstracts, cosine similarity with the predicted abstract embedding
  erank   effective rank (participation ratio) of the document representation

Usage:  python3 exp05_hierarchical_sim.py --shard k n   (writes part_exp05_k.json)
        python3 exp05_hierarchical_sim.py               (aggregates)
"""
import copy
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))

SEEDS = list(range(1, 11))
KS = [2, 4, 8, 16]
METHODS = ["H-JEPA", "H-JEPA-noSec", "Flat-first", "Flat-concat", "Raw-mean"]
D_S, D_U, DX, EMB = 4, 6, 32, 16
NOISE = 0.5
STEPS = 1500
BATCH = 128
LR = 1e-3
EMA_M = 0.99
LAMBDA = 0.5
VAR_W = 1.0
N_TRAIN_R, N_TEST_R, N_RET = 2000, 1000, 500


def make_world(rng):
    Ws = rng.standard_normal((DX, D_S)) / np.sqrt(D_S) * 1.5
    Wu = rng.standard_normal((DX, D_U)) / np.sqrt(D_U) * 1.5
    return Ws, Wu


def sample_docs(n, K, world, rng):
    Ws, Wu = world
    s = rng.standard_normal((n, D_S)).astype(np.float32)
    u = rng.standard_normal((n, K, D_U)).astype(np.float32)
    base = (s @ Ws.T)[:, None, :] + u @ Wu.T
    x = np.tanh(base) + NOISE * rng.standard_normal((n, K, DX))
    a = np.tanh(s @ Ws.T)
    return s, x.astype(np.float32), a.astype(np.float32)


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def var_term(z):
    z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


class Model(nn.Module):
    def __init__(self, method, K):
        super().__init__()
        self.method, self.K = method, K
        if method == "Flat-concat":
            self.f = mlp(K * DX, EMB, 256)          # context encoder over the concatenation
            self.fa = mlp(DX, EMB)                  # section-shaped encoder for the abstract target
        else:
            self.f = mlp(DX, EMB)
        self.g = mlp(EMB, EMB)                      # top-level predictor
        self.g2 = mlp(EMB, EMB)                     # section-level predictor
        tgt_src = self.fa if method == "Flat-concat" else self.f
        self.ft = copy.deepcopy(tgt_src)
        for p in self.ft.parameters():
            p.requires_grad_(False)

    def doc_repr(self, x):
        if self.method == "Flat-first":
            return self.f(x[:, 0])
        if self.method == "Flat-concat":
            return self.f(x.reshape(x.shape[0], -1))
        return self.f(x).mean(dim=1)

    def loss(self, x, a):
        with torch.no_grad():
            ta = self.ft(a)
        if self.method in ("Flat-first", "Flat-concat"):
            r = self.doc_repr(x)
            return F.mse_loss(self.g(r), ta) + VAR_W * var_term(r)
        z = self.f(x)                                # B, K, EMB
        top = F.mse_loss(self.g(z.mean(dim=1)), ta)
        loss = top + VAR_W * var_term(z)
        lam = LAMBDA if self.method == "H-JEPA" else 0.0
        if lam > 0 and x.shape[1] > 1:
            with torch.no_grad():
                tz = self.ft(x)
            csum = torch.cumsum(z, dim=1)[:, :-1] / torch.arange(1, x.shape[1], device=x.device).view(1, -1, 1)
            sec = F.mse_loss(self.g2(csum), tz[:, 1:])
            loss = loss + lam * sec
        return loss

    @torch.no_grad()
    def ema(self):
        src = self.fa if self.method == "Flat-concat" else self.f
        for pt, ps in zip(self.ft.parameters(), src.parameters()):
            pt.mul_(EMA_M).add_(ps.detach(), alpha=1 - EMA_M)


def effective_rank(Z):
    Zc = Z - Z.mean(axis=0)
    eig = np.maximum(np.linalg.eigvalsh(np.cov(Zc, rowvar=False)), 0)
    return float((eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12))


def ridge_r2(Xtr, ytr, Xte, yte, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1])
    b = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean(0)))
    pred = Xte @ b + ytr.mean(0)
    return float(1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean(0)) ** 2).sum())


def run_one(job):
    method, K, seed = job
    rng = np.random.default_rng(10_000 * K + seed)
    world = make_world(np.random.default_rng(777 + seed))
    torch.manual_seed(seed)
    model = None
    if method != "Raw-mean":
        model = Model(method, K)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
        for _ in range(STEPS):
            _, x, a = sample_docs(BATCH, K, world, rng)
            loss = model.loss(torch.from_numpy(x), torch.from_numpy(a))
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.ema()
    er = np.random.default_rng(55_000 + seed)
    s_tr, x_tr, _ = sample_docs(N_TRAIN_R, K, world, er)
    s_te, x_te, a_te = sample_docs(N_TEST_R, K, world, er)
    row = {"method": method, "K": K, "seed": seed, "rec1": float("nan"), "rec10": float("nan")}
    if method == "Raw-mean":
        Rtr, Rte = x_tr.mean(1), x_te.mean(1)
    else:
        with torch.no_grad():
            Rtr = model.doc_repr(torch.from_numpy(x_tr)).numpy()
            Rte = model.doc_repr(torch.from_numpy(x_te)).numpy()
            pred = model.g(torch.from_numpy(Rte)).numpy() if method in ("Flat-first", "Flat-concat") else \
                model.g(torch.from_numpy(Rte)).numpy()
            tgt = model.ft(torch.from_numpy(a_te)).numpy()
        P = pred[:N_RET] / (np.linalg.norm(pred[:N_RET], axis=1, keepdims=True) + 1e-9)
        T = tgt[:N_RET] / (np.linalg.norm(tgt[:N_RET], axis=1, keepdims=True) + 1e-9)
        sim = P @ T.T
        rank = (sim > np.diag(sim)[:, None]).sum(axis=1)
        row["rec1"] = float(np.mean(rank < 1))
        row["rec10"] = float(np.mean(rank < 10))
    row["r2"] = ridge_r2(Rtr, s_tr, Rte, s_te)
    row["erank"] = effective_rank(Rte)
    print(f"{method:13s} K={K:2d} seed={seed:2d} R2={row['r2']:.3f} rec1={row['rec1']:.3f} "
          f"erank={row['erank']:.2f}", flush=True)
    return row


def jobs():
    return [(m, K, s) for s in SEEDS for K in KS for m in METHODS]


def run_shard(k, n):
    path = os.path.join(HERE, f"part_exp05_{k}.jsonl")
    open(path, "w").close()
    for j in jobs()[k::n]:
        row = run_one(j)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def boot_ci(x, rng, n=5000):
    x = np.asarray(x, float)
    b = [np.mean(rng.choice(x, size=len(x))) for _ in range(n)]
    return float(np.mean(x)), *np.percentile(b, [2.5, 97.5])


def aggregate():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "part_exp05_*.jsonl"))):
        rows.extend(json.loads(l) for l in open(p) if l.strip())
    rows.sort(key=lambda r: (METHODS.index(r["method"]), r["K"], r["seed"]))
    with open(os.path.join(HERE, "results_exp05.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    lines = [f"Experiment 5 (simulation arm), {len(rows)} runs. Mean [95% bootstrap CI] over seeds.", ""]
    for K in KS:
        for m in METHODS:
            sel = [r for r in rows if r["method"] == m and r["K"] == K]
            if not sel:
                continue
            r2 = boot_ci([r["r2"] for r in sel], rng)
            rec = [r["rec1"] for r in sel]
            rtxt = "   n/a" if np.isnan(rec[0]) else "%.3f [%.3f, %.3f]" % boot_ci(rec, rng)
            lines.append(f"K={K:2d} {m:13s} n={len(sel)}  R2={r2[0]:.3f} [{r2[1]:.3f}, {r2[2]:.3f}]  "
                         f"rec@1={rtxt}  erank={np.mean([r['erank'] for r in sel]):.2f}")
        lines.append("")
    # paired differences H-JEPA vs others
    for other in ["H-JEPA-noSec", "Flat-first", "Flat-concat", "Raw-mean"]:
        for K in KS:
            a = {r["seed"]: r["r2"] for r in rows if r["method"] == "H-JEPA" and r["K"] == K}
            b = {r["seed"]: r["r2"] for r in rows if r["method"] == other and r["K"] == K}
            d = [a[s] - b[s] for s in a if s in b]
            mu, lo, hi = boot_ci(d, rng)
            lines.append(f"paired R2(H-JEPA - {other}) K={K:2d}: {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    open(os.path.join(HERE, "summary_exp05.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--shard"]:
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    else:
        aggregate()
