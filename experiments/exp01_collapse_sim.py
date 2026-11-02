"""
Experiment 1 (simulation arm) for artigo_01_collapse.tex.

Compares three collapse-prevention strategies for a JEPA-style predictor on a
synthetic manifold with known intrinsic dimension d:

  EMA    - EMA target encoder, no regularizer
  VICReg - stop-gradient target, VICReg variance + covariance terms
  SIGReg - stop-gradient target, moment-matching isotropic-Gaussian penalty
           (approximation written here; NOT the official SIGReg implementation)

Collapse is measured as the gap between the MLE intrinsic-dimension estimate
(Levina-Bickel, k=10) of the held-out embeddings and the true d, plus the
effective rank (participation ratio) of the embedding covariance.

Usage:  python3 exp01_collapse_sim.py
Outputs: results_exp01.csv and summary_exp01.txt next to this script.
"""
import copy
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 2, 3, 4, 5]
DIMS = [2, 8, 32]
METHODS = ["EMA", "VICReg", "SIGReg"]
AMBIENT = 256
HIDDEN = 64          # width of the latent space the manifold is embedded in
EMB = 64             # encoder output size
STEPS = 1500
BATCH = 256
LR = 1e-3
NOISE = 0.05         # augmentation noise for the two views
EMA_M = 0.996
LAMBDA = 1.0
N_EVAL = 2000
K_ID = 10

# CPU is faster than MPS for these tiny MLPs (MPS kernel-launch overhead dominates).
DEVICE = torch.device("cpu")


def make_manifold(d, rng):
    """Fixed nonlinear embedding of a d-dimensional latent into R^AMBIENT."""
    B = rng.standard_normal((HIDDEN, d)) / np.sqrt(d)
    M = rng.standard_normal((AMBIENT, HIDDEN)) / np.sqrt(HIDDEN)
    return B, M


def sample(n, d, B, M, rng):
    u = rng.standard_normal((n, d))
    x = np.tanh(u @ B.T) @ M.T
    return x.astype(np.float32)


def encoder():
    return nn.Sequential(
        nn.Linear(AMBIENT, 256), nn.GELU(),
        nn.Linear(256, EMB),
    )


def predictor():
    return nn.Sequential(
        nn.Linear(EMB, 128), nn.GELU(),
        nn.Linear(128, EMB),
    )


def vicreg_reg(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / z.shape[1]
    return var_loss + cov_loss


def sigreg_reg(z):
    """Moment-matching penalty toward an isotropic standard Gaussian."""
    mu = z.mean(dim=0)
    zc = z - mu
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    eye = torch.eye(z.shape[1], device=z.device)
    return (mu ** 2).mean() + ((cov - eye) ** 2).mean()


def train(method, d, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 * d + seed)
    B, M = make_manifold(d, rng)

    enc = encoder().to(DEVICE)
    pred = predictor().to(DEVICE)
    if method == "EMA":
        tgt = copy.deepcopy(enc)
        for p in tgt.parameters():
            p.requires_grad_(False)
    else:
        tgt = enc  # stop-gradient target, same network

    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=LR)

    for _ in range(STEPS):
        x = sample(BATCH, d, B, M, rng)
        x1 = torch.from_numpy(x + NOISE * rng.standard_normal(x.shape).astype(np.float32)).to(DEVICE)
        x2 = torch.from_numpy(x + NOISE * rng.standard_normal(x.shape).astype(np.float32)).to(DEVICE)

        z1 = enc(x1)
        with torch.no_grad():
            z2 = tgt(x2)
        loss = F.mse_loss(pred(z1), z2)

        if method == "VICReg":
            loss = loss + LAMBDA * vicreg_reg(z1)
        elif method == "SIGReg":
            loss = loss + LAMBDA * sigreg_reg(z1)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if method == "EMA":
            with torch.no_grad():
                for pt, pe in zip(tgt.parameters(), enc.parameters()):
                    pt.mul_(EMA_M).add_(pe.detach(), alpha=1 - EMA_M)

    return enc, rng, B, M


@torch.no_grad()
def embed(enc, x):
    return enc(torch.from_numpy(x).to(DEVICE)).cpu().numpy()


def intrinsic_dim_mle(X, k=K_ID):
    """Levina-Bickel MLE intrinsic-dimension estimate."""
    from sklearn.neighbors import NearestNeighbors
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn_.kneighbors(X)
    dist = dist[:, 1:]
    dist = np.maximum(dist, 1e-12)
    inv = np.mean(np.log(dist[:, -1:] / dist[:, :-1]), axis=1)
    return float(np.mean(1.0 / np.maximum(inv, 1e-12)))


def effective_rank(Z):
    Zc = Z - Z.mean(axis=0)
    eig = np.linalg.eigvalsh(np.cov(Zc, rowvar=False))
    eig = np.maximum(eig, 0)
    return float((eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12))


def run_one(job):
    """One training run; executed in a worker process."""
    method, d, seed = job
    enc, rng, B, M = train(method, d, seed)
    eval_rng = np.random.default_rng(99999 + seed)
    x_eval = sample(N_EVAL, d, B, M, eval_rng)
    Z = embed(enc, x_eval)
    id_hat = intrinsic_dim_mle(Z)
    row = {
        "method": method, "d": d, "seed": seed,
        "id_hat": id_hat, "id_gap": abs(id_hat - d),
        "eff_rank": effective_rank(Z),
    }
    print(f"{method:7s} d={d:2d} seed={seed} id_hat={id_hat:6.2f} "
          f"gap={row['id_gap']:6.2f} erank={row['eff_rank']:6.2f}", flush=True)
    return row


def run_shard(k, n):
    """Runs jobs k, k+n, k+2n, ... and writes one part file (sharded across processes)."""
    import json
    jobs = [(m, d, s) for m in METHODS for d in DIMS for s in SEEDS]
    rows = [run_one(j) for j in jobs[k::n]]
    path = os.path.join(HERE, f"part_exp01_{k}.json")
    with open(path, "w") as f:
        json.dump(rows, f)
    print(f"shard {k} wrote {len(rows)} rows -> {path}", flush=True)


def aggregate():
    import glob
    import json
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "part_exp01_*.json"))):
        with open(path) as f:
            rows.extend(json.load(f))
    rows.sort(key=lambda r: (METHODS.index(r["method"]), r["d"], r["seed"]))
    csv_path = os.path.join(HERE, "results_exp01.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"Experiment 1 (simulation arm), {len(rows)} runs. Mean and 95% bootstrap CI over seeds.", ""]
    rng = np.random.default_rng(0)
    for method in METHODS:
        for d in DIMS:
            sel = [r for r in rows if r["method"] == method and r["d"] == d]
            gaps = np.array([r["id_gap"] for r in sel])
            erk = np.array([r["eff_rank"] for r in sel])
            boots = [np.mean(rng.choice(gaps, size=len(gaps))) for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            lines.append(
                f"{method:7s} d={d:2d} n={len(sel)}  id_gap={gaps.mean():6.2f} "
                f"[{lo:6.2f}, {hi:6.2f}]  eff_rank={erk.mean():6.2f}"
            )
    summary_path = os.path.join(HERE, "summary_exp01.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten: {csv_path}\nWritten: {summary_path}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--shard"]:
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    else:
        aggregate()
