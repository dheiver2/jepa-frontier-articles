"""
Runs 10 experiments per article (100 total), each with a fixed 60 s wall-clock
budget. Within an experiment, seeded trials repeat until the budget is spent and
the metrics are averaged over trials (mean and 95% CI).

Each article's experiments are PROXIES that run on CPU with synthetic or
small public-style data. They test the article's central question at small
scale. They are not the full benchmarks named in the protocols (ImageNet,
LLM reasoning suites, PTB-XL, etc.), and the articles must say so.

Usage:  python3 run_all.py            (10 workers, ~10 min wall time)
Outputs: results/results_all.json and results/summary.md
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
BUDGET = 60.0  # seconds per experiment


# ---------------------------------------------------------------- budget loop
def timed(trial, cfg):
    t0 = time.time()
    rows, i = [], 0
    while True:
        rows.append(trial(cfg, seed=i))
        i += 1
        if time.time() - t0 >= BUDGET:
            break
    keys = list(rows[0].keys())
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[k] = {"mean": float(v.mean()), "ci95": float(half)}
    out["trials"] = len(rows)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- article 01
def trial_01(cfg, seed):
    import torch
    torch.set_num_threads(1)
    import exp01_collapse_sim as E
    E.STEPS = 300
    enc, rng, B, M = E.train(cfg["method"], cfg["d"], seed)
    x_eval = E.sample(1000, cfg["d"], B, M, np.random.default_rng(10_000 + seed))
    Z = E.embed(enc, x_eval)
    idh = E.intrinsic_dim_mle(Z)
    return {"id_gap": abs(idh - cfg["d"]), "eff_rank": E.effective_rank(Z)}


CFG_01 = [
    {"method": "EMA", "d": 2}, {"method": "EMA", "d": 8}, {"method": "EMA", "d": 32},
    {"method": "VICReg", "d": 2}, {"method": "VICReg", "d": 8}, {"method": "VICReg", "d": 32},
    {"method": "SIGReg", "d": 2}, {"method": "SIGReg", "d": 8}, {"method": "SIGReg", "d": 32},
    {"method": "EMA", "d": 16},
]


# ---------------------------------------------------------------- article 02
def trial_02(cfg, seed):
    """Linear world model on a noisy point mass; rollout horizon h."""
    rng = np.random.default_rng(seed)
    dt, n = 0.1, 3000
    A = np.array([[1, dt], [0, 1.0]])
    Bm = np.array([[0.5 * dt * dt], [dt]])
    s = rng.standard_normal((n, 2))
    a = rng.uniform(-1, 1, (n, 1))
    s_next = s @ A.T + a @ Bm.T + 0.01 * rng.standard_normal((n, 2))
    X = np.hstack([s, a])
    W, *_ = np.linalg.lstsq(X, s_next, rcond=None)  # learned linear dynamics
    h = cfg["h"]
    s0 = rng.standard_normal((500, 2))
    acts = rng.uniform(-1, 1, (500, h))
    obs0 = s0 + 0.05 * rng.standard_normal(s0.shape)
    pred, true = obs0.copy(), s0.copy()
    for t in range(h):
        pred = np.hstack([pred, acts[:, t:t + 1]]) @ W
        true = true @ A.T + acts[:, t:t + 1] @ Bm.T
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean(0)) ** 2).sum()
    return {"r2_h": float(1 - ss_res / ss_tot)}


CFG_02 = [{"h": h} for h in range(1, 11)]


# ---------------------------------------------------------------- article 03
def trial_03(cfg, seed):
    """Confounded linear SCM: observational slope vs true do-effect."""
    rng = np.random.default_rng(seed)
    n = 5000
    c = cfg["c"]
    U = rng.standard_normal(n)
    X1 = 1.0 * U + rng.standard_normal(n)
    Xn = 1.0 * X1 + c * U + rng.standard_normal(n)  # true causal effect of X1 is 1.0
    slope = np.polyfit(X1, Xn, 1)[0]              # observational (correlational) slope
    x_do = rng.uniform(-2, 2, 500)
    true_mean = 1.0 * x_do
    corr_pred = slope * x_do
    return {"corr_do_mse": float(np.mean((corr_pred - true_mean) ** 2)),
            "obs_slope": float(slope)}


CFG_03 = [{"c": round(c, 2)} for c in np.linspace(0.0, 2.0, 10)]


# ---------------------------------------------------------------- article 04
def trial_04(cfg, seed):
    """Answer readout accuracy vs reasoning depth on synthetic arithmetic."""
    from sklearn.neural_network import MLPClassifier
    rng = np.random.default_rng(seed)
    k, N = cfg["k"], 3000
    ops = rng.choice([-1, 1], size=(N, k))
    vals = rng.integers(0, 10, size=(N, k))
    feats = np.concatenate([ops, vals], axis=1).astype(float)
    y = (np.sum(ops * vals, axis=1)) % 10
    tr, te = slice(0, 2400), slice(2400, N)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed)
    clf.fit(feats[tr], y[tr])
    return {"acc": float(clf.score(feats[te], y[te]))}


CFG_04 = [{"k": k} for k in range(1, 11)]


# ---------------------------------------------------------------- article 05
def trial_05(cfg, seed):
    """Top-level embedding recovers slow state; hierarchical vs flat pooling."""
    rng = np.random.default_rng(seed)
    K, N, dz, df = cfg["K"], 2000, 4, 16
    s = rng.standard_normal((N, dz))
    W = rng.standard_normal((dz, df))
    F = np.stack([s @ W + 0.8 * rng.standard_normal((N, df)) for _ in range(K)], axis=1)
    hier = F.mean(axis=1)                  # section-level pooling
    flat = F.reshape(N, -1)[:, :df]        # flat: first section only (no pooling)
    def r2(X):
        Xb = np.hstack([X, np.ones((N, 1))])
        beta, *_ = np.linalg.lstsq(Xb[:1500], s[:1500], rcond=None)
        pred = Xb[1500:] @ beta
        return float(1 - ((pred - s[1500:]) ** 2).sum() / ((s[1500:] - s[1500:].mean(0)) ** 2).sum())
    return {"r2_hier": r2(hier), "r2_flat": r2(flat)}


CFG_05 = [{"K": K} for K in range(2, 12)]


# ---------------------------------------------------------------- article 06
def trial_06(cfg, seed):
    """Random-feature regression: test MSE vs width (proxy for scaling)."""
    rng = np.random.default_rng(seed)
    N, d, w = 4000, 8, cfg["w"]
    X = rng.standard_normal((N, d))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.standard_normal(N)
    Wr = rng.standard_normal((d, w))
    Phi = np.tanh(X @ Wr)
    beta = np.linalg.solve(Phi[:3000].T @ Phi[:3000] + 1e-3 * np.eye(w), Phi[:3000].T @ y[:3000])
    mse = np.mean((Phi[3000:] @ beta - y[3000:]) ** 2)
    return {"test_mse": float(mse), "params": float(d * w + w)}


CFG_06 = [{"w": w} for w in [4, 8, 16, 32, 64, 96, 128, 192, 256, 384]]


# ---------------------------------------------------------------- article 07
def trial_07(cfg, seed):
    """Bimodal future: coverage of unimodal vs mixture predictive intervals."""
    rng = np.random.default_rng(seed)
    sig, N = cfg["sigma"], 4000
    mode = rng.choice([-1.0, 1.0], size=N)
    y = mode + sig * rng.standard_normal(N)
    # unimodal Gaussian interval (mean 0, std from data)
    s_hat = y.std()
    cov_uni = np.mean(np.abs(y - 0.0) <= 1.645 * s_hat)
    # mixture: two modes, interval = union of +-1.645*sig around each mode
    cov_mix = np.mean((np.abs(y - 1.0) <= 1.645 * sig) | (np.abs(y + 1.0) <= 1.645 * sig))
    return {"cov_unimodal": float(cov_uni), "cov_mixture": float(cov_mix)}


CFG_07 = [{"sigma": round(s, 3)} for s in np.linspace(0.05, 0.5, 10)]


# ---------------------------------------------------------------- article 08
def trial_08(cfg, seed):
    """Low-label readout on synthetic ECG-like beats: PCA features vs raw."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    T, N = 200, 3000
    t = np.linspace(0, 1, T)
    y = rng.integers(0, 2, N)  # 0 normal, 1 anomalous morphology
    freq = 4 + 2 * y + 0.3 * rng.standard_normal(N)
    X = np.sin(2 * np.pi * freq[:, None] * t) + 0.2 * rng.standard_normal((N, T))
    pca = PCA(n_components=16).fit(X[:2000])  # 'pretraining' on unlabeled data
    Z = pca.transform(X)
    frac = cfg["frac"]
    n_lab = max(10, int(frac * 2000))
    idx = slice(2000, 2000 + n_lab)
    te = slice(2500, N)
    clf_z = LogisticRegression(max_iter=500).fit(Z[idx], y[idx])
    clf_r = LogisticRegression(max_iter=500).fit(X[idx], y[idx])
    return {"acc_pca": float(clf_z.score(Z[te], y[te])),
            "acc_raw": float(clf_r.score(X[te], y[te]))}


CFG_08 = [{"frac": round(f, 3)} for f in np.linspace(0.02, 0.5, 10)]


# ---------------------------------------------------------------- article 09
def trial_09(cfg, seed):
    """Shift severity: clean-trained vs adapted-with-replay accuracy."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    N, d = 3000, 20
    w = rng.standard_normal(d)
    X = rng.standard_normal((N, d))
    y = (X @ w > 0).astype(int)
    sev = cfg["s"]
    Xs = X + 0.1 * sev * rng.standard_normal((N, d)) + 0.3 * sev * np.roll(X, 1, axis=1)
    ys = (Xs @ w > 0).astype(int)
    clean = LogisticRegression(max_iter=500).fit(X[:2000], y[:2000])
    acc_clean_on_shift = clean.score(Xs[2500:], ys[2500:])
    # adaptation with replay: clean samples + 200 shifted samples
    rep_X = np.vstack([X[:1000], Xs[2000:2200]])
    rep_y = np.concatenate([y[:1000], ys[2000:2200]])
    adapted = LogisticRegression(max_iter=500).fit(rep_X, rep_y)
    return {"acc_no_adapt": float(acc_clean_on_shift),
            "acc_replay": float(adapted.score(Xs[2500:], ys[2500:])),
            "acc_clean_source": float(clean.score(X[2500:], y[2500:]))}


CFG_09 = [{"s": s} for s in range(1, 11)]


# ---------------------------------------------------------------- article 10
def trial_10(cfg, seed):
    """Probe selectivity: true labels minus random-label control, linear probe."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    N, m = 2000, cfg["m"]
    z = rng.standard_normal((N, 8))            # latent state
    R = rng.standard_normal((8, m))
    rep = np.tanh(z @ R)                         # frozen 'encoder' output of width m
    y = (z[:, 0] > 0).astype(int)                # property encoded in latent
    ctrl = rng.integers(0, 2, N)                 # control task: random labels
    tr, te = slice(0, 1500), slice(1500, N)
    acc_true = LogisticRegression(max_iter=500).fit(rep[tr], y[tr]).score(rep[te], y[te])
    acc_ctrl = LogisticRegression(max_iter=500).fit(rep[tr], ctrl[tr]).score(rep[te], ctrl[te])
    return {"acc_true": float(acc_true), "acc_control": float(acc_ctrl),
            "selectivity": float(acc_true - acc_ctrl)}


CFG_10 = [{"m": m} for m in [2, 4, 6, 8, 10, 12, 16, 20, 32, 64]]


# ---------------------------------------------------------------- dispatch
TRIALS = {1: trial_01, 2: trial_02, 3: trial_03, 4: trial_04, 5: trial_05,
          6: trial_06, 7: trial_07, 8: trial_08, 9: trial_09, 10: trial_10}
CONFIGS = {1: CFG_01, 2: CFG_02, 3: CFG_03, 4: CFG_04, 5: CFG_05,
           6: CFG_06, 7: CFG_07, 8: CFG_08, 9: CFG_09, 10: CFG_10}


def run_article(art):
    """Runs the 10 experiments of one article sequentially (one worker per article)."""
    import sys
    sys.path.insert(0, HERE)
    trial = TRIALS[art]
    results = []
    for j, cfg in enumerate(CONFIGS[art], start=1):
        res = timed(trial, cfg)
        results.append({"article": art, "exp": j, "cfg": cfg, "result": res})
        print(f"[art {art:02d} exp {j:02d}] {cfg} trials={res['trials']} "
              f"sec={res['seconds']}", flush=True)
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with Pool(processes=10) as pool:
        per_article = pool.map(run_article, list(range(1, 11)))
    flat = [r for block in per_article for r in block]
    with open(os.path.join(OUT_DIR, "results_all.json"), "w") as f:
        json.dump(flat, f, indent=2)
    print(f"done: {len(flat)} experiments -> {OUT_DIR}/results_all.json")


if __name__ == "__main__":
    main()
