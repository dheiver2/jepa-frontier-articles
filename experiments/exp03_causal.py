"""
Experiment 3 (simulation arm) for artigo_03_causal.tex.

Random structural causal models (SCMs) with n in {5,10,20} observed variables,
linear or nonlinear mechanisms, and two hidden confounders. One hidden confounder
loads on the target and on one of the target's parents (the "confounded parent").

Models are trained on OBSERVATIONAL samples only, to predict the last variable
(the target) from all the other observed variables (the context):
  Reg    - MLP regression of the target value (correlation-based).
  VAE    - VAE over the observed vector with the target hidden from the encoder,
           decoder reconstructs all variables; prediction = decoded target.
  JEPA   - context encoder f, predictor g, EMA target encoder over the target
           scalar (stop-gradient) + variance term; a small value readout h is fit
           on frozen target latents so that predictions can be compared in value space.
  Oracle - true structural equation of the target applied to the observed parents.

Interventional error. For each parent i of the target and 9 values x on a grid
spanning +-2 sd of X_i, the simulator draws 2000 samples under do(X_i = x). A model
predicts the target from the context of each sample; error = squared difference
between the mean prediction and the mean simulated target, averaged over the grid,
and divided by the observational variance of the target.
JEPA is also scored in latent space (squared distance between the mean predicted
latent and the mean target latent, divided by the total target-latent variance).
Structural recovery: parents of the target recovered from the model's sensitivity
(mean |d yhat / d c_j| scaled by the sd of c_j and of y, edge if > 0.10); we report
the number of wrong incoming edges of the target (SHD of the target's in-edges).

Not included (pending): text-derived causal statements as conditioning.

Usage: python3 exp03_causal.py [--shard k n | --agg]
"""
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(1, 11))
NS = [5, 10, 20]
KINDS = ["linear", "nonlinear"]
N_OBS = 20000
STEPS = 2000
BATCH = 256
LR = 1e-3
HID = 128
ZD = 8
EMA_M = 0.99
SIGMA = 0.5
LOAD = 1.5
N_DO = 2000
GRID = 9


class SCM:
    def __init__(self, n, kind, seed):
        rng = np.random.default_rng(31 * seed + n + (0 if kind == "linear" else 1000))
        p = {5: 0.45, 10: 0.3, 20: 0.2}[n]
        self.n, self.kind = n, kind
        self.pa = []
        for j in range(n):
            pa = [i for i in range(j) if rng.random() < p]
            if j == n - 1 and len(pa) < 2:           # target needs >=2 parents
                pa = sorted(set(pa) | set(rng.choice(n - 1, size=2, replace=False).tolist()))
            self.pa.append(pa)
        self.W = [rng.choice([-1, 1], len(pa)) * rng.uniform(0.5, 1.2, len(pa)) / np.sqrt(max(len(pa), 1)) * 1.4
                  for pa in self.pa]
        tp = self.pa[n - 1]
        self.conf_parent = int(rng.choice(tp))
        others = [i for i in range(n - 1) if i != self.conf_parent]
        self.loads = np.zeros((2, n))
        self.loads[0, n - 1] = LOAD * rng.choice([-1, 1])
        self.loads[0, self.conf_parent] = LOAD * rng.choice([-1, 1])
        for i in rng.choice(others, size=2, replace=False):
            self.loads[1, i] = LOAD * rng.choice([-1, 1])
        self.loads[1, n - 1] = LOAD * 0.5 * rng.choice([-1, 1])

    def f(self, u):
        return u if self.kind == "linear" else np.tanh(u) + 0.25 * u

    def sample(self, N, rng, do=None):
        X = np.zeros((N, self.n))
        U = rng.standard_normal((N, 2))
        for j in range(self.n):
            if do is not None and j == do[0]:
                X[:, j] = do[1]
                continue
            u = X[:, self.pa[j]] @ self.W[j] if self.pa[j] else 0.0
            u = u + U @ self.loads[:, j]
            X[:, j] = self.f(u) + SIGMA * rng.standard_normal(N)
        return X

    def oracle(self, C):
        """Structural mean of the target given the observed parents (context C = X[:, :n-1])."""
        j = self.n - 1
        return self.f(C[:, self.pa[j]] @ self.W[j])


def mlp(i, o, h=HID):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


def var_reg(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def fit_reg(C, y, seed):
    torch.manual_seed(seed)
    net = mlp(C.shape[1], 1)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(STEPS):
        i = torch.randint(0, len(C), (BATCH,))
        loss = F.mse_loss(net(C[i]).squeeze(-1), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda c: net(c).squeeze(-1), None


def fit_vae(C, y, seed, beta=0.1):
    torch.manual_seed(seed)
    enc = mlp(C.shape[1], 2 * ZD)
    dec = mlp(ZD, C.shape[1] + 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    full = torch.cat([C, y[:, None]], 1)
    for _ in range(STEPS):
        i = torch.randint(0, len(C), (BATCH,))
        h = enc(C[i])
        mu, lv = h[:, :ZD], h[:, ZD:].clamp(-6, 3)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        rec = F.mse_loss(dec(z), full[i])
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1).mean()
        loss = rec + beta * kl / full.shape[1]
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda c: dec(enc(c)[:, :ZD])[:, -1], None


def fit_jepa(C, y, seed):
    torch.manual_seed(seed)
    f = mlp(C.shape[1], ZD)
    g = mlp(ZD, ZD, 64)
    ft = mlp(1, ZD, 64)
    tgt = copy.deepcopy(ft)
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()) + list(ft.parameters()), lr=LR)
    # tgt = EMA(ft). ft is the online target branch: it is pulled toward the predicted latent
    # (so it carries information the context can predict) and held away from collapse by a variance term.
    for _ in range(STEPS):
        i = torch.randint(0, len(C), (BATCH,))
        zp = g(f(C[i]))
        with torch.no_grad():
            zt = tgt(y[i, None])
        zo = ft(y[i, None])
        loss = F.mse_loss(zp, zt) + var_reg(zp) + var_reg(zo) + 0.5 * F.mse_loss(zo, zp.detach())
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pt, pe in zip(tgt.parameters(), ft.parameters()):
                pt.mul_(EMA_M).add_(pe.detach(), alpha=1 - EMA_M)
    # value readout fit on frozen target latents
    h = mlp(ZD, 1, 64)
    oh = torch.optim.Adam(h.parameters(), lr=LR)
    with torch.no_grad():
        Zt = tgt(y[:, None])
    for _ in range(1500):
        i = torch.randint(0, len(C), (BATCH,))
        l2 = F.mse_loss(h(Zt[i]).squeeze(-1), y[i])
        oh.zero_grad(); l2.backward(); oh.step()
    with torch.no_grad():
        ro = F.mse_loss(h(Zt).squeeze(-1), y).item() / float(y.var())
    jepa = {"f": f, "g": g, "tgt": tgt, "h": h, "readout_resid": ro}
    return (lambda c: h(g(f(c))).squeeze(-1)), jepa


def sensitivity(pred, C, y_sd, c_sd):
    c = C[:3000].clone().requires_grad_(True)
    out = pred(c).sum()
    gr, = torch.autograd.grad(out, c)
    return (gr.abs().mean(0) * c_sd / y_sd).detach().numpy()


def run_one(n, kind, seed):
    scm = SCM(n, kind, seed)
    rng = np.random.default_rng(10_000 + 97 * seed + n)
    X = scm.sample(N_OBS, rng)
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    C = torch.tensor(Xs[:, :-1], dtype=torch.float32)
    y = torch.tensor(Xs[:, -1], dtype=torch.float32)
    # held-out observational set
    Xv = scm.sample(5000, rng)
    Cv = torch.tensor((Xv[:, :-1] - mu[:-1]) / sd[:-1], dtype=torch.float32)
    yv = (Xv[:, -1] - mu[-1]) / sd[-1]
    parents = scm.pa[n - 1]
    models = {}
    for name, fn in [("Reg", fit_reg), ("VAE", fit_vae), ("JEPA", fit_jepa)]:
        models[name] = fn(C, y, seed)
    rows = []
    c_sd = torch.tensor(np.ones(n - 1), dtype=torch.float32)
    for name in ["Reg", "VAE", "JEPA", "Oracle"]:
        if name == "Oracle":
            pred_np = lambda Cn: (scm.oracle(Cn * sd[:-1] + mu[:-1]) - mu[-1]) / sd[-1]
            obs_mse = float(np.mean((pred_np(Cv.numpy()) - yv) ** 2))
            shd = 0
            sens = None
        else:
            pred = models[name][0]
            with torch.no_grad():
                obs_mse = float(np.mean((pred(Cv).numpy() - yv) ** 2))
            pred_np = lambda Cn, pred=pred: pred(torch.tensor(Cn, dtype=torch.float32)).detach().numpy()
            s = sensitivity(pred, C, 1.0, c_sd)
            edges = s > 0.10
            truth = np.zeros(n - 1, bool)
            truth[parents] = True
            shd = int((edges != truth).sum())
        errs = {}
        lat = None
        for i in parents:
            xs = np.linspace(mu[i] - 2 * sd[i], mu[i] + 2 * sd[i], GRID)
            pm, tm, lm_p, lm_t = [], [], [], []
            for x in xs:
                Xd = scm.sample(N_DO, np.random.default_rng([555, seed, abs(int(round(x * 1000)))]), do=(i, x))
                Cd = (Xd[:, :-1] - mu[:-1]) / sd[:-1]
                pm.append(pred_np(Cd).mean())
                tm.append(((Xd[:, -1] - mu[-1]) / sd[-1]).mean())
                if name == "JEPA":
                    J = models["JEPA"][1]
                    with torch.no_grad():
                        lm_p.append(J["g"](J["f"](torch.tensor(Cd, dtype=torch.float32))).mean(0).numpy())
                        lm_t.append(J["tgt"](torch.tensor(((Xd[:, -1] - mu[-1]) / sd[-1])[:, None], dtype=torch.float32)).mean(0).numpy())
            errs[i] = float(np.mean((np.array(pm) - np.array(tm)) ** 2))   # target already standardized: var=1
            if name == "JEPA":
                with torch.no_grad():
                    Zt = models["JEPA"][1]["tgt"](y[:, None])
                tv = float(Zt.var(0).sum())
                lat = lat if lat is not None else []
                lat.append(float(np.mean(((np.array(lm_p) - np.array(lm_t)) ** 2).sum(1)) / tv))
        e_conf = errs[scm.conf_parent]
        un = [errs[i] for i in parents if i != scm.conf_parent]
        rows.append({"n": n, "kind": kind, "seed": seed, "method": name,
                     "int_err": float(np.mean(list(errs.values()))), "int_err_conf": e_conf,
                     "int_err_unconf": float(np.mean(un)) if un else float("nan"),
                     "obs_mse": obs_mse, "shd_in": shd,
                     "lat_err": float(np.mean(lat)) if lat else float("nan")})
    return rows


def aggregate():
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "part_exp03_*.json"))):
        rows.extend(json.load(open(p)))
    order = ["Reg", "VAE", "JEPA", "Oracle"]
    rows.sort(key=lambda r: (r["kind"], r["n"], order.index(r["method"]), r["seed"]))
    with open(os.path.join(HERE, "results_exp03.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        if len(x) == 0:
            return float("nan"), float("nan"), float("nan")
        b = [np.mean(rng.choice(x, size=len(x))) for _ in range(5000)]
        return float(x.mean()), *np.percentile(b, [2.5, 97.5])

    lines = [f"Experiment 3 (simulation arm), {len(rows)} rows. Mean [95% bootstrap CI over SCM seeds]. Errors are normalised by the target variance.", ""]
    for kind in KINDS:
        for n in NS:
            lines.append(f"-- {kind}, n={n}")
            for mth in order:
                sel = [r for r in rows if r["kind"] == kind and r["n"] == n and r["method"] == mth]
                if not sel:
                    continue
                p = []
                for k in ["int_err", "int_err_conf", "obs_mse", "shd_in", "lat_err"]:
                    mu, lo, hi = boot([r[k] for r in sel])
                    p.append(f"{k}={mu:.3f} [{lo:.3f},{hi:.3f}]")
                lines.append(f"  {mth:6s} k={len(sel)} " + " ".join(p))
    lines.append("\n-- pooled over all SCMs (per method)")
    for mth in order:
        sel = [r for r in rows if r["method"] == mth]
        p = []
        for k in ["int_err", "int_err_conf", "int_err_unconf", "obs_mse", "shd_in", "lat_err"]:
            mu, lo, hi = boot([r[k] for r in sel])
            p.append(f"{k}={mu:.3f} [{lo:.3f},{hi:.3f}]")
        lines.append(f"  {mth:6s} k={len(sel)} " + " ".join(p))
    # paired differences (JEPA - Reg), JEPA - VAE
    lines.append("\n-- paired differences in int_err over the same SCMs")
    key = lambda r: (r["kind"], r["n"], r["seed"])
    by = {m: {key(r): r["int_err"] for r in rows if r["method"] == m} for m in order}
    for a, b in [("JEPA", "Reg"), ("JEPA", "VAE"), ("Reg", "VAE")]:
        d = [by[a][k] - by[b][k] for k in by[a] if k in by[b]]
        mu, lo, hi = boot(d)
        lines.append(f"  {a} - {b}: {mu:.4f} [{lo:.4f}, {hi:.4f}] (k={len(d)})")
    open(os.path.join(HERE, "summary_exp03.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if "--agg" in sys.argv:
        aggregate()
    else:
        k, m = (int(sys.argv[2]), int(sys.argv[3])) if "--shard" in sys.argv else (0, 1)
        jobs = [(n, kd, s) for kd in KINDS for n in NS for s in SEEDS]
        out = []
        for (n, kd, s) in jobs[k::m]:
            t0 = time.time()
            rows = run_one(n, kd, s)
            out.extend(rows)
            print(f"{kd} n={n} seed={s} " + " ".join(f"{r['method']}:{r['int_err']:.3f}" for r in rows) +
                  f" [{time.time() - t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(HERE, f"part_exp03_{k}.json"), "w"))
        if "--shard" not in sys.argv:
            aggregate()
