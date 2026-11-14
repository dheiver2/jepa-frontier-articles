"""
Experiment 2 (simulation arm) for artigo_02_world_model.tex.

Latent world model + CEM-MPC on a simulated torque-limited pendulum with a
nonlinear 12-d observation map and 16 distractor dimensions that are fresh
unit-variance noise at every step (unpredictable by construction).

Methods (same encoder/predictor sizes, same optimizer steps, same unroll length):
  JEPA    - latent predictor trained against an EMA target encoder (stop-grad)
            plus a variance term; no decoder, no reconstruction.
  Recon   - encoder + decoder + latent dynamics, trained to reconstruct the
            observations ("pixel-reconstruction world model").
  Known   - MPC with the true simulator and the true-state cost (upper bound).
  Random  - random actions (lower bound).

Planning: CEM-MPC over horizon H, re-planned every step. Learned models score
candidates by the summed squared distance between predicted latents and the
goal latent f(o_goal). Success = |theta_T - theta_goal| < 0.15 and |omega_T| < 0.6
after T steps. Dynamics fidelity = held-out R^2 of a ridge probe from z to the
true state (theta, omega).

Not included (pending): MuJoCo tasks and language-specified goals.

Usage:  python3 exp02_world_model.py            # run all seeds, then aggregate
        python3 exp02_world_model.py --agg      # aggregate only
"""
import copy
import csv
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
DT = 0.1
OBS_STATE = 12
OBS_DIST = 16
OBS = OBS_STATE + OBS_DIST
ZD = 16
HID = 128
N_EP, T_EP = 300, 40          # training episodes, steps per episode
UNROLL = 4
STEPS = 3000
BATCH = 256
LR = 1e-3
EMA_M = 0.99
HORIZON = 12
CEM_N, CEM_ELITE, CEM_ITERS = 256, 16, 3
N_TEST = 40                    # planning episodes per seed
T_PLAN = 40
OBS_NOISE = 0.05


def step_true(s, a):
    """s: (...,2) [theta, omega]; a in [-1,1]. Torque-limited pendulum (stable at 0)."""
    th, om = s[..., 0], s[..., 1]
    om2 = om + DT * (-4.0 * torch.sin(th) - 0.5 * om + 4.0 * a)
    th2 = th + DT * om2
    return torch.stack([th2, om2], dim=-1)


def make_obs_map(seed):
    g = torch.Generator().manual_seed(7000 + seed)
    return torch.randn(2, OBS_STATE, generator=g) * 1.5, torch.randn(OBS_STATE, generator=g) * 0.5


def observe(s, Mw, Mb, noise=True, distract=True):
    base = torch.tanh(s @ Mw + Mb)
    if noise:
        base = base + OBS_NOISE * torch.randn_like(base)
    d = torch.randn(*s.shape[:-1], OBS_DIST) if distract else torch.zeros(*s.shape[:-1], OBS_DIST)
    return torch.cat([base, d], dim=-1)


def collect(Mw, Mb):
    s = torch.cat([torch.empty(N_EP, 1).uniform_(-1.2, 1.2), torch.empty(N_EP, 1).uniform_(-1, 1)], -1)
    S, A = [s], []
    for _ in range(T_EP):
        # temporally correlated random actions so that large angles are visited
        a = torch.empty(N_EP).uniform_(-1, 1) if not A else (0.8 * A[-1] + 0.2 * torch.empty(N_EP).uniform_(-1, 1) * 2).clamp(-1, 1)
        s = step_true(s, a)
        A.append(a)
        S.append(s)
    S = torch.stack(S, 1)                  # (N,T+1,2)
    A = torch.stack(A, 1)                  # (N,T)
    O = observe(S, Mw, Mb)
    return S, A, O


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, o))


class Model(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.enc = mlp(OBS, ZD)
        self.pred = mlp(ZD + 1, ZD)
        if kind == "Recon":
            self.dec = mlp(ZD, OBS)
        else:
            self.tgt = copy.deepcopy(self.enc)
            for p in self.tgt.parameters():
                p.requires_grad_(False)

    def rollout(self, z, A):
        """z: (B,ZD); A: (B,K). returns (B,K,ZD) predicted latents."""
        out = []
        for k in range(A.shape[1]):
            z = z + self.pred(torch.cat([z, A[:, k:k + 1]], -1))   # residual dynamics
            out.append(z)
        return torch.stack(out, 1)


def var_reg(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    zc = z - z.mean(0)
    cov = zc.T @ zc / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return F.relu(1.0 - std).mean() + (off ** 2).sum() / z.shape[1]


def train(kind, S, A, O):
    m = Model(kind)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
    N = O.shape[0]
    for it in range(STEPS):
        i = torch.randint(0, N, (BATCH,))
        t = torch.randint(0, T_EP - UNROLL + 1, (BATCH,))
        idx = t[:, None] + torch.arange(UNROLL + 1)[None]
        o = O[i[:, None], idx]                                      # (B,K+1,OBS)
        a = A[i[:, None], t[:, None] + torch.arange(UNROLL)[None]]  # (B,K)
        z0 = m.enc(o[:, 0])
        zp = m.rollout(z0, a)
        if kind == "JEPA":
            with torch.no_grad():
                zt = m.tgt(o[:, 1:])
            loss = F.mse_loss(zp, zt) + 1.0 * var_reg(z0)
        else:
            rec0 = F.mse_loss(m.dec(z0), o[:, 0])
            recp = F.mse_loss(m.dec(zp), o[:, 1:])
            loss = rec0 + recp
        opt.zero_grad()
        loss.backward()
        opt.step()
        if kind == "JEPA":
            with torch.no_grad():
                for pt, pe in zip(m.tgt.parameters(), m.enc.parameters()):
                    pt.mul_(EMA_M).add_(pe.detach(), alpha=1 - EMA_M)
    return m


@torch.no_grad()
def probe_r2(m, S, O):
    """Ridge probe z -> (theta, omega), fit on 70% of episodes, scored on held-out 30%."""
    n = O.shape[0]
    ntr = int(0.7 * n)
    Z = m.enc(O[:, :-1].reshape(-1, OBS)).reshape(n, T_EP, ZD)
    Y = S[:, :-1]
    Ztr, Ytr = Z[:ntr].reshape(-1, ZD), Y[:ntr].reshape(-1, 2)
    Zte, Yte = Z[ntr:].reshape(-1, ZD), Y[ntr:].reshape(-1, 2)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    X1 = torch.cat([(Ztr - mu) / sd, torch.ones(len(Ztr), 1)], 1)
    W = torch.linalg.solve(X1.T @ X1 + 1e-2 * torch.eye(ZD + 1), X1.T @ Ytr)
    P = torch.cat([(Zte - mu) / sd, torch.ones(len(Zte), 1)], 1) @ W
    r2 = 1 - ((P - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    eff = float(((torch.linalg.eigvalsh(torch.cov(Zte.T)).clamp(min=0).sum()) ** 2) /
                ((torch.linalg.eigvalsh(torch.cov(Zte.T)).clamp(min=0) ** 2).sum() + 1e-12))
    return float(r2[0]), float(r2[1]), eff


def cem_plan(cost_fn, B, rng):
    """Generic CEM. cost_fn(actions (B,N,H)) -> (B,N) costs. Returns first action (B,)."""
    mu = torch.zeros(B, HORIZON)
    sd = torch.full((B, HORIZON), 0.7)
    for _ in range(CEM_ITERS):
        a = (mu[:, None] + sd[:, None] * torch.randn(B, CEM_N, HORIZON, generator=rng)).clamp(-1, 1)
        c = cost_fn(a)
        idx = c.topk(CEM_ELITE, dim=1, largest=False).indices
        el = torch.gather(a, 1, idx[..., None].expand(-1, -1, HORIZON))
        mu, sd = el.mean(1), el.std(1) + 0.05
    return mu[:, 0]


@torch.no_grad()
def plan_eval(method, m, Mw, Mb, seed):
    g = torch.Generator().manual_seed(555 + seed)
    s = torch.cat([torch.empty(N_TEST, 1).uniform_(-1.2, 1.2, generator=g),
                   torch.empty(N_TEST, 1).uniform_(-1, 1, generator=g)], -1)
    goal = torch.empty(N_TEST).uniform_(-1.2, 1.2, generator=g)
    goal_s = torch.stack([goal, torch.zeros(N_TEST)], -1)
    if m is not None:
        zg = m.enc(observe(goal_s, Mw, Mb, noise=False, distract=False))      # (B,ZD)
    for _ in range(T_PLAN):
        if method == "Random":
            a = torch.empty(N_TEST).uniform_(-1, 1, generator=g)
        elif method == "Known":
            def cost(acts, s=s):
                x = s[:, None].expand(-1, CEM_N, -1)
                c = torch.zeros(acts.shape[:2])
                for h in range(HORIZON):
                    x = step_true(x, acts[..., h])
                    c = c + (x[..., 0] - goal[:, None]) ** 2 + 0.1 * x[..., 1] ** 2
                return c
            a = cem_plan(cost, N_TEST, g)
        else:
            z0 = m.enc(observe(s, Mw, Mb))
            def cost(acts, z0=z0):
                B = acts.shape[0]
                z = z0[:, None].expand(-1, CEM_N, -1).reshape(B * CEM_N, ZD)
                A_ = acts.reshape(B * CEM_N, HORIZON)
                c = torch.zeros(B * CEM_N)
                for h in range(HORIZON):
                    z = z + m.pred(torch.cat([z, A_[:, h:h + 1]], -1))
                    c = c + ((z.reshape(B, CEM_N, ZD) - zg[:, None]) ** 2).sum(-1).reshape(-1)
                return c.reshape(B, CEM_N)
            a = cem_plan(cost, N_TEST, g)
        s = step_true(s, a)
    ok = ((s[:, 0] - goal).abs() < 0.15) & (s[:, 1].abs() < 0.6)
    err = (s[:, 0] - goal).abs()
    return float(ok.float().mean()), float(err.mean())


def run_seed(seed):
    torch.manual_seed(seed)
    Mw, Mb = make_obs_map(seed)
    S, A, O = collect(Mw, Mb)
    rows = []
    for method in ["JEPA", "Recon", "Known", "Random"]:
        t0 = time.time()
        if method in ("JEPA", "Recon"):
            m = train(method, S, A, O)
            Sv, Av, Ov = collect(Mw, Mb)
            r2_th, r2_om, er = probe_r2(m, Sv, Ov)
        else:
            m, r2_th, r2_om, er = None, float("nan"), float("nan"), float("nan")
        succ, err = plan_eval(method, m, Mw, Mb, seed)
        rows.append({"method": method, "seed": seed, "success": succ, "final_err": err,
                     "r2_theta": r2_th, "r2_omega": r2_om, "eff_rank": er})
        print(f"seed={seed} {method:6s} success={succ:.3f} err={err:.3f} r2=({r2_th:.3f},{r2_om:.3f}) "
              f"erank={er:.2f} [{time.time() - t0:.0f}s]", flush=True)
    return rows


def boot(x, rng, n=5000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    b = [np.mean(rng.choice(x, size=len(x))) for _ in range(n)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def aggregate():
    import glob
    import json
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "part_exp02_*.json"))):
        rows.extend(json.load(open(p)))
    order = ["JEPA", "Recon", "Known", "Random"]
    rows.sort(key=lambda r: (order.index(r["method"]), r["seed"]))
    with open(os.path.join(HERE, "results_exp02.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    lines = [f"Experiment 2 (simulation arm), {len(rows)} rows, {len(set(r['seed'] for r in rows))} seeds. Mean [95% bootstrap CI over seeds].", ""]
    for mth in order:
        sel = [r for r in rows if r["method"] == mth]
        parts = []
        for k in ["success", "final_err", "r2_theta", "r2_omega", "eff_rank"]:
            mu, lo, hi = boot([r[k] for r in sel], rng)
            parts.append(f"{k}={mu:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"{mth:6s} n={len(sel)}  " + "  ".join(parts))
    # paired JEPA - Recon success difference
    j = {r["seed"]: r["success"] for r in rows if r["method"] == "JEPA"}
    c = {r["seed"]: r["success"] for r in rows if r["method"] == "Recon"}
    d = [j[s] - c[s] for s in j if s in c]
    mu, lo, hi = boot(d, rng)
    lines.append(f"\nPaired JEPA - Recon success: {mu:.3f} [{lo:.3f}, {hi:.3f}] (n={len(d)} seeds)")
    open(os.path.join(HERE, "summary_exp02.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import json
    if "--agg" in sys.argv:
        aggregate()
    elif "--shard" in sys.argv:
        k, n = int(sys.argv[2]), int(sys.argv[3])
        rows = []
        for s in SEEDS[k::n]:
            rows.extend(run_seed(s))
        json.dump(rows, open(os.path.join(HERE, f"part_exp02_{k}.json"), "w"))
    else:
        rows = []
        for s in SEEDS:
            rows.extend(run_seed(s))
        json.dump(rows, open(os.path.join(HERE, "part_exp02_0.json"), "w"))
        aggregate()
