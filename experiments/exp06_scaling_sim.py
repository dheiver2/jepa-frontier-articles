"""
Experiment 6 (synthetic arm) for artigo_06_scaling.tex.

Compute-matched grid for JEPA, MAE and contrastive pretraining on a synthetic family with a
known label-noise floor.

Data (generated on the fly, so training is single-epoch and D = examples seen):
  z ~ mixture of N_COMP isotropic Gaussians in R^8 (complexity = N_COMP)
  x = tanh(W2 tanh(W1 z))  in R^32 ;  a view = x + 0.1 eps with 30% of the dimensions zeroed
  y = a.z + N(0, SIGMA_Y^2)   -> irreducible MSE of the downstream task = SIGMA_Y^2

Methods (same encoder family: 32 -> w -> w -> 16 MLP; size axis = hidden width w):
  JEPA  context view -> encoder -> predictor, target = EMA encoder on the other view, VICReg variance term
  MAE   50% of dimensions masked -> encoder -> decoder, MSE on the masked dimensions
  CL    two views -> encoder -> projection head, symmetric InfoNCE (tau = 0.2)

Budget: for each method, N = number of trainable parameters (encoder + head), compute C = 6 N D,
so D = C / (6 N) examples (batch 128, steps = D / 128). Same C grid for all methods.
Downstream: frozen encoder (16-d) + ridge probe on 4000 labeled examples, test MSE on 4000 held-out.

Usage:  python3 exp06_scaling_sim.py --shard k n   (writes part_exp06_k.json)
        python3 exp06_scaling_sim.py               (aggregates, fits L(N,D))
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

SEEDS = [1, 2, 3, 4, 5]
METHODS = ["JEPA", "MAE", "CL"]
WIDTHS = [8, 16, 32, 64, 96]
BUDGETS = [2e8, 6e8, 2e9, 6e9]
DZ, DX, EMB = 8, 32, 16
N_COMP = 6
SIGMA_Y = 0.3
NOISE = 0.1
BATCH = 128
LR = 2e-3
N_PROBE_TR, N_PROBE_TE = 4000, 4000
E_FLOOR = SIGMA_Y ** 2


def make_world(seed):
    r = np.random.default_rng(31_000 + seed)
    f = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    return {
        "W1": f(r.standard_normal((24, DZ)) / np.sqrt(DZ) * 4.0),
        "W": f(r.standard_normal((DX, 24)) / np.sqrt(24) * 4.0),
        "mu": f(r.standard_normal((N_COMP, DZ)) * 1.5),
        "a": f(r.standard_normal(DZ) / np.sqrt(DZ) * 0.6),
        "b": f(r.standard_normal(DZ) / np.sqrt(DZ)),
    }


def draw(n, world, g):
    k = torch.randint(0, N_COMP, (n,), generator=g)
    z = world["mu"][k] + 0.7 * torch.randn(n, DZ, generator=g)
    x = torch.tanh(torch.tanh(z @ world["W1"].T) @ world["W"].T)
    return z, x


def view(x, g, drop=0.3):
    m = (torch.rand(x.shape, generator=g) > drop).float()
    return (x + NOISE * torch.randn(x.shape, generator=g)) * m


def enc(w):
    return nn.Sequential(nn.Linear(DX, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, EMB))


def head(w, out):
    return nn.Sequential(nn.Linear(EMB, w), nn.GELU(), nn.Linear(w, out))


def build(method, w):
    e = enc(w)
    h = head(w, DX if method == "MAE" else EMB)
    return e, h


def n_params(*mods):
    return sum(p.numel() for m in mods for p in m.parameters())


def var_term(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def info_nce(a, b, tau=0.2):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.T / tau
    lab = torch.arange(len(a))
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))


def train(method, w, budget, seed):
    torch.manual_seed(seed)
    world = make_world(seed)
    g = torch.Generator().manual_seed(seed * 7919 + w)
    e, h = build(method, w)
    N = n_params(e, h)
    D = budget / (6 * N)
    steps = max(1, int(round(D / BATCH)))
    params = list(e.parameters()) + list(h.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    et = None
    if method == "JEPA":
        et = copy.deepcopy(e)
        for p in et.parameters():
            p.requires_grad_(False)
    for t in range(steps):
        _, x = draw(BATCH, world, g)
        if method == "JEPA":
            v1, v2 = view(x, g), view(x, g)
            z = e(v1)
            with torch.no_grad():
                tz = et(v2)
            loss = F.mse_loss(h(z), tz) + var_term(z)
        elif method == "MAE":
            m = (torch.rand(x.shape, generator=g) > 0.5).float()
            xin = (x + NOISE * torch.randn(x.shape, generator=g)) * m
            rec = h(e(xin))
            loss = (((rec - x) ** 2) * (1 - m)).sum() / (1 - m).sum().clamp(min=1)
        else:
            loss = info_nce(h(e(view(x, g))), h(e(view(x, g))))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if et is not None:
            mom = min(0.99, (1 + t) / (10 + t))
            with torch.no_grad():
                for pt, ps in zip(et.parameters(), e.parameters()):
                    pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
    return e, world, N, D, steps


@torch.no_grad()
def probe_mse(e, world, seed):
    g = torch.Generator().manual_seed(900_000 + seed)

    def data(n):
        z, x = draw(n, world, g)
        y = z @ world["a"] + SIGMA_Y * torch.randn(n, generator=g)
        return e(x + NOISE * torch.randn(x.shape, generator=g)).numpy(), y.numpy()

    Xtr, ytr = data(N_PROBE_TR)
    Xte, yte = data(N_PROBE_TE)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    A = Xtr.T @ Xtr + 1e-2 * len(Xtr) * np.eye(EMB)
    b = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
    pred = Xte @ b + ytr.mean()
    return float(((pred - yte) ** 2).mean())


def run_one(job):
    method, w, bi, seed = job
    e, world, N, D, steps = train(method, w, BUDGETS[bi], seed)
    mse = probe_mse(e, world, seed)
    row = {"method": method, "width": w, "budget": BUDGETS[bi], "seed": seed,
           "N": N, "D": D, "steps": steps, "mse": mse}
    print(f"{method:5s} w={w:3d} C={BUDGETS[bi]:.0e} seed={seed} N={N:6d} D={D:9.0f} steps={steps:6d} mse={mse:.4f}",
          flush=True)
    return row


def jobs():
    j = [(m, w, bi, s) for m in METHODS for w in WIDTHS for bi in range(len(BUDGETS)) for s in SEEDS]
    # heaviest first (small N, large C), then deal round-robin so shards are balanced
    j.sort(key=lambda t: (t[3], -(BUDGETS[t[2]] / 1.0) / (t[1] + 1)))
    return j


def run_shard(k, n):
    path = os.path.join(HERE, f"part_exp06_{k}.jsonl")
    open(path, "w").close()
    for j in jobs()[k::n]:
        row = run_one(j)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def boot_ci(x, rng, n=4000):
    x = np.asarray(x, float)
    b = [np.mean(rng.choice(x, size=len(x))) for _ in range(n)]
    return float(np.mean(x)), *np.percentile(b, [2.5, 97.5])


def fit_surface(Ns, Ds, L):
    """L = E + A N^-alpha + B D^-beta, E >= E_FLOOR. Multi-start least squares on relative error."""
    from scipy.optimize import least_squares
    Ns, Ds, L = map(np.asarray, (Ns, Ds, L))

    def model(p, N, D):
        E, lA, al, lB, be = p
        return E + np.exp(lA) * N ** (-al) + np.exp(lB) * D ** (-be)

    def res(p):
        return (model(p, Ns, Ds) - L) / L

    best = None
    rs = np.random.default_rng(0)
    for _ in range(60):
        p0 = [E_FLOOR + rs.uniform(0, 0.2), rs.uniform(-2, 4), rs.uniform(0.1, 1.2), rs.uniform(-2, 6), rs.uniform(0.1, 1.2)]
        try:
            r = least_squares(res, p0, bounds=([E_FLOOR, -20, 0.01, -20, 0.01], [L.min(), 20, 3, 20, 3]))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x, model


def aggregate():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "part_exp06_*.jsonl"))):
        rows.extend(json.loads(l) for l in open(p) if l.strip())
    rows.sort(key=lambda r: (METHODS.index(r["method"]), r["width"], r["budget"], r["seed"]))
    with open(os.path.join(HERE, "results_exp06.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    rng = np.random.default_rng(0)
    lines = [f"Experiment 6 (synthetic arm), {len(rows)} runs, label-noise floor E = {E_FLOOR:.3f}. "
             "Test MSE mean [95% bootstrap CI] over seeds.", ""]
    fits = {}
    frontier = {}
    for m in METHODS:
        pts = []
        for w in WIDTHS:
            for C in BUDGETS:
                sel = [r for r in rows if r["method"] == m and r["width"] == w and r["budget"] == C]
                if not sel:
                    continue
                mu, lo, hi = boot_ci([r["mse"] for r in sel], rng)
                pts.append((sel[0]["N"], sel[0]["D"], C, w, mu, lo, hi))
        P = np.array(pts)
        lines.append(f"-- {m}: grid (width, N, C, D, MSE [CI])")
        for p in pts:
            lines.append(f"   w={int(p[3]):3d} N={int(p[0]):6d} C={p[2]:.0e} D={p[1]:10.0f}  {p[4]:.4f} [{p[5]:.4f}, {p[6]:.4f}]")
        popt, model = fit_surface(P[:, 0], P[:, 1], P[:, 4])
        fit = model(popt, P[:, 0], P[:, 1])
        ss_res = ((fit - P[:, 4]) ** 2).sum()
        ss_tot = ((P[:, 4] - P[:, 4].mean()) ** 2).sum()
        inband = float(np.mean((fit >= P[:, 5]) & (fit <= P[:, 6])))
        E, lA, al, lB, be = popt
        Ngrid = np.exp(np.linspace(np.log(300), np.log(2e5), 600))
        nstar = []
        for C in BUDGETS:
            Lg = model(popt, Ngrid, C / (6 * Ngrid))
            nstar.append(float(Ngrid[np.argmin(Lg)]))
        slope = float(np.polyfit(np.log(BUDGETS), np.log(nstar), 1)[0])
        fits[m] = dict(E=float(E), alpha=float(al), beta=float(be), r2=float(1 - ss_res / ss_tot),
                       inband=inband, nstar=nstar, slope=slope)
        lines.append(f"   fit: E={E:.3f} alpha={al:.2f} beta={be:.2f}  R2(fit)={fits[m]['r2']:.3f}  "
                     f"points within CI band={inband:.2f}  N*(C)={[int(x) for x in nstar]}  d log N*/d log C={slope:.2f}")
        lines.append("")
        # compute frontier: best width per budget (by mean MSE), with its CI
        fr = []
        for C in BUDGETS:
            cand = [p for p in pts if p[2] == C]
            b = min(cand, key=lambda p: p[4])
            fr.append((C, int(b[3]), b[4], b[5], b[6]))
        frontier[m] = fr
    lines.append("== compute frontier (best width per budget)")
    for i, C in enumerate(BUDGETS):
        lines.append(f"C={C:.0e}  " + "   ".join(
            f"{m}: w={frontier[m][i][1]:3d} {frontier[m][i][2]:.4f} [{frontier[m][i][3]:.4f},{frontier[m][i][4]:.4f}]"
            for m in METHODS))
    lines.append("")
    lines.append("== paired seed differences at the best width of each pair, frontier MSE (A - B), mean [95% CI]")
    for a, b in [("JEPA", "MAE"), ("JEPA", "CL"), ("MAE", "CL")]:
        for C in BUDGETS:
            def best_seedwise(m):
                w = [f for f in frontier[m] if f[0] == C][0][1]
                return {r["seed"]: r["mse"] for r in rows if r["method"] == m and r["width"] == w and r["budget"] == C}
            da, db = best_seedwise(a), best_seedwise(b)
            d = [da[s] - db[s] for s in da if s in db]
            mu, lo, hi = boot_ci(d, rng)
            lines.append(f"{a}-{b} C={C:.0e}: {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    json.dump({"fits": fits, "frontier": frontier}, open(os.path.join(HERE, "fits_exp06.json"), "w"), indent=1)
    open(os.path.join(HERE, "summary_exp06.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--shard"]:
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    else:
        aggregate()
