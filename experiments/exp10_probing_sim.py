"""
Experiment 10 (simulated-state arm) for artigo_10_probing.tex.

A simulated system with known state s in R^8 (four 2-D rotating blocks with different
frequencies, a mild tanh coupling and process noise) is rendered into a 128-d observation
o = tanh(s B) M + noise. A JEPA encoder is trained to predict the embedding of o_{t+1} from o_t
(EMA target encoder + variance term). Representation width m in {2,4,8,16,32,64}.

Compared at each width: JEPA-trained encoder vs the SAME architecture untrained (random).
Probes (frozen encoder): linear (logistic) and a one-hidden-layer MLP (64 units).
  Fidelity    - R^2 of a linear probe recovering the true state s from the latent
  Property    - y = 1[s_1 > 0]
  Control     - labels random per cell of an 8x8 grid over (s_1, s_2) (a control task in the sense
                of Hewitt & Liang): measures how much a probe can fit arbitrary structure
  Selectivity - Acc(property) - Acc(control)  (held-out accuracy)
  Sufficiency - accuracy of a linear probe predicting 1[s_{1,t+5} > 0] from the latent at t

Usage:  python3 exp10_probing_sim.py
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
WIDTHS = [2, 4, 8, 16, 32, 64]
ENCODERS = ["JEPA", "Random"]
PROBES = ["linear", "mlp"]
S_DIM, AMB = 8, 128
STEPS, BATCH, LR = 1500, 256, 1e-3
N_TRAIN, N_TEST = 3000, 3000
HORIZON = 5
GRID = 8


def make_world(rng):
    freqs = np.array([0.05, 0.11, 0.19, 0.31])
    R = np.zeros((S_DIM, S_DIM))
    for k, f in enumerate(freqs):
        c, s = np.cos(f), np.sin(f)
        R[2 * k:2 * k + 2, 2 * k:2 * k + 2] = [[c, -s], [s, c]]
    B = rng.standard_normal((S_DIM, 64)) / np.sqrt(S_DIM)
    M = rng.standard_normal((64, AMB)) / 8.0
    table = rng.integers(0, 2, size=(GRID, GRID))
    return R, B, M, table


def trajectories(n_traj, length, W, rng):
    R = W[0]
    s = rng.standard_normal((n_traj, S_DIM))
    out = [s]
    for _ in range(length - 1):
        s = s @ R.T
        s = s + 0.1 * np.tanh(s[:, ::-1]) * 0.2 + 0.05 * rng.standard_normal(s.shape)
        s = s / np.maximum(1.0, np.linalg.norm(s, axis=1, keepdims=True) / 3.5)
        out.append(s)
    return np.stack(out, 1)      # (n_traj, length, S_DIM)


def render(s, W, rng):
    B, M = W[1], W[2]
    return (np.tanh(s @ B) @ M + 0.05 * rng.standard_normal((*s.shape[:-1], AMB))).astype(np.float32)


def mk_enc(m):
    return nn.Sequential(nn.Linear(AMB, 256), nn.GELU(), nn.Linear(256, m))


def train_jepa(enc, O0, O1, seed):
    g = torch.Generator().manual_seed(seed)
    m = enc[-1].out_features
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(m, 128), nn.GELU(), nn.Linear(128, m))
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    X0, X1 = torch.from_numpy(O0), torch.from_numpy(O1)
    for _ in range(STEPS):
        i = torch.randint(0, X0.shape[0], (BATCH,), generator=g)
        z = enc(X0[i])
        with torch.no_grad():
            t = tgt(X1[i])
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


def cell(s):
    q = np.clip(((s[:, :2] + 3.5) / 7.0 * GRID).astype(int), 0, GRID - 1)
    return q[:, 0], q[:, 1]


def probe_acc(kind, Ztr, ytr, Zte, yte, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ztr)
    A, B = sc.transform(Ztr), sc.transform(Zte)
    if kind == "linear":
        clf = LogisticRegression(max_iter=2000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=150, random_state=seed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(A, ytr)
    return float((clf.predict(B) == yte).mean())


def r2_linear(Ztr, Str, Zte, Ste):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ztr)
    reg = Ridge(alpha=1.0).fit(sc.transform(Ztr), Str)
    P = reg.predict(sc.transform(Zte))
    return float(1 - ((P - Ste) ** 2).sum() / ((Ste - Ste.mean(0)) ** 2).sum())


def run_seed(seed):
    rng = np.random.default_rng(101 * seed)
    W = make_world(rng)
    T = HORIZON + 2
    tr = trajectories(N_TRAIN * 2 // T + 200, T, W, rng)
    te = trajectories(N_TEST // 2 + 200, T, W, rng)
    # training pairs for JEPA (all consecutive steps)
    S_tr = tr[:, :-1].reshape(-1, S_DIM)
    O0 = render(S_tr, W, rng)
    O1 = render(tr[:, 1:].reshape(-1, S_DIM), W, rng)
    # probe sets: state at t=0 of each trajectory plus t+HORIZON for sufficiency
    s_pr = tr[:N_TRAIN, 0]
    s_te = te[:N_TEST, 0]
    o_pr, o_te = render(s_pr, W, rng), render(s_te, W, rng)
    y_pr, y_te = (s_pr[:, 0] > 0).astype(int), (s_te[:, 0] > 0).astype(int)
    f_pr, f_te = (tr[:N_TRAIN, HORIZON, 0] > 0).astype(int), (te[:N_TEST, HORIZON, 0] > 0).astype(int)
    a, b = cell(s_pr), cell(s_te)
    c_pr, c_te = W[3][a[0], a[1]], W[3][b[0], b[1]]
    rows = []
    for m in WIDTHS:
        torch.manual_seed(seed * 100 + m)
        enc0 = mk_enc(m)
        encs = {"Random": copy.deepcopy(enc0), "JEPA": train_jepa(copy.deepcopy(enc0), O0, O1, seed)}
        for name in ENCODERS:
            with torch.no_grad():
                Zp = encs[name](torch.from_numpy(o_pr)).numpy()
                Zt = encs[name](torch.from_numpy(o_te)).numpy()
            fid = r2_linear(Zp, s_pr, Zt, s_te)
            suff = probe_acc("linear", Zp, f_pr, Zt, f_te, seed)
            for pk in PROBES:
                at = probe_acc(pk, Zp, y_pr, Zt, y_te, seed)
                ac = probe_acc(pk, Zp, c_pr, Zt, c_te, seed)
                rows.append({"encoder": name, "width": m, "probe": pk, "seed": seed,
                             "acc_true": at, "acc_control": ac, "selectivity": at - ac,
                             "fidelity_r2": fid, "sufficiency": suff})
            print(f"seed={seed} m={m:2d} {name:6s} fid={fid:.3f} suff={suff:.3f} "
                  f"lin sel={rows[-2]['selectivity']:+.3f} mlp sel={rows[-1]['selectivity']:+.3f}", flush=True)
    return rows


def boot(v, rng, n=5000):
    v = np.asarray(v, float)
    b = [np.mean(rng.choice(v, size=len(v))) for _ in range(n)]
    return float(v.mean()), *np.percentile(b, [2.5, 97.5])


def aggregate_and_plot(rows):
    with open(os.path.join(HERE, "results_exp10.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    lines = [f"Experiment 10 (simulated-state arm), {len(rows)} rows, {len(SEEDS)} seeds. Mean [95% bootstrap CI].", ""]
    st = {}
    for pk in PROBES:
        for name in ENCODERS:
            for m in WIDTHS:
                sel = [r for r in rows if r["probe"] == pk and r["encoder"] == name and r["width"] == m]
                d = {k: boot([r[k] for r in sel], rng) for k in
                     ["acc_true", "acc_control", "selectivity", "fidelity_r2", "sufficiency"]}
                st[(pk, name, m)] = d
                lines.append(f"{pk:6s} {name:6s} m={m:2d}  true={d['acc_true'][0]:.3f} control={d['acc_control'][0]:.3f} "
                             f"sel={d['selectivity'][0]:+.3f} [{d['selectivity'][1]:+.3f}, {d['selectivity'][2]:+.3f}]  "
                             f"R2={d['fidelity_r2'][0]:.3f} [{d['fidelity_r2'][1]:.3f}, {d['fidelity_r2'][2]:.3f}]  "
                             f"suff={d['sufficiency'][0]:.3f}")
        lines.append("")
    lines.append("Paired JEPA - Random, fidelity R^2 (mean [95% CI over seeds]) per width:")
    for m in WIDTHS:
        a = {r["seed"]: r["fidelity_r2"] for r in rows if r["encoder"] == "JEPA" and r["width"] == m and r["probe"] == "linear"}
        b = {r["seed"]: r["fidelity_r2"] for r in rows if r["encoder"] == "Random" and r["width"] == m and r["probe"] == "linear"}
        mm, lo, hi = boot([a[s] - b[s] for s in a], rng)
        lines.append(f"  m={m:2d}: {mm:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    with open(os.path.join(HERE, "summary_exp10.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.4), sharex=True)
    for name, mk in zip(ENCODERS, ["o", "s"]):
        for k, (key, pk, ax) in enumerate([("fidelity_r2", "linear", axs[0]), ("selectivity", "mlp", axs[1])]):
            m_ = np.array([st[(pk, name, m)][key][0] for m in WIDTHS])
            lo = np.array([st[(pk, name, m)][key][1] for m in WIDTHS])
            hi = np.array([st[(pk, name, m)][key][2] for m in WIDTHS])
            ax.errorbar(WIDTHS, m_, yerr=[m_ - lo, hi - m_], marker=mk, ms=2.5, capsize=1.5, lw=0.9,
                        label=name if k == 0 else None)
    axs[0].set_ylabel("Fidelity $R^2$ (linear)")
    axs[1].set_ylabel("Selectivity (MLP probe)")
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.set_xticks(WIDTHS)
        ax.set_xticklabels([str(w) for w in WIDTHS], fontsize=6)
        ax.set_xlabel("Width $m$")
    axs[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig10_probing.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    allrows = []
    for s in SEEDS:
        allrows.extend(run_seed(s))
    aggregate_and_plot(allrows)
