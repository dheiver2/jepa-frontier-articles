"""
Experiment 4 (synthetic arithmetic arm) for artigo_04_language_reasoning.tex.

Task: y = (sum_i o_i v_i) mod 10, o_i in {-1,+1}, v_i in {0..9}, depth k in {1,2,3,4}.
A question is a token sequence [CLS, (op, value) x k]. Surface forms: 3 synonyms per
operator, numeric or word tokens per value, and terms may be reordered (the sum is
commutative); a "paraphrase" is another random surface form of the same problem.
Training questions are canonical with prob 0.5, otherwise a random paraphrase.

All methods share the same question encoder (2-layer Transformer, d=64, CLS pooling),
the same batch size, optimizer, learning rate and number of optimizer steps. A model
is trained separately for each depth.

  AR        - answer-token cross-entropy (linear head on CLS). With a one-token answer
              this is the autoregressive objective.
  JEPA      - g(f(q)) regressed on the EMA-target embedding of the answer tokens
              (answers appear as numeric or word tokens), variance/covariance terms
              against collapse, plus an agreement term between g(f(q)) and g(f(q'))
              for a paraphrase q' of the same problem (the "paraphrase signal").
  JEPA-nopara - same without the agreement term.

JEPA answers are decoded two ways: (readout) a linear head trained on the frozen
predictions, with the same number of steps for every method (1000, backbone frozen);
(nearest) nearest answer embedding, with no readout head.

Reported: exact-match accuracy on 2000 fresh test problems, in canonical form and in
paraphrased form, 95% bootstrap CIs over seeds. Chance = 0.10.

Not included (pending): GSM8K-style word problems (English / Portuguese).
Usage: python3 exp04_language_reasoning.py [--shard k n | --agg]
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
SEEDS = list(range(1, 6))
DEPTHS = [1, 2, 3, 4]
METHODS = ["AR", "JEPA", "JEPA-nopara"]
STEPS = 1500
READOUT_STEPS = 1000
BATCH = 128
LR = 1e-3
D = 64
ZD = 8
EMA_M = 0.99
LAM_PARA = 1.0
N_TEST = 2000

CLS = 1
PLUS = [2, 3, 4]
MINUS = [5, 6, 7]
VNUM = list(range(8, 18))
VWORD = list(range(18, 28))
ANUM = list(range(28, 38))
AWORD = list(range(38, 48))
VOCAB = 48
MAXLEN = 1 + 2 * 4


def make_problems(n, k, rng):
    v = rng.integers(0, 10, (n, k))
    o = rng.choice([-1, 1], (n, k))
    y = (o * v).sum(1) % 10
    return v, o, y


def tokens(v, o, rng, canonical):
    n, k = v.shape
    toks = np.zeros((n, MAXLEN), dtype=np.int64)
    toks[:, 0] = CLS
    if canonical:
        order = np.tile(np.arange(k), (n, 1))
        optok = np.where(o > 0, PLUS[0], MINUS[0])
        valtok = np.array(VNUM)[v]
    else:
        order = np.argsort(rng.random((n, k)), axis=1)
        syn = rng.integers(0, 3, (n, k))
        optok = np.where(o > 0, np.array(PLUS)[syn], np.array(MINUS)[syn])
        word = rng.random((n, k)) < 0.5
        valtok = np.where(word, np.array(VWORD)[v], np.array(VNUM)[v])
        optok = np.take_along_axis(optok, order, 1)
        valtok = np.take_along_axis(valtok, order, 1)
    toks[:, 1:1 + 2 * k:2] = optok
    toks[:, 2:2 + 2 * k:2] = valtok
    return torch.from_numpy(toks)


def answer_tokens(y, rng):
    word = rng.random(len(y)) < 0.5
    return torch.from_numpy(np.where(word, np.array(AWORD)[y], np.array(ANUM)[y]))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(MAXLEN, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)

    def forward(self, t):
        h = self.emb(t) + self.pos[None]
        h = self.tr(h, src_key_padding_mask=(t == 0))
        return self.norm(h[:, 0])


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def vc_reg(z, cov=True):
    std = torch.sqrt(z.var(0) + 1e-4)
    loss = F.relu(1.0 - std).mean()
    if cov:
        zc = z - z.mean(0)
        c = zc.T @ zc / (z.shape[0] - 1)
        off = c - torch.diag(torch.diag(c))
        loss = loss + (off ** 2).sum() / z.shape[1]
    return loss


def batch(k, rng, n=BATCH):
    v, o, y = make_problems(n, k, rng)
    canon = rng.random() < 0.5
    q = tokens(v, o, rng, canon)
    qp = tokens(v, o, rng, False)
    a = answer_tokens(y, rng)
    return q, qp, a, torch.from_numpy(y)


def train(method, k, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(900 + 17 * seed + k)
    f = Encoder()
    if method == "AR":
        head = nn.Linear(D, 10)
        params = list(f.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=LR)
        for _ in range(STEPS):
            q, qp, a, y = batch(k, rng)
            loss = F.cross_entropy(head(f(q)), y)
            opt.zero_grad(); loss.backward(); opt.step()
        return {"f": f, "head": head}
    g = mlp(D, ZD)
    ae = nn.Sequential(nn.Embedding(VOCAB, 32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, ZD))   # online answer encoder
    tgt = copy.deepcopy(ae)
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()) + list(ae.parameters()), lr=LR)
    for _ in range(STEPS):
        q, qp, a, y = batch(k, rng)
        zp = g(f(q))
        with torch.no_grad():
            zt = tgt(a)
        za = ae(a)
        loss = F.mse_loss(zp, zt) + 0.5 * F.mse_loss(za, zp.detach()) + vc_reg(za, cov=True) + vc_reg(zp, cov=False)
        if method == "JEPA":
            loss = loss + LAM_PARA * F.mse_loss(zp, g(f(qp)))
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pt, pe in zip(tgt.parameters(), ae.parameters()):
                pt.mul_(EMA_M).add_(pe.detach(), alpha=1 - EMA_M)
    # answer embeddings of the ten classes (mean of numeric and word form), for "nearest" decoding
    with torch.no_grad():
        E = 0.5 * (tgt(torch.tensor(ANUM)) + tgt(torch.tensor(AWORD)))
    # linear readout on frozen predictions
    head = nn.Linear(ZD, 10)
    oh = torch.optim.Adam(head.parameters(), lr=LR)
    for _ in range(READOUT_STEPS):
        q, qp, a, y = batch(k, rng)
        with torch.no_grad():
            z = g(f(q))
        loss = F.cross_entropy(head(z), y)
        oh.zero_grad(); loss.backward(); oh.step()
    return {"f": f, "g": g, "head": head, "E": E}


@torch.no_grad()
def evaluate(method, M, k, seed):
    rng = np.random.default_rng(777000 + 31 * seed + k)
    v, o, y = make_problems(N_TEST, k, rng)
    yt = torch.from_numpy(y)
    out = {}
    for form, canon in [("canon", True), ("para", False)]:
        q = tokens(v, o, rng, canon)
        if method == "AR":
            out[f"acc_{form}"] = float((M["head"](M["f"](q)).argmax(1) == yt).float().mean())
        else:
            z = M["g"](M["f"](q))
            out[f"acc_{form}"] = float((M["head"](z).argmax(1) == yt).float().mean())
            d = torch.cdist(z, M["E"])
            out[f"acc_{form}_nearest"] = float((d.argmin(1) == yt).float().mean())
    return out


def nparams(M, method):
    mods = [M["f"], M["head"]] + ([M["g"]] if method != "AR" else [])
    return sum(p.numel() for m in mods for p in m.parameters())


def run_job(method, k, seed):
    t0 = time.time()
    M = train(method, k, seed)
    r = evaluate(method, M, k, seed)
    row = {"method": method, "depth": k, "seed": seed, "acc_canon": r["acc_canon"], "acc_para": r["acc_para"],
           "acc_canon_nearest": r.get("acc_canon_nearest", float("nan")),
           "acc_para_nearest": r.get("acc_para_nearest", float("nan")),
           "params": nparams(M, method)}
    print(f"{method:11s} k={k} seed={seed} canon={row['acc_canon']:.3f} para={row['acc_para']:.3f} "
          f"nearest=({row['acc_canon_nearest']:.3f},{row['acc_para_nearest']:.3f}) [{time.time() - t0:.0f}s]", flush=True)
    return row


def aggregate():
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "part_exp04_*.json"))):
        rows.extend(json.load(open(p)))
    rows.sort(key=lambda r: (r["depth"], METHODS.index(r["method"]), r["seed"]))
    with open(os.path.join(HERE, "results_exp04.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        if len(x) == 0:
            return float("nan"), float("nan"), float("nan")
        b = [np.mean(rng.choice(x, size=len(x))) for _ in range(5000)]
        return float(x.mean()), *np.percentile(b, [2.5, 97.5])

    lines = [f"Experiment 4 (synthetic arithmetic arm), {len(rows)} runs. Exact-match accuracy, mean [95% bootstrap CI over seeds]. Chance 0.10.", ""]
    for k in DEPTHS:
        lines.append(f"-- depth k={k}")
        for m in METHODS:
            sel = [r for r in rows if r["depth"] == k and r["method"] == m]
            if not sel:
                continue
            p = []
            for key in ["acc_canon", "acc_para", "acc_canon_nearest", "acc_para_nearest"]:
                mu, lo, hi = boot([r[key] for r in sel])
                if not np.isnan(mu):
                    p.append(f"{key}={mu:.3f} [{lo:.3f},{hi:.3f}]")
            lines.append(f"  {m:11s} n={len(sel)} params={sel[0]['params']} " + " ".join(p))
    lines.append("\n-- paired differences in acc_para (same seed and depth)")
    by = {m: {(r["depth"], r["seed"]): r["acc_para"] for r in rows if r["method"] == m} for m in METHODS}
    for a, b in [("JEPA", "AR"), ("JEPA", "JEPA-nopara"), ("JEPA-nopara", "AR")]:
        for k in DEPTHS:
            d = [by[a][key] - by[b][key] for key in by[a] if key in by[b] and key[0] == k]
            if d:
                mu, lo, hi = boot(d)
                lines.append(f"  k={k} {a} - {b}: {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(d)})")
    open(os.path.join(HERE, "summary_exp04.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if "--agg" in sys.argv:
        aggregate()
    else:
        k0, m0 = (int(sys.argv[2]), int(sys.argv[3])) if "--shard" in sys.argv else (0, 1)
        jobs = [(m, k, s) for k in DEPTHS for s in SEEDS for m in METHODS]
        out = [run_job(*j) for j in jobs[k0::m0]]
        json.dump(out, open(os.path.join(HERE, f"part_exp04_{k0}.json"), "w"))
        if "--shard" not in sys.argv:
            aggregate()
