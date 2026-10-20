"""Key pool (Aremu et al.) with the EXACT transformers colours (lefthash, torch randperm) for the two domains where the
prf run left the pool above the ideal level: code and wiki_ru. 1000 fresh keys (seed 2027), E2 texts, rule uniq,
exact binomial per-key test at the Sidak level. Writes results/key_pool_lefthash_z.npz and results/key_pool_lefthash.json.
Run from the project root:  PYTHONPATH=. uv run --with torch,numpy,scipy python explore/key_pool_lefthash.py
"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.getcwd())
from kl.common import GAMMA, make_keys, Colourer
from kl.nullstats import TextSets, crit_table

DOMS = ["wiki_ru", "code"]
keys = make_keys(1001, seed=2027)[1:]
_G = {}

def _init():
    import torch
    torch.set_num_threads(1)

def _work(idx):
    out = {}
    for i in idx:
        g = _G["col"].green(int(keys[i])).astype(np.float32)
        out[i] = {s: (np.asarray(_G["ts"].B[s] @ g).ravel()).astype(np.int16) for s in _G["ts"].names}   # green counts
    return out

if __name__ == "__main__":
    t0 = time.time()
    sets = {f"{d}/e2": np.load(f"data/{d}_e2.npz")["ids"] for d in DOMS}
    ts = TextSets(sets, h=1)
    _G["ts"] = ts
    _G["col"] = Colourer(ts.U, 1, backend="lefthash")         # built once in the parent, shared by fork
    print(f"distinct tuples {len(ts.U)}; contexts per key: {len(_G['col'].impl.ctx)}", flush=True)
    T = {s: np.asarray(ts.B[s].sum(axis=1)).ravel().astype(int) for s in sets}
    Gc = {s: np.zeros((len(keys), ts.n[s]), np.int16) for s in sets}
    W = max(2, min(12, (os.cpu_count() or 4) - 3))
    chunks = [list(range(i, min(i + 10, len(keys)))) for i in range(0, len(keys), 10)]
    done = 0
    with mp.get_context("fork").Pool(W, initializer=_init) as pool:
        for res in pool.imap_unordered(_work, chunks):
            for i, d in res.items():
                for s in sets:
                    Gc[s][i] = d[s]
            done += len(res)
            if done % 100 == 0:
                print(f"  {done}/{len(keys)} keys, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed("results/key_pool_lefthash_z.npz", **{s.replace("/", "|") + "|G": Gc[s] for s in sets}, **{s.replace("/", "|") + "|T": T[s] for s in sets})

    rng = np.random.default_rng(0); N_POOLS = 20000
    out = {}
    print(f"\n{'domain':8s} {'afw':>6s} {'r':>3s} | {'pool mean':>9s} {'pool p99':>8s} {'pool max':>8s} | {'single p99':>10s} | {'ideal':>5s}")
    for alpha_fw in (1e-3, 1e-2):
        for r in (4, 10):
            alpha_k = 1 - (1 - alpha_fw) ** (1 / r)
            crit = crit_table(alphas=np.array([alpha_k]))[:, 0]
            pools = np.stack([rng.choice(len(keys), r, replace=False) for _ in range(N_POOLS)])
            for d in DOMS:
                s = f"{d}/e2"
                F = Gc[s] >= crit[np.minimum(T[s], len(crit) - 1)]
                cnt = F[pools].sum(1); p1 = (cnt == 1).mean(1); single = F.mean(1)
                floor = binom.ppf(0.99, ts.n[s], alpha_fw) / (ts.n[s] * alpha_fw)
                v = dict(alpha_k=alpha_k, pool_mean=float(p1.mean() / alpha_fw), pool_p99=float(np.quantile(p1, .99) / alpha_fw),
                         pool_max=float(p1.max() / alpha_fw), single_mean=float(single.mean() / alpha_k),
                         single_p99=float(np.quantile(single, .99) / alpha_k), ideal=float(floor))
                out[f"{d}|afw={alpha_fw:g}|r={r}"] = v
                print(f"{d:8s} {alpha_fw:>6g} {r:>3d} | {v['pool_mean']:9.2f} {v['pool_p99']:8.2f} {v['pool_max']:8.1f} | {v['single_p99']:10.2f} | {floor:5.2f}", flush=True)
    json.dump(dict(keys_seed=2027, n_keys=len(keys), n_pools=N_POOLS, backend="lefthash", rule="uniq", results=out),
              open("results/key_pool_lefthash.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")
