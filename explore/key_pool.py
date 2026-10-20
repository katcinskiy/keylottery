"""
Randomized key selection (Aremu et al., arXiv:2507.07871v5, Algorithm 1) run on our human texts.

Rule: pool of r keys; detect with every key at a common Sidak-corrected threshold tau; a text is accepted as
"genuine" iff exactly one key fires (s = 1); s >= 2 is called "forgery". For a human text both verdicts are
false accusations, so we report P(s = 1) (the paper's FPR) and P(s >= 1).

Threshold: alpha_k = 1 - (1 - alpha_fw)^(1/r); tau = Phi^{-1}(1 - alpha_k) on the per-key uniq z-score
(the paper calibrates tau on the z statistic; it also offers an empirical quantile, which we report too,
estimated on the reference texts pooled over keys).

Keys: the 1000 fresh keys of the screening run. Texts: E2 (6000 fresh texts per domain). Colours: prf
(idealised hash, numpy only; C8 showed it matches lefthash qualitatively). Pools: 20000 random subsets of
r keys. Output: results/key_pool_prf.json and per-text z in results/key_pool_prf_z.npz.
Run from the project root:  uv run --with numpy,scipy python explore/key_pool.py
"""
import json, os, sys, time
import numpy as np
from scipy.stats import norm, binom

sys.path.insert(0, os.getcwd())
from kl.common import GAMMA, DOMAINS, make_keys, tuple_codes, Colourer
from kl.nullstats import TextSets

OUT = "results/key_pool_prf"
keys = make_keys(1001, seed=2027)[1:]
sets = {f"{d}/e2": np.load(f"data/{d}_e2.npz")["ids"] for d in DOMAINS}
sets.update({f"{d}/ref": np.load(f"data/{d}.npz")["ids"][6000:] for d in DOMAINS})
t0 = time.time()
ts = TextSets(sets, h=1)
col = Colourer(ts.U, 1, backend="prf")
print(f"distinct tuples {len(ts.U)}, built in {time.time() - t0:.0f}s", flush=True)

T = {s: np.asarray(ts.B[s].sum(axis=1)).ravel() for s in sets}
Z = {s: np.zeros((len(keys), ts.n[s]), np.float16) for s in sets}
for i, k in enumerate(keys):
    g = col.green(int(k)).astype(np.float32)
    for s in sets:
        G = np.asarray(ts.B[s] @ g).ravel()
        Z[s][i] = (G - GAMMA * T[s]) / np.sqrt(GAMMA * (1 - GAMMA) * T[s])
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(keys)} keys, {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(OUT + "_z.npz", **{s.replace("/", "|"): Z[s] for s in sets})

rng = np.random.default_rng(0)
N_POOLS = 20000
res = {}
for alpha_fw in (1e-2, 1e-3):
    for r in (4, 10):
        alpha_k = 1 - (1 - alpha_fw) ** (1 / r)
        tau_gauss = norm.isf(alpha_k)
        pools = np.stack([rng.choice(len(keys), r, replace=False) for _ in range(N_POOLS)])
        for d in DOMAINS:
            z = Z[f"{d}/e2"].astype(np.float32)
            # empirical tau as in the paper's Step 2: (1 - alpha_k) quantile of the null statistic, pooled over keys, on ref
            tau_emp = float(np.quantile(Z[f"{d}/ref"].astype(np.float32).ravel(), 1 - alpha_k))
            for tag, tau in (("gauss", tau_gauss), ("empirical", tau_emp)):
                F = (z > tau)                                   # [key, text] flags at the per-key level
                cnt = F[pools].sum(1)                           # [pool, text] number of keys that fire
                p1 = (cnt == 1).mean(1); pany = (cnt >= 1).mean(1)
                single = F.mean(1)                              # single-key FPR at alpha_k, per key
                floor = binom.ppf(0.99, 6000, alpha_fw) / (6000 * alpha_fw)
                res[f"{d}|afw={alpha_fw:g}|r={r}|{tag}"] = dict(
                    alpha_k=alpha_k, tau=float(tau), ideal_p99_over_afw=float(floor),
                    pool_exactly_one=dict(mean=float(p1.mean() / alpha_fw), p99=float(np.quantile(p1, .99) / alpha_fw), max=float(p1.max() / alpha_fw)),
                    pool_any=dict(mean=float(pany.mean() / alpha_fw), p99=float(np.quantile(pany, .99) / alpha_fw), max=float(pany.max() / alpha_fw)),
                    single_key_at_alpha_k=dict(mean=float(single.mean() / alpha_k), p99=float(np.quantile(single, .99) / alpha_k), max=float(single.max() / alpha_k)))
json.dump(dict(keys_seed=2027, n_keys=len(keys), n_pools=N_POOLS, backend="prf", rule="uniq", results=res), open(OUT + ".json", "w"), indent=1)

print("\nPool FPR / alpha_fw (p99 over 20000 random pools; 'one' = accepted as genuine, 'any' = any key fires):")
print(f"{'domain':8s} {'afw':>6s} {'r':>3s} {'tau':>9s} | {'one p99':>8s} {'one max':>8s} | {'any p99':>8s} | {'single-key p99':>14s} | {'ideal':>5s}")
for key, v in res.items():
    d, a, r, tag = key.split("|")
    print(f"{d:8s} {a[4:]:>6s} {r[2:]:>3s} {tag:>9s} | {v['pool_exactly_one']['p99']:8.2f} {v['pool_exactly_one']['max']:8.1f} | {v['pool_any']['p99']:8.2f} | {v['single_key_at_alpha_k']['p99']:14.2f} | {v['ideal_p99_over_afw']:5.2f}")
print(f"done in {time.time() - t0:.0f}s")
