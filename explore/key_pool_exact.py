"""Key pool (Aremu et al.) with the exact binomial per-key test at the Sidak level, from the per-text z saved by key_pool.py.
Same test as every other point on slide 2. Run: uv run --with numpy,scipy python explore/key_pool_exact.py"""
import json, numpy as np
from scipy.stats import binom
from kl.common import GAMMA, DOMAINS
from kl.nullstats import crit_table
Z = np.load("results/key_pool_prf_z.npz")
T = {d: np.load(f"data/{d}_e2.npz")["ids"][:, 3:] for d in DOMAINS}          # T = distinct pairs per text
def T_uniq(ids):
    codes = ids[:, :-1].astype(np.int64) * 50272 + ids[:, 1:]
    return np.array([len(np.unique(r)) for r in codes])
rng = np.random.default_rng(0); N_POOLS = 20000
out = {}
print(f"{'domain':8s} {'afw':>6s} {'r':>3s} | {'pool mean':>9s} {'pool p99':>8s} {'pool max':>8s} | {'single mean':>11s} {'single p99':>10s} | {'ideal':>5s}")
for alpha_fw in (1e-3, 1e-2):
    for r in (4, 10):
        alpha_k = 1 - (1 - alpha_fw) ** (1 / r)
        crit = crit_table(alphas=np.array([alpha_k]))[:, 0]
        pools = np.stack([rng.choice(1000, r, replace=False) for _ in range(N_POOLS)])
        for d in DOMAINS:
            z = Z[f"{d}|e2"].astype(np.float32); Tx = T_uniq(T[d])
            G = np.rint(z * np.sqrt(GAMMA * (1 - GAMMA) * Tx) + GAMMA * Tx).astype(int)   # z -> green count (integer)
            F = G >= crit[np.minimum(Tx, len(crit) - 1)]
            cnt = F[pools].sum(1); p1 = (cnt == 1).mean(1); single = F.mean(1)
            floor = binom.ppf(0.99, 6000, alpha_fw) / (6000 * alpha_fw)
            out[f"{d}|afw={alpha_fw:g}|r={r}"] = dict(alpha_k=alpha_k, pool_mean=float(p1.mean() / alpha_fw), pool_p99=float(np.quantile(p1, .99) / alpha_fw),
                pool_max=float(p1.max() / alpha_fw), single_mean=float(single.mean() / alpha_k), single_p99=float(np.quantile(single, .99) / alpha_k), ideal=float(floor))
            v = out[f"{d}|afw={alpha_fw:g}|r={r}"]
            print(f"{d:8s} {alpha_fw:>6g} {r:>3d} | {v['pool_mean']:9.2f} {v['pool_p99']:8.2f} {v['pool_max']:8.1f} | {v['single_mean']:11.2f} {v['single_p99']:10.2f} | {floor:5.2f}")
json.dump(out, open("results/key_pool_prf_exact.json", "w"), indent=1)
