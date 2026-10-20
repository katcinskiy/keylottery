"""
Local test of the null-experiment machinery on synthetic corpora with the idealised PRF colours.
Checks (1) matrix-based z equals a direct per-text loop, (2) Var_k(zbar) matches the formula,
(3) E_k[zbar] = 0, (4) the semi-analytic per-key FPR predictor matches counted flags,
(5) masking frequent tuples shrinks Var_k(zbar) below the frequency cutoff.
Run:  python -m tests.test_nullstats
"""
import time

import numpy as np
from scipy.stats import chi2

from kl.common import GAMMA, START, VOCAB_SIZE, WINDOW, make_keys, prf_ranks, tuple_codes, split_code
from kl import nullstats as ns


def synthetic_domain(n, rng, n_types, zipf_a, stickiness, n_succ=8):
    """Markov text: each token has a few preferred successors (shared bigrams across documents)."""
    types = rng.choice(np.arange(4, VOCAB_SIZE - 10), size=n_types, replace=False)
    p = 1.0 / np.arange(1, n_types + 1) ** zipf_a
    p /= p.sum()
    succ = rng.integers(0, n_types, size=(n_types, n_succ))
    texts = np.empty((n, WINDOW), dtype=np.int64)
    for i in range(n):
        cur = rng.choice(n_types, p=p)
        for t in range(WINDOW):
            texts[i, t] = types[cur]
            cur = succ[cur, rng.integers(n_succ)] if rng.random() < stickiness else rng.choice(n_types, p=p)
    return texts


def direct_z(text, key, h, rule, keep_codes=None):
    codes = tuple_codes(text[None, :], h)[0]
    if rule != "all":
        _, first = np.unique(codes, return_index=True)
        codes = codes[np.sort(first)]
        if keep_codes is not None:
            codes = codes[np.isin(codes, keep_codes)]
    ctx, tok = split_code(codes, h)
    g = prf_ranks(key, ctx, tok) < int(GAMMA * VOCAB_SIZE)
    T = len(codes)
    return 0.0 if T == 0 else (g.sum() - GAMMA * T) / np.sqrt(T * GAMMA * (1 - GAMMA))


def main():
    rng = np.random.default_rng(0)
    specs = {"plain": (3000, 1.05, 0.35), "sticky": (800, 1.1, 0.8)}   # n_types, zipf, stickiness
    sets = {}
    for d, (nt, a, st) in specs.items():
        x = synthetic_domain(2300, rng, nt, a, st)
        sets[f"{d}/eval"], sets[f"{d}/ref"] = x[:1500], x[1500:]
    keys = make_keys(400)
    h = 1
    t0 = time.time()
    ts = ns.TextSets(sets, h)
    plan = ns.build_plan(ts, eval_sets=[f"{d}/eval" for d in specs], ref_sets=[f"{d}/ref" for d in specs],
                         eps=[0.1, 0.03, 0.01],
                         cal_map={f"{d}/eval": {"same": f"{d}/ref", "plain": "plain/ref"} for d in specs})
    print(f"distinct tuples {len(ts.U)}, plan {time.time() - t0:.1f}s")
    res = ns.run_keys(ts, plan, keys, backend="prf", workers=2, z_keys=keys[:3], keep_z_for=list(sets))
    A, zs, meta = ns.stack(res, keys, plan)
    print(f"keys done in {time.time() - t0:.1f}s")

    # (1) matrix-based z == direct loop, incl. a mask rule
    keep01 = ts.U[plan["keep"]["mask0.03"] > 0]
    for k in keys[:3]:
        for s in sets:
            for r in ("all", "uniq"):
                z = zs[f"z|{k}|{s}|{r}"]
                for i in (0, 7, 123):
                    zd = direct_z(sets[s][i], int(k), h, r)
                    assert abs(z[i] - zd) < 1e-4, (k, s, r, i, z[i], zd)
    # mask rule via direct loop vs zsum consistency on one key
    k = int(keys[0])
    si, ri = meta["sets"].index("sticky/eval"), meta["rules"].index("mask0.03")
    zd = np.array([direct_z(t, k, h, "mask", keep01) for t in sets["sticky/eval"]])
    assert abs(zd.sum() - A["zsum"][0, si, ri]) < 1e-2, (zd.sum(), A["zsum"][0, si, ri])
    print("(1) matrix z == direct z: OK")

    K = len(keys)
    lo, hi = chi2.ppf([0.0005, 0.9995], K - 1) / (K - 1)
    for s in [f"{d}/eval" for d in specs]:
        si = meta["sets"].index(s)
        for ri, r in enumerate(meta["rules"]):
            zbar = A["zsum"][:, si, ri] / meta["n"][s]
            vp = meta["var_pred"][f"{s}|{r}"]
            ratio = zbar.var(ddof=1) / vp
            tmean = zbar.mean() / np.sqrt(vp / K)
            print(f"  {s:12s} {r:9s} Var_pred={vp:.4f} ratio={ratio:.3f} (99.9% band {lo:.2f}-{hi:.2f}) mean/se={tmean:+.2f}")
            assert lo < ratio < hi, "variance formula failed"
            assert abs(tmean) < 3.6, "mean of zbar not zero"
    print("(2,3) Var_k(zbar) formula and zero mean: OK")

    for s in [f"{d}/eval" for d in specs]:
        ei, si = meta["eval_sets"].index(s), meta["sets"].index(s)
        for r in ["uniq", "mask0.03"]:
            ri = meta["rules"].index(r)
            for a in (0, 1):                       # alpha 1e-2, 1e-3
                meas = A["bflags"][:, si, ri, a]
                pred = A["pred"][:, ei, ri, a] * meta["n"][s]
                ratio = meas.sum() / pred.sum()
                # goodness of fit: Poisson deviance-ish standardised residual of the total
                zt = (meas.sum() - pred.sum()) / np.sqrt(pred.sum())
                corr = np.corrcoef(meas, pred)[0, 1]
                print(f"  {s:12s} {r:9s} alpha={meta['alphas'][a]:.0e} measured={meas.sum()} predicted={pred.sum():.1f} "
                      f"ratio={ratio:.3f} z={zt:+.2f} corr(per key)={corr:.2f} nominal={meta['alphas'][a] * meta['n'][s] * K:.0f}")
                assert abs(zt) < 4, "predictor does not match counts"
    print("(4) semi-analytic per-key FPR predictor: OK")

    for s in [f"{d}/eval" for d in specs]:
        v = [meta["var_pred"][f"{s}|{r}"] for r in meta["rules"]]
        print(f"  {s}: Var_pred by rule {dict(zip(meta['rules'], np.round(v, 4)))}")
        for r, e in zip(["mask0.1", "mask0.03", "mask0.01"], [0.1, 0.03, 0.01]):
            assert meta["var_pred"][f"{s}|{r}"] <= meta["var_pred"][f"{s}|uniq"] + 1e-12
    print("(5) masking shrinks Var_pred: OK")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
