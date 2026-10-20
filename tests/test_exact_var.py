"""
Exact Var_k(zbar) for fixed-size green lists (lefthash) vs the i.i.d. formula, on synthetic corpora.
For h = 0 every tuple is coloured by one permutation per key, so the i.i.d. formula overstates the variance
of sparse rules (masks); the exact formula (V sum a^2 - sum_c S_c^2)/(V-1) should match.
  python -m tests.test_exact_var   (needs torch; run in a Slurm job)
"""
import time

import numpy as np
from scipy.stats import chi2

from kl.common import make_keys
from kl import nullstats as ns
from tests.test_nullstats import synthetic_domain


def main():
    rng = np.random.default_rng(3)
    specs = {"plain": (3000, 1.05, 0.35), "sticky": (800, 1.1, 0.8)}
    sets = {}
    for d, (nt, a, st) in specs.items():
        x = synthetic_domain(1600, rng, nt, a, st)
        sets[f"{d}/eval"], sets[f"{d}/ref"] = x[:1000], x[1000:]
    K = 400
    keys = make_keys(K)
    lo, hi = chi2.ppf([0.0005, 0.9995], K - 1) / (K - 1)
    bad = []
    for h, backend in [(0, "lefthash"), (1, "lefthash"), (0, "prf")]:
        t0 = time.time()
        ts = ns.TextSets(sets, h)
        plan = ns.build_plan(ts, [f"{d}/eval" for d in specs], [f"{d}/ref" for d in specs],
                             eps=[0.1, 0.03, 0.01], backend=backend)
        res = ns.run_keys(ts, plan, keys, backend=backend, workers=8)
        A, _, meta = ns.stack(res, keys, plan)
        for s in meta["eval_sets"]:
            si = meta["sets"].index(s)
            for ri, r in enumerate(meta["rules"]):
                vm = (A["zsum"][:, si, ri] / meta["n"][s]).var(ddof=1)
                vi, ve = meta["var_pred"][f"{s}|{r}"], meta["var_pred_exact"][f"{s}|{r}"]
                if vi <= 0:
                    continue
                ok = lo < vm / ve < hi
                print(f"h={h} {backend:8s} {s:12s} {r:9s} iid {vi:.5f} exact {ve:.5f} (exact/iid {ve / vi:.3f}) "
                      f"meas/iid {vm / vi:.3f} meas/exact {vm / ve:.3f} {'ok' if ok else 'OUT'}", flush=True)
                if not ok:
                    bad.append((h, backend, s, r))
        print(f"  h={h} {backend}: {time.time() - t0:.0f}s", flush=True)
    print(f"99.9% band {lo:.3f}-{hi:.3f}; outside (exact): {bad}")


if __name__ == "__main__":
    main()
