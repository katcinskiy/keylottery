"""
Power side: detection of OPT-1.3B watermarked text under each scoring rule, with and without
random token substitution, plus the null on unwatermarked OPT text under many keys.

  python -m kl.power --workers 32
Needs data/generated.npz (kl.generate) and the reference text of every domain (kl.data).
Writes results/power.json and results/null_opt0_h1_lefthash_K{K}.npz (+ .json).
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.stats import norm

from .common import DOMAINS, GAMMA, START, VOCAB_SIZE, Colourer, make_keys, tuple_codes
from .data import load_source
from . import nullstats as ns

RHOS = [0.0, 0.1, 0.2, 0.3]
TPR_ALPHAS = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
TOKEN_LO, TOKEN_HI = 4, 50265            # ordinary OPT tokens for random substitutions


def reference_df(domains):
    """max over domains of the document frequency of each tuple in the reference text (h = 1)."""
    ref = {f"{d}/ref": load_source(d)[1] for d in domains}
    ts = ns.TextSets(ref, 1)
    df = np.max(np.stack([ts.doc_freq(s) for s in ref]), axis=0)
    return ts.U, df


def lookup(codes, U, df):
    i = np.searchsorted(U, codes)
    i = np.clip(i, 0, len(U) - 1)
    return np.where(U[i] == codes, df[i], 0.0)


def substitute(window, rho, rng):
    """window: (n, 204) = 4 context + 200 generated tokens; replace round(rho*200) random generated tokens."""
    out = window.copy()
    L = out.shape[1] - START
    m = int(round(rho * L))
    if m == 0:
        return out
    for i in range(len(out)):
        pos = START + rng.choice(L, size=m, replace=False)
        out[i, pos] = rng.integers(TOKEN_LO, TOKEN_HI, size=m)
    return out


def detect_rows(window, key, U_ref, df_ref, eps, backend="lefthash"):
    """Per-text green counts G and scored counts T for every rule. window: (n, 204)."""
    codes = tuple_codes(window, 1)                              # (n, 200)
    U = np.unique(codes)
    green = Colourer(U, 1, backend=backend).green(int(key))
    df = lookup(U, U_ref, df_ref)
    idx = np.searchsorted(U, codes)
    G = {"all": green[idx].sum(1).astype(np.int64)}
    T = {"all": np.full(len(codes), codes.shape[1], dtype=np.int64)}
    Gu, Tu = np.zeros(len(codes), np.int64), np.zeros(len(codes), np.int64)
    Gm = {e: np.zeros(len(codes), np.int64) for e in eps}
    Tm = {e: np.zeros(len(codes), np.int64) for e in eps}
    for r in range(len(codes)):
        _, first = np.unique(codes[r], return_index=True)
        j = idx[r, first]
        Gu[r], Tu[r] = green[j].sum(), len(j)
        for e in eps:
            kept = j[df[j] <= e]
            Gm[e][r], Tm[e][r] = green[kept].sum(), len(kept)
    G["uniq"], T["uniq"] = Gu, Tu
    for e in eps:
        G[f"mask{e:g}"], T[f"mask{e:g}"] = Gm[e], Tm[e]
    return G, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--null_keys", type=int, default=1000)
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--backend", default="lefthash", choices=["lefthash", "prf"])
    ap.add_argument("--gen", default="data/generated.npz")
    ap.add_argument("--out", default="results/power.json")
    args = ap.parse_args()
    ns._single_thread()
    t0 = time.time()
    d = np.load(args.gen)
    ids, delta, gkey, P = d["ids"].astype(np.int64), d["delta"], d["key"], int(d["prompt_len"])
    window = ids[:, P - START:P + 200]
    assert window.shape[1] == START + 200
    U_ref, df_ref = reference_df(args.domains)
    print(f"reference tuples {len(U_ref):,}; generated rows {len(ids)}; {time.time() - t0:.0f}s", flush=True)
    eps = ns.EPS
    crit = ns.crit_table(alphas=TPR_ALPHAS)
    zcut = norm.isf(TPR_ALPHAS)
    rows = []
    for dl in sorted(set(delta.tolist())):
        for k in sorted(set(gkey[delta == dl].tolist())):
            sel = (delta == dl) & (gkey == k)
            for rho in RHOS:
                rng = np.random.default_rng(int(1000 * rho) + 7)
                win = substitute(window[sel], rho, rng)
                G, T = detect_rows(win, k, U_ref, df_ref, eps, backend=args.backend)
                for r in G:
                    z = ns.z_of(G[r], T[r])
                    Ti = np.minimum(T[r], ns.TMAX)
                    for a, al in enumerate(TPR_ALPHAS):
                        rows.append(dict(delta=float(dl), key=int(k), rho=rho, rule=r, alpha=float(al),
                                         tpr_bin=float(np.mean((G[r] >= crit[Ti, a]) & (T[r] > 0))),
                                         tpr_z=float(np.mean(z > zcut[a])), n=int(sel.sum()),
                                         mean_T=float(T[r].mean()), mean_z=float(z.mean())))
            print(f"delta={dl} key={k}: done ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(rows=rows, rhos=RHOS, alphas=TPR_ALPHAS.tolist(), eps=eps,
                       seconds=round(time.time() - t0, 1)), f)
    print(f"power done ({time.time() - t0:.0f}s)", flush=True)

    # null on unwatermarked OPT text (delta = 0) under many keys
    X0 = window[delta == 0]
    if len(X0) and args.null_keys > 0:
        ts = ns.TextSets({"opt0/eval": X0}, 1)
        plan = ns.build_plan(ts, ["opt0/eval"], [], df_ref=lookup(ts.U, U_ref, df_ref), backend=args.backend)
        keys = make_keys(args.null_keys)
        res = ns.run_keys(ts, plan, keys, backend=args.backend, workers=args.workers, z_keys=keys[:10],
                          keep_z_for=["opt0/eval"])
        A, zs, meta = ns.stack(res, keys, plan)
        meta.update(backend=args.backend, seconds=round(time.time() - t0, 1))
        ns.save(f"results/null_opt0_h1_{args.backend}_K{len(keys)}.npz", A, zs, meta)
    print(f"all done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
