"""
Null experiment: false positives on human text under MANY fixed keys.

For every key we colour every distinct (context, token) tuple once, then get green counts of all
texts with sparse matrix-vector products. Scoring rules (gamma = 0.25, h given):
  all      every scored position counts (KGW z-test as in the paper)
  uniq     each distinct tuple of a text counts once (KGW reliability paper / Three Bricks)
  mask<e>  like uniq, but tuples whose document frequency in the REFERENCE text exceeds e are
           not scored at all (our fix)
For every rule we record, per key and text set: sum z, sum z^2, flags of the z-test at several
thresholds, flags of the exact binomial test at several levels alpha, a semi-analytic prediction
of the per-key false-positive rate (also at levels too small to count), and a per-key calibrated
baseline (threshold standardised on the provider's reference text of a domain).

Theory checked here (colours = i.i.d. Bernoulli(gamma) over keys):
  zbar_k = mean_x z_x(k) has  E_k[zbar] = 0  and  Var_k(zbar) = sum_u a_u^2,
  a_u = (1/n) sum_x A[x,u] / sqrt(T_x),  A = counts (all) or indicators (uniq / mask).
If every scored tuple has document frequency <= e and all T_x are equal, Var_k(zbar) <= e.
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.stats import binom, norm

from .common import GAMMA, START, VOCAB_SIZE, Colourer, split_code, tuple_codes

Z_THRESH = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
ALPHAS = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
EPS = [0.3, 0.1, 0.03, 0.01, 0.003]
CAL_ALPHAS = np.array([1e-2, 1e-3])
TMAX = 512


def crit_table(tmax=TMAX, gamma=GAMMA, alphas=ALPHAS):
    """c[T, a] = smallest m with P(Binomial(T, gamma) >= m) <= alpha[a]  (exact one-sided test)."""
    c = np.zeros((tmax + 1, len(alphas)), dtype=np.int64)
    for T in range(tmax + 1):
        tail = binom.sf(np.arange(T + 2) - 1, T, gamma)          # tail[m] = P(X >= m), m = 0..T+1
        for a, al in enumerate(alphas):
            c[T, a] = int(np.argmax(tail <= al))                   # first m with tail <= alpha
    return c


def tail_table(tmax=TMAX, gamma=GAMMA):
    """tail[n, m + tmax + 1] = P(Binomial(n, gamma) >= m) for m in [-tmax-1, tmax+1]."""
    off = tmax + 1
    tab = np.zeros((tmax + 1, 2 * tmax + 3))
    for n in range(tmax + 1):
        m = np.arange(-off, tmax + 2)
        tab[n] = np.where(m <= 0, 1.0, binom.sf(m - 1, n, gamma))
    return tab, off


class TextSets:
    """Distinct tuples of several text sets and their incidence matrices."""

    def __init__(self, sets, h, start=START):
        self.h, self.names = h, list(sets)
        codes = {s: tuple_codes(x, h, start) for s, x in sets.items()}
        self.U = np.unique(np.concatenate([c.ravel() for c in codes.values()]))
        self.C, self.B, self.n = {}, {}, {}
        for s, c in codes.items():
            n, L = c.shape
            cols = np.searchsorted(self.U, c.ravel())
            rows = np.repeat(np.arange(n), L)
            C = sp.csr_matrix((np.ones(n * L, dtype=np.float32), (rows, cols)), shape=(n, len(self.U)))
            C.sum_duplicates()
            B = C.copy()
            B.data[:] = 1.0
            self.C[s], self.B[s], self.n[s] = C, B, n

    def doc_freq(self, s):
        return np.asarray(self.B[s].sum(axis=0)).ravel() / self.n[s]


def predicted_var(A, T, groups=None, vocab_size=VOCAB_SIZE):
    """Var over random colours of the set mean of z = sum_u A[x,u](g_u - gamma)/sqrt(T_x gamma(1-gamma)).
    groups=None: i.i.d. Bernoulli(gamma) colours, Var = sum_u a_u^2 (the formula fixed before the runs).
    groups = context id of every tuple (lefthash): tuples that share a context are coloured by one permutation,
    so exactly gamma*V of the V tokens are green (sampling without replacement) and
    Cov(g_u, g_v) = -gamma(1-gamma)/(V-1) within a context. Then
    Var = (V sum_u a_u^2 - sum_c S_c^2) / (V - 1),  S_c = sum_{u in context c} a_u   (exact)."""
    w = np.where(T > 0, 1.0 / np.sqrt(np.maximum(T, 1)), 0.0)
    a = np.asarray(A.T @ w).ravel() / A.shape[0]
    v = float(a @ a)
    if groups is None:
        return v
    S = np.bincount(groups, weights=a)
    return float((vocab_size * v - S @ S) / (vocab_size - 1))


def build_plan(ts, eval_sets, ref_sets, eps=EPS, cal_map=None, df_ref=None, backend=None):
    """Everything that does not depend on the key. df_ref (aligned with ts.U) overrides the reference
    document frequencies computed from ref_sets (used when the reference text is not in ts).
    backend only selects the exact variance formula (fixed-size green lists for lefthash)."""
    plan = dict(h=ts.h, sets=ts.names, eval_sets=eval_sets, ref_sets=ref_sets, eps=list(eps), backend=backend)
    if df_ref is None:
        df_ref = np.max(np.stack([ts.doc_freq(s) for s in ref_sets]), axis=0) if ref_sets else np.zeros(len(ts.U))
    keep = {f"mask{e:g}": (df_ref <= e).astype(np.float32) for e in eps}
    plan["rules"] = ["all", "uniq"] + list(keep)
    plan["keep"] = keep
    plan["mats"], plan["T"] = {}, {}
    for s in ts.names:
        plan["mats"][(s, "all")] = ts.C[s]
        plan["T"][(s, "all")] = np.asarray(ts.C[s].sum(axis=1)).ravel()
        plan["mats"][(s, "uniq")] = ts.B[s]
        plan["T"][(s, "uniq")] = np.asarray(ts.B[s].sum(axis=1)).ravel()
        for r, kp in keep.items():
            plan["T"][(s, r)] = np.asarray(ts.B[s] @ kp).ravel()
    # semi-analytic predictor: tuples shared by >= 2 texts of an eval set are treated exactly,
    # the rest as fresh coins (they belong to one text only)
    plan["shared"] = {s: (np.asarray(ts.B[s].sum(axis=0)).ravel() >= 2).astype(np.float32) for s in eval_sets}
    plan["T_shared"] = {}
    for s in eval_sets:
        f = plan["shared"][s]
        plan["T_shared"][(s, "uniq")] = np.asarray(ts.B[s] @ f).ravel()
        for r, kp in keep.items():
            plan["T_shared"][(s, r)] = np.asarray(ts.B[s] @ (f * kp)).ravel()
    # theory
    groups = None
    if backend == "lefthash":                      # one permutation per context (h = 0: one for all tuples)
        groups = split_code(ts.U, ts.h)[0] if ts.h == 1 else np.zeros(len(ts.U), dtype=np.int64)
    plan["var_pred"], plan["var_pred_exact"], plan["ez2_pred"] = {}, {}, {}
    for s in ts.names:
        for r in plan["rules"]:
            A = plan["mats"][(s, "all")] if r == "all" else ts.B[s]
            if r.startswith("mask"):
                A = ts.B[s] @ sp.diags(keep[r])
            T = plan["T"][(s, r)]
            plan["var_pred"][(s, r)] = predicted_var(A, T)
            plan["var_pred_exact"][(s, r)] = predicted_var(A, T, groups)
            A2 = A.multiply(A)
            plan["ez2_pred"][(s, r)] = float(np.mean(np.asarray(A2.sum(axis=1)).ravel() / np.maximum(T, 1)))
    plan["crit"] = crit_table()
    plan["tail"], plan["tail_off"] = tail_table()
    plan["cal_map"] = cal_map or {}
    plan["df_ref"] = df_ref
    return plan


def z_of(G, T, gamma=GAMMA):
    return np.where(T > 0, (G - gamma * T) / np.sqrt(np.maximum(T, 1) * gamma * (1 - gamma)), 0.0)


def key_stats(green, plan, ts, gamma=GAMMA, keep_z_for=()):
    """All per-key statistics from the colour vector `green` (float32 0/1 over ts.U)."""
    out = {}
    crit, tail, off = plan["crit"], plan["tail"], plan["tail_off"]
    zcache = {}
    gk = {"all": green, "uniq": green}
    gk.update({r: green * kp for r, kp in plan["keep"].items()})
    for s in ts.names:
        for r in plan["rules"]:
            A = plan["mats"][(s, "all")] if r == "all" else ts.B[s]
            G = np.asarray(A @ gk[r]).ravel()
            T = plan["T"][(s, r)]
            z = z_of(G, T, gamma)
            zcache[(s, r)] = z
            Ti = np.minimum(T, TMAX).astype(np.int64)
            out[("zsum", s, r)] = z.sum()
            out[("z2sum", s, r)] = (z * z).sum()
            out[("zflags", s, r)] = (z[:, None] > Z_THRESH[None, :]).sum(0)
            out[("bflags", s, r)] = ((G[:, None] >= crit[Ti]) & (T[:, None] > 0)).sum(0)
            if s in plan["eval_sets"]:
                out[("zflags_head", s, r)] = (z[:2000, None] > Z_THRESH[None, :]).sum(0)
                if r != "all":                                         # semi-analytic prediction
                    f = plan["shared"][s] * (plan["keep"][r] if r.startswith("mask") else 1.0)
                    SF = np.asarray(ts.B[s] @ (green * f)).ravel()
                    TR = (T - plan["T_shared"][(s, r)]).astype(np.int64)
                    m = crit[Ti] - SF[:, None].astype(np.int64)        # greens still needed among the rest
                    m = np.clip(m, -off, TMAX + 1)
                    p = tail[np.clip(TR, 0, TMAX)[:, None], m + off]
                    p = np.where(T[:, None] > 0, p, 0.0)
                    out[("pred", s, r)] = p.mean(0)
            if s in keep_z_for and r in ("all", "uniq"):
                out[("z", s, r)] = z.astype(np.float32)
    # per-key calibration baseline on the uniq statistic: z' = (z - mu) / sd with mu, sd from a reference set
    for s_eval, cal in plan["cal_map"].items():
        for tag, s_ref in cal.items():
            zr = zcache[(s_ref, "uniq")]
            mu, sd = zr.mean(), zr.std(ddof=1)
            zc = (zcache[(s_eval, "uniq")] - mu) / sd
            out[("calflags", s_eval, tag)] = (zc[:, None] > norm.isf(CAL_ALPHAS)[None, :]).sum(0)
            out[("calzsum", s_eval, tag)] = zc.sum()
    return out


# ------------------------------------------------------------------ parallel driver

_G = {}


def _single_thread():
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _init(backend):
    _single_thread()
    if "col" not in _G:                     # normally built once in the parent and shared by fork
        _G["col"] = Colourer(_G["ts"].U, _G["ts"].h, backend=backend)


def _work(job):
    keys, gamma = job
    res = []
    for k in keys:
        t0 = time.time()
        green = _G["col"].green(int(k), gamma).astype(np.float32)
        t1 = time.time()
        st = key_stats(green, _G["plan"], _G["ts"], gamma, keep_z_for=_G.get("keep_z_for", ()) if k in _G.get("z_keys", ()) else ())
        st[("time_colour",)] = t1 - t0
        st[("time_stats",)] = time.time() - t1
        res.append((int(k), st))
    return res


def run_keys(ts, plan, keys, backend="lefthash", workers=1, gamma=GAMMA, chunk=4, z_keys=(), keep_z_for=()):
    """Statistics for every key. Returns {key: stats-dict}."""
    import multiprocessing as mp
    _single_thread()                        # no intra-op thread pools before fork; one process per core
    _G.clear()
    _G.update(ts=ts, plan=plan, z_keys=set(int(k) for k in z_keys), keep_z_for=tuple(keep_z_for))
    _G["col"] = Colourer(ts.U, ts.h, backend=backend)
    jobs = [(keys[i:i + chunk], gamma) for i in range(0, len(keys), chunk)]
    out, t0 = {}, time.time()
    if workers <= 1:
        _init(backend)
        it = map(_work, jobs)
        for r in it:
            out.update(dict(r))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init, initargs=(backend,)) as pool:
            for i, r in enumerate(pool.imap_unordered(_work, jobs)):
                out.update(dict(r))
                if (i + 1) % max(1, len(jobs) // 20) == 0:
                    print(f"  {len(out)}/{len(keys)} keys, {time.time() - t0:.0f}s", flush=True)
    return out


def stack(results, keys, plan):
    """Turn {key: stats} into arrays indexed [key, set, rule, ...]."""
    sets, rules, ev = plan["sets"], plan["rules"], plan["eval_sets"]
    K, S, R, E = len(keys), len(sets), len(rules), len(ev)
    A = dict(zsum=np.zeros((K, S, R)), z2sum=np.zeros((K, S, R)),
             zflags=np.zeros((K, S, R, len(Z_THRESH)), np.int64),
             bflags=np.zeros((K, S, R, len(ALPHAS)), np.int64),
             zflags_head=np.zeros((K, E, R, len(Z_THRESH)), np.int64),
             pred=np.full((K, E, R, len(ALPHAS)), np.nan),
             time_colour=np.zeros(K), time_stats=np.zeros(K))
    cal_tags = sorted({t for c in plan["cal_map"].values() for t in c})
    A["calflags"] = np.zeros((K, E, max(1, len(cal_tags)), len(CAL_ALPHAS)), np.int64)
    A["calzsum"] = np.zeros((K, E, max(1, len(cal_tags))))
    zs = {}
    for i, k in enumerate(keys):
        st = results[int(k)]
        A["time_colour"][i], A["time_stats"][i] = st[("time_colour",)], st[("time_stats",)]
        for si, s in enumerate(sets):
            for ri, r in enumerate(rules):
                A["zsum"][i, si, ri] = st[("zsum", s, r)]
                A["z2sum"][i, si, ri] = st[("z2sum", s, r)]
                A["zflags"][i, si, ri] = st[("zflags", s, r)]
                A["bflags"][i, si, ri] = st[("bflags", s, r)]
                if ("z", s, r) in st:
                    zs[f"z|{k}|{s}|{r}"] = st[("z", s, r)]
        for ei, s in enumerate(ev):
            for ri, r in enumerate(rules):
                A["zflags_head"][i, ei, ri] = st[("zflags_head", s, r)]
                if ("pred", s, r) in st:
                    A["pred"][i, ei, ri] = st[("pred", s, r)]
            for ti, t in enumerate(cal_tags):
                if ("calflags", s, t) in st:
                    A["calflags"][i, ei, ti] = st[("calflags", s, t)]
                    A["calzsum"][i, ei, ti] = st[("calzsum", s, t)]
    meta = dict(keys=[int(k) for k in keys], sets=sets, rules=rules, eval_sets=ev, cal_tags=cal_tags,
                z_thresh=Z_THRESH.tolist(), alphas=ALPHAS.tolist(), cal_alphas=CAL_ALPHAS.tolist(),
                eps=plan["eps"], h=plan["h"], n={s: int(len(plan["T"][(s, "all")])) for s in sets},
                var_pred={f"{s}|{r}": v for (s, r), v in plan["var_pred"].items()},
                var_pred_exact={f"{s}|{r}": v for (s, r), v in plan["var_pred_exact"].items()},
                ez2_pred={f"{s}|{r}": v for (s, r), v in plan["ez2_pred"].items()},
                mean_T={f"{s}|{r}": float(np.mean(t)) for (s, r), t in plan["T"].items()},
                masked_tuple_share={f"{s}|{r}": float(1 - np.mean(plan['T'][(s, r)]) / np.mean(plan['T'][(s, 'uniq')]))
                                    for s in sets for r in rules if r.startswith('mask')})
    return A, zs, meta


def save(path, A, zs, meta):
    np.savez_compressed(path, **A)
    if zs:
        np.savez_compressed(path.replace(".npz", "_z.npz"), **zs)
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
