"""
Window-size sweep of the key lottery with the idealised hash (prf).  EXPLORATORY / POST HOC, not pre-registered.

Question: how does the per-key FPR spread under Three Bricks scoring depend on the KGW window h?
For each h in {1, 2, 3, 4, 8}:
  * tuple u = (previous h tokens, token); each distinct tuple is scored once per text (rule "uniq"),
    T_x = number of distinct tuples in text x;
  * scored span: positions 4..203 of the 204-token window (200 tokens, as in the main runs) for h <= 4;
    for h = 8 positions 8..203 (the first 8 tokens are context only, so T <= 196);
  * colour of u under key k: the prf backend of kl/common.py (splitmix64 of key, context, token;
    green <=> hash % V < int(gamma V)).  For h = 1 the context is the previous token id itself, so the
    colours are bit-identical to Colourer(backend="prf"); for h >= 2 the h context tokens are first folded
    into one 64-bit context id with a splitmix64 chain (collisions ~ n^2 / 2^65, negligible);
  * exact one-sided binomial test: flag iff G_x >= c_alpha(T_x) (kl.nullstats.crit_table).
Keys: make_keys(1001, seed=2027)[1:] (the 1000 fresh keys of the screening runs).  Texts: E2 (6000 unseen
texts per domain, primary) and E1 (the eval sets, replication).

Sanity check (--check): h = 1, E1, make_keys(300) (seed 2026) must reproduce results/null_h1_prf_K300.npz
(rule uniq: bflags at all five alphas and zsum) exactly; and h = 1, E2, fresh keys must match the per-text
z of results/key_pool_prf_z.npz (stored as float16).

Outputs: results/window_sweep_prf.json (summary), results/window_sweep_prf.npz (per-key flags and zbar).
Run from the project root:
  uv run --quiet --no-project --with numpy,scipy python explore/window_sweep.py [--check] [--hs 1,2,3,4,8]
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import binom, chi2

sys.path.insert(0, os.getcwd())
from kl.common import VOCAB_SIZE, START, WINDOW, GAMMA, DOMAINS, make_keys
from kl.nullstats import crit_table, ALPHAS

U64 = np.uint64
C1, C2, C3 = U64(0x9E3779B97F4A7C15), U64(0xBF58476D1CE4E5B9), U64(0x94D049BB133111EB)
TOKMUL = U64(0x2545F4914F6CDD1D)
S30, S27, S31 = U64(30), U64(27), U64(31)
GREEN_CUT = int(GAMMA * VOCAB_SIZE)
REPORT_ALPHAS = [1e-2, 1e-3]
AI = [int(np.argmin(np.abs(ALPHAS - a))) for a in REPORT_ALPHAS]
CRIT = crit_table()                                   # [T, alpha] over ALPHAS = 1e-2 .. 1e-6


def splitmix64(x):
    """Same function as kl.common._splitmix64 (uint64 arithmetic wraps, so the masks there are no-ops)."""
    z = x + C1
    z ^= z >> S30
    z *= C2
    z ^= z >> S27
    z *= C3
    z ^= z >> S31
    return z


def span(h):
    s = max(START, h)
    return s, WINDOW - s


def context_ids(X, h):
    """64-bit context id of every scored position, (n, L) uint64.  h = 1: the previous token id (as in prf)."""
    s, L = span(h)
    X = X.astype(np.int64)
    if h == 1:
        return X[:, s - 1:s - 1 + L].astype(U64)
    with np.errstate(over="ignore"):
        c = X[:, s - h:s - h + L].astype(U64) + U64(1)
        for j in range(1, h):
            c = splitmix64(c * U64(VOCAB_SIZE) + X[:, s - h + j:s - h + j + L].astype(U64))
    return c


class Corpus:
    """Distinct tuples of one text set at window h, the (text x tuple) incidence matrix and theory terms."""

    def __init__(self, X, h):
        self.X, self.h = X, h
        s, L = span(h)
        n = len(X)
        tok = X[:, s:s + L].astype(np.int64)
        ctx = context_ids(X, h)
        with np.errstate(over="ignore"):
            if h == 1:
                tid = ctx.astype(np.int64) * VOCAB_SIZE + tok          # exact code (as kl.common.tuple_codes)
            else:
                tid = splitmix64(splitmix64(ctx + U64(7)) ^ (tok.astype(U64) * TOKMUL)).view(np.int64)
        order = np.argsort(tid, axis=1, kind="stable")
        ts = np.take_along_axis(tid, order, 1)
        first = np.ones_like(ts, dtype=bool)
        first[:, 1:] = ts[:, 1:] != ts[:, :-1]                         # first occurrence of each tuple in its text
        rows = np.repeat(np.arange(n), L).reshape(n, L)[first]
        pos = order[first]                                               # earliest position (stable sort)
        U, idx, inv = np.unique(ts[first], return_index=True, return_inverse=True)
        self.n, self.nU, self.L, self.start = n, len(U), L, s
        self.T = np.bincount(rows, minlength=n).astype(np.int64)
        self.B = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, inv)), shape=(n, len(U)))
        self.rep_row, self.rep_pos = rows[idx], pos[idx]               # one occurrence of every distinct tuple
        r, p = self.rep_row, self.rep_pos
        with np.errstate(over="ignore"):
            self.P = splitmix64(ctx[r, p] + U64(7))                      # key-independent parts of the prf hash
            self.Q = tok[r, p].astype(U64) * TOKMUL
        self.df = np.bincount(inv, minlength=len(U))                    # number of texts containing the tuple
        self.a = np.asarray(self.B.T @ (1.0 / np.sqrt(self.T))).ravel() / n   # a_u = (1/n) sum_x B_xu / sqrt(T_x)

    def green(self, keys):
        """(nU, len(keys)) float32 0/1 colours.  Identical to kl.common.prf_ranks for h = 1."""
        out = np.empty((self.nU, len(keys)), np.float32)
        with np.errstate(over="ignore"):
            for j, k in enumerate(keys):
                b = splitmix64(splitmix64(U64(int(k)) ^ self.P) ^ self.Q)
                out[:, j] = (b % U64(VOCAB_SIZE)) < U64(GREEN_CUT)
        return out

    def score(self, keys, batch=25):
        """Per key: flags at every alpha in ALPHAS (K, 5) and the corpus-mean z (K,)."""
        Ti = np.minimum(self.T, CRIT.shape[0] - 1)
        crit = CRIT[Ti]                                                  # (n, 5)
        sd = np.sqrt(GAMMA * (1 - GAMMA) * self.T)
        flags = np.zeros((len(keys), len(ALPHAS)), np.int64)
        zbar = np.zeros(len(keys))
        for b0 in range(0, len(keys), batch):
            kb = keys[b0:b0 + batch]
            G = np.asarray(self.B @ self.green(kb))                      # (n, kb) green counts
            flags[b0:b0 + len(kb)] = (G[:, :, None] >= crit[:, None, :]).sum(0)
            zbar[b0:b0 + len(kb)] = ((G - GAMMA * self.T[:, None]) / sd[:, None]).mean(0)
        return flags, zbar


def dispersion(counts, n):
    """Same statistic as kl.analyze.dispersion: index of dispersion vs identical binomial keys, chi2 p."""
    p = counts.sum() / (len(counts) * n)
    if p == 0:
        return float("nan"), float("nan")
    D = ((counts - counts.mean()) ** 2).sum() / (n * p * (1 - p))
    return float(D / (len(counts) - 1)), float(chi2.sf(D, len(counts) - 1))


def realised_level(T, ai):
    """alpha_D = mean over texts of the exact size P(Bin(T_x, gamma) >= c_alpha(T_x)) of the discrete test."""
    Ti = np.minimum(T, CRIT.shape[0] - 1)
    return float(np.mean(binom.sf(CRIT[Ti, ai] - 1, T, GAMMA)))


# ------------------------------------------------------------------ OPT decoding (offline, no tokenizer library)

def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    m = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + m)
            m += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_decoder():
    import glob
    paths = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-1.3b/snapshots/*/vocab.json"))
    if not paths:
        return None
    vocab = json.load(open(paths[0], encoding="utf-8"))
    inv = {i: t for t, i in vocab.items()}
    bd = _byte_decoder()

    def decode(ids):
        out = bytearray()
        for i in ids:
            t = inv.get(int(i), f"<{int(i)}>")
            if t.startswith("<") and t.endswith(">"):
                out += t.encode()
            else:
                out += bytes(bd.get(ch, ord("?")) for ch in t)
        return out.decode("utf-8", errors="replace")
    return decode


# ------------------------------------------------------------------ main

def summarise(corp, flags, zbar, decode, top=5):
    n = corp.n
    var_pred = float((corp.a ** 2).sum())
    shared = corp.df >= 2
    occ_shared = float((corp.df[shared]).sum() / corp.T.sum())          # share of scored (text, tuple) slots
    res = dict(n=n, start=corp.start, mean_T=float(corp.T.mean()), min_T=int(corp.T.min()),
               distinct_tuples=int(corp.nU),
               share_tuples_df2=float(shared.mean()), share_slots_df2=occ_shared,
               var_pred=var_pred, n_var_pred=var_pred * n, var_meas=float(zbar.var(ddof=1)),
               var_ratio=float(zbar.var(ddof=1) / var_pred), share_var_df2=float((corp.a[shared] ** 2).sum() / var_pred),
               mean_zbar=float(zbar.mean()), alphas={})
    K = len(zbar)
    rng = np.random.default_rng(0)
    for a, ai in zip(REPORT_ALPHAS, AI):
        c = flags[:, ai]
        f = c / n
        aD = realised_level(corp.T, ai)
        q99 = binom.ppf(0.99, n, aD)
        sims = rng.binomial(n, aD, size=(2000, K))                      # 1000 ideal keys, 2000 replicates
        band = np.quantile(np.quantile(sims, 0.99, axis=1) / (n * a), [0.025, 0.975])
        D, Dp = dispersion(c, n)
        over = int((c > q99).sum())
        res["alphas"][str(a)] = dict(
            pooled_over_alpha=float(f.mean() / a), alpha_D_over_alpha=aD / a,
            median_over_alpha=float(np.median(f) / a),
            p99_over_alpha=float(np.quantile(f, 0.99) / a), max_over_alpha=float(f.max() / a),
            dispersion=D, dispersion_p=Dp,
            ideal_p99_over_alpha=float(q99 / (n * a)), ideal_p99_band95=[float(band[0]), float(band[1])],
            keys_above_ideal_q99=over, keys_above_ideal_q99_p=float(binom.sf(over - 1, K, 0.01)))
    tops = []
    for u in np.argsort(-corp.a ** 2)[:top]:
        r, p = corp.rep_row[u], corp.rep_pos[u] + corp.start
        ids = corp.X[r, p - corp.h:p + 1].tolist()
        tops.append(dict(ids=ids, df=int(corp.df[u]), df_frac=float(corp.df[u] / n),
                         share_var=float(corp.a[u] ** 2 / var_pred),
                         text=None if decode is None else [decode(ids[:-1]), decode(ids[-1:])],
                         joined=None if decode is None else decode(ids)))     # split byte-level pieces show as U+FFFD
    res["top_tuples"] = tops
    return res


def load_sets(which):
    out = {}
    for d in DOMAINS:
        if which == "e2":
            out[d] = np.load(f"data/{d}_e2.npz")["ids"]
        else:
            z = np.load(f"data/{d}.npz")
            out[d] = z["ids"][:int(z["n_eval"])]
    return out


def check():
    """Bit-exact reproduction of the existing prf run at h = 1 (E1, old keys) and of key_pool_prf_z (E2, fresh keys)."""
    ref = np.load("results/null_h1_prf_K300.npz")
    meta = json.load(open("results/null_h1_prf_K300.json"))
    keys = np.array(meta["keys"], dtype=np.int64)
    assert np.array_equal(keys, make_keys(300)), "K300 keys differ from make_keys(300)"
    ri = meta["rules"].index("uniq")
    ok = True
    out = {}
    for d, X in load_sets("e1").items():
        si = meta["sets"].index(f"{d}/eval")
        t0 = time.time()
        corp = Corpus(X, 1)
        flags, zbar = corp.score(keys)
        same_flags = np.array_equal(flags, ref["bflags"][:, si, ri, :])
        dz = float(np.max(np.abs(zbar * corp.n - ref["zsum"][:, si, ri])))
        out[d] = dict(flags_identical=bool(same_flags), max_abs_dzsum=dz)
        ok &= same_flags and dz < 1e-2
        print(f"check E1 h=1 {d}: flags identical={same_flags}, max|dzsum|={dz:.2e} ({time.time() - t0:.0f}s)", flush=True)
    zp = np.load("results/key_pool_prf_z.npz")
    fresh = make_keys(1001, seed=2027)[1:]
    for d, X in load_sets("e2").items():
        corp = Corpus(X, 1)
        _, zbar = corp.score(fresh)
        zr = zp[f"{d}|e2"].astype(np.float64).mean(1)
        dz = float(np.max(np.abs(zbar - zr)))
        out[d]["e2_fresh_max_abs_dzbar_vs_key_pool"] = dz
        ok &= dz < 1e-3
        print(f"check E2 h=1 {d}: max|dzbar| vs key_pool_prf_z = {dz:.2e} (float16 storage)", flush=True)
    out["ok"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", default="1,2,3,4,8")
    ap.add_argument("--texts", default="e2,e1")
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="results/window_sweep_prf")
    args = ap.parse_args()
    t_all = time.time()
    chk = check() if args.check else None
    if chk is not None and not chk["ok"]:
        print("SANITY CHECK FAILED", json.dumps(chk, indent=1))
        sys.exit(1)
    hs = [int(h) for h in args.hs.split(",")]
    keys = make_keys(args.K + 1, seed=2027)[1:]
    decode = load_decoder()
    res, arrays = {}, {}
    for which in args.texts.split(","):
        sets = load_sets(which)
        for h in hs:
            for d, X in sets.items():
                t0 = time.time()
                corp = Corpus(X, h)
                flags, zbar = corp.score(keys)
                s = summarise(corp, flags, zbar, decode)
                s["seconds"] = round(time.time() - t0, 1)
                res[f"{which}|h{h}|{d}"] = s
                arrays[f"{which}|h{h}|{d}|flags"] = flags.astype(np.int32)
                arrays[f"{which}|h{h}|{d}|zbar"] = zbar.astype(np.float32)
                a2, a3 = s["alphas"]["0.01"], s["alphas"]["0.001"]
                print(f"{which} h={h} {d:8s} T={s['mean_T']:.1f} nVar={s['n_var_pred']:.2f} meas/pred={s['var_ratio']:.2f} "
                      f"| 1e-2 p99={a2['p99_over_alpha']:.2f} (ideal {a2['ideal_p99_over_alpha']:.2f}) D={a2['dispersion']:.1f} "
                      f"| 1e-3 p99={a3['p99_over_alpha']:.2f} (ideal {a3['ideal_p99_over_alpha']:.2f}) D={a3['dispersion']:.1f} "
                      f"[{s['seconds']}s]", flush=True)
    meta = dict(note="EXPLORATORY / POST HOC window-size sweep, idealised hash (prf); not pre-registered",
                backend="prf", rule="uniq", test="exact binomial", gamma=GAMMA, vocab_size=VOCAB_SIZE,
                keys="make_keys(1001, seed=2027)[1:]", K=args.K, hs=hs, texts=args.texts.split(","),
                span="positions 4..203 (T<=200) for h<=4; positions h..203 for h>4 (first h tokens are context only)",
                context_hash="h=1: previous token id (bit-identical to kl.common prf); h>=2: splitmix64 chain of the h tokens",
                alphas=REPORT_ALPHAS, ideal_p99="binom.ppf(0.99, n, alpha_D)/(n alpha), alpha_D = realised level of the discrete test",
                ideal_p99_band95="2.5-97.5% of np.quantile(.99) over 1000 Bin(n, alpha_D) keys, 2000 replicates",
                decoded=decode is not None, sanity_check=chk, seconds=round(time.time() - t_all, 1))
    json.dump(dict(meta=meta, results=res), open(args.out + ".json", "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(args.out + ".npz", keys=keys, **arrays)
    print(f"wrote {args.out}.json/.npz in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
