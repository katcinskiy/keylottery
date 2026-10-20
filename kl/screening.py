"""
Key-screening follow-up (CLAUDE.md Section 9), written before the runs.
  python -m kl.screening --select     -> results/screen_gen_keys.json (+ _union.json for kl.generate --key_file)
  python -m kl.screening              -> report/screening.json (S1-S5 verdicts) and a printed summary
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import binom

from .common import DOMAINS

RUN = "results/screen_h1_lefthash_K1000"
Q_PRIMARY, Q_ALL = 0.2, (0.2, 0.5)
ALPHAS_S1 = (1e-2, 1e-3)
MASK_LOSS_PP = 0.087          # mask0.01 vs uniq TPR loss at delta=1, rho=0, alpha=1e-4 (Section-6 run)


def load(run=RUN):
    return dict(np.load(run + ".npz")), json.load(open(run + ".json"))


def zbar(A, m, s, rule):
    return A["zsum"][:, m["sets"].index(s), m["rules"].index(rule)] / m["n"][s]


def score(A, m, rule, doms=DOMAINS):
    """s(k) = max_d zbar_ref(k, d) / sd_k zbar_ref(., d)."""
    Z = np.stack([zbar(A, m, f"{d}/ref", rule) for d in doms], 1)
    return (Z / Z.std(0, ddof=1)).max(1)


def score_flags(A, m, doms=DOMAINS, al=1e-2):
    """S6: s_F(k) = max_d standardised exact-binomial flag count of rule uniq at alpha=1e-2 on d/ref."""
    ri, ai = m["rules"].index("uniq"), m["alphas"].index(al)
    C = np.stack([A["bflags"][:, m["sets"].index(f"{d}/ref"), ri, ai].astype(float) for d in doms], 1)
    return ((C - C.mean(0)) / C.std(0, ddof=1)).max(1)


def kept(s, q):
    return s <= np.quantile(s, q)


def floor(n, al):
    return float(binom.ppf(0.99, n, al) / (n * al))


def fpr_stats(A, m, s, rule, al, sel):
    n = m["n"][s]
    f = A["bflags"][sel, m["sets"].index(s), m["rules"].index(rule), m["alphas"].index(al)] / n
    return dict(p99_over_alpha=float(np.quantile(f, .99) / al), max_over_alpha=float(f.max() / al),
                pooled_over_alpha=float(f.mean() / al), n_keys=int(sel.sum()))


def select(A, m, out="results/screen_gen_keys.json"):
    keys = np.array(m["keys"])
    k = kept(score(A, m, "uniq"), Q_PRIMARY)
    rng = np.random.default_rng(7)
    screened = rng.choice(keys[k], 10, replace=False).tolist()      # screened draw first (pre-registered)
    random = rng.choice(keys, 10, replace=False).tolist()
    union = sorted(set(screened) | set(random))
    json.dump(dict(screened=screened, random=random, q=Q_PRIMARY, rng_seed=7), open(out, "w"), indent=1)
    json.dump(union, open(out.replace(".json", "_union.json"), "w"))
    print(f"screened {screened}\nrandom {random}\nunion {len(union)} keys -> {out}")


def s1_s5(A, m, rule, label, s=None):
    """p99/alpha over kept keys vs the ideal-key floor on E2 (and E1), for q in Q_ALL and all keys."""
    s = score(A, m, rule) if s is None else s
    cells, rows = [], []
    for d in DOMAINS:
        for part in ("e2", "eval"):
            st = f"{d}/{part}"
            n = m["n"][st]
            for al in ALPHAS_S1:
                F = floor(n, al)
                row = dict(domain=d, set=part, alpha=al, floor=F,
                           all=fpr_stats(A, m, st, rule, al, np.ones(len(s), bool)))
                for q in Q_ALL:
                    row[f"q{q:g}"] = fpr_stats(A, m, st, rule, al, kept(s, q))
                row["pass"] = row[f"q{Q_PRIMARY:g}"]["p99_over_alpha"] <= F
                rows.append(row)
                if part == "e2":
                    cells.append(row["pass"])
    npass = int(sum(cells))
    verdict = "supported" if npass == len(cells) else ("partial" if npass >= 8 else "refuted")
    return dict(rule=rule, verdict=verdict, cells_passed=npass, cells=len(cells), rows=rows, label=label)


def s2(A, m):
    c = {d: float(np.corrcoef(zbar(A, m, f"{d}/ref", "uniq"), zbar(A, m, f"{d}/e2", "uniq"))[0, 1]) for d in DOMAINS}
    c1 = {d: float(np.corrcoef(zbar(A, m, f"{d}/ref", "uniq"), zbar(A, m, f"{d}/eval", "uniq"))[0, 1]) for d in DOMAINS}
    npass = sum(v >= 0.8 for v in c.values())
    return dict(verdict="supported" if npass == 5 else ("partial" if npass == 4 else "refuted"),
                corr_ref_e2=c, corr_ref_e1=c1)


def s3(path="results/power_screen.json", keys_path="results/screen_gen_keys.json"):
    if not (os.path.exists(path) and os.path.exists(keys_path)):
        return dict(verdict="missing")
    P, G = json.load(open(path)), json.load(open(keys_path))
    rows = P["rows"]

    def tpr(key, rho, rule="uniq", al=1e-4, f="tpr_bin"):
        t = [x[f] for x in rows if x["key"] == key and x["rule"] == rule and abs(x["delta"] - 1) < 1e-9
             and abs(x["rho"] - rho) < 1e-9 and abs(x["alpha"] - al) / al < 1e-6]
        assert len(t) == 1, (key, rho, rule, len(t))
        return t[0]
    rng = np.random.default_rng(0)
    out = {}
    for rule in ("uniq", "mask0.01"):
        for rho in (0.0, 0.2):
            sc = np.array([tpr(k, rho, rule) for k in G["screened"]])
            ra = np.array([tpr(k, rho, rule) for k in G["random"]])
            boot = np.array([rng.choice(ra, len(ra)).mean() - rng.choice(sc, len(sc)).mean() for _ in range(10000)])
            out[f"{rule}|rho={rho:g}"] = dict(tpr_screened=float(sc.mean()), tpr_random=float(ra.mean()),
                                              delta_tpr=float(ra.mean() - sc.mean()),
                                              ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                                              per_key_screened=sc.tolist(), per_key_random=ra.tolist())
    main = out["uniq|rho=0"]
    if main["ci95"][1] < MASK_LOSS_PP:
        v = "supported"
    elif main["delta_tpr"] >= MASK_LOSS_PP:
        v = "refuted"
    else:
        v = "partial"
    return dict(verdict=v, threshold_pp=MASK_LOSS_PP, table=out)


def dispersion_index(c, n):
    """Same statistic as kl.analyze.dispersion: sum (c - cbar)^2 / (n p (1-p)) / (K - 1)."""
    p = c.sum() / (len(c) * n)
    return float(((c - c.mean()) ** 2).sum() / (n * p * (1 - p)) / (len(c) - 1)) if p > 0 else float("nan")


def s7(A, m, al=1e-2, limit=1.5):
    """Index of dispersion of per-key counts on E2 among kept keys (q=0.2) for each screening score, and all keys."""
    sel = {"S1 (mean z, uniq)": kept(score(A, m, "uniq"), Q_PRIMARY),
           "S6 (flags, uniq)": kept(score_flags(A, m), Q_PRIMARY),
           "all keys": np.ones(len(m["keys"]), bool)}
    res = {}
    for name, k in sel.items():
        res[name] = {d: dispersion_index(A["bflags"][k, m["sets"].index(f"{d}/e2"), m["rules"].index("uniq"),
                                                     m["alphas"].index(al)], m["n"][f"{d}/e2"]) for d in DOMAINS}
    k5 = kept(score(A, m, "mask0.01"), Q_PRIMARY)
    res["S5 (mask0.01 + screening)"] = {d: dispersion_index(A["bflags"][k5, m["sets"].index(f"{d}/e2"),
                                        m["rules"].index("mask0.01"), m["alphas"].index(al)], m["n"][f"{d}/e2"])
                                        for d in DOMAINS}
    ok = all(v <= limit for v in res["S1 (mean z, uniq)"].values())
    return dict(verdict="removes the lottery" if ok else "shifts keys to conservative ones", limit=limit,
                alpha=al, dispersion=res)


def s4(A, m):
    res = {}
    for held in DOMAINS:
        s = score(A, m, "uniq", [d for d in DOMAINS if d != held])
        k = kept(s, Q_PRIMARY)
        st = f"{held}/e2"
        res[held] = {f"{al:g}": dict(all=fpr_stats(A, m, st, "uniq", al, np.ones(len(s), bool))["p99_over_alpha"],
                                     kept=fpr_stats(A, m, st, "uniq", al, k)["p99_over_alpha"],
                                     floor=floor(m["n"][st], al)) for al in ALPHAS_S1}
    return dict(verdict="reported (not blind)", per_held_out_domain=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out", default="report/screening.json")
    args = ap.parse_args()
    A, m = load(args.run)
    if args.select:
        select(A, m)
        return
    S5 = s1_s5(A, m, "mask0.01", "S5")
    S5["verdict"] = "reported (no pass/fail)"           # pre-registration: S5 has no pass/fail
    out = dict(S1=s1_s5(A, m, "uniq", "S1"), S2=s2(A, m), S3=s3(), S4=s4(A, m), S5=S5,
               S6=s1_s5(A, m, "uniq", "S6 (flag score)", s=score_flags(A, m)), S7=s7(A, m),
               keys=dict(n=len(m["keys"]), seed=m.get("key_seed"), entropy_cost_bits=float(np.log2(1 / Q_PRIMARY))))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        print(k, out[k]["verdict"], {kk: out[k][kk] for kk in ("cells_passed", "corr_ref_e2") if kk in out[k]})
    for r in out["S1"]["rows"]:
        if r["set"] == "e2":
            print(f"  S1 {r['domain']:8s} a={r['alpha']:g} floor {r['floor']:.2f} | all {r['all']['p99_over_alpha']:.2f} "
                  f"| q0.5 {r['q0.5']['p99_over_alpha']:.2f} | q0.2 {r['q0.2']['p99_over_alpha']:.2f} "
                  f"{'ok' if r['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
