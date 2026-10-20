"""
Mechanical evaluation of the Section-6 claims (CLAUDE.md) from results/ and report/ files.
Written before any real result was seen; where the claim text is vague, the reading used is fixed here:
  * "at alpha <= 1e-4" means each of 1e-4, 1e-5, 1e-6.
  * dispersion / "keys significantly above pooled" are judged on the exact binomial test at alpha = 1e-2 and 1e-3
    (the levels where per-key counts are large enough to say anything); both levels must pass.
  * ">> alpha" is reported as the ratio; we call it ">>" if the ratio is >= 10.
  python -m kl.claims --results results --report report
Writes report/claims.json and prints a table.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import binom, chi2, poisson

from .analyze import calibration_table, perkey_table, predicted_table, theory_table

DISP_ALPHAS = (1e-2, 1e-3)
SMALL_ALPHAS = (1e-4, 1e-5, 1e-6)
MAIN = "null_h1_lefthash_K1000"


def load(results, tag):
    p = os.path.join(results, tag + ".npz")
    if not os.path.exists(p):
        return None, None
    return dict(np.load(p)), json.load(open(p.replace(".npz", ".json")))


def close(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300)


def dom(s):
    return s.split("/")[0]


def pk_row(pk, s, r, test, level):
    for x in pk:
        if x["set"] == s and x["rule"] == r and x["test"] == test and close(x["level"], level):
            return x
    return None


def pr_row(pr, s, r, al):
    for x in pr:
        if x["set"] == s and x["rule"] == r and close(x["alpha"], al):
            return x
    return None


def c0(results, out):
    if not os.path.exists(os.path.join(results, "checks.json")):
        out["C0"] = dict(verdict="missing")
        return
    ck = json.load(open(os.path.join(results, "checks.json")))
    col = all(c["identical_green_lists"] == c["contexts"] for c in ck["colours"])
    det = ck["detector"]["max_abs_dz"] < 1e-6
    lab4 = ck.get("lab4_texts") or {}
    texts = bool(lab4) and all(isinstance(v, dict) and v["identical"] and v["compared"] >= 2000 for v in lab4.values()) \
        and len(lab4) == 5
    A, meta = load(results, "null_h1_lefthash_K10_lab4keys")
    reg = dict(available=A is not None)
    ok_reg = False
    if A is not None:
        ev, rules = meta["eval_sets"], meta["rules"]
        zi = meta["z_thresh"].index(4.0)
        want_all = dict(news=17, wiki_en=21, lists=182, code=1188, wiki_ru=74)
        want_uniq = dict(news=5, wiki_en=1, lists=0, code=3, wiki_ru=3)
        got_all = {dom(s): int(A["zflags_head"][:, e, rules.index("all"), zi].sum()) for e, s in enumerate(ev)}
        got_uniq = {dom(s): int(A["zflags_head"][:, e, rules.index("uniq"), zi].sum()) for e, s in enumerate(ev)}
        ec, er = ev.index("code/eval"), ev.index("wiki_ru/eval")
        per = dict(code_K3=int(A["zflags_head"][3, ec, rules.index("all"), zi]),
                   code_K6=int(A["zflags_head"][6, ec, rules.index("all"), zi]),
                   wiki_ru_K1=int(A["zflags_head"][1, er, rules.index("all"), zi]))
        want_per = dict(code_K3=672, code_K6=459, wiki_ru_K1=47)
        ok_reg = got_all == want_all and got_uniq == want_uniq and per == want_per
        reg.update(all=got_all, uniq=got_uniq, per_key=per, want_all=want_all, want_uniq=want_uniq, want_per=want_per)
    xc_p = os.path.join(results, "crosscheck.json")
    xc = json.load(open(xc_p)) if os.path.exists(xc_p) else None
    ok = col and det and texts and ok_reg
    out["C0"] = dict(verdict="supported" if ok else "refuted", colours=col, detector_max_abs_dz=ck["detector"]["max_abs_dz"],
                     lab4_texts=lab4, regression=reg, regression_exact=ok_reg,
                     crosscheck=None if xc is None else dict(passed=xc["passed"], rows=xc["rows"]))


def c1(A, meta, pk, out):
    K, rows, ok_all, ok_uniq = len(meta["keys"]), [], {}, True
    for s in meta["eval_sets"]:
        si, n = meta["sets"].index(s), meta["n"][s]
        for a, al in enumerate(meta["alphas"]):
            if al < 1e-4 - 1e-12:
                continue
            c_all = int(A["bflags"][:, si, meta["rules"].index("all"), a].sum())
            c_u = int(A["bflags"][:, si, meta["rules"].index("uniq"), a].sum())
            lam = al * n * K
            p_u = float(poisson.sf(c_u - 1, lam))
            rows.append(dict(set=s, alpha=al, all_pooled_over_alpha=c_all / lam, uniq_pooled_over_alpha=c_u / lam,
                             uniq_count=c_u, expected_at_alpha=lam, uniq_p_one_sided=p_u, uniq_ok=p_u > 0.01))
            ok_uniq &= p_u > 0.01
            if dom(s) in ("code", "lists", "wiki_ru"):
                ok_all.setdefault(dom(s), []).append(c_all / lam)
    big = {d: min(v) for d, v in ok_all.items()}
    all_ok = all(v >= 10 for v in big.values())
    out["C1"] = dict(verdict="supported" if (all_ok and ok_uniq) else ("partial" if (all_ok or ok_uniq) else "refuted"),
                     all_min_ratio_on_repetitive=big, uniq_within_poisson_everywhere=ok_uniq, rows=rows)


def lottery_rows(A, meta, pk, pr, rule="uniq"):
    res = {}
    for s in meta["eval_sets"]:
        d = {}
        for al in DISP_ALPHAS:
            x = pk_row(pk, s, rule, "bin", al)
            d[f"disp_p@{al:g}"] = x["dispersion_p"]
            d[f"disp_p_mc@{al:g}"] = x["dispersion_p_mc"]
            d[f"dispersion@{al:g}"] = x["dispersion"]
            d[f"frac_sig_above@{al:g}"] = x["frac_keys_sig_above_pooled"]
            d[f"p99_over_alpha_meas@{al:g}"] = x["p99"] / al
        for al in SMALL_ALPHAS:
            y = pr_row(pr, s, rule, al)
            d[f"pred_p99_over_alpha@{al:g}"] = y["pred_p99"] / al
        d["overdispersed"] = all(d[f"disp_p@{al:g}"] < 0.01 for al in DISP_ALPHAS)
        d["overdispersed_mc"] = all(d[f"disp_p_mc@{al:g}"] < 0.01 for al in DISP_ALPHAS)
        d["sig_keys_gt_1pct"] = all(d[f"frac_sig_above@{al:g}"] > 0.01 for al in DISP_ALPHAS)
        d["pred_p99_ge_2"] = all(d[f"pred_p99_over_alpha@{al:g}"] >= 2 for al in SMALL_ALPHAS)
        d["lottery"] = d["overdispersed"] and d["sig_keys_gt_1pct"] and d["pred_p99_ge_2"]
        res[dom(s)] = d
    return res


def c2(A, meta, pk, pr, out):
    res = lottery_rows(A, meta, pk, pr)
    v = "supported" if res["code"]["lottery"] else ("refuted" if not any(r["lottery"] for r in res.values()) else "partial")
    out["C2"] = dict(verdict=v, domains_with_lottery=[d for d, r in res.items() if r["lottery"]], per_domain=res)


def c3(runs, out):
    """Primary (decided before the real run, after a code review): ratio to the exact formula for fixed-size green
    lists (lefthash; identical to sum_u a_u^2 for prf). Also reported: the pre-registered i.i.d. sum_u a_u^2.
    Band and +-10% allowance unchanged."""
    res, exc = {}, {"iid": [], "exact": []}
    for label, (A, meta) in runs.items():
        rows = theory_table(A, meta, sets=meta["sets"])
        rows = [r for r in rows if r["var_pred"] > 0]
        for r in rows:
            for kind, rk, wk in (("iid", "ratio", "within_ci"), ("exact", "ratio_exact", "within_ci_exact")):
                if not (r[wk] or abs(r[rk] - 1) <= 0.10):
                    exc[kind].append(dict(run=label, set=r["set"], rule=r["rule"], ratio=r[rk],
                                          ci=[r["ci95_lo"], r["ci95_hi"]]))
        ev = [r for r in rows if r["set"].endswith("/eval")]
        res[label] = dict(n_checked=len(rows),
                          eval_ratio_range_iid=[min(r["ratio"] for r in ev), max(r["ratio"] for r in ev)],
                          eval_ratio_range_exact=[min(r["ratio_exact"] for r in ev), max(r["ratio_exact"] for r in ev)],
                          max_abs_mean_over_se=max(abs(r["mean_over_se"]) for r in ev),
                          n_outside_95_exact=sum(not r["within_ci_exact"] for r in rows))
    # Aggregation (fixed before the real numbers, after the synthetic dry run): with ~200 rows some chance
    # exceptions are expected even if the theory is exact. Per row, the chance of an exception is
    # q = P(ratio < min(lo, 0.9)) + P(ratio > max(hi, 1.1)) with ratio ~ chi2_{K-1}/(K-1). "supported": the number of
    # exceptions is not significantly above sum(q) (one-sided binomial p > 0.01, rows treated as independent) and
    # no row is outside the 99.9% band and +-10%; "partial": some strong exceptions, but <= 5% of rows;
    # otherwise "refuted". The strict reading (no exception at all) is reported as verdict_strict.
    def chance_q(K):
        lo, hi = chi2.ppf([0.025, 0.975], K - 1) / (K - 1)
        return float(chi2.cdf(min(lo, 0.9) * (K - 1), K - 1) + chi2.sf(max(hi, 1.1) * (K - 1), K - 1))

    def strong(e, K):
        a, b = chi2.ppf([0.0005, 0.9995], K - 1) / (K - 1)
        return not (min(a, 0.9) <= e["ratio"] <= max(b, 1.1))
    v, agg = {}, {}
    for kind, e in exc.items():
        Ks = {lab: len(m["keys"]) for lab, (_, m) in runs.items()}
        q = {lab: chance_q(K) for lab, K in Ks.items()}
        n_rows = {lab: res[lab]["n_checked"] for lab in runs}
        expected = sum(q[lab] * n_rows[lab] for lab in runs)
        N = sum(n_rows.values())
        p_excess = float(binom.sf(len(e) - 1, N, expected / N)) if e else 1.0
        n_strong = sum(strong(x, Ks[x["run"]]) for x in e)
        v[kind] = "supported" if (p_excess > 0.01 and n_strong == 0) else ("partial" if n_strong <= 0.05 * N else "refuted")
        agg[kind] = dict(n_rows=N, n_exceptions=len(e), expected_by_chance=expected, p_excess=p_excess,
                         n_strong=n_strong, verdict_strict="supported" if not e else "refuted")
    out["C3"] = dict(verdict=v["exact"], verdict_iid_preregistered=v["iid"], aggregate_exact=agg["exact"],
                     aggregate_iid=agg["iid"], exceptions_iid=exc["iid"], exceptions_exact=exc["exact"], per_run=res)


def c4(A, meta, pk, pr, out):
    rows, bad = [], []
    for s in meta["eval_sets"]:
        for r in meta["rules"]:
            if r == "all":
                continue
            for al in DISP_ALPHAS:
                y, x = pr_row(pr, s, r, al), pk_row(pk, s, r, "bin", al)
                if y is None or not np.isfinite(y["total_z"]):
                    continue
                sig = x["dispersion_p"] is not None and np.isfinite(x["dispersion_p"]) and x["dispersion_p"] < 0.01
                ok_t = abs(y["total_z"]) < 3
                ok_c = (not sig) or (np.isfinite(y["corr_per_key"]) and y["corr_per_key"] > 0.5)
                row = dict(set=s, rule=r, alpha=al, measured=y["measured_total"], predicted=y["predicted_total"],
                           total_z=y["total_z"], dispersion_sig=sig, corr=y["corr_per_key"], ok=ok_t and ok_c)
                rows.append(row)
                if not row["ok"]:
                    bad.append(row)
    out["C4"] = dict(verdict="supported" if not bad else ("partial" if len(bad) <= max(1, len(rows) // 10) else "refuted"),
                     n_checked=len(rows), failures=bad, rows=rows)


def c5(A, meta, pk, pr, out):
    res, ok_all = {}, True
    th = {(r["set"], r["rule"]): r for r in theory_table(A, meta)}
    for eps in (0.01, 0.003):
        rule = f"mask{eps:g}"
        for s in meta["eval_sets"]:
            t = th[(s, rule)]
            disp = {al: pk_row(pk, s, rule, "bin", al)["dispersion_p"] for al in DISP_ALPHAS}
            disp_mc = {al: pk_row(pk, s, rule, "bin", al)["dispersion_p_mc"] for al in DISP_ALPHAS}
            p99 = {al: pr_row(pr, s, rule, al)["pred_p99"] / al for al in SMALL_ALPHAS}
            ok_v = t["var_meas"] <= eps
            ok_d = all((not np.isfinite(p)) or p >= 0.01 for p in disp.values())
            ok_p = all(v <= 1.5 for v in p99.values())
            res[f"{dom(s)}|{rule}"] = dict(var_meas=t["var_meas"], var_pred=t["var_pred"], var_ok=ok_v,
                                          disp_p={f"{k:g}": v for k, v in disp.items()}, disp_ok=ok_d,
                                          disp_p_mc={f"{k:g}": v for k, v in disp_mc.items()},
                                          pred_p99_over_alpha={f"{k:g}": v for k, v in p99.items()}, p99_ok=ok_p,
                                          masked_share=meta["masked_tuple_share"][f"{s}|{rule}"])
            ok_all &= ok_v and ok_d and ok_p
    nfail = sum(not (v["var_ok"] and v["disp_ok"] and v["p99_ok"]) for v in res.values())
    out["C5"] = dict(verdict="supported" if ok_all else ("partial" if nfail <= 2 else "refuted"), n_fail=nfail, cells=res)


def c6(P, out):
    if P is None:
        out["C6"] = dict(verdict="missing")
        return
    rows = P["rows"]

    def tpr(rule, delta, rho, al=1e-4):
        t = [x["tpr_bin"] for x in rows if x["rule"] == rule and close(x["delta"], delta) and close(x["rho"], rho)
             and close(x["alpha"], al)]
        return float(np.mean(t)) if t else float("nan")
    table = {}
    for delta in (1.0, 2.0):
        for rho in (0.0, 0.2):
            base = tpr("uniq", delta, rho)
            table[f"delta={delta:g},rho={rho:g}"] = dict(all=tpr("all", delta, rho), uniq=base, **{
                f"mask{e:g}": dict(tpr=tpr(f"mask{e:g}", delta, rho), loss_vs_uniq=base - tpr(f"mask{e:g}", delta, rho))
                for e in P["eps"]})
    out["C6"] = dict(verdict="reported", alpha=1e-4, table=table)


def c7(A, meta, out):
    ca = calibration_table(A, meta)
    res = {}
    for r in ca:
        res.setdefault(dom(r["set"]), {})[f"{r['calibrated_on']}@{r['alpha']:g}"] = dict(
            pooled_over_alpha=r["pooled_over_alpha"], p95_over_alpha=r["p95_over_alpha"],
            ideal_p95_over_alpha=r["ideal_p95_over_alpha"], mean_calz=r["mean_calz_over_keys"],
            sd_calz_over_keys=r["sd_calz_over_keys"])
    cross = {d: min(res[d][f"news@{al:g}"]["p95_over_alpha"] for al in (1e-2, 1e-3)) for d in ("code", "lists")}
    same_p = {d: {k: v["pooled_over_alpha"] for k, v in r.items() if k.startswith("same")} for d, r in res.items()}
    out["C7"] = dict(verdict="supported" if all(v >= 10 for v in cross.values()) else
                     ("partial" if any(v >= 10 for v in cross.values()) else "refuted"),
                     news_cal_min_p95_over_alpha=cross, same_domain_pooled_over_alpha=same_p, table=res)


def c8(main, prf, out):
    (A1, m1), (A2, m2) = main, prf
    pk1, pr1 = perkey_table(A1, m1), predicted_table(A1, m1)
    pk2, pr2 = perkey_table(A2, m2), predicted_table(A2, m2)
    l1, l2 = lottery_rows(A1, m1, pk1, pr1), lottery_rows(A2, m2, pk2, pr2)
    agree = {d: dict(lefthash=l1[d]["overdispersed"], prf=l2[d]["overdispersed"],
                     lefthash_pred_p99_1e4=l1[d]["pred_p99_over_alpha@0.0001"], prf_pred_p99_1e4=l2[d]["pred_p99_over_alpha@0.0001"])
             for d in l1}
    same = all(v["lefthash"] == v["prf"] for v in agree.values())
    out["C8"] = dict(verdict="supported" if same else "partial", dispersion_verdicts=agree,
                     note="K=300 for prf vs 1000 for lefthash: less power to detect overdispersion")


def c9(A, meta, out):
    res = {}
    ri = meta["rules"].index("uniq")
    for s in meta["eval_sets"]:
        si, n = meta["sets"].index(s), meta["n"][s]
        zbar = A["zsum"][:, si, ri] / n
        for a, al in enumerate(meta["alphas"][:3]):
            f = A["bflags"][:, si, ri, a] / n
            res.setdefault(dom(s), {})[f"{al:g}"] = dict(median_over_alpha=float(np.median(f) / al),
                                                        p90_over_alpha=float(np.quantile(f, .9) / al),
                                                        p99_over_alpha=float(np.quantile(f, .99) / al),
                                                        max_over_alpha=float(f.max() / al), frac_keys_zero=float(np.mean(f == 0)))
        res[dom(s)]["sd_zbar"] = float(zbar.std(ddof=1))
    out["C9"] = dict(verdict="see numbers", per_domain=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--report", default="report")
    ap.add_argument("--main", default=MAIN)
    ap.add_argument("--h0", default="null_h0_lefthash_K1000")
    ap.add_argument("--prf", default="null_h1_prf_K300")
    args = ap.parse_args()
    out = {}
    c0(args.results, out)
    A, meta = load(args.results, args.main)
    pk, pr = perkey_table(A, meta), predicted_table(A, meta)
    c1(A, meta, pk, out)
    c2(A, meta, pk, pr, out)
    runs = {"h1": (A, meta)}
    for label, tag in [("h0", args.h0), ("prf", args.prf)]:
        B = load(args.results, tag)
        if B[0] is not None:
            runs[label] = B
    c3(runs, out)
    c4(A, meta, pk, pr, out)
    c5(A, meta, pk, pr, out)
    pj = os.path.join(args.results, "power.json")
    c6(json.load(open(pj)) if os.path.exists(pj) else None, out)
    c7(A, meta, out)
    if "prf" in runs:
        c8(runs["h1"], runs["prf"], out)
    if "h0" in runs:
        c9(*runs["h0"], out)
    os.makedirs(args.report, exist_ok=True)
    with open(os.path.join(args.report, "claims.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    for c, v in out.items():
        print(c, v.get("verdict"))


if __name__ == "__main__":
    main()
