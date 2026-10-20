"""
Tables, figures and pass/fail verdicts from the results/ files.
  python -m kl.analyze --results results --out report
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import binom, chi2, norm, poisson


def load(results, name):
    p = os.path.join(results, name)
    if not os.path.exists(p):
        return None, None
    return dict(np.load(p)), json.load(open(p.replace(".npz", ".json")))


def q(x, p):
    return float(np.quantile(x, p))


def theory_table(A, meta, sets=None):
    rows = []
    K = len(meta["keys"])
    lo, hi = chi2.ppf([0.025, 0.975], K - 1) / (K - 1)
    for s in sets or meta["eval_sets"]:
        si = meta["sets"].index(s)
        for ri, r in enumerate(meta["rules"]):
            zbar = A["zsum"][:, si, ri] / meta["n"][s]
            vp = meta["var_pred"][f"{s}|{r}"]
            vm = float(zbar.var(ddof=1))
            ratio = vm / vp if vp > 0 else float("nan")
            # exact formula for fixed-size green lists (lefthash); equals var_pred for i.i.d. colours (prf)
            ve = meta.get("var_pred_exact", {}).get(f"{s}|{r}", vp)
            ratio_e = vm / ve if ve > 0 else float("nan")
            rows.append(dict(set=s, rule=r, var_pred=vp, var_meas=vm, ratio=ratio, ci95_lo=lo, ci95_hi=hi,
                             within_ci=bool(lo <= ratio <= hi), var_pred_exact=ve, ratio_exact=ratio_e,
                             within_ci_exact=bool(lo <= ratio_e <= hi), mean_zbar=float(zbar.mean()),
                             mean_over_se=float(zbar.mean() / np.sqrt(ve / K)) if ve > 0 else float("nan"),
                             sd_zbar=float(np.sqrt(vm)), max_zbar=float(zbar.max()), min_zbar=float(zbar.min()),
                             mean_T=meta["mean_T"][f"{s}|{r}"]))
    return rows


def dispersion(counts, n):
    """Overdispersion of per-key flag counts relative to identical binomial keys."""
    p = counts.sum() / (len(counts) * n)
    if p == 0:
        return np.nan, np.nan
    exp_var = n * p * (1 - p)
    D = ((counts - counts.mean()) ** 2).sum() / exp_var
    return float(D / (len(counts) - 1)), float(chi2.sf(D, len(counts) - 1))


def dispersion_mc(counts, n, reps=1000, seed=0):
    """Monte-Carlo p-value of the same dispersion statistic under K identical Binomial(n, p_hat) keys
    (the chi2 reference is anti-conservative when the expected count per key is << 1)."""
    K = len(counts)
    p = counts.sum() / (K * n)
    if p == 0:
        return np.nan
    def stat(c):
        ph = c.sum(-1) / (K * n)
        ss = ((c - c.mean(-1, keepdims=True)) ** 2).sum(-1)
        return np.where(ph > 0, ss / np.maximum(n * ph * (1 - ph), 1e-300), 0.0)
    sims = np.random.default_rng(seed).binomial(n, p, size=(reps, K))
    return float((1 + np.sum(stat(sims) >= stat(counts.astype(float)))) / (reps + 1))


def perkey_table(A, meta):
    rows = []
    for s in meta["eval_sets"]:
        si, n = meta["sets"].index(s), meta["n"][s]
        for r in meta["rules"]:
            ri = meta["rules"].index(r)
            tests = [("z", j, t, float(norm.sf(t))) for j, t in enumerate(meta["z_thresh"])]
            tests += [("bin", j, a, a) for j, a in enumerate(meta["alphas"])]
            for kind, j, lev, nominal in tests:
                c = A["zflags" if kind == "z" else "bflags"][:, si, ri, j]
                f = c / n
                # keys whose count is significantly above the pooled rate (one-sided Poisson, 1%)
                pooled = c.mean()
                sig_hi = float(np.mean(poisson.sf(c - 1, pooled) < 0.01)) if pooled > 0 else 0.0
                disp, disp_p = dispersion(c, n)
                disp_mc = dispersion_mc(c, n) if kind == "bin" else np.nan
                rows.append(dict(set=s, rule=r, test=kind, level=lev, nominal=nominal, n_texts=n,
                                 pooled=float(f.mean()), median=q(f, .5), p90=q(f, .9), p95=q(f, .95), p99=q(f, .99),
                                 max=float(f.max()), frac_keys_over_2x_nominal=float(np.mean(f > 2 * nominal)),
                                 frac_keys_sig_above_pooled=sig_hi, dispersion=disp, dispersion_p=disp_p,
                                 dispersion_p_mc=disp_mc))
    return rows


def predicted_table(A, meta):
    rows = []
    for ei, s in enumerate(meta["eval_sets"]):
        si, n = meta["sets"].index(s), meta["n"][s]
        for ri, r in enumerate(meta["rules"]):
            if r == "all":
                continue
            for a, al in enumerate(meta["alphas"]):
                pr = A["pred"][:, ei, ri, a]
                meas = A["bflags"][:, si, ri, a]
                tot_p, tot_m = pr.sum() * n, meas.sum()
                rows.append(dict(set=s, rule=r, alpha=al, pred_mean=float(pr.mean()), pred_median=q(pr, .5),
                                 pred_p90=q(pr, .9), pred_p95=q(pr, .95), pred_p99=q(pr, .99), pred_max=float(pr.max()),
                                 pred_p95_over_alpha=q(pr, .95) / al, pred_max_over_alpha=float(pr.max()) / al,
                                 frac_keys_pred_over_2x=float(np.mean(pr > 2 * al)),
                                 measured_total=int(tot_m), predicted_total=float(tot_p),
                                 meas_over_pred=float(tot_m / tot_p) if tot_p > 0 else np.nan,
                                 total_z=float((tot_m - tot_p) / np.sqrt(tot_p)) if tot_p > 0 else np.nan,
                                 corr_per_key=float(np.corrcoef(meas, pr)[0, 1]) if meas.std() > 0 and pr.std() > 0 else np.nan))
    return rows


def calibration_table(A, meta):
    rows = []
    for ei, s in enumerate(meta["eval_sets"]):
        n = meta["n"][s]
        for ti, t in enumerate(meta["cal_tags"]):
            for a, al in enumerate(meta["cal_alphas"]):
                f = A["calflags"][:, ei, ti, a] / n
                mz = A["calzsum"][:, ei, ti] / n if "calzsum" in A else np.full(len(f), np.nan)
                rows.append(dict(set=s, calibrated_on=t, alpha=al, pooled=float(f.mean()), median=q(f, .5),
                                 p95=q(f, .95), max=float(f.max()), pooled_over_alpha=float(f.mean() / al),
                                 p95_over_alpha=q(f, .95) / al,
                                 ideal_p95_over_alpha=float(binom.ppf(0.95, n, al) / (n * al)),
                                 mean_calz_over_keys=float(np.mean(mz)), sd_calz_over_keys=float(np.std(mz, ddof=1))))
    return rows


def power_rows(P):
    return P["rows"] if P else []


def write_csv(path, rows):
    if not rows:
        return
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for name in sorted(os.listdir(args.results)):
        if not (name.startswith("null_") and name.endswith(".npz")) or name.endswith("_z.npz"):
            continue
        A, meta = load(args.results, name)
        tag = name[:-4]
        th, pk = theory_table(A, meta, sets=meta["sets"]), perkey_table(A, meta)
        pr, ca = predicted_table(A, meta), (calibration_table(A, meta) if meta.get("cal_tags") else [])
        for t, rows in [("theory", th), ("perkey", pk), ("predicted", pr), ("calibration", ca)]:
            write_csv(os.path.join(args.out, f"{tag}_{t}.csv"), rows)
        summary[tag] = dict(theory=th, perkey=pk, predicted=pr, calibration=ca, meta={k: meta[k] for k in
                            ("keys", "sets", "rules", "n", "eps", "h") if k in meta})
    P = json.load(open(os.path.join(args.results, "power.json"))) if os.path.exists(os.path.join(args.results, "power.json")) else None
    write_csv(os.path.join(args.out, "power.csv"), power_rows(P))
    summary["power"] = power_rows(P)
    ck = os.path.join(args.results, "checks.json")
    summary["checks"] = json.load(open(ck)) if os.path.exists(ck) else None
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
