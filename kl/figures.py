"""
Figures from results/.   python -m kl.figures --results results --out report/fig
  fig1_perkey.png      per-key false-positive counts over many keys vs. what identical keys would give
  fig2_variance.png    Var_k(mean z) measured vs. predicted (all sets, rules, h)
  fig3_predictor.png   semi-analytic per-key FPR: predicted vs. measured, and the tail at tiny alpha
  fig4_pareto.png      power (TPR) vs. worst-key FPR for each scoring rule
  fig5_calibration.png per-key calibration baseline, in-domain vs. cross-domain
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

DOM_LABEL = {"news": "news (C4)", "wiki_en": "Wikipedia EN", "lists": "Wikipedia lists", "code": "Python code",
             "wiki_ru": "Wikipedia RU", "opt0": "OPT text, δ=0"}


def load(results, name):
    p = os.path.join(results, name)
    return dict(np.load(p)), json.load(open(p.replace(".npz", ".json")))


def find(results, prefix):
    """Untagged main run for a prefix, e.g. null_h1_lefthash_K1000.npz (not ..._K10_lab4keys.npz); largest K wins."""
    import re
    pat = re.compile(rf"^{re.escape(prefix)}(?:[A-Za-z0-9]*_)?K?(\d+)\.npz$")
    c = [(int(m.group(1)), f) for f in os.listdir(results) if (m := pat.match(f))]
    return max(c)[1] if c else None


def dom(s):
    return s.split("/")[0]


def fig_perkey(A, meta, out, alpha_idx=1, rules=("all", "uniq", "mask0.01")):
    """Sorted per-key flag counts (binomial test) for each domain, against the band of identical keys."""
    ev = meta["eval_sets"]
    al = meta["alphas"][alpha_idx]
    fig, axes = plt.subplots(1, len(ev), figsize=(3.2 * len(ev), 3.0), sharey=False)
    K = len(meta["keys"])
    for ax, s in zip(np.atleast_1d(axes), ev):
        si, n = meta["sets"].index(s), meta["n"][s]
        for r in rules:
            if r not in meta["rules"]:
                continue
            c = np.sort(A["bflags"][:, si, meta["rules"].index(r), alpha_idx])[::-1]
            ax.plot(np.arange(1, K + 1), np.maximum(c, 0.5), label=r, lw=1.4)
        # identical keys at the nominal level: sorted Binomial(n, alpha) order statistics (median and 95% band)
        rng = np.random.default_rng(0)
        sims = np.sort(rng.binomial(n, al, size=(400, K)), axis=1)[:, ::-1]
        lo, med, hi = np.quantile(sims, [0.025, 0.5, 0.975], axis=0)
        ax.fill_between(np.arange(1, K + 1), np.maximum(lo, 0.5), np.maximum(hi, 0.5), color="0.85", label=f"ideal keys (α={al:g})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{DOM_LABEL.get(dom(s), s)} (n={n})", fontsize=9)
        ax.set_xlabel("keys, ranked")
    np.atleast_1d(axes)[0].set_ylabel(f"flagged human texts per key\n(binomial test, α={al:g})")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_variance(runs, out):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    mk = {"all": "o", "uniq": "s"}
    lo_all, hi_all = [], []
    for label, (A, meta) in runs.items():
        K = len(meta["keys"])
        for s in meta["eval_sets"] + [x for x in meta["sets"] if x.endswith("/ref")]:
            si = meta["sets"].index(s)
            for ri, r in enumerate(meta["rules"]):
                vp = meta.get("var_pred_exact", meta["var_pred"])[f"{s}|{r}"]
                if vp <= 0:
                    continue
                vm = (A["zsum"][:, si, ri] / meta["n"][s]).var(ddof=1)
                ax.scatter(vp, vm, s=14, marker=mk.get(r, "^"), alpha=0.7,
                           color={"h1": "C0", "h0": "C3", "prf": "C2", "opt0": "C4"}.get(label, "C5"))
                lo_all.append(min(vp, vm))
                hi_all.append(max(vp, vm))
    lo, hi = min(lo_all) / 2, max(hi_all) * 2
    xs = np.geomspace(lo, hi, 50)
    ax.plot(xs, xs, "k-", lw=1)
    K = len(runs["h1"][1]["keys"])
    from scipy.stats import chi2
    a, b = chi2.ppf([0.025, 0.975], K - 1) / (K - 1)
    ax.fill_between(xs, a * xs, b * xs, color="0.9", zorder=0, label=f"95% band, K={K}")
    for label, c in {"h1": "C0", "h0": "C3", "prf": "C2", "opt0": "C4"}.items():
        if label in runs:
            ax.scatter([], [], color=c, label={"h1": "h=1 (lefthash)", "h0": "h=0 (fixed list)", "prf": "h=1 (ideal PRF)",
                                               "opt0": "OPT text, h=1"}[label])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"predicted $\mathrm{Var}_k(\bar z)$ (fixed-size green lists)")
    ax.set_ylabel(r"measured $\mathrm{Var}_k(\bar z)$ over keys")
    ax.set_title("corpus-mean z across keys: theory vs. measurement", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_predictor(A, meta, out, rule="uniq"):
    ev, ri = meta["eval_sets"], meta["rules"].index(rule)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    for ei, s in enumerate(ev):
        si, n = meta["sets"].index(s), meta["n"][s]
        for a in (0, 1):
            pr = A["pred"][:, ei, ri, a] * n
            me = A["bflags"][:, si, ri, a]
            ax.scatter(pr, me + 0.3 * (me == 0), s=6, alpha=0.4, color=f"C{ei}", marker="o" if a == 0 else "x",
                       label=DOM_LABEL.get(dom(s), s) if a == 0 else None)
    m = ax.get_xlim()[1]
    ax.plot([0.3, m], [0.3, m], "k-", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted flags per key (o: α=1e-2, x: α=1e-3)")
    ax.set_ylabel("measured flags per key")
    ax.set_title(f"per-key predictor, rule={rule}", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    for ei, s in enumerate(ev):
        q = [np.quantile(A["pred"][:, ei, ri, a] / al, [0.5, 0.99, 1.0]) for a, al in enumerate(meta["alphas"])]
        q = np.array(q)
        ax.plot(meta["alphas"], q[:, 1], "-o", ms=3, color=f"C{ei}", label=DOM_LABEL.get(dom(s), s))
        ax.plot(meta["alphas"], q[:, 2], ":", color=f"C{ei}")
    ax.axhline(1, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("test level α")
    ax.set_ylabel("per-key FPR / α  (solid: 99th pct key, dotted: worst key)")
    ax.set_title(f"predicted per-key FPR tail, rule={rule}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_pareto(A, meta, P, out, alpha=1e-4, delta=2.0, rhos=(0.0, 0.2)):
    rules = meta["rules"][1:]
    a = meta["alphas"].index(alpha)
    rows = P["rows"]
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.4 * len(rhos), 3.8), sharey=True)
    for ax, rho in zip(np.atleast_1d(axes), rhos):
        for ei, s in enumerate(meta["eval_sets"]):
            xs, ys = [], []
            for r in rules:
                ri = meta["rules"].index(r)
                worst = np.quantile(A["pred"][:, ei, ri, a], 0.99) / alpha
                t = [x["tpr_bin"] for x in rows if x["rule"] == r and x["delta"] == delta and abs(x["rho"] - rho) < 1e-9
                     and abs(x["alpha"] - alpha) / alpha < 1e-6]
                xs.append(worst)
                ys.append(np.mean(t) if t else np.nan)
            ax.plot(xs, ys, "-o", ms=3, color=f"C{ei}", label=DOM_LABEL.get(dom(s), s))
            for x, y, r in zip(xs, ys, rules):
                if ei == 0:
                    ax.annotate(r.replace("mask", "ε="), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.axvline(1, color="k", lw=0.8)
        ax.set_xlabel(f"99th-pct key FPR / α  (α={alpha:g}, predicted)")
        ax.set_title(f"δ={delta:g}, {int(rho * 100)}% tokens substituted", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("TPR on OPT-1.3B watermarked text")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_calibration(A, meta, out):
    if not meta.get("cal_tags"):
        return
    ev = meta["eval_sets"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    w = 0.38
    for ti, t in enumerate(meta["cal_tags"]):
        for a, al in enumerate(meta["cal_alphas"]):
            vals = [np.quantile(A["calflags"][:, ei, ti, a] / meta["n"][s], 0.95) / al for ei, s in enumerate(ev)]
            ax.bar(np.arange(len(ev)) + (2 * ti + a - 1.5) * w / 2, vals, width=w / 2, bottom=0,
                   label=f"calibrated on {t} ref, α={al:g}", alpha=0.6 + 0.3 * a)
    ax.axhline(1, color="k", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(ev)))
    ax.set_xticklabels([DOM_LABEL.get(dom(s), s) for s in ev], fontsize=7)
    ax.set_ylabel("95th-pct key FPR / α")
    ax.set_title("per-key calibration baseline (uniq z standardised on reference text)", fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/fig")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = {}
    for label, prefix in [("h1", "null_h1_lefthash_K"), ("h0", "null_h0_lefthash_K"), ("prf", "null_h1_prf_K"),
                          ("opt0", "null_opt0_h1_")]:
        f = find(args.results, prefix)
        if f:
            runs[label] = load(args.results, f)
    if "h1" not in runs and "prf" in runs:      # synthetic dry run: prf-only files
        runs["h1"] = runs.pop("prf")
    if "h0" not in runs and find(args.results, "null_h0_prf_K"):
        runs["h0"] = load(args.results, find(args.results, "null_h0_prf_K"))
    A, meta = runs["h1"]
    fig_perkey(A, meta, os.path.join(args.out, "fig1_perkey.png"))
    if "h0" in runs:
        fig_perkey(*runs["h0"], os.path.join(args.out, "fig1b_perkey_h0.png"), rules=("all", "uniq"))
    fig_variance(runs, os.path.join(args.out, "fig2_variance.png"))
    fig_predictor(A, meta, os.path.join(args.out, "fig3_predictor.png"))
    pj = os.path.join(args.results, "power.json")
    if os.path.exists(pj):
        fig_pareto(A, meta, json.load(open(pj)), os.path.join(args.out, "fig4_pareto.png"))
    fig_calibration(A, meta, os.path.join(args.out, "fig5_calibration.png"))
    print("figures in", args.out, sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
