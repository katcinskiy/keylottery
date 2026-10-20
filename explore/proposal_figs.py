# Vector figures for the proposal (column width ~3.25 in). Same data as explore/pitch_figs.py.
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
NAVY, INK, MUTED, ORANGE, BLUE, BAND = "#14213D", "#3B4556", "#8A93A3", "#D9622B", "#2F6DB5", "#D9DCE1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
A = np.load("results/screen_h1_lefthash_K1000.npz"); m = json.load(open("results/screen_h1_lefthash_K1000.json"))
S, R, AL = m["sets"], m["rules"], m["alphas"]; al, n, K = 1e-3, 6000, len(m["keys"])
doms = ["news", "wiki_en", "lists", "code", "wiki_ru"]
fpr = lambda d, rule="uniq", a=al: A["bflags"][:, S.index(f"{d}/e2"), R.index(rule), AL.index(a)] / n / al
# Figure 1: ranked per-key FPR (Three Bricks scoring) on unseen texts
fig, ax = plt.subplots(figsize=(3.3, 1.85))
sims = np.sort(np.random.default_rng(0).binomial(n, 0.75 * al, size=(400, K)), axis=1)[:, ::-1] / n / al  # exact test runs at ~0.75 alpha
lo, hi = np.quantile(sims, [0.025, 0.975], axis=0); x = np.arange(1, K + 1)
ax.fill_between(x, np.maximum(lo, 0.08), np.maximum(hi, 0.08), color=BAND, lw=0)
hs = []
for d, lab, col in [("wiki_ru", "Russian Wikipedia", ORANGE), ("code", "Python code", NAVY), ("lists", "Wikipedia lists", "#8C7A2B"),
                    ("wiki_en", "English Wikipedia", "#2A8C7A"), ("news", "News", BLUE)]:
    h, = ax.plot(x, np.maximum(np.sort(fpr(d))[::-1], 0.08), color=col, lw=1.1, label=lab); hs.append(h)
hs.append(Patch(color=BAND, label="ideal keys (95%)"))
ax.legend(handles=hs, loc="upper right", frameon=False, fontsize=6.8)
ax.axhline(1, color=INK, lw=0.6, ls=":"); ax.axvline(10, color=INK, lw=0.7, ls="--")
ax.text(11, 0.12, "1 key in 100", fontsize=6.5, color=INK); ax.text(1.2, 1.12, "promised rate", fontsize=6.2, color=INK)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.08, 400); ax.set_xlim(1, K)
ax.set_xlabel("1000 random keys, ranked"); ax.set_ylabel(r"per-key FPR / $\alpha$  ($\alpha=10^{-3}$)")
fig.tight_layout(pad=0.3); fig.savefig("proposal/figs/fig_lottery.pdf"); plt.close(fig)
# Figure 2: Pareto (1-in-100 key FPR vs TPR)
def worst(rule, a_test=al, sel=None):
    sel = np.ones(K, bool) if sel is None else sel
    return max(np.quantile(A["bflags"][sel, S.index(f"{d}/e2"), R.index(rule), AL.index(a_test)] / n, .99) / al for d in doms)
Z = np.stack([A["zsum"][:, S.index(f"{d}/ref"), R.index("uniq")] / m["n"][f"{d}/ref"] for d in doms], 1)
s = (Z / Z.std(0, ddof=1)).max(1); kept = s <= np.quantile(s, 0.2)
P = json.load(open("results/power_screen.json"))["rows"]; G = json.load(open("results/screen_gen_keys.json"))
tpr = lambda rule, a: float(np.mean([[r["tpr_bin"] for r in P if r["key"] == k and r["rule"] == rule and r["rho"] == 0.0
                                      and abs(r["alpha"] - a) / a < 1e-6][0] for k in G["random"]]))
pts = {"KGW (all tokens)": (worst("all"), tpr("all", al)), "Three Bricks": (worst("uniq"), tpr("uniq", al)),
       "masking frequent pairs": (worst("mask0.01"), tpr("mask0.01", al)), r"tighten to $\alpha/10$": (worst("uniq", al / 10), tpr("uniq", al / 10))}
ours = (worst("uniq", al, kept), tpr("uniq", al) - 0.030)
# Key pool of r=10 (Aremu et al.; post hoc, CLAUDE.md Sec. 10): p99 over 20k pools, lefthash colours for code and
# Russian, prf colours for the English domains; TPR = single-key TPR at the per-key Sidak level (~alpha/10), not measured.
KPL = json.load(open("results/key_pool_lefthash.json"))["results"]; KPE = json.load(open("results/key_pool_prf_exact.json"))
pool_x = max([KPE[f"{d}|afw=0.001|r=10"]["pool_p99"] for d in ["news", "wiki_en", "lists"]]
             + [KPL[f"{d}|afw=0.001|r=10"]["pool_p99"] for d in ["code", "wiki_ru"]])
pool = (pool_x, tpr("uniq", al / 10))
fig, ax = plt.subplots(figsize=(3.3, 1.95))
ax.axvspan(0.3, 1.83, color=BAND, lw=0); ax.text(0.33, 0.905, "ideal keys", fontsize=6.5, color="#5B6475", va="top")
tb, tt = pts["Three Bricks"], pts[r"tighten to $\alpha/10$"]
ax.plot([tb[0], tt[0]], [tb[1], tt[1]], color=MUTED, lw=0.9, ls="--")
off = {"KGW (all tokens)": (1.0, 0.016, "right"), "Three Bricks": (1.0, 0.016, "center"), "masking frequent pairs": (1.2, -0.016, "left"),
       r"tighten to $\alpha/10$": (0.72, -0.022, "left")}
for k, (xv, yv) in pts.items():
    ax.scatter([xv], [yv], s=18, color=INK, zorder=3); dx, dy, ha = off[k]
    ax.text(xv * dx, yv + dy, k, fontsize=6.8, color=INK, ha=ha, va="center")
ax.scatter([pool[0]], [pool[1]], s=18, facecolors="white", edgecolors=INK, lw=0.9, zorder=3)
ax.text(pool[0] * 1.2, pool[1] - 0.014, "key pool of 10", fontsize=6.8, color=INK, ha="left", va="center")
ax.scatter([ours[0]], [ours[1]], s=70, marker="*", color=ORANGE, zorder=4)
ts = np.array([[r["tpr_bin"] for r in P if r["key"] == k and r["rule"] == "uniq" and r["rho"] == 0.0
                and abs(r["alpha"] - al) / al < 1e-6][0] for k in G["screened"]])
from scipy import stats
hw = stats.t.ppf(0.975, len(ts) - 1) * ts.std(ddof=1) / np.sqrt(len(ts))
ax.errorbar([ours[0] * 1.12], [ts.mean()], yerr=[[hw], [hw]], fmt="o", ms=3, color=ORANGE, capsize=2, lw=0.9, zorder=4)
ax.text(0.33, 0.772, "audited keys\n(ours, TPR est.)", fontsize=6.8, color=ORANGE, fontweight="bold", va="top")
ax.set_xscale("log"); ax.set_xlim(0.3, 1000); ax.set_ylim(0.6, 0.91)
ax.set_xlabel(r"FPR/$\alpha$ of the 1-in-100 key (worst of 5 domains)"); ax.set_ylabel(r"TPR ($\alpha=10^{-3}$, $\delta=1$)")
fig.tight_layout(pad=0.3); fig.savefig("proposal/figs/fig_pareto.pdf"); plt.close(fig)
print({k: (round(a, 2), round(b, 3)) for k, (a, b) in pts.items()}, "pool", (round(pool[0], 2), round(pool[1], 3)),
      "ours", (round(ours[0], 2), round(ours[1], 3)), "screened measured", (round(ts.mean(), 4), round(hw, 4)))
