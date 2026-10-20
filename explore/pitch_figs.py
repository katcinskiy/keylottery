# Figures for the 3-slide pitch (numbers from results/screen_h1_lefthash_K1000 = 1000 fresh keys on fresh texts E2,
# and results/power_screen.json = TPR of 10 random + 10 screened keys). Run from the project root with uv.
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom
NAVY, INK, MUTED, ORANGE, BLUE, BAND, BG = "#14213D", "#3B4556", "#8A93A3", "#D9622B", "#2F6DB5", "#D9DCE1", "#F7F6F2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
A = np.load("results/screen_h1_lefthash_K1000.npz"); m = json.load(open("results/screen_h1_lefthash_K1000.json"))
S, R, AL = m["sets"], m["rules"], m["alphas"]; al = 1e-3; n = 6000; K = len(m["keys"])
def fpr(dom, rule="uniq"):
    return A["bflags"][:, S.index(f"{dom}/e2"), R.index(rule), AL.index(al)] / n / al
# --- fig 1: per-key FPR of Three Bricks (uniq) across 1000 keys, ranked
fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor=BG); ax.set_facecolor(BG)
rng = np.random.default_rng(0)
sims = np.sort(rng.binomial(n, al, size=(400, K)), axis=1)[:, ::-1] / n / al
lo, hi = np.quantile(sims, [0.025, 0.975], axis=0); x = np.arange(1, K + 1)
ax.fill_between(x, np.maximum(lo, 0.08), np.maximum(hi, 0.08), color=BAND, lw=0)
from matplotlib.patches import Patch
handles = []
for dom, lab, col in [("wiki_ru", "Russian Wikipedia", ORANGE), ("code", "Python code", NAVY), ("news", "News", BLUE)]:
    y = np.sort(fpr(dom))[::-1]; h, = ax.plot(x, np.maximum(y, 0.08), color=col, lw=2.6, label=lab); handles.append(h)
handles.append(Patch(color=BAND, label="ideal keys (95% band)"))
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=14)
ax.axhline(1, color=INK, lw=1, ls=":"); ax.text(1.2, 0.82, "promised rate α", color=INK, fontsize=13)
ax.axvline(10, color=INK, lw=1.2, ls="--"); ax.text(11, 0.13, "1 key in 100", color=INK, fontsize=14)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.08, 400); ax.set_xlim(1, K)
ax.set_xlabel("1000 random keys, ranked from worst"); ax.set_ylabel("false-positive rate / α  (α = 10⁻³)")
fig.tight_layout(); fig.savefig("report/pitch/fig_lottery.png", dpi=200, facecolor=BG); plt.close(fig)
# --- Pareto: worst-key FPR (99th pct over keys, worst domain) vs TPR (delta=1, clean), alpha=1e-3
doms = ["news", "wiki_en", "lists", "code", "wiki_ru"]
def worst(rule, al_test=al, sel=None):
    sel = np.ones(K, bool) if sel is None else sel
    return max(np.quantile(A["bflags"][sel, S.index(f"{d}/e2"), R.index(rule), AL.index(al_test)] / n, .99) / al for d in doms)
Z = np.stack([A["zsum"][:, S.index(f"{d}/ref"), R.index("uniq")] / m["n"][f"{d}/ref"] for d in doms], 1)
s = (Z / Z.std(0, ddof=1)).max(1); kept = s <= np.quantile(s, 0.2)
P = json.load(open("results/power_screen.json"))["rows"]; G = json.load(open("results/screen_gen_keys.json"))
def tpr(rule, a, rho=0.0, ks=None):
    ks = ks or G["random"]
    return float(np.mean([[x["tpr_bin"] for x in P if x["key"] == k and x["rule"] == rule and x["rho"] == rho
                           and abs(x["alpha"] - a) / a < 1e-6][0] for k in ks]))
pts = {"KGW, all tokens": (worst("all"), tpr("all", al)), "Three Bricks (each pair once)": (worst("uniq"), tpr("uniq", al)),
       "Masking frequent pairs (our pilot)": (worst("mask0.01"), tpr("mask0.01", al)), "Tighten threshold ×10": (worst("uniq", al / 10), tpr("uniq", al / 10))}
ours = (worst("uniq", al, kept), tpr("uniq", al) - 0.030)        # model-based screening cost at alpha=1e-3, rho=0: 3.0 pp
print({k: (round(a, 2), round(b, 3)) for k, (a, b) in pts.items()}, "ours", (round(ours[0], 2), round(ours[1], 3)))
for with_ours in (False, True):
    fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor=BG); ax.set_facecolor(BG)
    ax.axvspan(0.3, 2.0, color=BAND, lw=0); ax.text(0.34, 0.905, "as good as\nan ideal key", color="#5B6475", fontsize=13, va="top")
    off = {"KGW, all tokens": (-0.0, 0.014, "right"), "Three Bricks (each pair once)": (1.0, 0.014, "center"),
           "Masking frequent pairs (our pilot)": (1.18, -0.018, "left"), "Tighten threshold ×10": (1.22, 0.0, "left")}
    tb, tt = pts["Three Bricks (each pair once)"], pts["Tighten threshold ×10"]
    ax.plot([tb[0], tt[0]], [tb[1], tt[1]], color=MUTED, lw=1.6, ls="--", zorder=2)
    ax.text(0.62, 0.676, "tightening the threshold (interpolated)", color="#5B6475", fontsize=12, rotation=29)
    for k, (xv, yv) in pts.items():
        ax.scatter([xv], [yv], s=140, color=INK, zorder=3)
        dx, dy, ha = off[k]; ax.text(xv * (1.0 if dx == 0 else dx), yv + dy, k, fontsize=14, color=INK, ha=ha, va="center")
    if with_ours:
        ax.scatter([ours[0]], [ours[1]], s=420, marker="*", color=ORANGE, zorder=4)
        ax.text(0.33, 0.797, "Screened keys\n(ours, TPR est.)", fontsize=15, color=ORANGE, fontweight="bold", va="top")
    else:
        ax.annotate("", xy=(1.6, 0.835), xytext=(7.5, 0.835), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.4))
        ax.text(1.7, 0.79, "gap", color=ORANGE, fontsize=17, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(0.3, 1000); ax.set_ylim(0.6, 0.91)
    ax.set_xlabel("FPR / α of the 1-in-100 key  (worst of 5 domains)")
    ax.set_ylabel("detection rate (TPR) at α = 10⁻³")
    fig.tight_layout(); fig.savefig(f"report/pitch/fig_pareto{'_ours' if with_ours else ''}.png", dpi=200, facecolor=BG); plt.close(fig)
print("ok")
