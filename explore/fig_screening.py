# Figure for the pre-registered key-screening follow-up (CLAUDE.md Section 9): p99/alpha of per-key FPR on fresh texts E2.
# uv run --no-project --with numpy,scipy,matplotlib python explore/fig_screening.py
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open("report/screening.json"))
doms = ["news", "wiki_en", "lists", "code", "wiki_ru"]
lab = {"news": "новости", "wiki_en": "Википедия EN", "lists": "списки", "code": "код Python", "wiki_ru": "Википедия RU"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, al in zip(axes, (1e-2, 1e-3)):
    rows = {(x["domain"]): x for x in r["S1"]["rows"] if x["set"] == "e2" and x["alpha"] == al}
    rows5 = {(x["domain"]): x for x in r["S5"]["rows"] if x["set"] == "e2" and x["alpha"] == al}
    rows6 = {(x["domain"]): x for x in r["S6"]["rows"] if x["set"] == "e2" and x["alpha"] == al}
    xs = np.arange(len(doms)); w = 0.16
    series = [("все ключи, uniq", [rows[d]["all"]["p99_over_alpha"] for d in doms], "0.6"),
              ("отбор 20% по среднему z (S1)", [rows[d]["q0.2"]["p99_over_alpha"] for d in doms], "C0"),
              ("отбор 20% по флагам на ref (S6)", [rows6[d]["q0.2"]["p99_over_alpha"] for d in doms], "C4"),
              ("все ключи, маска ε=0.01", [rows5[d]["all"]["p99_over_alpha"] for d in doms], "C1"),
              ("отбор 20% + маска", [rows5[d]["q0.2"]["p99_over_alpha"] for d in doms], "C2")]
    for i, (name, v, c) in enumerate(series):
        ax.bar(xs + (i - 2) * w, v, width=w, color=c, label=name)
    F = rows["news"]["floor"]
    ax.axhline(F, color="k", ls="--", lw=1, label="идеальные ключи (p99)")
    ax.text(len(doms) - 0.5, F * 1.04, f"идеальные ключи: {F:.2f}", ha="right", va="bottom", fontsize=7)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([lab[d] for d in doms], fontsize=8)
    ax.set_yscale("log"); ax.set_title(f"свежие тексты E2 (без пересечений), α={al:g}", fontsize=9)
axes[0].set_ylabel("99-й перцентиль FPR ключа / α\n(1000 новых ключей)")
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("report/fig/fig6_screening.png", dpi=160)
print("report/fig/fig6_screening.png")
