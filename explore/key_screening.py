# POST-HOC exploration (2026-10-04), not part of the pre-registered Section-6 analysis.
# Key screening: rank keys by worst-domain zbar on REFERENCE text, measure per-key FPR on EVAL text.
# Run from the project root: uv run --no-project --with numpy,scipy python explore/key_screening.py
import json, numpy as np
from scipy.stats import chi2, spearmanr
A = dict(np.load("results/null_h1_lefthash_K1000.npz")); m = json.load(open("results/null_h1_lefthash_K1000.json"))
S, R = m["sets"], m["rules"]; doms = [s.split("/")[0] for s in m["eval_sets"]]
def zbar(d, r): return A["zsum"][:, S.index(f"{d}/{('ref' if d else '')}"), R.index(r)] / m["n"][f"{d}/ref"]
def zb(d, part, r): s = f"{d}/{part}"; return A["zsum"][:, S.index(s), R.index(r)] / m["n"][s]
def disp(c, n):
    p = c.sum() / (len(c) * n); D = ((c - c.mean())**2).sum() / (n*p*(1-p)); return D / (len(c)-1)
for rule in ("uniq", "mask0.01"):
    ri = R.index(rule)
    Zr = np.stack([zb(d, "ref", rule) for d in doms], 1); Ze = np.stack([zb(d, "eval", rule) for d in doms], 1)
    print(f"\n=== rule {rule}: corr over keys of zbar(ref) vs zbar(eval):", {d: round(np.corrcoef(Zr[:, i], Ze[:, i])[0,1], 3) for i, d in enumerate(doms)})
    C = np.corrcoef(Zr.T); print("    corr of zbar(ref) ACROSS domains:\n", np.round(C, 2))
    score = (Zr / Zr.std(0)).max(1)              # worst domain, in sd units, on reference text only
    for keep in (1.0, 0.5, 0.2):
        acc = score <= np.quantile(score, keep)
        print(f"  keep {keep:.0%} of keys ({acc.sum()}):")
        for i, d in enumerate(doms):
            si, n = S.index(f"{d}/eval"), m["n"][f"{d}/eval"]
            row = []
            for a, al in ((0, 1e-2), (1, 1e-3)):
                c = A["bflags"][acc, si, ri, a]; f = c / n
                row.append(f"a={al:g}: pooled/a {f.mean()/al:.2f} p99/a {np.quantile(f,.99)/al:.2f} max/a {f.max()/al:.1f} D {disp(c, n):.2f}")
            pr = A["pred"][acc, m['eval_sets'].index(f'{d}/eval'), ri, 2]
            print(f"    {d:8s} " + " | ".join(row) + f" | pred p99/a@1e-4 {np.quantile(pr,.99)/1e-4:.2f} | sd zbar_eval {Ze[acc, i].std():.3f}")
# single-domain screening (news ref only) -> does it transfer to other domains?
ri = R.index("uniq"); zr = zb("news", "ref", "uniq"); acc = zr <= np.quantile(zr, 0.5)
print("\n=== uniq, screen on NEWS ref only, keep 50%: eval p99/a at 1e-2 per domain (all keys -> kept):")
for d in doms:
    si, n = S.index(f"{d}/eval"), m["n"][f"{d}/eval"]; f = A["bflags"][:, si, ri, 0] / n
    print(f"    {d:8s} {np.quantile(f,.99)/1e-2:.2f} -> {np.quantile(f[acc],.99)/1e-2:.2f}")
