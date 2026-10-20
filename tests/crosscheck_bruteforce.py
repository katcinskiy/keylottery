"""
Independent cross-check of the per-text z (CLAUDE.md, Section 6). Imports nothing from kl/:
green lists come from transformers' WatermarkLogitsProcessor at every position, repeated tuples are
removed with a Python set, z is computed in plain Python. Compares with the per-text z that run_null
saved for its first 10 keys (results/null_h1_lefthash_K*_z.npz).

  python -m tests.crosscheck_bruteforce --set code --n 300 --runs results/null_h1_lefthash_K1000_z.npz:0,1,2 \
         results/null_h1_lefthash_K10_lab4keys_z.npz:3,6
Writes results/crosscheck.json.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from transformers.generation.logits_process import WatermarkLogitsProcessor

V, GAMMA, START, WINDOW = 50272, 0.25, 4, 204


def brute_z(text, proc):
    """(z_all, z_uniq) of one 204-token text; scored positions START..WINDOW-1, context = previous token."""
    g_all, seen, g_u = 0, set(), 0
    for i in range(START, WINDOW):
        prev, tok = int(text[i - 1]), int(text[i])
        green = proc._get_greenlist_ids(torch.tensor([prev]))
        g = int(bool((green == tok).any()))
        g_all += g
        if (prev, tok) not in seen:
            seen.add((prev, tok))
            g_u += g
    T_all, T_u = WINDOW - START, len(seen)
    z = lambda G, T: (G - GAMMA * T) / math.sqrt(T * GAMMA * (1 - GAMMA))
    return z(g_all, T_all), z(g_u, T_u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="code")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--runs", nargs="+", required=True, help="path_to_z.npz:i,j,...  (indices into that run's keys)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    d = np.load(f"data/{args.set}.npz")
    texts = d["ids"][: int(d["n_eval"])][: args.n].astype(np.int64)
    out, t0 = [], time.time()
    for spec in args.runs:
        path, idx = spec.split(":")
        meta = json.load(open(path.replace("_z.npz", ".json")))
        Z = np.load(path)
        for i in map(int, idx.split(",")):
            key = int(meta["keys"][i])
            proc = WatermarkLogitsProcessor(vocab_size=V, device="cpu", greenlist_ratio=GAMMA, bias=2.0,
                                            hashing_key=key, seeding_scheme="lefthash", context_width=1)
            zb = np.array([brute_z(x, proc) for x in texts])
            row = dict(run=path, key_index=i, key=key, set=args.set, texts=int(len(texts)))
            for j, r in enumerate(("all", "uniq")):
                zs = Z[f"z|{key}|{args.set}/eval|{r}"][: len(texts)].astype(np.float64)
                row[f"max_abs_dz_{r}"] = float(np.max(np.abs(zs - zb[:, j])))
                row[f"flags_z4_{r}_brute"] = int((zb[:, j] > 4).sum())
                row[f"flags_z4_{r}_saved"] = int((zs > 4).sum())
                row[f"mean_z_{r}"] = float(zb[:, j].mean())
            out.append(row)
            print(json.dumps(row), f"({time.time() - t0:.0f}s)", flush=True)
    res = dict(rows=out, passed=bool(all(r["max_abs_dz_all"] < 1e-4 and r["max_abs_dz_uniq"] < 1e-4 for r in out)),
               tolerance=1e-4, note="saved z is float32")
    with open("results/crosscheck.json", "w") as f:
        json.dump(res, f, indent=1)
    print("passed:", res["passed"])


if __name__ == "__main__":
    main()
