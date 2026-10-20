"""
Key-screening follow-up (CLAUDE.md Section 9): 1000 FRESH keys scored on the reference texts (screening), the old
eval texts E1 and the new far-away texts E2.
  python -m kl.run_screen --workers 32
Writes results/screen_h1_lefthash_K1000.npz (+ .json).
"""
import argparse
import json
import os
import time

import numpy as np

from .common import DOMAINS, make_keys
from .data import load_source
from .data_e2 import load_e2
from . import nullstats as ns

SCREEN_SEED = 2027
EPS_SCREEN = [0.01]


def fresh_keys(n=1000):
    keys = make_keys(n + 1, seed=SCREEN_SEED)[1:]           # drop transformers' default key
    old = set(make_keys(1000).tolist())
    if os.path.exists("lab4_keys.json"):
        old |= set(json.load(open("lab4_keys.json")))
    assert not (set(keys.tolist()) & old), "fresh keys overlap the keys used before"
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--keys", type=int, default=1000)
    ap.add_argument("--backend", default="lefthash", choices=["lefthash", "prf"])
    args = ap.parse_args()
    t0 = time.time()
    sets = {}
    for d in DOMAINS:
        ev, ref = load_source(d)
        sets[f"{d}/ref"], sets[f"{d}/eval"], sets[f"{d}/e2"] = ref, ev, load_e2(d)
        print(f"{d}: ref {ref.shape} eval {ev.shape} e2 {sets[f'{d}/e2'].shape}", flush=True)
        assert len(sets[f"{d}/e2"]) == 6000, f"E2 for {d} has {len(sets[f'{d}/e2'])} texts, pre-registered 6000"
    keys = fresh_keys(args.keys)
    ts = ns.TextSets(sets, 1)
    eval_sets = [f"{d}/{p}" for p in ("eval", "e2") for d in DOMAINS]
    plan = ns.build_plan(ts, eval_sets, [f"{d}/ref" for d in DOMAINS], eps=EPS_SCREEN, backend=args.backend)
    print(f"{len(ts.U):,} distinct tuples; plan {time.time() - t0:.0f}s; {len(keys)} fresh keys "
          f"(seed {SCREEN_SEED}), workers={args.workers}", flush=True)
    res = ns.run_keys(ts, plan, keys, backend=args.backend, workers=args.workers)
    A, zs, meta = ns.stack(res, keys, plan)
    meta.update(backend=args.backend, seconds=round(time.time() - t0, 1), key_seed=SCREEN_SEED,
                distinct_tuples=int(len(ts.U)))
    os.makedirs("results", exist_ok=True)
    out = f"results/screen_h1_{args.backend}_K{len(keys)}.npz"
    ns.save(out, A, zs, meta)
    print(f"saved {out}; total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
