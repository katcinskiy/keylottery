"""
Null experiment driver.  python -m kl.run_null --h 1 --keys 1000 --backend lefthash --workers 32
Writes results/null_h{h}_{backend}_K{keys}.npz (+ .json metadata, + _z.npz per-text z for keys 0..9).
"""
import argparse
import json
import os
import time

import numpy as np

from .common import DOMAINS, make_keys
from .data import load_source
from . import nullstats as ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--keys", type=int, default=1000)
    ap.add_argument("--key_file", default=None, help="optional JSON list of keys (overrides --keys)")
    ap.add_argument("--backend", default="lefthash", choices=["lefthash", "prf"])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    sets = {}
    for d in args.domains:
        ev, ref = load_source(d)
        sets[f"{d}/eval"], sets[f"{d}/ref"] = ev, ref
        print(f"{d}: eval {ev.shape} ref {ref.shape}", flush=True)
    keys = np.array(json.load(open(args.key_file)), dtype=np.int64) if args.key_file else make_keys(args.keys)
    ts = ns.TextSets(sets, args.h)
    eval_sets = [f"{d}/eval" for d in args.domains]
    ref_sets = [f"{d}/ref" for d in args.domains]
    cal_map = {f"{d}/eval": {"same": f"{d}/ref", "news": "news/ref"} for d in args.domains if "news" in args.domains}
    plan = ns.build_plan(ts, eval_sets, ref_sets, cal_map=cal_map, backend=args.backend)
    print(f"h={args.h}: {len(ts.U):,} distinct tuples; plan built in {time.time() - t0:.0f}s; "
          f"{len(keys)} keys, backend={args.backend}, workers={args.workers}", flush=True)
    for s in eval_sets:
        print("  Var_pred", s, {r: round(plan['var_pred'][(s, r)], 4) for r in plan['rules']}, flush=True)
        print("  exact/iid", s, {r: round(plan['var_pred_exact'][(s, r)] / plan['var_pred'][(s, r)], 4)
                                 for r in plan['rules'] if plan['var_pred'][(s, r)] > 0}, flush=True)
    res = ns.run_keys(ts, plan, keys, backend=args.backend, workers=args.workers,
                      z_keys=keys[:10], keep_z_for=eval_sets)
    A, zs, meta = ns.stack(res, keys, plan)
    meta.update(backend=args.backend, seconds=round(time.time() - t0, 1), domains=args.domains,
                distinct_tuples=int(len(ts.U)))
    os.makedirs("results", exist_ok=True)
    out = f"results/null_h{args.h}_{args.backend}_K{len(keys)}{args.tag}.npz"
    ns.save(out, A, zs, meta)
    # tuple frequencies for the power experiment (only tuples that can be masked at all)
    if args.h == 1 and args.backend == "lefthash":
        sel = plan["df_ref"] > min(plan["eps"])
        np.savez_compressed("results/df_ref_h1.npz", codes=ts.U[sel], df=plan["df_ref"][sel].astype(np.float32))
    print(f"saved {out}; total {time.time() - t0:.0f}s; mean per-key colour {A['time_colour'].mean():.2f}s, "
          f"stats {A['time_stats'].mean():.2f}s", flush=True)


if __name__ == "__main__":
    main()
