"""
Correctness checks that need torch/transformers (run on the cluster):
  1. our lefthash colours == transformers' WatermarkLogitsProcessor green lists (several keys)
  2. our all-token z == transformers' WatermarkDetector z on human text
  3. our first 2,000 eval texts == the Lab 04 texts (if the Lab 04 data cache is given)
  python -m kl.checks [--lab4_data /path/to/lab04/data]
Writes results/checks.json.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoConfig, WatermarkingConfig
from transformers.generation import WatermarkDetector
from transformers.generation.logits_process import WatermarkLogitsProcessor

from .common import DEFAULT_KEY, DOMAINS, GAMMA, START, VOCAB_SIZE, Colourer, make_keys, tuple_codes
from .data import TOKENIZER, load_source


def check_colours(keys, n_ctx=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in keys:
        proc = WatermarkLogitsProcessor(vocab_size=VOCAB_SIZE, device="cpu", greenlist_ratio=GAMMA, bias=2.0,
                                        hashing_key=int(k), seeding_scheme="lefthash", context_width=1)
        ctxs = rng.integers(4, 50265, size=n_ctx)
        codes = (ctxs[:, None] * VOCAB_SIZE + np.arange(VOCAB_SIZE)[None, :]).ravel()
        green = Colourer(codes, 1, backend="lefthash").green(int(k)).reshape(n_ctx, VOCAB_SIZE)
        agree = 0
        for i, c in enumerate(ctxs):
            ref = proc._get_greenlist_ids(torch.tensor([1, int(c)]))
            mine = np.flatnonzero(green[i])
            agree += int(set(ref.tolist()) == set(mine.tolist()))
        out.append(dict(key=int(k), contexts=int(n_ctx), identical_green_lists=agree))
    return out


def check_detector(n=300, key=DEFAULT_KEY):
    cfg = AutoConfig.from_pretrained(TOKENIZER)
    wcfg = WatermarkingConfig(greenlist_ratio=GAMMA, bias=2.0, hashing_key=int(key), seeding_scheme="lefthash",
                              context_width=1)
    det = WatermarkDetector(model_config=cfg, device="cpu", watermarking_config=wcfg, ignore_repeated_ngrams=False)
    ev, _ = load_source("news")
    X = ev[:n]
    # transformers scores every token after the first context_width tokens of the given sequence;
    # feed tokens START-1 .. end so exactly the 200 positions we score are scored with the same contexts
    res = det(torch.tensor(X[:, START - 1:]), return_dict=True)
    z_ref = np.asarray(res.z_score, dtype=float)
    codes = tuple_codes(X, 1)
    U = np.unique(codes)
    green = Colourer(U, 1, backend="lefthash").green(int(key))
    G = green[np.searchsorted(U, codes)].sum(1)
    T = codes.shape[1]
    z = (G - GAMMA * T) / np.sqrt(T * GAMMA * (1 - GAMMA))
    return dict(texts=int(n), max_abs_dz=float(np.max(np.abs(z - z_ref))), mean_z=float(z.mean()))


def check_lab4_texts(lab4_dir):
    out = {}
    for d in DOMAINS:
        p = os.path.join(lab4_dir, f"{d}.npz")
        if not os.path.exists(p):
            out[d] = "missing"
            continue
        old = np.load(p)["ids"].astype(np.int64)
        ev, _ = load_source(d)
        n = min(len(old), len(ev))
        out[d] = dict(lab4=int(len(old)), compared=int(n), identical=bool(np.array_equal(old[:n], ev[:n])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab4_data", default=None)
    args = ap.parse_args()
    res = dict(colours=check_colours(make_keys(1000)[:4]), detector=check_detector())
    if args.lab4_data:
        res["lab4_texts"] = check_lab4_texts(args.lab4_data)
    os.makedirs("results", exist_ok=True)
    with open("results/checks.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
