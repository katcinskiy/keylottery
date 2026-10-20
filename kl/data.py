"""
Human-written text (the null) and prompts for generation.

Same sources, filters and order as Lab 04 (data.py there), so the first 2,000 evaluation texts of each
source are exactly the Lab 04 texts. Each source is split into
  eval: the first N_EVAL qualifying documents  (used to measure false positives)
  ref:  the next  N_REF  qualifying documents  (the provider's "public" reference text: tuple
        frequencies for masking, and per-key calibration)
Every text is cut to its first WINDOW OPT tokens (no special tokens); shorter documents are skipped.
"""
import json
import os
import sys
import time

import numpy as np

from .common import WINDOW

TOKENIZER = "facebook/opt-1.3b"
N_EVAL, N_REF = 6000, 3000

SOURCES = {
    # name: (HF dataset, config, split, text field, filter on the record, max records to scan)
    "news":    ("allenai/c4", "realnewslike", "validation", "text", None, 400_000),
    "wiki_en": ("wikimedia/wikipedia", "20231101.en", "train", "text",
                lambda r: not r["title"].startswith("List of"), 400_000),
    "lists":   ("wikimedia/wikipedia", "20231101.en", "train", "text",
                lambda r: r["title"].startswith("List of"), 3_000_000),
    "code":    ("codeparrot/codeparrot-clean-valid", None, "train", "content", None, 400_000),
    "wiki_ru": ("wikimedia/wikipedia", "20231101.ru", "train", "text", None, 400_000),
}


def build_source(name, tok, n_total=N_EVAL + N_REF, out_dir="data"):
    path = f"{out_dir}/{name}.npz"
    if os.path.exists(path):
        return path
    from datasets import load_dataset
    ds_name, config, split, field, keep, max_scan = SOURCES[name]
    ds = load_dataset(ds_name, config, split=split, streaming=True)
    ids, meta, scanned, t0 = [], [], 0, time.time()
    for rec in ds:
        scanned += 1
        if scanned > max_scan:
            break
        if keep is not None and not keep(rec):
            continue
        toks = tok(rec[field][:20_000], add_special_tokens=False)["input_ids"]
        if len(toks) < WINDOW:
            continue
        ids.append(toks[:WINDOW])
        meta.append(rec.get("title") or rec.get("url") or rec.get("path") or "")
        if len(ids) == n_total:
            break
    os.makedirs(out_dir, exist_ok=True)
    n = len(ids)
    n_eval = min(N_EVAL, n - min(N_REF, n // 3))   # keep at least a third for reference if short
    np.savez_compressed(path, ids=np.array(ids, dtype=np.int32), n_eval=n_eval)
    with open(f"{out_dir}/{name}.json", "w") as f:
        json.dump(dict(source=name, dataset=ds_name, config=config, split=split, n=n, n_eval=int(n_eval),
                       records_scanned=scanned, seconds=round(time.time() - t0, 1), items=meta), f)
    return path


def load_source(name, out_dir="data"):
    d = np.load(f"{out_dir}/{name}.npz")
    ids, n_eval = d["ids"].astype(np.int64), int(d["n_eval"])
    return ids[:n_eval], ids[n_eval:]


def build_prompts(tok, n=400, plen=100, out="data/prompts.npz"):
    """BOS + the first plen-1 tokens of C4 realnewslike *train* documents (disjoint from the null texts)."""
    if os.path.exists(out):
        return out
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    rows = []
    for rec in ds:
        t = tok(rec["text"][:5_000], add_special_tokens=False)["input_ids"]
        if len(t) >= plen - 1:
            rows.append([tok.bos_token_id] + t[:plen - 1])
        if len(rows) == n:
            break
    np.savez_compressed(out, ids=np.array(rows, dtype=np.int32))
    return out


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    names = sys.argv[1:] or list(SOURCES) + ["prompts"]
    for name in names:
        t0 = time.time()
        if name == "prompts":
            build_prompts(tok)
            print("prompts:", np.load("data/prompts.npz")["ids"].shape, flush=True)
            continue
        build_source(name, tok)
        ev, ref = load_source(name)
        print(f"{name}: eval {ev.shape}, ref {ref.shape} in {time.time() - t0:.0f}s", flush=True)
