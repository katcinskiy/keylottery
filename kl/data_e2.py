"""
Fresh evaluation texts E2 for the key-screening follow-up (CLAUDE.md Section 9): 6000 texts per source from far-away
parts of the same sources, with the same filters, tokenizer and 204-token window as kl.data.
  python -m kl.data_e2 [names...]     -> data/<name>_e2.npz (+ .json)
"""
import json
import os
import sys
import time

import numpy as np

from .common import WINDOW
from .data import SOURCES, TOKENIZER

N_E2 = 6000
E2 = {
    # name: (HF dataset, config, split, records to skip, max records to scan after the skip)
    "news":    ("allenai/c4", "realnewslike", "train", 100_000, 400_000),
    "wiki_en": ("wikimedia/wikipedia", "20231101.en", "train", 400_000, 400_000),
    "lists":   ("wikimedia/wikipedia", "20231101.en", "train", 1_000_000, 3_000_000),
    "code":    ("codeparrot/codeparrot-clean-valid", None, "train", 30_000, 400_000),
    "wiki_ru": ("wikimedia/wikipedia", "20231101.ru", "train", 400_000, 400_000),
}


def build_e2(name, tok, n=N_E2, out_dir="data"):
    path = f"{out_dir}/{name}_e2.npz"
    if os.path.exists(path):
        return path
    from datasets import load_dataset
    ds_name, config, split, skip, max_scan = E2[name]
    field, keep = SOURCES[name][3], SOURCES[name][4]
    # skip by iterating: IterableDataset.skip(N) on titan skipped only ~1000 records (CLAUDE.md Section 9a)
    it, t0 = iter(load_dataset(ds_name, config, split=split, streaming=True)), time.time()
    for _ in range(skip):
        next(it)
    print(f"{name}: skipped {skip} records in {time.time() - t0:.0f}s", flush=True)
    # texts whose 204-token window also occurs in the main stream (eval + ref) are skipped: templated files (licence
    # headers, stubs) repeat verbatim; the first fresh build had 142 such windows in code and 14 in wiki_ru (9a)
    main_rows = {r.tobytes() for r in np.load(f"{out_dir}/{name}.npz")["ids"].astype(np.int32)}
    ids, meta, scanned, dropped = [], [], 0, 0
    for rec in it:
        scanned += 1
        if scanned > max_scan:
            break
        if keep is not None and not keep(rec):
            continue
        toks = tok(rec[field][:20_000], add_special_tokens=False)["input_ids"]
        if len(toks) < WINDOW:
            continue
        if np.asarray(toks[:WINDOW], dtype=np.int32).tobytes() in main_rows:
            dropped += 1
            continue
        ids.append(toks[:WINDOW])
        meta.append(rec.get("title") or rec.get("url") or rec.get("path") or "")
        if len(ids) == n:
            break
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids, dtype=np.int32))
    with open(f"{out_dir}/{name}_e2.json", "w") as f:
        json.dump(dict(source=name, dataset=ds_name, config=config, split=split, skipped=skip, n=len(ids),
                       dropped_shared_windows=dropped,
                       records_scanned=scanned, seconds=round(time.time() - t0, 1), items=meta), f)
    return path


def check_fresh(name, out_dir="data"):
    """E2 must not overlap the main stream data/<name>.npz (eval + ref) (CLAUDE.md 9a): no shared 204-token window,
    and no shared title/url, except for code, whose items are in-repo paths (setup.py, ...) that repeat across repos."""
    main = np.load(f"{out_dir}/{name}.npz")["ids"]
    e2 = np.load(f"{out_dir}/{name}_e2.npz")["ids"]
    rows = {r.tobytes() for r in main}
    shared_rows = int(sum(r.tobytes() in rows for r in e2))
    items = set(json.load(open(f"{out_dir}/{name}.json"))["items"])
    shared_items = int(sum(i in items for i in json.load(open(f"{out_dir}/{name}_e2.json"))["items"]))
    print(f"{name}_e2: {len(e2)} texts, shared rows {shared_rows}, shared titles/paths {shared_items}", flush=True)
    return len(e2) == N_E2 and shared_rows == 0 and (name == "code" or shared_items == 0)


def load_e2(name, out_dir="data"):
    return np.load(f"{out_dir}/{name}_e2.npz")["ids"].astype(np.int64)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        ok = [check_fresh(n) for n in sys.argv[2:] or list(E2)]
        sys.exit(0 if all(ok) else 1)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    for name in sys.argv[1:] or list(E2):
        t0 = time.time()
        build_e2(name, tok)
        print(f"{name}_e2: {load_e2(name).shape} in {time.time() - t0:.0f}s", flush=True)
