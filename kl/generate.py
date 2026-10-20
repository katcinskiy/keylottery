"""
Watermarked text for the power side: OPT-1.3B with transformers' KGW processor (lefthash, h = 1,
gamma = 0.25), green lists computed on the CPU so they match our CPU colours exactly (torch.randperm
differs between CUDA and CPU for the same seed; see Lab 04).

  python -m kl.generate --n_keys 5 --deltas 0 1 2 --n_prompts 400
Writes data/generated.npz: ids (rows, prompt_len + 200), delta, key, prompt_idx.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from transformers.generation.logits_process import WatermarkLogitsProcessor

from .common import GAMMA, make_keys

MODEL = "facebook/opt-1.3b"


class CPUGreenListWatermark(WatermarkLogitsProcessor):
    """transformers' KGW processor with the green lists computed on the CPU (as in Lab 04).
    cache=True memoises the green list per previous token: with lefthash and context width 1 the list depends only
    on (key, previous token), so the output is identical and generation is several times faster."""

    def __init__(self, vocab_size, bias, greenlist_ratio, hashing_key, cache=False, cache_size=4096):
        super().__init__(vocab_size=vocab_size, device="cpu", greenlist_ratio=greenlist_ratio,
                         bias=bias, hashing_key=hashing_key, seeding_scheme="lefthash", context_width=1)
        self.cache, self.cache_size, self._lists = cache, cache_size, {}

    def green_ids(self, seq, device):
        if not self.cache:
            return self._get_greenlist_ids(seq).to(device)
        prev = int(seq[-1])
        g = self._lists.get(prev)
        if g is None:
            if len(self._lists) >= self.cache_size:
                self._lists.clear()
            g = self._lists[prev] = self._get_greenlist_ids(seq).to(device)
        return g

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        for b, seq in enumerate(input_ids.cpu()):
            scores[b, self.green_ids(seq, scores.device)] += self.bias
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_keys", type=int, default=5)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_prompts", type=int, default=400)
    ap.add_argument("--new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--out", default="data/generated.npz")
    ap.add_argument("--key_file", default=None, help="JSON list of keys (overrides --n_keys)")
    ap.add_argument("--cache", action="store_true", help="memoise green lists per previous token (same output)")
    args = ap.parse_args()
    if os.path.exists(args.out):
        print("exists:", args.out)
        return
    prompts = np.load("data/prompts.npz")["ids"][:args.n_prompts]
    keys = np.array(json.load(open(args.key_file)), dtype=np.int64) if args.key_file else make_keys(1000)[:args.n_keys]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    V = model.config.vocab_size
    rows, meta_rows, t0 = [], [], time.time()
    jobs = [(0.0, int(keys[0]))] if 0 in args.deltas else []
    jobs += [(float(d), int(k)) for d in args.deltas if d > 0 for k in keys]
    for delta, key in jobs:
        torch.manual_seed(1234)                      # same sampling noise for every (delta, key)
        procs = LogitsProcessorList([CPUGreenListWatermark(V, delta, GAMMA, key, cache=args.cache)]) if delta > 0 else None
        for i in range(0, len(prompts), args.batch_size):
            batch = torch.tensor(prompts[i:i + args.batch_size], dtype=torch.long).cuda()
            with torch.no_grad():
                gen = model.generate(batch, attention_mask=torch.ones_like(batch), logits_processor=procs,
                                     do_sample=True, top_k=0, top_p=1.0, temperature=1.0,
                                     max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                                     pad_token_id=tok.pad_token_id)
            g = gen.cpu().numpy()
            rows.append(g)
            meta_rows.append(np.stack([np.full(len(g), delta), np.full(len(g), key),
                                       np.arange(i, i + len(g))], 1))
            print(f"delta={delta} key={key} batch {i // args.batch_size}: {time.time() - t0:.0f}s", flush=True)
    m = np.concatenate(meta_rows)
    np.savez_compressed(args.out, ids=np.concatenate(rows).astype(np.int32), delta=m[:, 0],
                        key=m[:, 1].astype(np.int64), prompt_idx=m[:, 2].astype(np.int64),
                        prompt_len=prompts.shape[1])
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump(dict(model=MODEL, keys=[int(k) for k in keys], deltas=args.deltas, gamma=GAMMA,
                       n_prompts=int(len(prompts)), new_tokens=args.new_tokens,
                       sampling="multinomial, temperature 1, no top-k/top-p, EOS suppressed",
                       seconds=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0),
                       gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       gpu_peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2)), f)


if __name__ == "__main__":
    main()
