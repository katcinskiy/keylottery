"""
The memoised green lists in kl.generate.CPUGreenListWatermark(cache=True) must give exactly the same logits as the
uncached path (which matches transformers' processor, Lab 4 and kl.checks). CPU only, no model.
  python -m tests.test_generate_cache
"""
import torch

from kl.common import GAMMA, VOCAB_SIZE, make_keys
from kl.generate import CPUGreenListWatermark


def main():
    g = torch.Generator().manual_seed(0)
    for key in make_keys(1000)[:3].tolist():
        plain = CPUGreenListWatermark(VOCAB_SIZE, 1.0, GAMMA, key)
        cached = CPUGreenListWatermark(VOCAB_SIZE, 1.0, GAMMA, key, cache=True, cache_size=50)
        for step in range(30):
            # few distinct last tokens so the cache is hit, and enough to force cache clears (size 50)
            ids = torch.randint(4, 50265, (32, 12), generator=g)
            ids[:, -1] = torch.randint(4, 120, (32,), generator=g)
            scores = torch.randn(32, VOCAB_SIZE, generator=g)
            a, b = plain(ids, scores), cached(ids, scores)
            assert torch.equal(a, b), (key, step)
        print(f"key {key}: identical over 30 steps, cache entries {len(cached._lists)}")
    print("cache test OK")


if __name__ == "__main__":
    main()
