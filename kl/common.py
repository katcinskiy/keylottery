"""
Shared constants and the KGW green-list colour function.

A "tuple" is (context, token). For context width h = 1 the context is the previous token, so a
tuple is a token pair; for h = 0 there is no context and a tuple is just the token.

Colour of a tuple under key k (green / red) follows transformers' KGW implementation
("lefthash", the default):  seed = k * previous_token  (h = 1)   or   seed = k  (h = 0),
perm = torch.randperm(V) from a CPU torch.Generator seeded with seed % (2**64 - 1),
green  <=>  rank of the token in perm  <  int(gamma * V).

Two colour backends:
  * "lefthash": the exact transformers scheme above (needs torch); used for all real runs.
  * "prf": an idealised random function (splitmix64 hash of key, context, token); numpy only.
    Used for unit tests and as a cross-check that results do not depend on torch's RNG.
"""
import numpy as np

VOCAB_SIZE = 50272          # OPT model.config.vocab_size (the tokenizer has 50265 entries)
DEFAULT_KEY = 15485863      # transformers' default hashing_key
START = 4                   # scored positions start here (4 context tokens, as in Lab 4)
WINDOW = 204                # 4 context tokens + 200 scored tokens
GAMMA = 0.25
DOMAINS = ["news", "wiki_en", "lists", "code", "wiki_ru"]
U64 = np.uint64


def make_keys(n, seed=2026):
    """Key 0 is transformers' default; the rest are distinct random 31-bit integers.
    31 bits keep key * token below 2**47, so the seed never wraps (same as transformers' Python ints)."""
    rng = np.random.default_rng(seed)
    keys, seen = [DEFAULT_KEY], {DEFAULT_KEY}
    while len(keys) < n:
        k = int(rng.integers(1, 2**31 - 1))
        if k not in seen:
            keys.append(k)
            seen.add(k)
    return np.array(keys, dtype=np.int64)


def tuple_codes(texts, h, start=START):
    """Integer code of the tuple at every scored position. texts: (n, L) int array.
    h = 1: code = prev * V + token;  h = 0: code = token.  Returns (n, L - start) int64."""
    texts = np.asarray(texts, dtype=np.int64)
    tok = texts[:, start:]
    if h == 0:
        return tok.copy()
    if h == 1:
        return texts[:, start - 1:-1] * VOCAB_SIZE + tok
    raise ValueError("only h in {0, 1} is supported here")


def split_code(code, h):
    """(context, token) of tuple codes (context = -1 for h = 0)."""
    code = np.asarray(code, dtype=np.int64)
    if h == 0:
        return np.full_like(code, -1), code
    return code // VOCAB_SIZE, code % VOCAB_SIZE


# ------------------------------------------------------------------ colour backends

def _splitmix64(x):
    x = (x + U64(0x9E3779B97F4A7C15)) & U64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = ((z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)) & U64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)) & U64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> U64(31))


def prf_ranks(key, ctx, tok, vocab_size=VOCAB_SIZE):
    """Idealised colours: a pseudo-random 'rank' in [0, V) for each (key, ctx, tok).
    green <=> rank < int(gamma V), so P(green) = int(gamma V) / V for every tuple, independently."""
    with np.errstate(over="ignore"):
        a = _splitmix64(U64(key) ^ _splitmix64(np.asarray(ctx).astype(np.int64).astype(U64) + U64(7)))
        b = _splitmix64(a ^ (np.asarray(tok).astype(np.int64).astype(U64) * U64(0x2545F4914F6CDD1D)))
    return (b % U64(vocab_size)).astype(np.int32)


class LeftHashColours:
    """Exact transformers lefthash ranks for many (context, token) tuples under one key.
    Tuples are grouped by context once; each distinct context costs one torch.randperm."""

    def __init__(self, ctx_sorted, starts, counts, tok_sorted, h, vocab_size=VOCAB_SIZE):
        import torch
        self.torch = torch
        self.ctx = ctx_sorted          # distinct contexts (int64), -1 for h = 0
        self.starts, self.counts = starts, counts
        self.tok = torch.from_numpy(np.ascontiguousarray(tok_sorted, dtype=np.int64))
        self.h, self.V = h, vocab_size
        self.arange = torch.arange(vocab_size, dtype=torch.int32)

    def ranks(self, key):
        torch = self.torch
        out = torch.empty(len(self.tok), dtype=torch.int32)
        g = torch.Generator(device="cpu")
        rank = torch.empty(self.V, dtype=torch.int32)
        for c, s, n in zip(self.ctx.tolist(), self.starts.tolist(), self.counts.tolist()):
            seed = int(key) if self.h == 0 else int(key) * int(c)
            g.manual_seed(seed % (2**64 - 1))
            perm = torch.randperm(self.V, generator=g)
            rank[perm] = self.arange
            out[s:s + n] = rank[self.tok[s:s + n]]
        return out.numpy()


def group_by_context(codes, h):
    """Sort distinct tuple codes by context. Returns (order, ctx_distinct, starts, counts, tok_sorted)
    so that ranks computed in sorted order can be scattered back with ranks[order] = sorted_ranks."""
    ctx, tok = split_code(codes, h)
    order = np.lexsort((tok, ctx))
    c_sorted = ctx[order]
    uniq, starts, counts = np.unique(c_sorted, return_index=True, return_counts=True)
    return order, uniq, starts, counts, tok[order]


class Colourer:
    """ranks(key) -> rank of every distinct tuple (in the order of `codes`)."""

    def __init__(self, codes, h, backend="lefthash", vocab_size=VOCAB_SIZE):
        self.codes, self.h, self.backend, self.V = np.asarray(codes, dtype=np.int64), h, backend, vocab_size
        if backend == "lefthash":
            self.order, ctx, starts, counts, tok = group_by_context(self.codes, h)
            self.impl = LeftHashColours(ctx, starts, counts, tok, h, vocab_size)
        elif backend == "prf":
            self.ctx, self.tok = split_code(self.codes, h)
        else:
            raise ValueError(backend)

    def ranks(self, key):
        if self.backend == "prf":
            return prf_ranks(key, self.ctx, self.tok, self.V)
        sorted_ranks = self.impl.ranks(key)
        out = np.empty_like(sorted_ranks)
        out[self.order] = sorted_ranks
        return out

    def green(self, key, gamma=GAMMA):
        return self.ranks(key) < int(gamma * self.V)
