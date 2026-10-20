"""
End-to-end dry run of the pipeline on synthetic data with the PRF colour backend (no torch needed):
data -> run_null (h=1, h=0) -> power -> analyze.   python -m tests.pipeline_synth
"""
import os
import subprocess
import sys

import numpy as np

from kl.common import DOMAINS, GAMMA, VOCAB_SIZE, WINDOW, make_keys, prf_ranks
from tests.test_nullstats import synthetic_domain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("KL_SYNTH_DIR", "/tmp/kl_synth")


def synth_generate(n, key, delta, rng, types, p, succ, plen=100, new=200):
    out = np.empty((n, plen + new), dtype=np.int64)
    for i in range(n):
        seq = list(rng.choice(types, size=plen, p=p))
        for t in range(new):
            prev = seq[-1]
            j = np.searchsorted(types, prev) if prev in set(types) else 0
            cand = np.concatenate([types[succ[j % len(succ)]], rng.choice(types, size=22, p=p)])
            w = np.concatenate([np.full(8, 3.0), np.ones(22)])
            g = prf_ranks(key, np.full(len(cand), prev), cand) < int(GAMMA * VOCAB_SIZE)
            logit = np.log(w) + delta * g
            pr = np.exp(logit - logit.max())
            seq.append(int(rng.choice(cand, p=pr / pr.sum())))
        out[i] = seq
    return out


def main():
    os.makedirs(f"{WORK}/data", exist_ok=True)
    rng = np.random.default_rng(1)
    params = {"news": (4000, 1.0, 0.3), "wiki_en": (5000, 1.0, 0.3), "lists": (1500, 1.1, 0.6),
              "code": (600, 1.2, 0.85), "wiki_ru": (2500, 1.05, 0.5)}
    for d in DOMAINS:
        nt, a, st = params[d]
        x = synthetic_domain(1500, rng, nt, a, st)
        np.savez_compressed(f"{WORK}/data/{d}.npz", ids=x.astype(np.int32), n_eval=1000)
    types = np.sort(rng.choice(np.arange(4, VOCAB_SIZE - 10), size=3000, replace=False))
    p = 1.0 / np.arange(1, 3001) ** 1.0
    p /= p.sum()
    succ = rng.integers(0, 3000, size=(3000, 8))
    keys = make_keys(1000)[:2]
    rows, meta = [], []
    for delta, ks in [(0.0, keys[:1]), (1.0, keys), (2.0, keys)]:
        for k in ks:
            g = synth_generate(60, int(k), delta, rng, types, p, succ)
            rows.append(g)
            meta.append(np.stack([np.full(len(g), delta), np.full(len(g), k), np.arange(len(g))], 1))
    m = np.concatenate(meta)
    np.savez_compressed(f"{WORK}/data/generated.npz", ids=np.concatenate(rows).astype(np.int32), delta=m[:, 0],
                        key=m[:, 1].astype(np.int64), prompt_idx=m[:, 2].astype(np.int64), prompt_len=100)
    env = dict(os.environ, PYTHONPATH=ROOT)
    cmds = [
        [sys.executable, "-m", "kl.run_null", "--h", "1", "--keys", "60", "--backend", "prf", "--workers", "2"],
        [sys.executable, "-m", "kl.run_null", "--h", "0", "--keys", "60", "--backend", "prf", "--workers", "2"],
        [sys.executable, "-m", "kl.power", "--backend", "prf", "--workers", "2", "--null_keys", "40"],
        [sys.executable, "-m", "kl.analyze", "--results", "results", "--out", "report"],
    ]
    for c in cmds:
        print(">>>", " ".join(c[1:]), flush=True)
        r = subprocess.run(c, cwd=WORK, env=env, capture_output=True, text=True)
        print(r.stdout[-2500:])
        if r.returncode != 0:
            print(r.stderr[-4000:])
            raise SystemExit(r.returncode)
    print(sorted(os.listdir(f"{WORK}/report")))


if __name__ == "__main__":
    main()
