# Pre-registration

The sections below are copied verbatim from the project's working notes (`CLAUDE.md`, not published), where they were
written before the corresponding runs:

- **Section 6** (claims C0–C9 and their pass/fail criteria): 3 October 2026, before the main runs.
- **Section 9** (key screening, hypotheses S1–S5): 4 October 2026, before the screening runs; **amendment 9a** (S6, S7,
  the E2 freshness bug and fix): 4 October 2026, after the first screening run and before any fresh E2 text existed.
- **Section 10** (randomized key pools): 5 October 2026, explicitly *not* pre-registered; reported as a post-hoc baseline.

This repository was created on 20 October 2026, so its git history does not timestamp these texts; the order of events
is recorded in the texts themselves. Verdicts computed by code written before the results: `report/claims.json`
(C0–C9, `kl/claims.py`) and `report/screening.json` (S1–S7, `kl/screening.py`). Later analyses not listed here are
post hoc and labelled as such (e.g. `explore/window_sweep.py`).

Notation used below: E1 = the original unseen evaluation texts, E2 = fresh texts from far-away parts of the same
sources, "uniq" = Three Bricks scoring (each (context, token) pair counted once, exact binomial test).

## 6. Claims and pass/fail criteria (fixed before the real runs)

Main configuration: h=1, lefthash, K=1000 keys, n=6000 eval texts per domain. "Per-key FPR" = flags / n
for one key.

- **C0. Implementation.** `checks.json`: identical green lists for all contexts and keys;
  max |Δz| vs WatermarkDetector < 1e-6; the first 2,000 texts are identical to Lab 4; the Section 4
  regression matches exactly.
- **C1. All-token scoring fails; uniq holds marginally.** Pooled-over-keys FPR of `all` is ≫ α on code,
  lists and wiki_ru. For `uniq` with the exact binomial test, the pooled FPR ≤ α within Poisson error
  (one-sided p > 0.01) at α = 1e-2, 1e-3, 1e-4 in every domain.
- **C2. Key lottery under uniq.** For at least code (expected also lists and wiki_ru): per-key counts are
  overdispersed (`dispersion_p` < 0.01 in `*_perkey.csv`), the fraction of keys significantly above the
  pooled rate is > 1%, and the predicted 99th-percentile key FPR/α ≥ 2 at α ≤ 1e-4. If this fails
  everywhere, the main claim is **refuted** and the honest story becomes "uniq is enough". Report that.
- **C3. Variance theory.** Measured/predicted Var$_k(\bar z)$ lies inside the 95% χ² band (or within
  ±10%) for every set × rule, for h=1 lefthash, h=0 and prf. List any exceptions.
- **C4. Predictor.** For uniq and mask rules at α = 1e-2 and 1e-3: |total_z| < 3 (measured vs predicted
  total flags), and per-key correlation > 0.5 where keys actually differ (dispersion significant).
- **C5. Masking removes the lottery.** For ε = 0.01 (and 0.003): measured Var$_k(\bar z)$ ≤ ε, dispersion
  not significant, and predicted 99th-percentile key FPR/α ≤ 1.5 at α = 1e-4 … 1e-6, in every domain.
- **C6. Power cost.** Report the TPR loss of mask ε vs uniq at δ=1 and δ=2, α=1e-4, ρ=0 and 0.2. No
  pre-set threshold; it is a Pareto trade-off, so state it plainly.
- **C7. Calibration baseline.** Same-domain calibration fixes the per-key mean, but news-calibrated
  (cross-domain) p95 key FPR/α ≫ 1 on code and lists.
- **C8. Not an RNG artefact.** prf (idealised hash) gives the same qualitative picture as lefthash (Var
  ratios, dispersion verdicts).
- **C9. h=0 is the extreme case.** With a fixed green list, uniq still has a huge per-key spread.

Independent cross-check (do at least one): for 2–3 keys, recompute the per-text uniq z on a few hundred
code texts with a separate, brute-force path (e.g. transformers `WatermarkLogitsProcessor` green lists per
position plus a Python set for dedup) inside a Slurm job, and compare with `_z.npz`.

## 9. Follow-up: key screening (pre-registered 2026-10-04, before any of these runs)

Context. The Section-6 runs refuted C5 (masking does not remove the lottery). A post-hoc look at the existing arrays
(`explore/key_screening.py`, old 1000 keys, old eval texts E1) suggested **key screening**: the per-key corpus-mean z on
the reference texts predicts it on new texts of the same domain (corr 0.87-0.996 for uniq), so a provider can reject
bad keys before deployment. That look used the old keys and E1 and also computed cross-domain correlations, so the
tests below use **fresh keys** and **fresh texts**, and S4 is marked as not blind.

Setup (fixed now):
- Fresh keys: `make_keys(1001, seed=2027)[1:]` (1000 keys; must be disjoint from the main 1000 keys and the Lab 4 keys).
- Screening reference: the existing ref sets (3000 texts per domain, same as the masking reference).
- Test texts: E1 = the existing eval sets; **E2 = 6000 new texts per domain from far-away parts of the sources**:
  news = C4 realnewslike *train* after skipping 100,000 records; wiki_en = Wikipedia 20231101.en after skipping 400,000
  records (non-"List of"); lists = the same dump after skipping 1,000,000 records ("List of" only); code =
  codeparrot-clean-valid after skipping 30,000 records; wiki_ru = 20231101.ru after skipping 400,000 records.
  Same tokenizer, filters and 204-token window as Section 1.
- Screening score of key k: s(k) = max over the 5 domains d of zbar_ref(k, d) / sd_k(zbar_ref(., d)), rule uniq
  (for S5: rule mask0.01). Kept set at fraction q: keys with s(k) <= the q-quantile of s. Primary q = 0.2 (costs
  log2(5) = 2.3 bits of key entropy); q = 0.5 reported.
- Ideal-key floor F(n, alpha) = (99th percentile of Binomial(n, alpha)) / (n alpha).

Hypotheses and pass/fail:
- **S1 (screening transfers to new texts).** q = 0.2, rule uniq, exact binomial test, E2: in every domain at
  alpha = 1e-2 and 1e-3, the 99th percentile over kept keys of per-key FPR / alpha <= F(6000, alpha).
  Supported if 10/10 cells pass, partial if >= 8/10, refuted otherwise. Also reported: E1, q = 0.5, all keys, max.
- **S2 (same mechanism on fresh keys and texts).** corr over the fresh keys of zbar_ref(k, d) vs zbar_E2(k, d), uniq,
  >= 0.8 in every domain. Supported if 5/5, partial if 4/5.
- **S3 (power cost is smaller than masking's).** OPT-1.3B, delta = 1, 400 prompts, 10 keys drawn uniformly from the
  kept set (q = 0.2) and 10 keys drawn uniformly from all fresh keys (numpy default_rng(7); screened draw first).
  Rule uniq, alpha = 1e-4, rho in {0, 0.2}. Delta_TPR = mean TPR(random keys) - mean TPR(screened keys), 95% bootstrap
  CI over keys. Supported if at rho = 0 the upper CI bound is < 8.7 pp (the measured loss of mask0.01 vs uniq at
  delta = 1, rho = 0); refuted if the point estimate is >= 8.7 pp; partial otherwise.
- **S4 (coverage; NOT blind).** Leave one domain out: screen on the other 4 (q = 0.2), report p99/alpha at 1e-2 and
  1e-3 on the held-out domain's E2 next to all keys. Expected: little or no gain on code and wiki_ru. No pass/fail;
  this is the attack surface (text from a domain the provider did not cover).
- **S5 (screening + masking, secondary).** As S1 but with rule mask0.01 and its own score. Reported, no pass/fail.

Analysis code (`kl/screening.py`) is written before the runs; any later change will be stated with its reason.

### 9a. Amendment (2026-10-04, after the first screening run, BEFORE any fresh E2 exists)

- **Bug.** `load_dataset(..., streaming=True).skip(N)` on titan skipped only ~1000 records, so the first "E2" was a
  shifted window of the main stream for wiki_en, lists, code and wiki_ru (6000/6000 texts shared with E1+ref; found by
  an independent check). Only news E2 was new (other split). The first S1/S2/S4/S5 numbers are therefore NOT a test on
  fresh texts; they are kept as `*_staleE2*` files. Fix: skip by iterating, then refuse to run if any E2 text (token
  row or title/url/path) occurs in data/<d>.npz. Keys, reference sets, the generation and S3 do not depend on E2 and
  are kept.
- **S6 (new, pre-registered now).** Flag-based screening score: s_F(k) = max_d (c_ref(k,d) - mean_k c_ref(., d)) /
  sd_k c_ref(., d), with c_ref the exact-binomial flags of rule uniq at alpha = 1e-2 on d/ref; keep q = 0.2. Same
  pass/fail as S1 on fresh E2. (Motivation, post hoc on E1/stale E2: the mean-z score misses template clusters in
  code, e.g. Apache/GPL headers; not blind on E1.)
- **S7 (new).** Does screening remove the lottery or only pick conservative keys? Among kept keys (S1 score, q = 0.2),
  index of dispersion of per-key flag counts on fresh E2 at alpha = 1e-2 (as in kl/analyze.dispersion) <= 1.5 in every
  domain => "removes the lottery"; otherwise "shifts keys to conservative ones". Also reported for S6 and S5 kept sets
  and for all keys.
- S3 caveat (independent check): the 10 random keys happen to be unusually low in zbar_ref(news) (p = 0.006); a
  regression of TPR on zbar_ref(news) gives a population screening cost of about 3.9 pp at rho = 0. Reported next to
  the pre-registered verdict.
- Freshness rule as implemented (decided after the first fresh build failed the check, before any analysis on it):
  no E2 204-token window may equal a window of the main stream; such verbatim windows occur for templated files
  (142 in code, 14 in wiki_ru on that build) and are skipped while building, the next texts are taken instead.
  Titles/urls must not repeat (not checked for code, whose items are in-repo paths such as setup.py).

## 10. Post-hoc baseline run (2026-10-05, NOT pre-registered): randomized key pools

Nils's key-pool method (Aremu et al., arXiv:2507.07871v5; PDF in `context/`) was added to the proposal table as a union
bound only. On 5 Oct it was run for real: `explore/key_pool.py` (prf colours, all 5 domains) and
`explore/key_pool_lefthash.py` (exact transformers colours, code and wiki_ru). Rule as in the paper's Algorithm 1: pool of
r keys, common Sidak-corrected level alpha_k = 1-(1-alpha_fw)^(1/r), per-key exact binomial test (same test as every
other baseline), a text counts as genuine iff exactly one key fires. 1000 fresh keys (seed 2027), E2 texts, 20000 random
pools. Result (alpha_fw = 1e-3, r = 10, p99 over pools / alpha_fw): prf colours: news 1.7, wiki_en 2.0, lists 1.8, code 3.0,
wiki_ru 10.7; exact lefthash colours (code, wiki_ru only): code 2.67 (max 22.3, the licence-header key), wiki_ru 3.00
(max 4.8). prf overstates the Russian tail (few contexts, fixed-size green lists matter there); the slide uses the
lefthash numbers for code and Russian. r = 4 (lefthash): code 2.8, wiki_ru 5.2. Pool means 0.3-0.8 alpha. TPR of the
pool = TPR at alpha/r (0.66 at r = 10 from power_screen.json). The earlier "1.8-3.0x" union bound was therefore right.
Not pre-registered; reported as a baseline, no pass/fail.
