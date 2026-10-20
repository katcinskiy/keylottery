# Key lottery: per-key false positives of LLM text watermarks

Course project for ML8502 Machine Learning Security (MBZUAI, Fall 2026). Proposal: [`proposal/proposal.pdf`](proposal/proposal.pdf).

The false-positive promise of keyed LLM watermarks (KGW and relatives) holds over a *random* key. A deployed detector
uses *one* key, and under it the false-positive rate on human text is a lottery: with Three Bricks scoring
(each (context, token) pair counted once, exact binomial test), window h = 1 and the OPT tokenizer, the FPR averaged
over 1000 keys stays below the promised alpha = 1e-3, but the 1-in-100 key flags 2.7–3.0x the promised rate on
English text, 5.7x on code and 9.9x on Russian Wikipedia. The project explains this exactly (variance over keys of the
corpus-mean z), predicts a key's FPR from public text, and tests an audit that deploys only keys that pass, together
with an adaptive attack on that audit.

## Layout

| path | what |
|---|---|
| `kl/` | library: keys and green lists (`common.py`, exact transformers `lefthash` colours and an idealised hash), human-text data (`data.py`, `data_e2.py`), null statistics, predictor and variance theory (`nullstats.py`), generation with OPT-1.3B (`generate.py`), power (`power.py`), checks against transformers (`checks.py`), claim verdicts (`claims.py`), key screening (`screening.py`), tables and figures |
| `tests/` | synthetic tests of the variance formula, predictor and masking; brute-force cross-check against transformers |
| `slurm/` | the Slurm jobs that produced every result (`submit.sh` chains data → null ∥ generation → power) |
| `explore/` | post-hoc analyses and figure scripts (key pools, window-size sweep, pitch/proposal figures) |
| `results/`, `report/` | result summaries (JSON/CSV) and figures; large per-key arrays (`.npz`) are not in the repository |
| `proposal/` | the proposal (LaTeX source, figures, PDF; ICML 2026 template) |
| `PREREGISTRATION.md` | claims and pass/fail criteria, written before the runs |

## Reproduce

Environment: Python 3.12, PyTorch 2.11, transformers 5.7, datasets; numpy, scipy and matplotlib for the analysis.

- Everything runs as Slurm jobs; `slurm/env.sh` sets the Python path and caches (adjust `KL_PY`, `KL_HF_HOME` and the
  `#SBATCH` account/partition lines for your cluster). Then `bash slurm/submit.sh`.
- Human texts (C4 realnewslike, Wikipedia 20231101 en/ru, codeparrot-clean-valid) are streamed from Hugging Face by
  `kl/data.py` and `kl/data_e2.py`; they are not included here.
- Green lists must be computed on the CPU: CUDA and CPU `torch.randperm` differ for the same seed.
- Synthetic tests: `python -m pytest tests/test_nullstats.py`.
- Figures: `python -m kl.figures --results results --out report/fig`, `python explore/proposal_figs.py`.

Verdicts of the pre-registered claims: `report/claims.json` (C0–C9) and `report/screening.json` (S1–S7).
