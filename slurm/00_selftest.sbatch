#!/bin/bash
#SBATCH --job-name=kl-selftest
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
# synthetic unit tests: Var formula / predictor / masking (prf), exact Var for fixed-size green lists (lefthash),
# and the end-to-end dry run incl. analyze + claims on synthetic data
"$PY" -m tests.test_nullstats
"$PY" -m tests.test_exact_var
export KL_SYNTH_DIR="$PWD/selftest_synth"
rm -rf "$KL_SYNTH_DIR"
"$PY" -m tests.pipeline_synth
cd "$KL_SYNTH_DIR"
"$PY" -m kl.claims --results results --report report --main null_h1_prf_K60 --h0 null_h0_prf_K60 --prf null_h1_prf_K60
"$PY" -m kl.figures --results results --out report/fig
