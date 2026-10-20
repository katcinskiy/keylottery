#!/bin/bash
#SBATCH --job-name=kl-report
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
# Section-6 verdicts (kl.claims, written before the real run) and figures (no matplotlib on the Mac)
"$PY" -m kl.analyze --results results --out report
"$PY" -m kl.claims --results results --report report
"$PY" -m kl.figures --results results --out report/fig
