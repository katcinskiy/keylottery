#!/bin/bash
#SBATCH --job-name=kl-screen
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
# 1000 fresh keys on ref + E1 + E2 (CLAUDE.md Section 9), then draw the 10 screened + 10 random keys for generation
"$PY" -m kl.run_screen --workers "${SLURM_CPUS_PER_TASK:-32}"
"$PY" -m kl.screening --select
