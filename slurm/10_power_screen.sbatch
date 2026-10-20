#!/bin/bash
#SBATCH --job-name=kl-pow-scr
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
"$PY" -m kl.power --gen data/gen_screen.npz --out results/power_screen.json --null_keys 0 --workers "${SLURM_CPUS_PER_TASK:-32}"
"$PY" -m kl.screening
