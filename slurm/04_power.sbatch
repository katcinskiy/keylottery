#!/bin/bash
#SBATCH --job-name=kl-power
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
"$PY" -m kl.power --workers "${SLURM_CPUS_PER_TASK:-32}" --null_keys 1000
"$PY" -m kl.analyze --results results --out report
