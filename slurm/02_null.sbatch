#!/bin/bash
#SBATCH --job-name=kl-null
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
W=${SLURM_CPUS_PER_TASK:-32}
[ -f lab4_keys.json ] && "$PY" -m kl.run_null --h 1 --key_file lab4_keys.json --backend lefthash --workers "$W" --tag _lab4keys
"$PY" -m kl.run_null --h 1 --keys 1000 --backend lefthash --workers "$W"
"$PY" -m kl.run_null --h 0 --keys 1000 --backend lefthash --workers "$W"
"$PY" -m kl.run_null --h 1 --keys 300 --backend prf --workers "$W"
