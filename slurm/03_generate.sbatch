#!/bin/bash
#SBATCH --job-name=kl-gen
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=mladen
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=shard:2
source slurm/env.sh
"$PY" -m kl.generate --n_keys 5 --deltas 0 1 2 --n_prompts 400
