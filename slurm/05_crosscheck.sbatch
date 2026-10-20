#!/bin/bash
#SBATCH --job-name=kl-xcheck
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
# brute-force per-text z (transformers green lists per position + Python set) vs the saved _z.npz
"$PY" -m tests.crosscheck_bruteforce --set code --n 300 \
  --runs results/null_h1_lefthash_K1000_z.npz:0,1,2 results/null_h1_lefthash_K10_lab4keys_z.npz:3,6
