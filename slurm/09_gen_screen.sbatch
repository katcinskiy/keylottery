#!/bin/bash
#SBATCH --job-name=kl-gen-scr
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=mladen
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=shard:4
source slurm/env.sh
# job 44639 was cancelled by titan-gpu-monitor at 34.4 GiB on shard:2 (19.1 GiB budget); shards = ceil(peak/9.6 GiB) = 4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -m tests.test_generate_cache
"$PY" -m kl.generate --key_file results/screen_gen_keys_union.json --deltas 1 --n_prompts 400 \
  --out data/gen_screen.npz --cache
