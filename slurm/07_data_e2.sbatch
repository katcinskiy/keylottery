#!/bin/bash
#SBATCH --job-name=kl-data-e2
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
# fresh far-away eval texts E2 (CLAUDE.md Section 9); same post-save abort tolerance as 01_data
# HF streaming without a token once hung on the network for 30 min (job 44629, code): per-source timeout + retries;
# finished sources are skipped (build_e2 returns if the file exists)
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
for s in news wiki_en lists code wiki_ru; do
  for attempt in 1 2 3; do
    [ -s "data/${s}_e2.npz" ] && break
    timeout 1200 "$PY" -m kl.data_e2 "$s" || echo "WARNING: kl.data_e2 $s attempt $attempt exited with status $?"
  done
  [ -s "data/${s}_e2.npz" ] || { echo "ERROR: data/${s}_e2.npz missing"; exit 1; }
done
