#!/bin/bash
#SBATCH --job-name=kl-e2-redo
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
# CLAUDE.md 9a: rebuild code and wiki_ru E2 without windows shared with the main stream; keep the failed build
for s in code wiki_ru; do
  for ext in npz json; do
    if [ -f "data/${s}_e2.$ext" ]; then mv "data/${s}_e2.$ext" "data/${s}_e2_shared.$ext"; fi
  done
done
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
for s in code wiki_ru; do
  for attempt in 1 2 3; do
    if [ -s "data/${s}_e2.npz" ]; then break; fi
    timeout 2400 "$PY" -m kl.data_e2 "$s" || echo "WARNING: kl.data_e2 $s attempt $attempt exited with status $?"
  done
  [ -s "data/${s}_e2.npz" ] || { echo "ERROR: data/${s}_e2.npz missing"; exit 1; }
done
"$PY" -m kl.data_e2 --check
