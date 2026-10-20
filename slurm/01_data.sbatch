#!/bin/bash
#SBATCH --job-name=kl-data
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
# one process per source. Lab 4's data job saw a pyarrow/HF-streaming abort (PyGILState_Release) at interpreter
# exit AFTER the file was saved, so a non-zero exit is tolerated only if the output file exists.
for s in news wiki_en lists code wiki_ru prompts; do
  "$PY" -m kl.data "$s" || echo "WARNING: kl.data $s exited with status $?"
  [ -s "data/$s.npz" ] || { echo "ERROR: data/$s.npz missing"; exit 1; }
done
"$PY" -m kl.checks ${KL_LAB4_DATA:+--lab4_data "$KL_LAB4_DATA"}
