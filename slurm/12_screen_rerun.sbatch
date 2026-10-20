#!/bin/bash
#SBATCH --job-name=kl-screen2
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=mladen
#SBATCH --partition=cpu
source slurm/env.sh
# rerun of the screening null on the fresh E2 (CLAUDE.md 9a); stale results are kept
for ext in npz json; do
  if [ -f "results/screen_h1_lefthash_K1000.$ext" ]; then
    mv "results/screen_h1_lefthash_K1000.$ext" "results/screen_h1_lefthash_K1000_staleE2.$ext"
  fi
done
if [ -f report/screening.json ]; then mv report/screening.json report/screening_staleE2.json; fi
cp results/screen_gen_keys.json results/screen_gen_keys_first.json
"$PY" -m kl.run_screen --workers "${SLURM_CPUS_PER_TASK:-32}"
"$PY" -m kl.screening --select
# the generation keys depend only on the reference sets, so they must be identical to the first draw
cmp results/screen_gen_keys.json results/screen_gen_keys_first.json && echo "generation keys unchanged"
"$PY" -m kl.screening
