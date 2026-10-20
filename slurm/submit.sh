#!/bin/bash
# run on the titan login node: only submits jobs (no compute here)
set -euo pipefail
cd "$HOME/keylottery"
mkdir -p logs
j1=$(sbatch --parsable slurm/01_data.sbatch)
j2=$(sbatch --parsable --dependency=afterok:$j1 slurm/02_null.sbatch)
j3=$(sbatch --parsable --dependency=afterok:$j1 slurm/03_generate.sbatch)
j4=$(sbatch --parsable --dependency=afterok:$j2:$j3 slurm/04_power.sbatch)
echo "data=$j1 null=$j2 gen=$j3 power=$j4" | tee -a logs/submitted.txt
