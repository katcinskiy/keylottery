# sourced by every job: python env, caches, provenance
set -euo pipefail
cd "${KL_DIR:-$HOME/keylottery}"
export PY="${KL_PY:-/shared/home/alexander.yukhimchuk/miniconda3/envs/venv/bin/python}"
# login-node HF_HOME=/datasets/huggingface is read-only on compute nodes; Lab 4 used this cache (OPT-1.3B is there)
export HF_HOME="${KL_HF_HOME:-$HOME/hf_cache}"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs results data report
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) | cpus=${SLURM_CPUS_PER_TASK:-?} | $(date -Is)"
"$PY" - <<'PYEOF'
import platform, numpy, scipy, torch, transformers
print("python", platform.python_version(), "| numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available() and torch.cuda.get_device_name(0))
PYEOF
