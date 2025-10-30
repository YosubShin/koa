#!/bin/bash
#SBATCH --job-name=koa-example
#SBATCH --partition=kill-shared
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Minimal sample SLURM script for KOA submissions. Update resources, modules,
# and commands to match your workload. The koa CLI automatically sets
# KOA_ML_CODE_ROOT to the synced project directory and, when configured,
# KOA_ML_RESULTS_ROOT for shared scratch space.

set -euo pipefail

module purge >/dev/null 2>&1 || true
module load lang/Python/3.11.5-GCCcore-13.2.0 >/dev/null 2>&1 || true
module load system/CUDA/12.2.0 >/dev/null 2>&1 || true

# KOA_ML_RESULTS_ROOT is injected by the CLI when configured; fall back to a local path.
RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "Running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Writing outputs to ${RESULTS_DIR}"

# Replace this with your workload. 'sleep' is only a placeholder.
srun --cpu-bind=cores sleep 60

echo "Job completed"
