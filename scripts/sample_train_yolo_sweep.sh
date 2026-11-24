#!/bin/bash
# -------------------- Script Info -------------------- #
# YOLO hyperparameter sweep via Weights & Biases.
#
# Before you run
#   1. Add a sweep_cfg block to the dataset YAML (not in the example configs by default).
#   2. Set WANDB_API_KEY and WANDB_ENTITY in my_vars.env.
#
# Run from the scripts/ directory (so my_vars.env and cd .. resolve).
#   bash sample_train_yolo_sweep.sh example_axial_muscle
#
# Cluster (edit the #SBATCH lines for your site, then submit from scripts/):
#   sbatch sample_train_yolo_sweep.sh example_axial_muscle
# Logs: scripts/logs/yolo-sweep-<jobid>.out
#
# For a single training run without a sweep, use sample_train_yolo.sh instead.

# -------------------- SLURM Job Configuration -------------------- #
#SBATCH --job-name=yolo-sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# -------------------- Environment --------------------  #
# Set bash to exit on error
set -e

echo "Running on node $HOSTNAME"

# Activate conda environment
module load SCS/anaconda/anaconda3 || conda --version
source $(conda info --base)/etc/profile.d/conda.sh

source "my_vars.env"

# Conda setup
if [ -f "${CONDA_ENV_NAME}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_ENV_NAME}/bin/activate"
  PYTHON="$(command -v python)"
elif [ -x "${CONDA_ENV_NAME}/bin/python" ]; then
  export PATH="${CONDA_ENV_NAME}/bin:${PATH}"
  PYTHON="${CONDA_ENV_NAME}/bin/python"
else
  echo "ERROR: Could not find python in CONDA_ENV_NAME (${CONDA_ENV_NAME})."
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_ENV_NAME}/lib:${LD_LIBRARY_PATH}"

echo "Using python: ${PYTHON}"
"${PYTHON}" -V || true

# -------------------- USER DEFINED PARAMS -------------------- #

CONFIG_NAME="${1}"

# -------------------- BEGIN CODE -------------------- #
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_STEP_GPUS: $SLURM_STEP_GPUS"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

if [ -z "${WANDB_API_KEY}" ] || [ -z "${WANDB_ENTITY}" ]; then
  echo "ERROR: W&B sweeps need WANDB_API_KEY and WANDB_ENTITY in my_vars.env."
  echo "Copy scripts/train_yolo.env.example to scripts/my_vars.env and fill those in."
  exit 1
fi

yolo settings wandb=True

echo "**************************************************"
echo "YOLO W&B sweep: ${CONFIG_NAME}"
echo "**************************************************"

## Navigate to the repository root
cd .. || exit

python3 src/yolo_finetuning/train_yolo_sweep.py "${CONFIG_NAME}"

echo "Done."
