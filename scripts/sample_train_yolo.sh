#!/bin/bash
# -------------------- Script Info -------------------- #
# Train the YOLO detector from a dataset YAML.
#
# Before you run
#   1. Finish sample_build_dataset.sh (needs slice_preprocessed/YOLO and yolo_data.yaml).
#   2. Set YOLO.model to a starting checkpoint (e.g. weights/YOLO/yolo26x.pt).
#   3. Optional W&B: cp train_yolo.env.example my_vars.env
#
# Run from the scripts/ directory (so my_vars.env and cd .. resolve).
#   bash sample_train_yolo.sh example_axial_muscle
#
# Cluster (edit the #SBATCH lines for your site, then submit from scripts/):
#   sbatch sample_train_yolo.sh example_axial_muscle
# Logs: scripts/logs/train-yolo-<jobid>.out
#
# Hyperparameter sweep instead: sample_train_yolo_sweep.sh
# Next: sample_train_sam.sh, then sample_infer.sh

# -------------------- SLURM Job Configuration -------------------- #
#SBATCH --job-name=train-yolo
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
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

# Enable Weights & Biases logging
yolo settings wandb=True

echo "**************************************************"
echo "YOLO training: ${CONFIG_NAME}"
echo "Output: work_dir/yolo/<YOLO.yolo_parameters.project>/  (see the YAML)"
echo "**************************************************"

## Navigate to the repository root
cd .. || exit

python3 src/yolo_finetuning/train_yolo.py "${CONFIG_NAME}"

echo "Done. Point inference.models.obj_det.model_path at the run folder (the one that contains best.pt)."
