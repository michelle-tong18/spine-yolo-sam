#!/bin/bash
# -------------------- Script Info -------------------- #
# Finetune SAM from a dataset YAML.
#
# Before you run
#   1. Finish sample_build_dataset.sh (needs slice_preprocessed/SAM).
#   2. Place the SAM ViT-B checkpoint at SAM.module.pretrain_model
#      (example configs use weights/SAM/sam_vit_b_01ec64.pth).
#   3. Optional W&B: set SAM.module.use_wandb: True in the YAML and fill my_vars.env.
#
# Run from the scripts/ directory (so my_vars.env and cd .. resolve).
#   bash sample_train_sam.sh example_axial_muscle
#
# Cluster (edit the #SBATCH lines for your site, then submit from scripts/):
#   sbatch sample_train_sam.sh example_axial_muscle
# Logs: scripts/logs/train-sam-<jobid>.out
#
# Next: sample_infer.sh

# -------------------- SLURM Job Configuration -------------------- #
#SBATCH --job-name=train-sam
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1                           # Request tasks (processes) per node
#SBATCH --mem=128G
#SBATCH --time=23:59:00                               # Set a time limit for the job
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

echo "**************************************************"
echo "SAM finetuning: ${CONFIG_NAME}"
echo "Output: work_dir/sam/<SAM.module.task_name>/  (see the YAML)"
echo "**************************************************"

## Navigate to the repository root
cd .. || exit

python3 src/sam_finetuning/finetune_main.py "${CONFIG_NAME}"
# python3 src/sam_finetuning/finetune_main.py "${CONFIG_NAME}" --kwargs datamodule.batch_size=2

echo "Done. Point inference.models.segmentation.model_path at the Run_N folder"
echo "(the one that contains finetuned_model_best_converted.pth)."
