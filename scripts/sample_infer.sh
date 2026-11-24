#!/bin/bash
# -------------------- Script Info -------------------- #
# Run inference: YOLO boxes, SAM masks, or both on the same Run_N.
#
# Before you run
#   1. Point inference.models.obj_det.model_path at a trained YOLO run (folder with best.pt).
#   2. Point inference.models.segmentation.model_path at a trained SAM Run_N folder.
#   3. Set inference.volume_dir (volumes to label) and output_data_dir in the YAML.
#
# Run from the scripts/ directory (so my_vars.env and cd .. resolve).
#   bash sample_infer.sh example_axial_muscle both
#   bash sample_infer.sh example_axial_muscle yolo
#   bash sample_infer.sh example_axial_muscle sam --boxes pred --run-num 1
#   bash sample_infer.sh example_axial_muscle sam --boxes gt
#
# Cluster (edit the #SBATCH lines for your site, then submit from scripts/):
#   sbatch sample_infer.sh example_axial_muscle both
# Logs: scripts/logs/infer-<jobid>.out

# -------------------- SLURM Job Configuration -------------------- #
#SBATCH --job-name=infer
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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
COMMAND="${2:-both}"

# -------------------- BEGIN CODE -------------------- #
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_STEP_GPUS: $SLURM_STEP_GPUS"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

echo "**************************************************"
echo "Inference: config=${CONFIG_NAME}  command=${COMMAND}"
echo "Writes: {output_data_dir}/inference/{task_name}/Run_N/{label_pred,seg_pred}"
echo "**************************************************"

## Navigate to the repository root
cd .. || exit

if [ "$#" -ge 1 ]; then
  shift
fi
if [ "$#" -ge 1 ]; then
  shift
fi

python3 src/inference/run_inference.py "${COMMAND}" --config "${CONFIG_NAME}" "$@"

echo "Done."
