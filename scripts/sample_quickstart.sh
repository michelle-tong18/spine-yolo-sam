#!/bin/bash
# -------------------- Script Info -------------------- #
# Label new scans with released (or your own) weights: preprocess image-only, YOLO + SAM on the preprocessed
# volumes into RUN_DIR, then (optional) measurements from the segmentations: METRICS is a comma-separated subset of
# discs (sagittal, {run}/disc_measurements.csv) and volumes ({run}/label_volumes.csv).
#
# Before you run
#   1. Copy a YAML from config/datasets/ and set general_data_dir, data_dir (a CSV of scans), image_column.
#      Leave mask_column empty (or set preprocessing_cfg.image_only: True): scans without labels are
#      preprocessed image-only.
#   2. Download the released weights into weights/spine-yolo-sam/ and the SAM base checkpoint into weights/SAM/
#      (README: Model weights); the example configs already point at them.
#   3. Keep inference.volume_dir at ${general_data_dir}/nifti_preprocessed/img (the preprocessed volumes).
#   4. cp train_yolo.env.example my_vars.env and set CONDA_ENV_NAME (this script sources it).
#
# Run from the scripts/ directory (so my_vars.env and cd .. resolve).
#   bash sample_quickstart.sh example_sagittal_vertdisc /path/to/runs/new_scans
#   bash sample_quickstart.sh example_sagittal_vertdisc /path/to/runs/new_scans discs,volumes
#
# Cluster (edit the #SBATCH lines for your site, then submit from scripts/):
#   sbatch sample_quickstart.sh example_sagittal_vertdisc /path/to/runs/new_scans discs,volumes
# Logs: scripts/logs/quickstart-<jobid>.out

# -------------------- SLURM Job Configuration -------------------- #
#SBATCH --job-name=quickstart
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
RUN_DIR="${2}"
METRICS="${3:-}"

# -------------------- BEGIN CODE -------------------- #
if [ -z "${CONFIG_NAME}" ] || [ -z "${RUN_DIR}" ]; then
  echo "Usage: bash sample_quickstart.sh <config> <run_dir> [metrics, e.g. discs,volumes]"
  exit 1
fi

echo "**************************************************"
echo "Quickstart: config=${CONFIG_NAME}  run=${RUN_DIR}  metrics=${METRICS:-none}"
echo "Writes: {general_data_dir}/nifti_preprocessed/{img,spacing.csv}"
echo "        ${RUN_DIR}/{label_pred,seg_pred,figures}"
echo "**************************************************"

## Navigate to the repository root
cd .. || exit

python3 src/preprocessing/full_standardization.py "${CONFIG_NAME}"
python3 src/inference/run_inference.py both --config "${CONFIG_NAME}" --run-dir "${RUN_DIR}" --figures
if [ -n "${METRICS}" ]; then
  python3 src/measurements/run_measurements.py "${CONFIG_NAME}" --run-dir "${RUN_DIR}" --metrics "${METRICS}"
fi

echo "Done."
