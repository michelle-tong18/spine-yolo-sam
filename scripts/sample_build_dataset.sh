#!/bin/bash
# -------------------- Script Info -------------------- #
# Build preprocessed volumes and YOLO/SAM train-val-test splits.
#
# Before you run
#   1. Copy a YAML from config/datasets/ and set general_data_dir, data_dir, mask_labels, split_id.
#   2. List the same subject ids under that split_id in config/splits/splits_registry.yaml.
#   3. Optional: cp train_yolo.env.example my_vars.env
#
# Run from the scripts/ directory (so my_vars.env and cd .. resolve).
#   bash sample_build_dataset.sh example_axial_muscle
#
# Cluster (edit the #SBATCH lines for your site, then submit from scripts/):
#   sbatch sample_build_dataset.sh example_axial_muscle
# Logs: scripts/logs/build-dataset-<jobid>.out
#
# Next: sample_train_yolo.sh and/or sample_train_sam.sh

# -------------------- SLURM Job Configuration -------------------- #
#SBATCH --job-name=build-dataset
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
echo "**************************************************"
echo "Building dataset from config: ${CONFIG_NAME}"
echo "**************************************************"

## Navigate to the repository root
cd .. || exit

python3 src/preprocessing/full_standardization.py "${CONFIG_NAME}"
python3 src/preprocessing/dataset_split.py "${CONFIG_NAME}"

echo "Done. Preprocessed volumes and slice folders are under general_data_dir in the YAML."

