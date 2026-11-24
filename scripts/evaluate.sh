#!/bin/bash
# -------------------- Script Info -------------------- #
# Score an inference run: detection + segmentation metrics for each patient, each series and
# each tissue level (single labels, evaluation.tissue_groups merged, and all tissues merged).
# Needs evaluation.gt_labels_dir / gt_masks_dir (and optionally tissue_groups) in the YAML.
#
# Run from anywhere. CONFIG is a name under config/datasets/ or a path to a YAML:
#   bash scripts/evaluate.sh sagittal_vertdisc /path/to/inference/sagittal_vertdisc/Run_2
#   bash scripts/evaluate.sh work_dir/sagittal_work/config/sagittal_vertdisc.yaml /path/to/Run_2
# Extra args go to run_evaluation.py, e.g. --output_dir DIR (default: <run_dir>/evaluation).
#
# Writes to the output dir:
#   detection_metrics_breakdown.csv      <- P / R / F1 / mAP, scope x level x class rows
#   segmentation_metrics_breakdown.csv   <- Dice / IoU / HD95, scope x level x class rows
#
# Python: CONDA_ENV_NAME from scripts/my_vars.env if that file exists, else python3 on PATH.

set -eo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/evaluate.sh <config> <run_dir> [run_evaluation.py args]"
  exit 1
fi

CONFIG="${1}"
RUN_DIR="${2}"
shift 2

# Resolve paths against the caller's cwd before moving to the repo root.
if [ -f "${CONFIG}" ]; then
  CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
fi
if [ ! -d "${RUN_DIR}" ]; then
  echo "ERROR: run dir not found: ${RUN_DIR}"
  exit 1
fi
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="python3"
if [ -f "${SCRIPT_DIR}/my_vars.env" ]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/my_vars.env"
  if [ -x "${CONDA_ENV_NAME}/bin/python" ]; then
    PYTHON="${CONDA_ENV_NAME}/bin/python"
  fi
fi
echo "Using python: ${PYTHON}"

echo "**************************************************"
echo "Evaluate: config=${CONFIG}"
echo "          run=${RUN_DIR}"
echo "**************************************************"

cd "${SCRIPT_DIR}/.."
"${PYTHON}" src/evaluation/run_evaluation.py "${CONFIG}" \
  --run "${RUN_DIR}" \
  --output_dir "${RUN_DIR}/evaluation" \
  "$@"

echo "Done."
