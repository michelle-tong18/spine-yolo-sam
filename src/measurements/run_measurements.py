"""
Measurements from a finished inference run (no ground truth, no models): reads ``{run}/seg_pred`` and, for disc
measurements, the preprocessed images inference ran on. Each volume is loaded once for every requested metric.

    python src/measurements/run_measurements.py <config> --run-dir RUN [--metrics discs,volumes] [--volumes DIR]

Metrics (``--metrics``, comma separated, default ``volumes``):
    discs    per disc level (src/measurements/disc_measurements.py), sagittal label maps only
             -> {run}/disc_measurements.csv: volume_stem, level, slices, n_slices_with_disc, row_mm, col_mm and
                every disc_measurements.MEASUREMENTS column (median over measurements.mid_slices midline slices)
    volumes  per label and per evaluation.tissue_groups entry (src/measurements/volumes.py)
             -> {run}/label_volumes.csv: volume_stem, level, class, voxels, volume_cm3

Pixel spacing is the one preprocessing recorded for the image volume (``nifti_preprocessed/spacing.csv``,
src/preprocessing/spacing.py). Volumes without it are still measured: mm and cm³ columns are nan, and the
scale-free disc columns (height indices, signal ratio, high-signal fraction) are measured in pixels, which assumes
square pixels. Optional YAML for discs:

    measurements:
      mid_slices: 3                       # default 3
      levels: [L3-L4, L4-L5, L5-S1]       # default: every disc in mask_labels with both vertebrae labelled
"""
import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.inference.inference_config import SEG_PRED_SUBDIR
from src.measurements.disc_measurements import MM_COLUMNS, measure_discs_in_volume, measurement_levels
from src.measurements.volumes import label_volume_rows
from src.preprocessing.spacing import load_grid_spacing
from src.utils.file_management.config_handler import get_config_name, load_main_config

METRICS = ("discs", "volumes")
OUTPUT_CSV = {"discs": "disc_measurements.csv", "volumes": "label_volumes.csv"}


def measure_run(cfg: dict, run_dir: str, volume_dir: str, metrics: list[str]) -> dict[str, Path]:
    """Measure every segmented volume of a run; writes one CSV per metric in ``run_dir`` and returns their paths."""
    mask_labels = {int(k): str(v) for k, v in cfg["mask_labels"].items()}
    measurements_cfg = cfg.get("measurements") or {}
    levels = measurement_levels(mask_labels, measurements_cfg) if "discs" in metrics else []
    n_mid = int(measurements_cfg.get("mid_slices", 3))
    tissue_groups = {str(k): [int(i) for i in v] for k, v in ((cfg.get("evaluation") or {}).get("tissue_groups") or {}).items()}

    rows = {m: [] for m in metrics}
    problems = []
    for seg_path in sorted((Path(run_dir) / SEG_PRED_SUBDIR).glob("*.nii.gz")):
        stem = seg_path.name.removesuffix(".nii.gz")
        seg = np.asanyarray(nib.load(seg_path).dataobj).astype(np.int16)
        spacing = load_grid_spacing(stem, volume_dir)
        if spacing is None:
            problems.append(f"{stem}: no pixel spacing in spacing.csv (preprocess it with full_standardization.py); mm / cm³ columns are nan")
        slice_mm, row_mm, col_mm = spacing if spacing is not None else (np.nan, np.nan, np.nan)

        if "volumes" in metrics:
            rows["volumes"] += [{"volume_stem": stem, **r}
                                for r in label_volume_rows(seg, mask_labels, tissue_groups, slice_mm * row_mm * col_mm)]

        if "discs" in metrics:
            img_path = Path(volume_dir) / seg_path.name
            img = np.asanyarray(nib.load(img_path).dataobj).astype(np.float32) if img_path.is_file() else None
            if img is None or img.shape != seg.shape:
                problems.append(f"{stem}: no image matching the segmentation in {volume_dir}; discs skipped (measure "
                                "on the preprocessed volumes inference ran on, nifti_preprocessed/img)")
                continue
            # Without spacing, measure in pixels: ratios stay valid (square pixels), mm columns are blanked.
            disc_rows = measure_discs_in_volume(seg, img, mask_labels, levels, n_mid,
                                                row_mm if spacing is not None else 1.0,
                                                col_mm if spacing is not None else 1.0)
            for row in disc_rows:
                if spacing is None:
                    row.update({c: np.nan for c in MM_COLUMNS + ["row_mm", "col_mm"]})
                rows["discs"].append({"volume_stem": stem, **row})

    paths = {}
    for metric in metrics:
        paths[metric] = Path(run_dir) / OUTPUT_CSV[metric]
        pd.DataFrame(rows[metric]).to_csv(paths[metric], index=False)
        print(f"{metric}: {len({r['volume_stem'] for r in rows[metric]})} volumes -> {paths[metric]}")
    for msg in problems:
        print(f"  {msg}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Measurements (no ground truth) from a finished inference run.")
    parser.add_argument("config", help="Dataset config name in config/datasets/ (or a YAML path).")
    parser.add_argument("--run-dir", required=True, help="Inference run directory with seg_pred/.")
    parser.add_argument("--metrics", default="volumes", help=f"Comma-separated subset of {', '.join(METRICS)}.")
    parser.add_argument("--volumes", default=None,
                        help="Preprocessed image volumes inference ran on (default: inference.volume_dir).")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in METRICS]
    if unknown or not metrics:
        parser.error(f"--metrics takes {', '.join(METRICS)}; got {args.metrics!r}")
    cfg = load_main_config(get_config_name(args.config), os.getcwd(), sub_dir="datasets")
    volume_dir = args.volumes or (cfg.get("inference") or {}).get("volume_dir")
    if not volume_dir:
        parser.error("set inference.volume_dir in the YAML or pass --volumes")
    measure_run(cfg, args.run_dir, volume_dir, metrics)


if __name__ == "__main__":
    main()
