"""
Axial tissue volume per vertebral level (src/measurements/levels.py) for patients with a sagittal and an axial scan,
from two finished inference runs (no ground truth, no models).

    python src/measurements/run_level_volumes.py --pairs pairs.csv \\
        --sagittal-config example_sagittal_vertdisc --sagittal-run RUN_S \\
        --axial-config example_axial_muscle --axial-run RUN_A [--sagittal-volumes DIR] [--axial-volumes DIR]

``pairs.csv``: one row per scan pair, columns ``patient_id``, ``sagittal_volume``, ``axial_volume`` (preprocessed
volume names). Volumes are read from each run's ``seg_pred``; their world geometry from ``spacing.csv`` next to the
preprocessed volumes inference ran on (default: each config's ``inference.volume_dir``). Tissue groups: the axial
config's ``evaluation.tissue_groups``.

Writes ``{axial run}/label_volumes_per_level.csv``: patient_id, sagittal_volume, axial_volume, level, disc_above,
disc_below, n_axial_slices, coverage_frac, level_kind (label | tissue), class, voxels, volume_cm3. A pair that cannot
be measured (missing volume or world geometry, no disc mid-planes) is reported and skipped.
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
from src.measurements.levels import axial_level_volume_rows, disc_planes, vertebral_levels
from src.preprocessing.spacing import load_world_affine
from src.utils.file_management.config_handler import get_config_name, load_main_config

OUTPUT_CSV = "label_volumes_per_level.csv"
PAIR_COLUMNS = ["patient_id", "sagittal_volume", "axial_volume"]


def load_scan(run_dir: str, volume_dir: str, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """(segmentation, world affine) of one volume; raises ValueError naming what is missing."""
    seg_path = Path(run_dir) / SEG_PRED_SUBDIR / f"{stem}.nii.gz"
    if not seg_path.is_file():
        raise ValueError(f"no segmentation {seg_path}")
    affine = load_world_affine(stem, volume_dir)
    if affine is None:
        raise ValueError(f"no world_affine for {stem} in {Path(volume_dir).parent / 'spacing.csv'} (preprocess the "
                         "scan with full_standardization.py; the header needs position and orientation)")
    return np.asanyarray(nib.load(seg_path).dataobj).astype(np.int16), affine


def measure_pairs(pairs: pd.DataFrame, sagittal_cfg: dict, sagittal_run: str, sagittal_volumes: str,
                  axial_cfg: dict, axial_run: str, axial_volumes: str) -> Path:
    """Measure every pair; writes and returns ``{axial_run}/label_volumes_per_level.csv``."""
    sagittal_labels = {int(k): str(v) for k, v in sagittal_cfg["mask_labels"].items()}
    axial_labels = {int(k): str(v) for k, v in axial_cfg["mask_labels"].items()}
    tissue_groups = {str(k): [int(i) for i in v]
                     for k, v in ((axial_cfg.get("evaluation") or {}).get("tissue_groups") or {}).items()}
    rows, problems = [], []
    for pair in pairs.itertuples(index=False):
        try:
            sag_seg, sag_affine = load_scan(sagittal_run, sagittal_volumes, pair.sagittal_volume)
            ax_seg, ax_affine = load_scan(axial_run, axial_volumes, pair.axial_volume)
        except ValueError as e:
            problems.append(f"{pair.patient_id}: {e}")
            continue
        planes = disc_planes(sag_seg, sag_affine, sagittal_labels)
        levels = vertebral_levels(sagittal_labels, planes)
        if not levels:
            problems.append(f"{pair.patient_id}: no vertebra with both disc mid-planes in {pair.sagittal_volume}")
            continue
        meta = {"patient_id": pair.patient_id, "sagittal_volume": pair.sagittal_volume,
                "axial_volume": pair.axial_volume}
        rows += [{**meta, **r} for r in
                 axial_level_volume_rows(ax_seg, ax_affine, axial_labels, tissue_groups, planes, levels)]

    out_path = Path(axial_run) / OUTPUT_CSV
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Level volumes for {len({(r['patient_id'], r['axial_volume']) for r in rows})} of {len(pairs)} pairs -> {out_path}")
    for msg in problems:
        print(f"  {msg}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Axial tissue volume per vertebral level from sagittal + axial runs.")
    parser.add_argument("--pairs", required=True, help=f"CSV with columns {', '.join(PAIR_COLUMNS)}.")
    for plane in ("sagittal", "axial"):
        parser.add_argument(f"--{plane}-config", required=True, help=f"{plane} dataset config name or YAML path.")
        parser.add_argument(f"--{plane}-run", required=True, help=f"{plane} inference run directory (seg_pred/).")
        parser.add_argument(f"--{plane}-volumes", default=None,
                            help=f"Preprocessed {plane} volumes inference ran on (default: inference.volume_dir).")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs, dtype=str)
    missing = [c for c in PAIR_COLUMNS if c not in pairs.columns]
    if missing:
        parser.error(f"{args.pairs} lacks columns {missing}")
    cfgs, volume_dirs = {}, {}
    for plane in ("sagittal", "axial"):
        cfgs[plane] = load_main_config(get_config_name(getattr(args, f"{plane}_config")), os.getcwd(), sub_dir="datasets")
        volume_dirs[plane] = getattr(args, f"{plane}_volumes") or (cfgs[plane].get("inference") or {}).get("volume_dir")
        if not volume_dirs[plane]:
            parser.error(f"set inference.volume_dir in the {plane} config or pass --{plane}-volumes")
    measure_pairs(pairs, cfgs["sagittal"], args.sagittal_run, volume_dirs["sagittal"],
                  cfgs["axial"], args.axial_run, volume_dirs["axial"])


if __name__ == "__main__":
    main()
