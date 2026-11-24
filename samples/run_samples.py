#!/usr/bin/env python3
"""
Label the sample scans with the released models, end to end (README: Try it on the sample scans):

    python samples/run_samples.py [--planes sagittal axial] [--no-levels]

For each plane, the scans ``samples/data/sample_<plane>_<contrast>*`` (T1 and T2; NIfTI / MHA files or DICOM
folders) are preprocessed image-only, segmented (YOLO + SAM, with ``--figures``) and measured together, with the
example config of that plane and the weights in ``weights/`` (README: Model weights). Outputs, per plane, in
``samples/outputs/<plane>/``:
    run/figures/sample_<plane>_<contrast>.png      the image, YOLO boxes and masks on 5 evenly spaced slices
    run/seg_pred/sample_<plane>_<contrast>.nii.gz  masks
    run/label_volumes.csv                          cm³ per label (and run/disc_measurements.csv for sagittal)
With both planes (the samples are one patient from one session), also the axial muscle volume per vertebral level,
sagittal and axial scans of the same contrast paired (src/measurements/run_level_volumes.py), in
``samples/outputs/axial/run/label_volumes_per_level.csv``; ``--no-levels`` skips it.
"""
import argparse
import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUTPUTS = SAMPLES / "outputs"
EXAMPLE_CONFIG = {"sagittal": "config/datasets/example_sagittal_vertdisc.yaml",
                  "axial": "config/datasets/example_axial_muscle.yaml"}
METRICS = {"sagittal": "discs,volumes", "axial": "volumes"}


def sample_scans(plane: str) -> dict[str, Path]:
    """``samples/data/sample_<plane>_<contrast>`` with any extension (or a DICOM folder of that name), by name."""
    paths = sorted((SAMPLES / "data").glob(f"sample_{plane}_*"))
    found = {p.name.split(".")[0]: p for p in paths if p.suffix != ".csv"}
    if not found:
        sys.exit(f"No sample scan samples/data/sample_{plane}_* (NIfTI, .mha or a DICOM folder).")
    return found


def write_config(plane: str, scans: dict[str, Path], out: Path) -> Path:
    """The example config with the samples' paths: image-only preprocessing, on the GPU if there is one."""
    with open(ROOT / EXAMPLE_CONFIG[plane]) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))  # keeps key order: image_config.transforms run in YAML order
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_id": list(scans), "image": [str(p) for p in scans.values()]}).to_csv(out / "scans.csv",
                                                                                             index=False)
    cfg.update(general_data_dir=str(out), output_dir=str(out), output_data_dir=str(out / "results"),
               work_dir=str(out / "runs"), data_dir=str(out / "scans.csv"), image_column="image", mask_column="",
               include_subject_accession=False)
    cfg["inference"]["volume_dir"] = str(out / "nifti_preprocessed" / "img")
    cfg["inference"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    path = out / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Label the sample scans with the released models.")
    parser.add_argument("--planes", nargs="+", choices=list(EXAMPLE_CONFIG), default=list(EXAMPLE_CONFIG))
    parser.add_argument("--no-levels", action="store_true",
                        help="Skip the axial muscle volume per vertebral level (needs both planes).")
    args = parser.parse_args()
    configs, names = {}, {}
    for plane in args.planes:
        out = OUTPUTS / plane
        scans = sample_scans(plane)
        names[plane] = list(scans)
        config = configs[plane] = str(write_config(plane, scans, out))
        run_dir = str(out / "run")
        run([sys.executable, "src/preprocessing/full_standardization.py", config])
        run([sys.executable, "src/inference/run_inference.py", "both", "--config", config, "--run-dir", run_dir,
             "--figures"])
        run([sys.executable, "src/measurements/run_measurements.py", config, "--run-dir", run_dir,
             "--metrics", METRICS[plane]])
        print(f"\n{plane}: figure {run_dir}/figures/, masks {run_dir}/seg_pred/, tables {run_dir}/*.csv")

    if args.no_levels:
        return
    if set(configs) != set(EXAMPLE_CONFIG):
        print("\nAxial muscle volume per level skipped: it needs both planes (run without --planes).")
        return
    pairs = OUTPUTS / "pairs.csv"
    contrasts = [n.removeprefix("sample_sagittal_") for n in names["sagittal"]
                 if n.replace("sagittal", "axial") in names["axial"]]
    pd.DataFrame([{"patient_id": "sample", "sagittal_volume": f"sample_sagittal_{c}",
                   "axial_volume": f"sample_axial_{c}"} for c in contrasts]).to_csv(pairs, index=False)
    run([sys.executable, "src/measurements/run_level_volumes.py", "--pairs", str(pairs),
         "--sagittal-config", configs["sagittal"], "--sagittal-run", str(OUTPUTS / "sagittal" / "run"),
         "--axial-config", configs["axial"], "--axial-run", str(OUTPUTS / "axial" / "run")])


if __name__ == "__main__":
    main()
