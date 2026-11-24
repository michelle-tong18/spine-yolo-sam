"""Tests for src/measurements/levels.py and run_level_volumes.py on a synthetic patient (sagittal + axial)."""
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.measurements.levels import axial_level_volume_rows, disc_planes, vertebral_levels
from src.measurements.run_level_volumes import OUTPUT_CSV, measure_pairs
from src.preprocessing.spacing import write_spacing_record

SAGITTAL_LABELS = {0: "background", 1: "L5", 2: "L4-L5", 3: "L4", 4: "L3-L4", 5: "L3", 12: "L5-S1", 13: "S1"}
AXIAL_LABELS = {0: "background", 1: "muscle_multifidus_r", 2: "muscle_multifidus_l"}
# Sagittal rows run caudally in patient coordinates (world y = row): disc mid-planes at y = 8.5, 20.5, 32.5.
ROWS = {5: (0, 8), 4: (8, 10), 3: (10, 20), 2: (20, 22), 1: (22, 32), 12: (32, 34), 13: (34, 44)}
SAGITTAL_AFFINE = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)  # (slice, row, col) -> (col, row, slice)
AXIAL_AFFINE = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)     # axial slice z -> world y


def sagittal_seg():
    seg = np.zeros((11, 50, 20), np.int16)
    for lid, (top, bottom) in ROWS.items():
        seg[:, top:bottom, 5:16] = lid
    return seg


def axial_seg(n_slices=50):
    """Axial volume (slice = world y): multifidus right on columns 2-5, left on 10-13, on every slice."""
    seg = np.zeros((n_slices, 11, 16), np.int16)
    seg[:, 3:8, 2:6] = 1
    seg[:, 3:8, 10:14] = 2
    return seg


def test_disc_mid_planes_point_caudally_through_each_disc():
    planes = disc_planes(sagittal_seg(), SAGITTAL_AFFINE, SAGITTAL_LABELS)
    centre, normal = planes["L4-L5"]
    assert centre[1] == pytest.approx(20.5) and abs(normal[1]) == pytest.approx(1.0) and normal[1] > 0
    assert vertebral_levels(SAGITTAL_LABELS, planes) == [("L5", "L4-L5", "L5-S1"), ("L4", "L3-L4", "L4-L5")]


def test_axial_voxels_go_to_the_level_between_the_two_disc_mid_planes():
    planes = disc_planes(sagittal_seg(), SAGITTAL_AFFINE, SAGITTAL_LABELS)
    levels = vertebral_levels(SAGITTAL_LABELS, planes)
    rows = axial_level_volume_rows(axial_seg(), AXIAL_AFFINE, AXIAL_LABELS, {"multifidus": [1, 2]},
                                   planes, levels)
    by = {(r["level"], r["class"]): r for r in rows}
    # L4: slices 9..20 (centre y >= 8.5 and < 20.5) -> 12 slices of 5 x 4 px per side.
    assert by[("L4", "muscle_multifidus_r")]["voxels"] == 12 * 20
    assert by[("L4", "multifidus")]["volume_cm3"] == pytest.approx(12 * 40 / 1000)
    assert by[("L4", "multifidus")]["coverage_frac"] == pytest.approx(1.0)
    assert by[("L5", "muscle_multifidus_l")]["n_axial_slices"] == 12


def test_partial_axial_coverage_is_reported():
    # An axial scan of 25 slices starting at world y = 15 covers only y 15-20 of L4 (8.5-20.5) and all of L5.
    planes = disc_planes(sagittal_seg(), SAGITTAL_AFFINE, SAGITTAL_LABELS)
    shorter = AXIAL_AFFINE.copy()
    shorter[1, 3] = 15.0
    rows = axial_level_volume_rows(axial_seg(25), shorter, AXIAL_LABELS, {}, planes,
                                   vertebral_levels(SAGITTAL_LABELS, planes))
    by = {(r["level"], r["class"]): r for r in rows}
    l4 = by[("L4", "muscle_multifidus_r")]
    assert l4["n_axial_slices"] == 6 and l4["coverage_frac"] == pytest.approx(0.5) and l4["voxels"] == 6 * 20
    assert by[("L5", "muscle_multifidus_r")]["coverage_frac"] == pytest.approx(1.0)


def write_run(tmp_path, plane, stem, seg, affine):
    """seg_pred/<stem>.nii.gz under <plane>/run and a spacing.csv row with its world_affine (None: no row)."""
    run, img_dir = tmp_path / plane / "run", tmp_path / plane / "nifti_preprocessed" / "img"
    (run / "seg_pred").mkdir(parents=True)
    img_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(seg, np.eye(4)), run / "seg_pred" / f"{stem}.nii.gz")
    if affine is not None:
        write_spacing_record(str(img_dir.parent), {"volume_stem": stem, "world_affine": ";".join(map(str, affine.ravel()))})
    return str(run), str(img_dir)


def test_pairs_are_measured_and_a_pair_without_world_geometry_is_skipped(tmp_path, capsys):
    sag_run, sag_dir = write_run(tmp_path, "sagittal", "P1_T2sag", sagittal_seg(), SAGITTAL_AFFINE)
    ax_run, ax_dir = write_run(tmp_path, "axial", "P1_T2ax", axial_seg(), AXIAL_AFFINE)
    nib.save(nib.Nifti1Image(axial_seg(), np.eye(4)), tmp_path / "axial" / "run" / "seg_pred" / "P2_T2ax.nii.gz")
    pairs = pd.DataFrame({"patient_id": ["P1", "P2"], "sagittal_volume": ["P1_T2sag", "P1_T2sag"],
                          "axial_volume": ["P1_T2ax", "P2_T2ax"]})
    cfg = {"mask_labels": SAGITTAL_LABELS}
    measure_pairs(pairs, cfg, sag_run, sag_dir, {"mask_labels": AXIAL_LABELS}, ax_run, ax_dir)
    table = pd.read_csv(tmp_path / "axial" / "run" / OUTPUT_CSV)
    assert set(table["patient_id"]) == {"P1"} and set(table["level"]) == {"L4", "L5"}
    assert "P2: no world_affine for P2_T2ax" in capsys.readouterr().out
