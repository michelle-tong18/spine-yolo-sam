"""Tests for src/evaluation: GT volume / slice lookup and the segmentation breakdown with HD95."""
import math

import nibabel as nib
import numpy as np
import pytest

from src.evaluation.detection_metrics import get_test_stems_from_gt_labels, volumes_without_predictions
from src.evaluation.segmentation_metrics import (
    compute_segmentation_metrics_by_scope,
    get_test_stems_from_gt_masks,
    volume_hd95,
)
from src.inference.sam_from_boxes import ordered_yolo_txt_stems
from src.preprocessing.spacing import write_spacing_record


def lines(n_slices, gt_slices, pred_slices, gt_col=5, pred_col=8):
    """(gt, pred) volumes of 1-px vertical lines (rows 5-15): GT at ``gt_col``, prediction at ``pred_col``."""
    gt, pred = np.zeros((2, n_slices, 20, 20), bool)
    gt[gt_slices, 5:16, gt_col] = True
    pred[pred_slices, 5:16, pred_col] = True
    return gt, pred


def test_hd95_is_the_in_plane_offset_in_mm():
    gt, pred = lines(1, [0], [0])
    out = volume_hd95(gt, pred, 0.5, 0.5)
    assert out["hd95"] == pytest.approx(1.5) and out["hd95_pooled"] == pytest.approx(1.5)
    assert out["hd95_undefined_frac"] == 0


def test_hd95_counts_slices_with_only_gt_or_only_prediction_as_undefined():
    gt, pred = lines(3, [0, 1], [0])
    out = volume_hd95(gt, pred, 0.5, 0.5)
    assert out["hd95"] == pytest.approx(1.5)
    assert out["hd95_undefined_frac"] == pytest.approx(0.5)


def test_hd95_is_nan_without_spacing():
    gt, pred = lines(1, [0], [0])
    assert all(math.isnan(v) for v in volume_hd95(gt, pred, float("nan"), float("nan")).values())


def test_gt_stems_accept_subject_and_subject_series_names(tmp_path):
    for name in ("P1.nii.gz", "P2_T1sag.nii.gz", "P2_T2sag.nii.gz", "P22_T2sag.nii.gz"):
        (tmp_path / name).touch()
    assert get_test_stems_from_gt_masks(tmp_path, ["P2", "P1"]) == ["P2_T1sag", "P2_T2sag", "P1"]
    for name in ("P1-002.txt", "P1-010.txt", "P2_T2sag-001.txt", "P22_T2sag-001.txt"):
        (tmp_path / name).touch()
    assert get_test_stems_from_gt_labels(tmp_path, ["P1", "P2"]) == ["P1-002", "P1-010", "P2_T2sag-001"]


def test_breakdown_reports_hd95_where_preprocessing_recorded_spacing(tmp_path, capsys):
    gt_dir, pred_dir = tmp_path / "nifti_preprocessed" / "mask", tmp_path / "seg_pred"
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir()
    gt, pred = lines(1, [0], [0])
    for stem in ("P1_T2sag", "P2_T2sag"):
        nib.save(nib.Nifti1Image(gt.astype(np.uint8), np.eye(4)), gt_dir / f"{stem}.nii.gz")
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.eye(4)), pred_dir / f"{stem}.nii.gz")
    write_spacing_record(str(gt_dir.parent), {"volume_stem": "P1_T2sag", "slice_mm": 4.0, "row_mm": 0.5, "col_mm": 0.5})

    df = compute_segmentation_metrics_by_scope(str(pred_dir), str(gt_dir), ["P1", "P2"], {0: "background", 1: "line"}, {})
    series = df[(df["scope"] == "series") & (df["level"] == "label")].set_index("patient_id")
    assert series.loc["P1", "hd95"] == pytest.approx(1.5)
    assert math.isnan(series.loc["P2", "hd95"])
    assert series.loc["P1", "dice"] == 0
    overall = df[(df["scope"] == "overall") & (df["level"] == "label")].iloc[0]
    assert overall["hd95"] == pytest.approx(1.5) and overall["n_series_scored"] == 2
    assert "no pixel spacing recorded for 1 volume" in capsys.readouterr().out


def test_volumes_whose_predictions_are_named_differently_from_gt_are_reported(tmp_path):
    for name in ("P1-004.txt", "P2_T2sag-000.txt"):
        (tmp_path / name).touch()
    gt_stems = ["P1-004", "P1-007", "P2_T2sag-012", "P3-001"]
    assert volumes_without_predictions(tmp_path, gt_stems) == ["P2_T2sag", "P3"]


def test_label_files_numbered_by_scan_slice_pair_with_volume_slices_in_order(tmp_path):
    for name in ("P1-010.txt", "P1-003.txt", "P1-007.txt", "P10-001.txt"):
        (tmp_path / name).touch()
    assert ordered_yolo_txt_stems("P1", str(tmp_path)) == ["P1-003", "P1-007", "P1-010"]

