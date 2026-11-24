"""Tests for YOLO predict settings (src/inference/yolo_predict.py) and gap fill (postprocess_volume_3d)."""
import pytest
import torch

from src.inference.yolo_predict import get_volume_id_and_slice, yolo_settings
from src.yolo_finetuning.postprocess_volume_3d.config import gap_fill_config_from_dict
from src.yolo_finetuning.postprocess_volume_3d.pipeline import VolumeDet, fill_gaps

RELEVEL = {"enabled": True, "chains": {"spine": {"order": [3, 2, 1]}}}
GAP_FILL = {"enabled": True, "type": "axial_spine_muscle", "gap_fill": {"enabled": True, "max_gap_slices": 2}}


def test_enabled_relevel_block_is_passed_to_predict():
    assert yolo_settings({"models": {"obj_det": {"relevel": RELEVEL}}})["relevel"] == RELEVEL


def test_disabled_or_missing_relevel_is_off():
    assert yolo_settings({"models": {"obj_det": {"relevel": {**RELEVEL, "enabled": False}}}})["relevel"] is None
    assert yolo_settings({"models": {"obj_det": {}}})["relevel"] is None


def test_gap_fill_runs_only_when_postprocess_and_gap_fill_are_enabled():
    assert yolo_settings({"models": {"obj_det": {"postprocess": GAP_FILL}}})["gap_fill"].max_gap_slices == 2
    assert gap_fill_config_from_dict({**GAP_FILL, "enabled": False}) is None
    assert gap_fill_config_from_dict({**GAP_FILL, "gap_fill": {"enabled": False}}) is None
    assert gap_fill_config_from_dict(None) is None


def test_removed_or_misspelled_postprocess_keys_are_errors():
    with pytest.raises(ValueError, match="clustering"):
        gap_fill_config_from_dict({**GAP_FILL, "clustering": {"enabled": False}})
    with pytest.raises(ValueError, match="max_gap"):
        gap_fill_config_from_dict({**GAP_FILL, "gap_fill": {"enabled": True, "max_gap": 4}})


def box(slice_idx, label, conf, x1, size=10.0):
    return VolumeDet(slice_idx, label, conf, torch.tensor([x1, 0.0, x1 + size, size]))


def test_gaps_up_to_max_slices_get_the_larger_boundary_box():
    dets = [box(0, 1, 0.9, 0.0), box(3, 1, 0.8, 5.0, size=20.0), box(9, 1, 0.9, 0.0)]  # gaps of 2 and 5 slices
    filled = fill_gaps(dets, n_slices=10, cfg=gap_fill_config_from_dict(GAP_FILL))
    synthetic = [d for d in dets if d.confidence == 0.05]
    assert filled == 2 and sorted(d.slice_idx for d in synthetic) == [1, 2]
    assert all(torch.equal(d.xyxy, dets[1].xyxy) for d in synthetic)  # larger area of slices 0 and 3


def test_volume_id_and_slice_from_a_label_stem():
    assert get_volume_id_and_slice("0102-00031_C513196_T1sag-007") == ("0102-00031_C513196_T1sag", 7)
