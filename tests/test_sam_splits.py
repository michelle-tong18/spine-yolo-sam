"""Tests for how SAM training picks each subject's slices (src/sam_finetuning/datamodule/dynamic_dataset_loader.py)."""
import os

import pytest
import yaml

from src.sam_finetuning.datamodule.dynamic_dataset_loader import get_splits, subject_slice_files


def test_subject_slices_match_the_subject_id_and_a_separator_only(tmp_path):
    for name in ("010-001.npy", "010_T2sag-004.npy", "0101-99999_X1_T1ax-002.npy", "0100-003.npy"):
        (tmp_path / name).touch()
    assert [os.path.basename(f) for f in subject_slice_files(str(tmp_path), "010")] == ["010-001.npy", "010_T2sag-004.npy"]
    assert [os.path.basename(f) for f in subject_slice_files(str(tmp_path), "0101-99999")] == ["0101-99999_X1_T1ax-002.npy"]


def test_splits_come_from_the_configured_registry_file(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"smoke": {"train": ["001", "002"], "val": ["003"], "test": []}}))
    splits = get_splits({"split_id": "smoke", "splits_registry": str(registry)})
    assert list(splits["train"]) == ["001", "002"] and list(splits["val"]) == ["003"] and len(splits["test"]) == 0
    with pytest.raises(ValueError, match="other"):
        get_splits({"split_id": "other", "splits_registry": str(registry)})

