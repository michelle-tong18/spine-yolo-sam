"""Tests for src/inference/figures.py (verification figures) and the --figures option of run_inference.py."""
import nibabel as nib
import numpy as np
import pytest

from src.inference.figures import FIGURES_SUBDIR, evenly_spaced_slices, save_run_figures, save_volume_figure
from src.inference.run_inference import _validate_args, build_parser
from src.preprocessing.spacing import write_spacing_record

LABELS = {0: "background", 1: "L5", 2: "L4-L5", 20: "spinal_canal"}


def test_evenly_spaced_slices_span_the_volume():
    assert evenly_spaced_slices(21, 5) == [0, 5, 10, 15, 20]
    assert evenly_spaced_slices(3, 5) == [0, 1, 2]


def test_volume_figure_is_written_for_the_chosen_slices(tmp_path):
    img = np.random.default_rng(0).random((9, 16, 16))
    seg = np.zeros((9, 16, 16), int)
    seg[:, 4:8, 4:8] = 1
    boxes = [[(1, 0.375, 0.375, 0.25, 0.25)]] * 9
    shown = save_volume_figure(img, seg, boxes, list(range(10, 19)), LABELS, tmp_path / "v.png", "v", n_slices=3)
    assert shown == [0, 4, 8]
    assert (tmp_path / "v.png").stat().st_size > 0


def test_run_figures_pair_label_files_numbered_by_scan_slice_with_the_volume(tmp_path):
    img_dir, run = tmp_path / "nifti_preprocessed" / "img", tmp_path / "run"
    img_dir.mkdir(parents=True)
    (run / "seg_pred").mkdir(parents=True)
    (run / "label_pred").mkdir()
    img = np.random.default_rng(1).integers(1, 1000, (3, 16, 16)).astype(np.uint16)
    nib.save(nib.Nifti1Image(img, np.eye(4)), img_dir / "P1_T2sag.nii.gz")
    nib.save(nib.Nifti1Image(np.ones((3, 16, 16), np.uint8), np.eye(4)), run / "seg_pred" / "P1_T2sag.nii.gz")
    write_spacing_record(str(img_dir.parent), {"volume_stem": "P1_T2sag", "kept_slices": "4;6;8"})
    for z in (4, 6, 8):
        (run / "label_pred" / f"P1_T2sag-{z:03d}.txt").write_text("1 0.5 0.5 0.25 0.25 0.9")
    cfg = {"mask_labels": LABELS, "preprocessing_cfg": {"image_size": 16}, "image_config": {}}
    saved = save_run_figures(cfg, str(run), str(img_dir))
    assert saved == [run / FIGURES_SUBDIR / "P1_T2sag.png"] and saved[0].is_file()


def test_figures_option_needs_masks():
    parser = build_parser()
    with pytest.raises(SystemExit):
        _validate_args(parser, parser.parse_args(["yolo", "--figures"]))
    _validate_args(parser, parser.parse_args(["both", "--figures"]))
