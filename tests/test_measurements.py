"""Tests for src/measurements: disc geometry and label volumes on synthetic sagittal data, and the per-run tables."""
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import rotate

import src.measurements.disc_measurements as dm
from src.measurements.run_measurements import OUTPUT_CSV, measure_run
from src.measurements.volumes import label_volume_rows
from src.preprocessing.spacing import write_spacing_record

UPPER, DISC, LOWER, CANAL = 3, 2, 1, 20  # L4, L4-L5, L5, spinal canal


def close(measured: dict, expected: dict, tol: float = 1e-6) -> bool:
    """Every expected measurement within ``tol`` (the disc frame can tilt by a rounding error)."""
    return all(abs(measured[k] - v) < tol for k, v in expected.items())


def _spine_slice():
    """Upper vertebra 20 px, disc 8 px, lower vertebra 20 px tall, 40 px deep; canal 10 px wide just posterior."""
    seg = np.zeros((80, 90), int)
    seg[10:30, 20:60], seg[30:38, 20:60], seg[38:58, 20:60], seg[0:70, 62:72] = UPPER, DISC, LOWER, CANAL
    img = np.full(seg.shape, 100.0)
    img[seg == DISC], img[seg == CANAL] = 50.0, 200.0
    return seg, img


def test_normal_disc_heights_signal_and_canal():
    seg, img = _spine_slice()
    m = dm.measure_disc_level(seg, img, DISC, UPPER, LOWER, CANAL, 1.0, 1.0)
    assert close(m, {"disc_height_mid_mm": 8, "vert_height_upper_mm": 20, "disc_height_index_mid": 0.4,
                     "disc_canal_signal_ratio": 0.25, "posterior_high_signal_fraction": 0, "posterior_displacement_mm": 0,
                     "area_beyond_line_mm2": 0})
    half = dm.measure_disc_level(seg, img, DISC, UPPER, LOWER, CANAL, 0.5, 0.5)  # 0.5 mm pixels: mm halve, ratios stay
    assert close(half, {"disc_height_mid_mm": 4, "area_beyond_line_mm2": 0, "disc_height_index_mid": 0.4})


def test_herniation_displacement_and_area_beyond_the_corner_line():
    seg, img = _spine_slice()
    seg[33:35, 60] = DISC   # 2 px just past the vertebral corners
    seg[30:42, 61] = DISC   # 12 px one column further, reaching below the lower vertebra's corner
    m = dm.measure_disc_level(seg, img, DISC, UPPER, LOWER, CANAL, 1.0, 1.0)
    assert close(m, {"posterior_displacement_mm": 2, "area_beyond_line_mm2": 14}, tol=1e-3)


def test_measurements_follow_a_tilted_spine():
    seg, img = _spine_slice()
    tilted = rotate(seg, 25, order=0, reshape=True)
    tilted_img = rotate(img, 25, order=0, reshape=True)
    m = dm.measure_disc_level(tilted, tilted_img, DISC, UPPER, LOWER, CANAL, 1.0, 1.0)
    assert abs(m["disc_height_index_mid"] - 0.4) < 0.06 and abs(m["disc_canal_signal_ratio"] - 0.25) < 0.02
    assert m["posterior_displacement_mm"] < 1.5


def test_highlight_of_bright_posterior_disc():
    seg, img = _spine_slice()
    img[30:38, 52:60] = 250.0  # posterior quarter of the disc brighter than the canal
    assert close(dm.measure_disc_level(seg, img, DISC, UPPER, LOWER, CANAL, 1.0, 1.0), {"posterior_high_signal_fraction": 0.8})


def test_missing_neighbours_give_nan_not_errors():
    seg, img = _spine_slice()
    seg[seg == LOWER] = 0
    m = dm.measure_disc_level(seg, img, DISC, UPPER, LOWER, CANAL, 1.0, 1.0)
    assert close(m, {"disc_height_index_mid": 8 / 20})  # upper vertebra alone
    assert np.isnan(dm.measure_disc_level(seg * 0, img, DISC, UPPER, LOWER, CANAL, 1.0, 1.0)["disc_height_mid_mm"])


def test_disc_neighbours_and_midline_slices():
    labels = {0: "background", 1: "L5", 2: "L4-L5", 3: "L4", 12: "L5-S1", 13: "S1", 20: "spinal_canal"}
    assert dm.disc_neighbours(labels) == {2: (3, 1), 12: (1, 13)}
    vol = np.zeros((9, 4, 4), int)
    vol[5, :, :2] = CANAL
    vol[4, :, :1] = CANAL
    assert dm.mid_sagittal_slices(vol, CANAL, 3) == [4, 5, 6] and dm.mid_sagittal_slices(vol, CANAL, 3)[1] == 5
    vol[:] = 0
    vol[8, :, :2] = CANAL
    assert dm.mid_sagittal_slices(vol, CANAL, 3) == [6, 7, 8]  # clipped at the volume edge


def test_measure_volume_takes_the_median_over_midline_slices():
    seg, img = _spine_slice()
    vol, ivol = np.stack([seg] * 5), np.stack([img] * 5)
    vol[2, 30:38, 20:60] = 0
    vol[2, 31:37, 20:60] = DISC  # one of the three mid slices has a 6 px disc
    vol[2, 0:70, 72:75] = CANAL  # widest canal: slice 2 is the midline, so slices 1-3 are measured
    labels = {1: "L5", 2: "L4-L5", 3: "L4", 20: "spinal_canal"}
    (row,) = dm.measure_discs_in_volume(vol, ivol, labels, ["L4-L5"], 3, 1.0, 1.0)
    assert row["level"] == "L4-L5" and row["n_slices_with_disc"] == 3 and close(row, {"disc_height_mid_mm": 8})


def test_disc_geometry_gives_the_corners_and_line_the_measurements_use():
    seg, _ = _spine_slice()
    g = dm.disc_geometry(seg, DISC, UPPER, LOWER, CANAL, 1.0, 1.0)
    assert len(g["corners"]) == 2 and g["line"] is not None
    assert g["u"] @ (g["pts"]["canal"].mean(axis=0) - g["centroid"]) > 0  # u points toward the canal
    assert dm.disc_geometry(np.zeros_like(seg), DISC, UPPER, LOWER, CANAL, 1.0, 1.0) is None


LABELS = {0: "background", 1: "L5", 2: "L4-L5", 3: "L4", 20: "spinal_canal"}


@pytest.fixture
def spine_run(tmp_path):
    """Run dir with seg_pred/ for volumes 'a' (spacing recorded, 0.5 mm) and 'b' (none), and their images."""
    seg, img = _spine_slice()
    run, img_dir = tmp_path / "run", tmp_path / "nifti_preprocessed" / "img"
    (run / "seg_pred").mkdir(parents=True)
    img_dir.mkdir(parents=True)
    for stem in ("a", "b"):
        nib.save(nib.Nifti1Image(np.stack([seg] * 3).astype(np.int16), np.eye(4)), run / "seg_pred" / f"{stem}.nii.gz")
        nib.save(nib.Nifti1Image(np.stack([img] * 3).astype(np.float32), np.eye(4)), img_dir / f"{stem}.nii.gz")
    write_spacing_record(str(img_dir.parent), {"volume_stem": "a", "slice_mm": 4.0, "row_mm": 0.5, "col_mm": 0.5})
    return run, img_dir


def test_disc_table_uses_recorded_spacing_and_leaves_mm_nan_without_it(spine_run, capsys):
    run, img_dir = spine_run
    measure_run({"mask_labels": LABELS}, str(run), str(img_dir), ["discs"])
    table = pd.read_csv(run / OUTPUT_CSV["discs"]).set_index("volume_stem")
    assert table.loc["a", "disc_height_mid_mm"] == pytest.approx(4.0)
    assert np.isnan(table.loc["b", "disc_height_mid_mm"]) and np.isnan(table.loc["b", "row_mm"])
    assert table.loc["b", "disc_height_index_mid"] == pytest.approx(0.4)  # scale free: measured in pixels
    assert "b: no pixel spacing" in capsys.readouterr().out
    assert not (run / OUTPUT_CSV["volumes"]).exists()  # only the requested metrics are written


def test_disc_table_skips_a_volume_whose_image_does_not_match(spine_run, capsys):
    run, img_dir = spine_run
    nib.save(nib.Nifti1Image(np.zeros((2, 4, 4), np.float32), np.eye(4)), img_dir / "b.nii.gz")
    measure_run({"mask_labels": LABELS}, str(run), str(img_dir), ["discs", "volumes"])
    assert set(pd.read_csv(run / OUTPUT_CSV["discs"])["volume_stem"]) == {"a"}
    assert set(pd.read_csv(run / OUTPUT_CSV["volumes"])["volume_stem"]) == {"a", "b"}  # volumes need no image
    assert "b: no image matching" in capsys.readouterr().out


def test_volume_table_in_cm3_where_spacing_was_recorded(spine_run):
    run, img_dir = spine_run
    cfg = {"mask_labels": LABELS, "evaluation": {"tissue_groups": {"vertebrae": [1, 3]}}}
    measure_run(cfg, str(run), str(img_dir), ["volumes"])
    table = pd.read_csv(run / OUTPUT_CSV["volumes"]).set_index(["volume_stem", "class"])
    disc_voxels = 3 * 8 * 40  # 3 slices of an 8 x 40 px disc
    assert table.loc[("a", "L4-L5"), "voxels"] == disc_voxels
    assert table.loc[("a", "L4-L5"), "volume_cm3"] == pytest.approx(disc_voxels * 4.0 * 0.5 * 0.5 / 1000)
    assert table.loc[("a", "vertebrae"), "voxels"] == 3 * 2 * 20 * 40
    assert np.isnan(table.loc[("b", "L4-L5"), "volume_cm3"])


def test_label_volume_rows_count_each_label_and_tissue():
    seg = np.array([[[0, 1, 1], [2, 2, 2]]])
    rows = label_volume_rows(seg, {0: "background", 1: "a", 2: "b", 3: "c"}, {"ab": [1, 2]}, 2.0)
    by_class = {r["class"]: r for r in rows}
    assert [by_class[c]["voxels"] for c in ("a", "b", "c", "ab")] == [2, 3, 0, 5]
    assert by_class["ab"]["volume_cm3"] == pytest.approx(5 * 2.0 / 1000) and by_class["ab"]["level"] == "tissue"


def test_measurement_levels_default_to_every_disc_and_reject_unusable_label_maps():
    assert dm.measurement_levels(LABELS, {}) == ["L4-L5"]
    with pytest.raises(ValueError, match="spinal_canal"):
        dm.measurement_levels({1: "L5", 2: "L4-L5", 3: "L4"}, {})
    with pytest.raises(ValueError, match="L5-S1"):
        dm.measurement_levels(LABELS, {"levels": ["L5-S1"]})
