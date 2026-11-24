"""Tests for src/preprocessing/spacing.py: header spacing, its fallbacks, and the preprocessed grid spacing."""
import math

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from src.preprocessing.spacing import (
    SPACING_CSV,
    affine_from_record,
    grid_inplane_mm,
    load_grid_spacing,
    loaded_index_from_transformed,
    read_header_spacing,
    read_header_world,
    slice_numbers,
    transformed_index_from_grid,
    spacing_record,
    transform_axes,
    write_spacing_record,
)

PREP = {"image_size": 1024, "make_square": True, "ratio_resize": False}


def write_dicom_series(folder, n_slices=3, pixel_spacing=(0.5, 0.6), slice_step=4.0, orientation=None):
    """Minimal MR series: InstanceNumber order, ImagePositionPatient ``slice_step`` mm apart along z (optional
    ImageOrientationPatient)."""
    folder.mkdir()
    for i in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{i}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0.0, 0.0, slice_step * i]
        if pixel_spacing is not None:
            ds.PixelSpacing = list(pixel_spacing)
        if orientation is not None:
            ds.ImageOrientationPatient = list(orientation)
        ds.Rows, ds.Columns = 4, 4
        ds.save_as(str(folder / f"{i}.dcm"), enforce_file_format=True)
    return folder


def test_dicom_series_spacing_from_positions_and_pixel_spacing(tmp_path):
    folder = write_dicom_series(tmp_path / "series")
    assert read_header_spacing(str(folder)) == pytest.approx((4.0, 0.5, 0.6))


def test_dicom_series_without_pixel_spacing_has_no_header_spacing(tmp_path):
    folder = write_dicom_series(tmp_path / "series", pixel_spacing=None)
    assert read_header_spacing(str(folder)) is None


def test_nifti_without_orientation_or_zooms_has_no_header_spacing(tmp_path):
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4), np.uint8), None), tmp_path / "bare.nii.gz")  # no qform / sform
    scanned = nib.Nifti1Image(np.zeros((2, 3, 4), np.uint8), np.diag([3.0, 0.4, 0.5, 1.0]))
    nib.save(scanned, tmp_path / "scanned.nii.gz")
    assert read_header_spacing(str(tmp_path / "bare.nii.gz")) is None
    assert read_header_spacing(str(tmp_path / "scanned.nii.gz")) == pytest.approx((3.0, 0.4, 0.5))


def test_metaimage_spacing_is_reversed_to_array_order(tmp_path):
    header = tmp_path / "vol.mhd"
    header.write_text("NDims = 3\nDimSize = 4 3 2\nElementSpacing = 0.5 0.6 4.0\nElementDataFile = vol.raw\n")
    assert read_header_spacing(str(header)) == pytest.approx((4.0, 0.6, 0.5))


def test_transform_axes_follows_transpose_swapaxes_rot90_and_flip():
    assert transform_axes((1, 2, 3), {"transpose": [2, 0, 1]}) == (3, 1, 2)
    assert transform_axes((1, 2, 3), {"swapaxes": [0, 2], "flip": 1}) == (3, 2, 1)
    assert transform_axes((1, 2, 3), {"rot90": [1, 2], "percentile": 99}) == (1, 3, 2)


def test_grid_inplane_mm_for_square_padding_and_ratio_resize():
    # 512 x 256 at 0.5 mm: padded to 512², resized to 1024² -> 0.25 mm both ways
    assert grid_inplane_mm((512, 256), (0.5, 0.5), 1024, True, False) == pytest.approx((0.25, 0.25))
    # no padding: each axis stretched to 1024
    assert grid_inplane_mm((512, 256), (0.5, 0.5), 1024, False, False) == pytest.approx((0.25, 0.125))
    # long side to 1024, aspect kept
    assert grid_inplane_mm((512, 256), (0.5, 0.5), 1024, False, True) == pytest.approx((0.25, 0.25))


def test_record_falls_back_to_config_then_missing(tmp_path, capsys):
    folder = write_dicom_series(tmp_path / "series", pixel_spacing=None)
    from_config = spacing_record("s1", str(folder), (3, 512, 512), {}, {**PREP, "pixel_spacing_mm": [4, 0.5, 0.5]}, [0, 1, 2])
    missing = spacing_record("s1", str(folder), (3, 512, 512), {}, PREP, [0, 1, 2])
    assert from_config["spacing_source"] == "config"
    assert from_config["row_mm"] == pytest.approx(0.25)
    assert missing["spacing_source"] == "missing"
    assert math.isnan(missing["row_mm"])
    assert "no voxel spacing" in capsys.readouterr().out
    assert np.array_equal(affine_from_record(missing), np.eye(4))


def test_record_spacing_follows_the_yaml_transforms(tmp_path):
    folder = write_dicom_series(tmp_path / "series")  # loaded (slice, row, col) = (4.0, 0.5, 0.6) mm
    record = spacing_record("s1", str(folder), (3, 512, 256), {"transpose": [2, 0, 1]}, PREP, [0, 1])
    assert record["shape"] == "256;3;512"
    assert record["slice_mm"] == pytest.approx(0.6)
    assert (record["row_mm"], record["col_mm"]) == pytest.approx((4.0 * 512 / 1024, 0.5 * 512 / 1024))


def test_load_grid_spacing_reads_spacing_csv_only(tmp_path):
    img_dir = tmp_path / "nifti_preprocessed" / "img"
    img_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros((2, 4, 4), np.uint8), np.diag([3.0, 0.4, 0.4, 1.0])), img_dir / "a.nii.gz")
    assert load_grid_spacing("a", str(img_dir)) is None  # an affine alone is not a record

    write_spacing_record(str(tmp_path / "nifti_preprocessed"), {"volume_stem": "a", "slice_mm": 5.0, "row_mm": 0.3, "col_mm": 0.3})
    write_spacing_record(str(tmp_path / "nifti_preprocessed"), {"volume_stem": "a", "slice_mm": 6.0, "row_mm": 0.2, "col_mm": 0.2})
    assert len(pd.read_csv(tmp_path / "nifti_preprocessed" / SPACING_CSV)) == 1
    assert load_grid_spacing("a", str(img_dir)) == pytest.approx((6.0, 0.2, 0.2))
    assert load_grid_spacing("b", str(img_dir)) is None


def test_slice_numbers_are_the_kept_slices_else_positions(tmp_path, capsys):
    img_dir = tmp_path / "nifti_preprocessed" / "img"
    img_dir.mkdir(parents=True)
    assert slice_numbers("a", str(img_dir), 3) == [0, 1, 2]  # no spacing.csv yet
    write_spacing_record(str(img_dir.parent), {"volume_stem": "a", "kept_slices": "4;7;9"})
    assert slice_numbers("a", str(img_dir), 3) == [4, 7, 9]
    assert slice_numbers("b", str(img_dir), 2) == [0, 1]  # volume not recorded
    assert slice_numbers("a", str(img_dir), 2) == [0, 1]  # count differs: positions, with a warning
    assert "2 in the volume" in capsys.readouterr().out


def test_dicom_world_geometry_follows_orientation_positions_and_pixel_spacing(tmp_path):
    # Columns along x (row direction cosines), rows along y, slices 4 mm apart along z.
    folder = write_dicom_series(tmp_path / "series", orientation=[1, 0, 0, 0, 1, 0])
    world = read_header_world(str(folder))
    assert world @ np.array([2, 3, 5, 1]) == pytest.approx([5 * 0.6, 3 * 0.5, 2 * 4.0, 1])
    assert read_header_world(str(write_dicom_series(tmp_path / "no_orientation"))) is None


def test_loaded_index_map_inverts_transpose_and_flip():
    data = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    transforms = {"transpose": [2, 0, 1], "flip": 1}
    out = np.flip(np.transpose(data, [2, 0, 1]), axis=1)
    m = loaded_index_from_transformed(data.shape, transforms)
    for idx in [(0, 0, 0), (4, 2, 1), (1, 1, 3)]:
        loaded = (m @ np.array([*idx, 1]))[:3].round().astype(int)
        assert out[idx] == data[tuple(loaded)]
    assert loaded_index_from_transformed(data.shape, {"rot90": [1, 2]}) is None


def test_grid_index_map_undoes_resize_and_square_padding():
    # 512 x 256 slices padded to 512 x 512 (128 columns each side), resized to 1024: pixel centres scale by 0.5.
    m = transformed_index_from_grid((512, 256), [4, 6, 8], 1024, make_square=True, ratio_resize=False)
    assert m @ np.array([2, 1023, 512, 1]) == pytest.approx([8, 511.25, 127.75, 1])
    assert transformed_index_from_grid((512, 256), [0, 2, 3], 1024, True, False) is None  # uneven kept slices


def test_nifti_affine_is_the_world_geometry_in_ras_when_recorded():
    world = np.array([[0, 0, 0.6, 10], [0, 0.5, 0, 20], [4, 0, 0, 30], [0, 0, 0, 1]], float)
    affine = affine_from_record({"world_affine": ";".join(map(str, world.ravel())), "slice_mm": 4, "row_mm": 0.5, "col_mm": 0.6})
    assert affine == pytest.approx(np.diag([-1, -1, 1, 1]) @ world)

