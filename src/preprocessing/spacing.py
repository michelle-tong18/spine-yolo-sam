"""
Voxel spacing of each preprocessed volume, recorded when preprocessing first reads the scan.

Preprocessing reorders axes (the YAML ``transforms``), keeps a subset of slices, pads and resizes every slice to
``image_size``², so the native header spacing no longer describes the saved array. ``spacing_record`` follows the
same steps and gives the spacing of the preprocessed grid. It is written twice:
    the NIfTI affine of ``nifti_preprocessed/{img,mask}/<volume>.nii.gz`` (diagonal: slice, row, col mm), and
    ``nifti_preprocessed/spacing.csv``, one row per volume: source scan, ``spacing_source`` (header | config |
    missing), ``shape`` / ``spacing_mm`` of the array after the YAML transforms (before cropping and resizing),
    ``kept_slices`` (indices into that array, in saved order) and ``slice_mm`` / ``row_mm`` / ``col_mm`` of the
    preprocessed grid. ``slice_mm`` is the native slice step: kept slices need not be adjacent. ``world_affine``
    (16 numbers, row-major) maps a preprocessed voxel index (slice, row, col) to patient coordinates in mm (DICOM
    LPS); the NIfTI affine is its RAS form, so viewers place sagittal and axial volumes of one patient together.

Header spacing, by input format (array order as ``src/preprocessing/data_loaders`` loads it):
    DICOM folder  rows / cols from PixelSpacing (else ImagerPixelSpacing); slices from the distance between the
                  first two ImagePositionPatient (InstanceNumber order), else SpacingBetweenSlices, else SliceThickness
    DICOM file    PixelSpacing (else ImagerPixelSpacing)
    NIfTI         header zooms, unless the header has no qform / sform and every zoom is 1 (no spacing stored)
    .mha / .mhd   ElementSpacing
    other formats none
Readers that silently default to 1 mm (SimpleITK, nibabel) are not trusted for this. When the header has no spacing,
``preprocessing_cfg.pixel_spacing_mm`` ([slice, row, col] mm of the array after the YAML transforms) is used; without
it, spacing is nan and ``spacing_source`` is ``missing``: mm measurements (HD95, disc heights) are then nan.

World geometry (``world_affine``) needs the scan's position and orientation: DICOM ImagePositionPatient and
ImageOrientationPatient, or the NIfTI / MetaImage header (read with SimpleITK, header only). It is left empty when the
header has none, when the YAML transforms include rot90 or drop_dim, or when the kept slices are not evenly spaced
(cropping to scattered labelled slices); measurements across two scans (src/measurements/levels.py) then skip the
volume. Image-only preprocessing keeps every slice.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

from src.preprocessing.data_loaders.raw_file_loaders import DICOM_EXTENSIONS

SPACING_CSV = "spacing.csv"
SPACING_COLUMNS = ["volume_stem", "source", "spacing_source", "shape", "spacing_mm",
                   "kept_slices", "slice_mm", "row_mm", "col_mm", "world_affine"]
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


# ---------------------------------------------------------------- header spacing, in loader array order
def _dicom_inplane(ds) -> tuple[float, float] | None:
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        value = getattr(ds, tag, None)
        if value is not None and len(value) == 2 and all(float(v) > 0 for v in value):
            return float(value[0]), float(value[1])
    return None


def _dicom_series_headers(dir_path: str) -> list:
    """Headers of a DICOM folder in InstanceNumber order, the slice order ``load_dcm`` stacks them in."""
    headers = []
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if name.startswith(".") or not os.path.isfile(path) or os.path.splitext(name)[1].lower() not in DICOM_EXTENSIONS:
            continue
        try:
            headers.append(pydicom.dcmread(path, stop_before_pixels=True))
        except pydicom.errors.InvalidDicomError:
            continue
    return sorted((h for h in headers if hasattr(h, "InstanceNumber")), key=lambda h: int(h.InstanceNumber))


def _dicom_dir_spacing(dir_path: str) -> tuple[float, float, float] | None:
    headers = _dicom_series_headers(dir_path)
    if not headers:
        return None
    inplane = _dicom_inplane(headers[0])
    if inplane is None:
        return None
    slice_mm = None
    positions = [getattr(h, "ImagePositionPatient", None) for h in headers[:2]]
    if len(headers) > 1 and all(p is not None for p in positions):
        slice_mm = float(np.linalg.norm(np.subtract([float(v) for v in positions[1]], [float(v) for v in positions[0]])))
    for tag in ("SpacingBetweenSlices", "SliceThickness"):
        if not slice_mm:
            value = getattr(headers[0], tag, None)
            slice_mm = float(value) if value is not None else None
    if not slice_mm or slice_mm <= 0:
        return None
    return slice_mm, inplane[0], inplane[1]


def _nifti_spacing(path: str) -> tuple[float, ...] | None:
    header = nib.load(path).header
    zooms = tuple(float(z) for z in header.get_zooms()[: len(header.get_data_shape())])
    has_orientation = int(header["qform_code"]) > 0 or int(header["sform_code"]) > 0
    if not has_orientation and all(z == 1.0 for z in zooms):
        return None
    return zooms


def _metaimage_spacing(path: str) -> tuple[float, ...] | None:
    """ElementSpacing (x y z) from a .mha / .mhd text header, reversed to array order (z, y, x)."""
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("latin-1").strip()
            key, _, value = line.partition("=")
            if key.strip() == "ElementSpacing":
                return tuple(float(v) for v in value.split())[::-1]
            if key.strip() == "ElementDataFile":
                return None
    return None


def read_header_spacing(path: str, no_dicom_ext_flag: bool = False) -> tuple[float, ...] | None:
    """Spacing (mm) per axis of the array the data loaders return for ``path``; None when the header has none."""
    if os.path.isdir(path):
        return _dicom_dir_spacing(path)
    lower = path.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return _nifti_spacing(path)
    if lower.endswith((".mha", ".mhd")):
        return _metaimage_spacing(path)
    if os.path.splitext(lower)[1] in DICOM_EXTENSIONS or no_dicom_ext_flag:
        return _dicom_inplane(pydicom.dcmread(path, stop_before_pixels=True))
    return None


# ---------------------------------------------------------------- header world geometry, in loader array order
def read_header_world(path: str) -> np.ndarray | None:
    """
    4x4 affine from an index of the array the data loaders return for ``path`` to patient coordinates (mm, LPS);
    None when the header has no position / orientation or the scan is 2D.
    DICOM folder: axis 0 = slices in InstanceNumber order (step between the first and last ImagePositionPatient),
    axis 1 = rows (ImageOrientationPatient column direction x PixelSpacing[0]), axis 2 = columns (row direction x
    PixelSpacing[1]). NIfTI (nibabel order i, j, k) and MetaImage (SimpleITK array order k, j, i) from SimpleITK's
    header, which is LPS.
    """
    affine = np.eye(4)
    if os.path.isdir(path):
        headers = _dicom_series_headers(path)
        h0 = headers[0] if headers else None
        positions = [getattr(h, "ImagePositionPatient", None) for h in headers]
        orientation = getattr(h0, "ImageOrientationPatient", None)
        inplane = _dicom_inplane(h0) if h0 is not None else None
        if len(headers) < 2 or orientation is None or inplane is None or any(p is None for p in positions):
            return None
        first, last = np.array(positions[0], float), np.array(positions[-1], float)
        row_dir, col_dir = np.array(orientation[:3], float), np.array(orientation[3:], float)
        affine[:3, 0] = (last - first) / (len(headers) - 1)
        affine[:3, 1] = col_dir * inplane[0]
        affine[:3, 2] = row_dir * inplane[1]
        affine[:3, 3] = first
        return affine
    lower = path.lower()
    if not lower.endswith((".nii", ".nii.gz", ".mha", ".mhd")):
        return None
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    if reader.GetDimension() != 3:
        return None
    direction = np.array(reader.GetDirection(), float).reshape(3, 3)
    spacing = np.array(reader.GetSpacing(), float)
    axes = [0, 1, 2] if lower.endswith((".nii", ".nii.gz")) else [2, 1, 0]  # array axis -> SimpleITK index axis
    for array_axis, itk_axis in enumerate(axes):
        affine[:3, array_axis] = direction[:, itk_axis] * spacing[itk_axis]
    affine[:3, 3] = reader.GetOrigin()
    return affine


def loaded_index_from_transformed(loaded_shape: tuple, transforms: dict) -> np.ndarray | None:
    """4x4 map from an index of the array after the YAML ``transforms`` back to the loaded array's index
    (transpose, swapaxes, flip); None for rot90 / drop_dim, which this does not follow."""
    m, shape = np.eye(4), list(loaded_shape)
    for name, args in (transforms or {}).items():
        if name in ("rot90", "drop_dim"):
            return None
        if name in ("transpose", "swapaxes"):
            axes = list(args) if name == "transpose" else _swapped(len(shape), *args)
            step = np.zeros((4, 4))
            step[3, 3] = 1.0
            for new_axis, old_axis in enumerate(axes):
                step[old_axis, new_axis] = 1.0
            m, shape = m @ step, [shape[a] for a in axes]
        elif name == "flip":
            step = np.eye(4)
            for axis in np.atleast_1d(args):
                step[axis, axis], step[axis, 3] = -1.0, shape[axis] - 1
            m = m @ step
    return m


def _swapped(n: int, a: int, b: int) -> list[int]:
    axes = list(range(n))
    axes[a], axes[b] = b, a
    return axes


def transformed_index_from_grid(shape_hw: tuple[int, int], kept_slices: list[int], image_size: int,
                                make_square: bool, ratio_resize: bool) -> np.ndarray | None:
    """
    4x4 map from a preprocessed voxel index (slice, row, col) to the index of the array after the YAML transforms:
    slice k -> kept_slices[k] (evenly spaced kept slices only, else None), and each pixel centre undone through the
    resize and the pad of ``ImagePrep.resize_images``.
    """
    steps = set(np.diff(kept_slices)) if len(kept_slices) > 1 else {1}
    if len(steps) != 1 or min(steps) < 1:
        return None
    (h, w) = shape_hw
    m = np.eye(4)
    m[0, 0], m[0, 3] = float(steps.pop()), float(kept_slices[0])
    if ratio_resize:
        r = image_size / max(h, w)
        sizes = [(n, min(math.ceil(n * r), image_size) if r != 1 else n, 0) for n in (h, w)]
    else:
        padded = max(h, w) if make_square else None
        sizes = [(padded or n, image_size, ((padded - n) // 2) if padded else 0) for n in (h, w)]
    for axis, (extent, resized, pad) in zip((1, 2), sizes):
        scale = extent / resized  # skimage / cv2 resize: pixel centres (i + 0.5) * scale - 0.5
        m[axis, axis], m[axis, 3] = scale, 0.5 * scale - 0.5 - pad
    return m


def world_affine(source: str, loaded_shape: tuple, transforms: dict, preprocessing_cfg: dict,
                 kept_slices: list[int]) -> np.ndarray | None:
    """Preprocessed voxel index -> patient mm (LPS), or None when any step cannot be followed (module docstring)."""
    if len(loaded_shape) != 3:
        return None
    native = read_header_world(source)
    to_loaded = loaded_index_from_transformed(tuple(loaded_shape), transforms)
    shape = transform_axes(tuple(loaded_shape), transforms)
    to_transformed = transformed_index_from_grid(shape[1:], list(kept_slices), int(preprocessing_cfg.get("image_size", 1024)),
                                                 bool(preprocessing_cfg.get("make_square", False)),
                                                 bool(preprocessing_cfg.get("ratio_resize", False)))
    if native is None or to_loaded is None or to_transformed is None:
        return None
    return native @ to_loaded @ to_transformed


# ---------------------------------------------------------------- preprocessing steps
def transform_axes(values: tuple, transforms: dict) -> tuple:
    """Per-axis values (spacing or shape) carried through the axis changes of ``transform_data``: transpose,
    swapaxes, rot90 (its two axes swap lengths) and drop_dim; flip and intensity transforms keep them."""
    values = list(values)
    for name, args in (transforms or {}).items():
        if name == "transpose":
            values = [values[i] for i in args]
        elif name in ("swapaxes", "rot90"):
            a, b = args
            values[a], values[b] = values[b], values[a]
        elif name == "drop_dim":
            del values[args[0]]
    return tuple(values)


def grid_inplane_mm(shape_hw: tuple[int, int], spacing_hw: tuple[float, float], image_size: int,
                    make_square: bool, ratio_resize: bool) -> tuple[float, float]:
    """(row, col) mm of one pixel after ``ImagePrep.resize_images``: pad to square (``make_square``) and resize to
    ``image_size``², or with ``ratio_resize`` scale the long side to ``image_size``."""
    (h, w), (sy, sx) = shape_hw, spacing_hw
    if ratio_resize:
        r = image_size / max(h, w)
        if r == 1:
            return sy, sx
        new_h, new_w = min(math.ceil(h * r), image_size), min(math.ceil(w * r), image_size)
        return sy * h / new_h, sx * w / new_w
    if make_square:
        h = w = max(h, w)
    return sy * h / image_size, sx * w / image_size


def spacing_record(volume_stem: str, source: str, loaded_shape: tuple, transforms: dict, preprocessing_cfg: dict,
                   kept_slices: list[int], no_dicom_ext_flag: bool = False) -> dict:
    """
    One ``spacing.csv`` row for a volume. ``loaded_shape``: array shape as loaded, before the YAML ``transforms``
    (``ImagePrep.get_prepped_data``'s original shape); ``kept_slices``: slice indices of the transformed array
    that preprocessing kept, in saved order.
    """
    shape = transform_axes(tuple(loaded_shape), transforms)
    header = read_header_spacing(source, no_dicom_ext_flag)
    if header is not None and len(header) == len(loaded_shape):
        spacing, spacing_source = transform_axes(header, transforms), "header"
    elif preprocessing_cfg.get("pixel_spacing_mm"):
        spacing, spacing_source = tuple(float(v) for v in preprocessing_cfg["pixel_spacing_mm"])[-len(shape):], "config"
    else:
        spacing, spacing_source = (float("nan"),) * len(shape), "missing"
        print(f"  WARNING {volume_stem}: no voxel spacing in {source} and no preprocessing_cfg.pixel_spacing_mm; "
              "mm measurements will be nan.")
    row_mm, col_mm = grid_inplane_mm(shape[-2:], spacing[-2:], int(preprocessing_cfg.get("image_size", 1024)),
                                     bool(preprocessing_cfg.get("make_square", False)),
                                     bool(preprocessing_cfg.get("ratio_resize", False)))
    world = world_affine(source, loaded_shape, transforms, preprocessing_cfg, kept_slices)
    return {
        "volume_stem": volume_stem,
        "source": source,
        "spacing_source": spacing_source,
        "shape": ";".join(str(int(v)) for v in shape),
        "spacing_mm": ";".join(f"{v:.6g}" for v in spacing),
        "kept_slices": ";".join(str(int(z)) for z in kept_slices),
        "slice_mm": float(spacing[0]) if len(shape) == 3 else float("nan"),
        "row_mm": float(row_mm),
        "col_mm": float(col_mm),
        "world_affine": ";".join(f"{v:.9g}" for v in world.ravel()) if world is not None else "",
    }


def affine_from_record(record: dict) -> np.ndarray:
    """NIfTI affine of a preprocessed volume: the RAS form of ``world_affine`` when recorded, else diagonal (slice,
    row, col mm), else identity when spacing is missing."""
    if record.get("world_affine"):
        return LPS_TO_RAS @ np.array([float(v) for v in record["world_affine"].split(";")]).reshape(4, 4)
    diag = [record["slice_mm"], record["row_mm"], record["col_mm"]]
    if not all(np.isfinite(diag)):
        return np.eye(4)
    return np.diag([*diag, 1.0])


def write_spacing_record(nifti_output_dir: str, record: dict) -> None:
    """Add or replace one volume's row in ``<nifti_output_dir>/spacing.csv`` (written per volume, so a stopped run
    keeps every finished row)."""
    path = Path(nifti_output_dir) / SPACING_CSV
    table = pd.DataFrame([record])
    if path.is_file():
        rows = pd.read_csv(path, dtype={"volume_stem": str})
        rows = rows[rows["volume_stem"] != record["volume_stem"]]
        table = pd.concat([rows, table], ignore_index=True) if not rows.empty else table
    table.reindex(columns=SPACING_COLUMNS).to_csv(path, index=False)


# ---------------------------------------------------------------- readers
def load_grid_spacing(volume_stem: str, volume_dir: str) -> tuple[float, float, float] | None:
    """
    (slice, row, col) mm of a preprocessed volume in ``volume_dir`` (``nifti_preprocessed/img`` or ``/mask``), from
    its row in ``../spacing.csv``; None when the volume has no row (not preprocessed by full_standardization.py) or its
    spacing is ``missing``.
    """
    csv_path = Path(volume_dir).parent / SPACING_CSV
    if csv_path.is_file():
        rows = pd.read_csv(csv_path, dtype={"volume_stem": str})
        row = rows[rows["volume_stem"] == volume_stem]
        if not row.empty:
            values = tuple(float(row.iloc[0][c]) for c in ("slice_mm", "row_mm", "col_mm"))
            return values if all(np.isfinite(values[1:])) else None
    return None


def slice_numbers(volume_stem: str, volume_dir: str, n_slices: int) -> list[int]:
    """
    Slice number of each slice of a preprocessed volume in ``volume_dir``, as used in slice file names
    (``{volume}-{number:03d}``): its index in the scan after the YAML transforms (``kept_slices`` in
    ``../spacing.csv``), the same numbering preprocessing gives GT slice labels. Falls back to the position in the
    volume (0..n_slices-1) for volumes without a row (not preprocessed by full_standardization.py), which is the
    same numbering whenever every slice was kept.
    """
    csv_path = Path(volume_dir).parent / SPACING_CSV
    if csv_path.is_file():
        rows = pd.read_csv(csv_path, dtype={"volume_stem": str, "kept_slices": str})
        row = rows[rows["volume_stem"] == volume_stem]
        if not row.empty and isinstance(row.iloc[0]["kept_slices"], str):
            kept = [int(z) for z in row.iloc[0]["kept_slices"].split(";")]
            if len(kept) == n_slices:
                return kept
            print(f"  WARNING {volume_stem}: {len(kept)} kept slices in {csv_path} vs {n_slices} in the volume; "
                  "slice files are numbered by position instead.")
    return list(range(n_slices))


def load_world_affine(volume_stem: str, volume_dir: str) -> np.ndarray | None:
    """``world_affine`` (preprocessed voxel index -> patient mm, LPS) of a volume in ``volume_dir`` from
    ``../spacing.csv``; None when not recorded."""
    csv_path = Path(volume_dir).parent / SPACING_CSV
    if not csv_path.is_file():
        return None
    rows = pd.read_csv(csv_path, dtype={"volume_stem": str, "world_affine": str})
    row = rows[rows["volume_stem"] == volume_stem]
    if row.empty or "world_affine" not in row or not isinstance(row.iloc[0]["world_affine"], str):
        return None
    return np.array([float(v) for v in row.iloc[0]["world_affine"].split(";")]).reshape(4, 4)

