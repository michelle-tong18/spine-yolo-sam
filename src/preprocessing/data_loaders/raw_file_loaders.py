"""
Load one scan as a numpy array: NIfTI (nibabel array order), MetaImage .mha / .mhd (SimpleITK array order, slice
first), a DICOM series folder (slices in InstanceNumber order) or a single DICOM file.
"""
import os

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk

# Siemens .IMA / .dicom, and extension-less files (common on PACS dumps), are DICOM.
DICOM_EXTENSIONS = {".dcm", ".dcm.gz", ".ima", ".dicom", ""}


def load_mhd(filename: str) -> np.ndarray:
    """MetaImage (.mha single file or .mhd + .raw)."""
    return sitk.GetArrayFromImage(sitk.ReadImage(filename))


def load_nifti(file_path: str) -> np.ndarray:
    return nib.load(file_path).get_fdata()


def _dicom_paths_in_dir(dcm_dirpath: str) -> list[str]:
    """GDCM series names, or the DICOM-named files in the folder if GDCM lists nothing."""
    names = list(sitk.ImageSeriesReader().GetGDCMSeriesFileNames(dcm_dirpath) or [])
    if names:
        return names
    return [
        os.path.join(dcm_dirpath, f)
        for f in os.listdir(dcm_dirpath)
        if not f.startswith(".") and os.path.isfile(os.path.join(dcm_dirpath, f))
        and os.path.splitext(f)[1].lower() in DICOM_EXTENSIONS
    ]


def load_dcm(dcm_dirpath: str) -> np.ndarray:
    """A DICOM series folder as (slice, row, col), slices sorted by InstanceNumber."""
    names = _dicom_paths_in_dir(dcm_dirpath)
    if not names:
        raise ValueError(f"No DICOM files found in {dcm_dirpath}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted(names, key=lambda p: int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber)))
    return sitk.GetArrayFromImage(reader.Execute())


def load_dicom_file(dcm_filepath: str) -> np.ndarray:
    """A single DICOM file's pixel array."""
    return pydicom.dcmread(dcm_filepath).pixel_array
