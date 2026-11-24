"""
Load one scan (a file or a DICOM series folder) and apply the YAML ``transforms`` (src/preprocessing/data_transforms.py).
"""
import os

import src.preprocessing.data_loaders.raw_file_loaders as FileLoader
from src.preprocessing.data_transforms import transform_data


def _extension(file_path: str) -> str:
    """Lower-case extension, with ``.gz`` kept as part of it (``.nii.gz``, ``.dcm.gz``)."""
    name = os.path.basename(file_path.lower())
    if name.endswith(".gz"):
        return os.path.splitext(name[:-3])[1] + ".gz"
    return os.path.splitext(name)[1]


def load_raw(data_path: str, no_dicom_ext_flag: bool = False):
    """The array a scan holds, before transforms: NIfTI, MetaImage, a DICOM file, or a DICOM series folder."""
    if os.path.isdir(data_path):
        return FileLoader.load_dcm(data_path)
    if not os.path.isfile(data_path):
        raise ValueError(f"File or DICOM folder does not exist: {data_path}")
    extension = _extension(data_path)
    if extension in (".nii", ".nii.gz"):
        return FileLoader.load_nifti(data_path)
    if extension in (".mha", ".mhd"):
        return FileLoader.load_mhd(data_path)
    if extension in FileLoader.DICOM_EXTENSIONS or no_dicom_ext_flag:
        return FileLoader.load_dicom_file(data_path)
    raise ValueError(f"Unsupported file format {extension!r}: {data_path} (NIfTI, .mha / .mhd, DICOM)")


def get_data(data_path: str, data_transforms: dict = {}, no_dicom_ext_flag: bool = False):
    """(array after the YAML ``transforms``, array shape as loaded) of one scan."""
    data = load_raw(data_path, no_dicom_ext_flag)
    return transform_data(data, data_transforms), data.shape
