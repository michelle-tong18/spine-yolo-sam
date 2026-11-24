"""Load a predicted or GT volume (NIfTI, .npy, .npz) and save NIfTI, for inference and evaluation."""
from src.utils.file_management.file_loaders_savers import load_nifti, load_npy, load_npz, save_nifti  # noqa: F401


def load_data(file_path: str, key: str = ''):
    """Load a .nii / .nii.gz, .npy or .npz (``key``) volume."""
    if file_path.endswith('.npz'):
        return load_npz(file_path, key=key)
    if file_path.endswith('.npy'):
        return load_npy(file_path, key=key)
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        return load_nifti(file_path)
    raise ValueError(f"Unsupported file format: {file_path}")
