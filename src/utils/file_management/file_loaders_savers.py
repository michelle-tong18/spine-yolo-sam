import nibabel as nib
import numpy as np

# -------------------- For Standardized NIfTI Data --------------------

def load_nifti(file_path:str) -> np.ndarray:
    """
    Load data from NIfTI file and return as a numpy array. 
    """
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    return data

def save_nifti(data, file_name, reference_nifti_path:str=None, affine:np.ndarray=None):
    """Save as NIfTI with the affine / header of ``reference_nifti_path``, else ``affine``, else identity."""
    if reference_nifti_path is not None:
        # Load the original NIfTI file to use as a template for affine/header
        original_nii = nib.load(reference_nifti_path)
        # Create a new NIfTI image with the segmentation data
        new_nii = nib.Nifti1Image(data, original_nii.affine, original_nii.header)
    else:
        new_nii = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    return nib.save(new_nii, file_name)


# -------------------- For Metadata --------------------


# ------------- For Standardized NPY & NPZ Data -------------

def load_npy(filename:str, key:str=''):
    """
    Load data from a NumPy file using a specified key.
    """
    data = np.load(filename, "r", allow_pickle=True)
    if key:
        data = data[key]
    return data

def load_npz(filename:str, key:str=''):
    """
    Load data from a NumPy NPZ file using a specified key.
    """
    data = np.load(filename, "rb", allow_pickle=True)
    if key:
        data = data[key]
    return data