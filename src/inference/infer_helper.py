"""Inference I/O and model helpers shared by YOLO predict and SAM from boxes: run folders, volume files, image prep,
the SAM predictor, and saving masks."""
import csv
import os
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pydicom
import torch
from segment_anything import sam_model_registry

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.preprocessing.data_prep import MaskPrep, ImagePrep
from src.preprocessing.full_standardization import image_data_prep  # noqa: F401  (re-exported for inference)
from src.sam_finetuning.engine.models.sam import finetunedSAM
from src.inference.inference_config import LABEL_PRED_SUBDIR, SEG_PRED_SUBDIR


# -------------------- config and run folders -------------------- #
def require_config_path(value, key_name: str) -> str:
    """Raise if a required inference path/key is missing. Does not default from general_data_dir."""
    if value is None or (isinstance(value, str) and not str(value).strip()):
        raise ValueError(
            f"{key_name} is required. Set it in the dataset YAML or pass the matching CLI flag. "
            "Inference does not default this path from general_data_dir."
        )
    return str(value).strip()


def resolve_maybe_path(base_dir: str, p) -> str | None:
    if p is None or (isinstance(p, str) and not str(p).strip()):
        return None
    p = str(p).strip()
    return p if os.path.isabs(p) else os.path.join(base_dir, p)


def load_valid_volume_ids(infer_cfg: dict, base_dir: str) -> set[str] | None:
    """Volume names listed in the ``inference.valid_list`` CSV (column ``id``); None when it is not set."""
    raw = infer_cfg.get("valid_list")
    if not raw:
        return None
    path = Path(resolve_maybe_path(base_dir, raw))
    if not path.is_file():
        raise FileNotFoundError(f"inference.valid_list not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        id_key = next((n for n in reader.fieldnames or [] if n and n.strip().lower() == "id"), None)
        if id_key is None:
            raise ValueError(f"inference.valid_list must have an 'id' column; got {reader.fieldnames!r} ({path})")
        ids = {str(row[id_key]).strip() for row in reader if str(row.get(id_key) or "").strip()}
    print(f"  valid_list: {path} ({len(ids)} volume id(s))")
    return ids


def determine_run_directory(base_dir, task_name):
    """Create and return the next ``{base_dir}/{task_name}/Run_N``."""
    base_path = os.path.join(base_dir, task_name)
    os.makedirs(base_path, exist_ok=True)
    runs = [int(d.split("_")[1]) for d in os.listdir(base_path)
            if d.startswith("Run_") and d.split("_")[1:] and d.split("_")[1].isdigit()
            and os.path.isdir(os.path.join(base_path, d))]
    full_run_path = os.path.join(base_path, f"Run_{max(runs, default=0) + 1}")
    os.makedirs(full_run_path, exist_ok=True)
    return full_run_path


def use_existing_run_directory(base_dir, task_name, run_num):
    """``{base_dir}/{task_name}/Run_{run_num}`` (created if missing)."""
    full_run_path = os.path.join(base_dir, task_name, f"Run_{int(run_num)}")
    os.makedirs(full_run_path, exist_ok=True)
    return full_run_path


def resolve_inference_io(
    cfg,
    *,
    volumes=None,
    output=None,
    run_dir=None,
    run_num=None,
    base_dir=None,
    require_volume_dir=True,
):
    """Resolve the single inference I/O contract.

    Run folder is always ``{output_data_dir}/inference/{task_name}/Run_N`` unless
    ``run_dir`` or ``run_num`` reuses an existing run. ``volume_dir``,
    ``output_data_dir``, and ``inference.task_name`` are required (CLI overrides YAML).
    """
    base_dir = base_dir or os.getcwd()
    infer = cfg.get("inference") or {}

    volume_raw = volumes if volumes is not None else infer.get("volume_dir")
    if require_volume_dir:
        volume_dir = resolve_maybe_path(base_dir, require_config_path(volume_raw, "inference.volume_dir"))
    else:
        volume_dir = resolve_maybe_path(base_dir, volume_raw) if volume_raw else None

    output_raw = output if output is not None else cfg.get("output_data_dir")
    output_data_dir = resolve_maybe_path(base_dir, require_config_path(output_raw, "output_data_dir"))

    task_name = require_config_path(infer.get("task_name"), "inference.task_name")

    if run_dir:
        resolved_run = resolve_maybe_path(base_dir, run_dir)
        os.makedirs(resolved_run, exist_ok=True)
    elif run_num is not None:
        resolved_run = use_existing_run_directory(os.path.join(output_data_dir, "inference"), task_name, run_num)
    else:
        resolved_run = determine_run_directory(os.path.join(output_data_dir, "inference"), task_name)

    return {
        "volume_dir": volume_dir,
        "output_data_dir": output_data_dir,
        "task_name": task_name,
        "run_dir": resolved_run,
        "label_pred_dir": os.path.join(resolved_run, LABEL_PRED_SUBDIR),
        "seg_pred_dir": os.path.join(resolved_run, SEG_PRED_SUBDIR),
    }


# -------------------- volume files and image prep -------------------- #
def is_nifti_file(file_name):
    return file_name.endswith('.nii') or file_name.endswith('.nii.gz')


def is_dicom(file_name):
    if file_name.lower().endswith((".dcm", ".ima", ".dicom")):
        return True
    return pydicom.misc.is_dicom(file_name)


def locate_files(directory_or_file):
    """Volumes in an inference input: one file, a folder of NIfTI files or of DICOM series folders, or one DICOM
    series folder."""
    if os.path.isfile(directory_or_file):
        return [directory_or_file]
    if not os.path.isdir(directory_or_file):
        raise ValueError(f"{directory_or_file} is not a valid file or directory")
    full_paths = [os.path.join(directory_or_file, f) for f in os.listdir(directory_or_file)]
    if all(os.path.isdir(p) for p in full_paths) or all(is_nifti_file(f) for f in full_paths):
        return full_paths
    return [directory_or_file]  # a single DICOM series folder


def extract_filename(img_file: str):
    """Volume name: a folder's name, or a file name without its extension (``.nii.gz`` / ``.dcm.gz`` included).
    Caution: fails for files with periods but no extension (ex. dicom file: "1.2.345")."""
    if os.path.isdir(img_file):
        return os.path.basename(os.path.normpath(img_file))
    file_name = os.path.basename(img_file)
    if file_name.endswith('.gz'):
        file_name = file_name[:-3]
    return os.path.splitext(file_name)[0]


def make_image_prep_from_cfg(cfg, verbose=False):
    """Build the same ImagePrep SAM and volume-based YOLO inference use."""
    prep_cfg = cfg.get("preprocessing_cfg") or {}
    img_cfg = cfg.get("image_config") or {}
    image_size = prep_cfg.get("image_size", 1024)
    return ImagePrep(
        image_size_tuple=(image_size, image_size),
        make_square=prep_cfg.get("make_square", False),
        ratio_resize=prep_cfg.get("ratio_resize", False),
        data_transforms=img_cfg.get("transforms", {}) or {},
        slice_normalization=prep_cfg.get("slice_normalization", True),
        verbose=verbose,
    )


# -------------------- SAM model and masks -------------------- #
def make_predictor(model_type: str, comp_cfg, initial_weights: str, finetuned_weights: str = None,
                   device: str = "cpu"):
    """SAM (``initial_weights``) wrapped in finetunedSAM, with the finetuned state dict loaded when given."""
    sam_model = sam_model_registry[model_type](checkpoint=initial_weights).to(device)
    finetuned_model = finetunedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        config=comp_cfg
    ).to(device)
    if finetuned_weights and finetuned_weights != initial_weights:
        if not os.path.isfile(finetuned_weights):
            raise FileNotFoundError(f"SAM finetuned weights not found: {finetuned_weights} "
                                    "(inference.models.segmentation.model_path + model_weights)")
        checkpoint = torch.load(finetuned_weights, map_location=device)
        finetuned_model.load_state_dict(checkpoint["model"])
    return finetuned_model


def postprocess_resize(mask, image_size_tuple: tuple[int, int], make_square, dtype=np.uint8):
    """Resize mask to new dimensions."""
    predMaskPrep = MaskPrep(make_square=make_square)
    return predMaskPrep.resize_mask(mask_data=mask.astype(dtype), image_size_tuple=image_size_tuple)


def postprocess_prediction_as_float(sam_pred, dtype=np.float32):
    """SAM probabilities -> 8-bit, 5x5 Gaussian blur, back to [0, 1] in ``dtype``."""
    sam_pred *= 255
    smooth_mask = cv2.GaussianBlur(sam_pred.astype(np.uint8), (5, 5), 0)
    return smooth_mask.astype(dtype) / 255


def extract_full_filename(save_dir, filename, output_ext, subfolder=SEG_PRED_SUBDIR):
    ext = "npz" if output_ext == "npz" else "nii.gz"
    return os.path.join(save_dir, subfolder, f"{filename}.{ext}")


def save_prediction(seg_3D, save_dir, filename, output_ext, subfolder=SEG_PRED_SUBDIR, affine=None):
    """Save prediction as .nii.gz (with ``affine``, default identity), or .npz file with key 'seg'."""
    output_dir = os.path.join(save_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    if output_ext == 'npz':
        np.savez_compressed(os.path.join(output_dir, f"{filename}.npz"), seg=seg_3D)
    else:
        nib.save(nib.Nifti1Image(seg_3D, np.eye(4) if affine is None else affine),
                 os.path.join(output_dir, f"{filename}.nii.gz"))
