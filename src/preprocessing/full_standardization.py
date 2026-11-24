"""
Preprocess the scans listed in ``data_dir`` (CSV: ``subject_id``, ``image_column``, optional ``mask_column`` and
``accession``) into ``{output_dir}/nifti_preprocessed/{img,mask}`` (cropped to labelled slices, resized to
``image_size``², with ``spacing.csv``, src/preprocessing/spacing.py) and, for scans with a mask, the training slices in
``{output_dir}/slice_preprocessed/SAM/{imgs,gts,png,labels}`` (``{volume}-{slice:03d}``, slice = index in the scan).
A scan without a mask (empty ``mask_column`` or ``preprocessing_cfg.image_only``) is preprocessed image-only.

    python src/preprocessing/full_standardization.py <config> [--shuffle] [--verbose]
"""
import argparse
import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    dotenv=True, # load environment variables from .env if exists in root directory
)

from src.preprocessing.data_prep import MaskPrep, ImagePrep, get_relabeled_bounding_boxes, write_list_to_file
from src.utils.file_management.config_handler import load_main_config
from src.utils.file_management.file_handler import save_nifti
from src.preprocessing.data_transforms import transform_data, find_non_zero_slices, extrapolate_z
from src.preprocessing.spacing import affine_from_record, spacing_record, transform_axes, write_spacing_record
from src.utils.file_management.path_info import extract_data_pair_paths
from src.utils.visualizers import plot_segmentation_overlay, plot_bbox_overlay


def data_standardization(config_name, shuffle=False, verbose=False):
    """Preprocess every scan of the config's ``data_dir`` CSV."""
    cfg = load_main_config(config_name, base_dir=root, sub_dir='datasets')
    prep_cfg = cfg.get('preprocessing_cfg')
    image_size_tuple = (prep_cfg.get('image_size'), prep_cfg.get('image_size'))

    dataMaskPrep = MaskPrep(
        remove_label_ids=prep_cfg.get('remove_label_ids'),
        voxel_threshold_3d=prep_cfg.get('voxel_num_thre3d'),
        pixel_threshold_2d=prep_cfg.get('voxel_num_thre2d'),
        image_size_tuple=image_size_tuple,
        crop_non_zero_slices_flag=prep_cfg.get('crop_non_zero_slices_flag', True),
        make_square=prep_cfg.get('make_square', False),
        ratio_resize=prep_cfg.get('ratio_resize', False),
        data_transforms=cfg.get('mask_config', {}).get('transforms', {}),
        z_sample=prep_cfg.get('add_sampling', 0.0),
        verbose=verbose,
    )
    dataImagePrep = ImagePrep(
        image_size_tuple=image_size_tuple,
        make_square=prep_cfg.get('make_square', False),
        ratio_resize=prep_cfg.get('ratio_resize', False),
        data_transforms=cfg.get('image_config', {}).get('transforms', {}),
        slice_normalization=prep_cfg.get('slice_normalization', True),
        verbose=verbose,
    )
    data_paths_info = extract_data_pair_paths(cfg.get("data_dir"), img_column=cfg.get("image_column"),
                                              mask_column=cfg.get("mask_column"),
                                              include_accession=cfg.get("include_subject_accession", False))
    if shuffle:
        random.shuffle(data_paths_info)

    process_data_paths(data_paths_info,
                       dataImagePrep,
                       dataMaskPrep,
                       output_dir=cfg.get("output_dir"),
                       img_properties=cfg.get("image_config", {}).get("data_properties", {}),
                       mask_properties=cfg.get("mask_config", {}).get("data_properties", {}),
                       overwrite_flag=cfg.get("overwrite_existing_flag"),
                       no_dicom_ext_flag=cfg.get("no_dicom_extension_flag", False),
                       yolo_relabel=prep_cfg.get("yolo_relabel") or {},
                       visualize=((cfg.get("nifti_fig_cfg") or {}).get("fig_type", ""), cfg.get("mask_labels", {})),
                       image_only=prep_cfg.get("image_only", False),
                       crop_non_zero_slices_flag=prep_cfg.get("crop_non_zero_slices_flag", True),
                       z_sample=prep_cfg.get("add_sampling", 0.0) or 0.0,
                       preprocessing_cfg=prep_cfg,
                       )


def process_data_paths(data_paths_info, dataImagePrep, dataMaskPrep, output_dir: str, img_properties: dict = {},
                       mask_properties: dict = {}, overwrite_flag: bool = True, no_dicom_ext_flag: bool = False,
                       yolo_relabel: dict = {}, visualize=('', {}), image_only: bool = False,
                       crop_non_zero_slices_flag: bool = True, z_sample: float = 0.0, preprocessing_cfg: dict = {}):
    """
    Preprocess each (image paths, mask paths, subject id) of ``data_paths_info`` (one image volume per row).
    - Saves nifti_preprocessed/{img,mask}/<subject>.nii.gz and the volume's row of nifti_preprocessed/spacing.csv
      (voxel spacing and world geometry, src/preprocessing/spacing.py; also the NIfTI affines)
    - For scans with a mask: slice_preprocessed/SAM training slices (``save_slice_data``)
    - With ``nifti_fig_cfg.fig_type: 2D_overlay``: QC overlays in nifti_preprocessed/fig/ and slice_preprocessed/SAM/figs/
    Volumes already preprocessed are skipped unless ``overwrite_flag``.
    """
    nifti_output_dir = os.path.join(output_dir, "nifti_preprocessed")
    img_save_dir = os.path.join(nifti_output_dir, "img")
    mask_save_dir = os.path.join(nifti_output_dir, "mask")
    os.makedirs(img_save_dir, exist_ok=True)

    for img_paths, mask_paths, subject_id in tqdm(data_paths_info):
        print(subject_id)
        img_path = img_paths[0]
        image_only_subject = image_only or not (mask_paths and mask_paths[0])
        img_save_path = os.path.join(img_save_dir, f"{subject_id}.nii.gz")
        mask_save_path = os.path.join(mask_save_dir, f"{subject_id}.nii.gz")
        if not overwrite_flag and os.path.isfile(img_save_path):
            continue

        if image_only_subject:
            print(img_path)
            z_indices, original_z = image_z_indices_prep(img_path, dataImagePrep, no_dicom_ext_flag,
                                                         crop_non_zero_slices_flag=crop_non_zero_slices_flag,
                                                         z_sample=z_sample)
            mask_data = None
        else:
            os.makedirs(mask_save_dir, exist_ok=True)
            if not overwrite_flag and os.path.isfile(mask_save_path):
                continue
            # saved below with the image's spacing
            mask_data, z_indices, original_z = mask_data_prep(mask_paths[0], dataMaskPrep, mask_properties, no_dicom_ext_flag)

        img_data, loaded_shape = image_data_prep(img_path, dataImagePrep, img_properties, no_dicom_ext_flag,
                                                 z_indices, original_z)
        record = _spacing_record(subject_id, img_path, loaded_shape, dataImagePrep, preprocessing_cfg,
                                 z_indices, original_z, no_dicom_ext_flag)
        save_nifti(img_data, img_save_path, affine=affine_from_record(record))
        if mask_data is not None:
            save_nifti(mask_data, mask_save_path, affine=affine_from_record(record))
        write_spacing_record(nifti_output_dir, record)

        if image_only_subject:
            continue

        if visualize[0] == '2D_overlay':
            fig_dir = os.path.join(nifti_output_dir, "fig")
            os.makedirs(fig_dir, exist_ok=True)
            plot_segmentation_overlay(img_data, mask_data, save_path=os.path.join(fig_dir, f"{subject_id}.png"),
                                      title=subject_id, labels_dict=visualize[1])

        slice_output_dir = os.path.join(output_dir, "slice_preprocessed", "SAM")
        os.makedirs(slice_output_dir, exist_ok=True)
        save_slice_data(image_data=img_data, mask_data=mask_data, z_indices=z_indices, output_dir=slice_output_dir,
                        base_name=subject_id, yolo_relabel=yolo_relabel, overwrite_flag=overwrite_flag,
                        visualize=visualize)


def image_z_indices_prep(data_filepath, dataImagePrep, no_dicom_ext_flag: bool = False,
                         crop_non_zero_slices_flag: bool = True, z_sample: float = 0.0):
    """Slice indices to keep for image-only preprocessing: non-zero image slices (plus an optional random share)."""
    image_data, _ = dataImagePrep.get_data(filepath=data_filepath, no_dicom_ext_flag=no_dicom_ext_flag)
    data_dims = dataImagePrep.data_dims(image_data)
    original_z = list(np.arange(np.shape(image_data)[0])) if data_dims == 3 else [0]
    if not (crop_non_zero_slices_flag and data_dims == 3):
        return original_z, original_z

    z_select = []
    if z_sample > 0:
        all_slices = np.arange(np.shape(image_data)[0])
        np.random.shuffle(all_slices)
        z_select = all_slices[0: int(len(all_slices) * z_sample)]
    z_indices = np.unique(np.sort(list(find_non_zero_slices(image_data)) + list(z_select)))
    if len(z_indices) == 0:
        raise ValueError(f"No non-zero slices found in image: {data_filepath}")
    return list(z_indices), original_z


def _spacing_record(subject_id, img_path, loaded_shape, dataImagePrep, preprocessing_cfg, z_indices, original_z,
                    no_dicom_ext_flag):
    """spacing.csv row for one volume; kept slices mirror ``ImagePrep.prep_image``'s crop (z_indices remapped
    when the image and mask have different slice counts)."""
    transforms = dataImagePrep.data_transforms
    n_slices = transform_axes(tuple(loaded_shape), transforms)[0]
    kept = list(z_indices)
    if not kept:
        kept = list(range(n_slices)) if len(loaded_shape) == 3 else []
    elif len(original_z) > 0 and n_slices != len(original_z):
        kept = list(extrapolate_z(np.asarray(kept), n_slices, len(original_z)))
    return spacing_record(subject_id, img_path, loaded_shape, transforms, preprocessing_cfg, kept, no_dicom_ext_flag)


def image_data_prep(data_filepath, dataImagePrep, expected_properties: dict = {}, no_dicom_ext_flag: bool = False,
                    z_indices: list[int] = [], original_z: list[int] = []):
    """
    Load, prepare (ImagePrep: crop to ``z_indices``, normalize, resize) and validate one image volume.
    ``original_z``: slice indices of the volume ``z_indices`` refer to (remapped when the image has another slice
    count). Returns (image, array shape as loaded, before the YAML transforms).
    """
    img_data, orig_img_shape = dataImagePrep.get_prepped_data(filepath=data_filepath, no_dicom_ext_flag=no_dicom_ext_flag,
                                                              z_indices=z_indices, original_z=original_z)
    dataImagePrep.validate_image(img_data, expected_properties)
    return img_data, orig_img_shape


def mask_data_prep(data_filepath, dataMaskPrep, expected_properties: dict = {}, no_dicom_ext_flag: bool = False):
    """Load, prepare (MaskPrep) and validate one mask volume; returns (mask, kept slice indices, all slice indices)."""
    mask_data, z_indices, original_z = dataMaskPrep.get_prepped_data(filepath=data_filepath, no_dicom_ext_flag=no_dicom_ext_flag)
    dataMaskPrep.validate_mask(mask_data, expected_properties)
    return mask_data, z_indices, original_z


def save_slice_data(image_data: np.ndarray, mask_data: np.ndarray, z_indices, output_dir: str, base_name,
                    yolo_relabel={}, overwrite_flag: bool = True, visualize=('', {})):
    """
    Save one volume's training slices (``{base_name}-{z:03d}``, z = the slice's index in the scan):
    - output_dir/imgs/*.npy  3-channel image, range 0-1 (SAM)
    - output_dir/png/*.png   3-channel image, 0-255 (YOLO)
    - output_dir/gts/*.npy   mask slice (slices with labels only)
    - output_dir/labels/*.txt  YOLO boxes per label (``yolo_relabel`` maps label IDs to YOLO classes)
    - output_dir/figs/<base_name>.png  box overlay, with ``visualize[0] == '2D_overlay'``
    """
    images_dir = os.path.join(output_dir, "imgs")
    masks_dir = os.path.join(output_dir, "gts")
    png_dir = os.path.join(output_dir, "png")
    annotations_dir = os.path.join(output_dir, "labels")
    for d in (images_dir, masks_dir, png_dir, annotations_dir):
        os.makedirs(d, exist_ok=True)

    volume_bbox_dict = {}
    for i, (img_slice, mask_slice) in enumerate(zip(image_data, mask_data)):
        image_base_name = f"{base_name}-{str(z_indices[i]).zfill(3)}"
        img_f = os.path.join(images_dir, image_base_name + ".npy")
        mask_f = os.path.join(masks_dir, image_base_name + ".npy")
        png_f = os.path.join(png_dir, image_base_name + ".png")

        # 3-channel grayscale, range 0-1 for SAM
        img_slice_3c = transform_data(np.repeat(img_slice[:, :, None], 3, axis=-1), {'normalize': '1', 'dtype': 'np.float64'})
        if overwrite_flag or not os.path.exists(img_f):
            np.save(img_f, img_slice_3c)
        if (overwrite_flag or not os.path.exists(mask_f)) and np.sum(mask_slice) > 0:
            np.save(mask_f, mask_slice)
        if overwrite_flag or not os.path.exists(png_f):
            # 0-255 for YOLO
            Image.fromarray(transform_data(img_slice_3c, {'normalize': 'np.uint8', 'dtype': 'np.uint8'}), 'RGB').save(png_f)

        annotations_f = os.path.join(annotations_dir, image_base_name + ".txt")
        bb_text = get_relabeled_bounding_boxes(mask_slice, yolo_relabel)
        if os.path.exists(annotations_f):  # written in append mode, so start from an empty file
            os.remove(annotations_f)
        write_list_to_file(annotations_f, bb_text, write_mode='a')
        volume_bbox_dict[i] = bb_text

    if visualize[0] == '2D_overlay':
        fig_dir = os.path.join(output_dir, "figs")
        os.makedirs(fig_dir, exist_ok=True)
        plot_bbox_overlay(image_data, volume_bbox_dict, save_path=os.path.join(fig_dir, base_name + ".png"),
                          title=base_name, labels_dict=visualize[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert medical image and mask data to model training file format.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle the data before processing")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")
    args = parser.parse_args()
    data_standardization(args.config_name, args.shuffle, args.verbose)
