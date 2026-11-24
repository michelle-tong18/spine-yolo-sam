"""SAM from boxes: per-slice boxes (predicted YOLO txt or GT masks) -> SAM -> one 3D mask per volume.

Called by ``run_inference.py`` (``sam`` / ``both``). The detection filters here (``_filter_detections``) are shared
with YOLO predict and gap fill, so the boxes written to ``label_pred`` are the boxes SAM receives.
"""
from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.preprocessing.data_transforms import transform_data
from src.utils.file_management.file_handler import load_data

from src.inference.inference_config import DetectionConfig, detection_config_from_dict
from src.inference.infer_helper import (
    image_data_prep,
    postprocess_prediction_as_float,
    make_predictor,
    load_valid_volume_ids,
    locate_files,
    extract_filename,
    extract_full_filename,
    save_prediction,
    postprocess_resize,
    make_image_prep_from_cfg,
    resolve_maybe_path,
    require_config_path,
    LABEL_PRED_SUBDIR,
    SEG_PRED_SUBDIR,
)


def get_det_items_from_gt_slice(mask_slice_2d: np.ndarray, device) -> list:
    """Convert a 2D GT mask slice to det_items: list of (label_id, bbox_tensor, confidence=1.0).
    bbox is xyxy format as tensor on device. Used when ``--boxes gt``.
    """
    mask_slice_2d = np.asarray(mask_slice_2d)
    out = []
    for label_id in np.unique(mask_slice_2d):
        if int(label_id) == 0:
            continue
        ys, xs = np.where(mask_slice_2d == label_id)
        if ys.size == 0 or xs.size == 0:
            continue
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).to(device)
        out.append((int(label_id), bbox, 1.0))
    return out


def _filter_detections(det_items_sorted, detection_cfg: DetectionConfig):
    """Apply single_cls, remove_label_ids, det_limit. det_items_sorted: list of (label_id, payload, confidence).
    Returns list of (label_id, payload, confidence) after filtering. Shared by prepare_detections_for_sam and prepare_detections_for_txt.
    """
    if not det_items_sorted:
        return []
    class_ids = [x[0] for x in det_items_sorted]
    det_lim = detection_cfg.det_limit.copy()
    for cls in np.unique(class_ids):
        if int(cls) not in det_lim:
            det_lim[int(cls)] = len(class_ids)
    out = []
    for label_id, payload, confidence in det_items_sorted:
        if label_id == 0:
            if detection_cfg.single_cls:
                label_id = 1
            else:
                raise ValueError(
                    "Detected class ID 0 found in multi-class mode, which is unexpected. "
                    "Set yaml config obj_det.single_cls to True if using a single class model."
                )
        if label_id in detection_cfg.remove_label_ids:
            continue
        if det_lim[label_id] <= 0:
            continue
        det_lim[label_id] -= 1
        out.append((label_id, payload, confidence))
    return out


def prepare_detections_for_sam(result, detection_cfg: DetectionConfig):
    """Apply single_cls, remove_label_ids, det_limit; sort by confidence. Returns list of (label_id, bbox, confidence).
    Detections with label_id in remove_label_ids are skipped (never passed to SAM).
    det_limit caps the number of detections per class per slice (max detections per class per slice).
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []
    class_ids = result.boxes.cls.int().tolist()
    if not class_ids:
        return []
    boxes = result.boxes.xyxy
    confs = result.boxes.conf.tolist()
    det_items = list(zip(class_ids, boxes, confs))
    det_items_sorted = sorted(det_items, key=lambda x: -x[2])
    return _filter_detections(det_items_sorted, detection_cfg)


def prepare_detections_for_txt(result, detection_cfg: DetectionConfig):
    """Same filtering as prepare_detections_for_sam but returns list of (cls_id, x_c, y_c, w, h, conf) normalized for YOLO .txt.
    Use when writing prediction labels so saved detections match what would be passed to SAM.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []
    class_ids = result.boxes.cls.int().tolist()
    if not class_ids:
        return []
    xywhn = result.boxes.xywhn.cpu().numpy()
    confs = result.boxes.conf.tolist()
    det_items = [(class_ids[i], xywhn[i], confs[i]) for i in range(len(class_ids))]
    det_items_sorted = sorted(det_items, key=lambda x: -x[2])
    filtered = _filter_detections(det_items_sorted, detection_cfg)
    out = []
    for label_id, whn, conf in filtered:
        x_c, y_c, w, h = float(whn[0]), float(whn[1]), float(whn[2]), float(whn[3])
        out.append((label_id, x_c, y_c, w, h, conf))
    return out



def label_file_slice_number(stem: str, img_name: str) -> int | None:
    """Slice number of a ``{img_name}-{slice}`` label stem; None for another volume's stem."""
    prefix = f"{img_name}-"
    if not stem.startswith(prefix) or not stem[len(prefix):].isdigit():
        return None
    return int(stem[len(prefix):])


def ordered_yolo_txt_stems(img_name: str, yolo_txt_labels_dir: str) -> list[str]:
    """Label stems of one volume in slice-number order. YOLO writes ``{volume}-{slice:03d}.txt`` numbered by each
    slice's index in the scan (``spacing.slice_numbers``), so the k-th stem belongs to the k-th volume slice."""
    if not yolo_txt_labels_dir or not os.path.isdir(yolo_txt_labels_dir):
        return []
    numbered = [(label_file_slice_number(name[:-4], img_name), name[:-4])
                for name in os.listdir(yolo_txt_labels_dir) if name.endswith(".txt")]
    return [stem for number, stem in sorted((n, s) for n, s in numbered if n is not None)]


def load_det_items_from_yolo_txt(txt_path: str, orig_shape, device) -> list:
    """YOLO label lines (class x_center y_center width height [confidence], normalized) -> list of
    (label_id, bbox_xyxy tensor, confidence), highest confidence first; a missing file has no boxes."""
    h, w = int(orig_shape[1]), int(orig_shape[2])
    rows = []
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                xc, yc, bw, bh = map(float, parts[1:5])
                bbox = torch.tensor([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h],
                                    dtype=torch.float32, device=device)
                rows.append((int(parts[0]), bbox, float(parts[5]) if len(parts) >= 6 else 1.0))
    return sorted(rows, key=lambda x: -x[2])


def run_sam_and_combine_for_slice(img_2D_sam, det_items, sam_model, orig_shape, detection_cfg: DetectionConfig,
                                  make_square, device, combined_mask, confidence_map):
    """Run SAM per box and combine into ``combined_mask`` in place.
    label_id in ignore_label_ids: run SAM but do not add to combined mask.
    label_id in subtract_label_ids: zero out that region in combined mask.
    label_id in priority_label_ids: overwrite combined mask with this label where SAM predicts; otherwise overwrite
    only where confidence is higher.
    """
    image_size_tuple = (orig_shape[1], orig_shape[2])
    for label_id, bbox, confidence in det_items:
        bbox = bbox.to(device)
        sam_pred = sam_model(img_2D_sam.unsqueeze(0), bbox.unsqueeze(0))
        sam_mask = postprocess_resize(torch.sigmoid(sam_pred).squeeze().detach().cpu().numpy(), image_size_tuple,
                                      make_square, dtype=np.float32)
        sam_mask = torch.tensor(postprocess_prediction_as_float(sam_mask)).to(device)

        if label_id in detection_cfg.ignore_label_ids:
            continue
        if label_id in detection_cfg.subtract_label_ids:
            mask_indices = sam_mask > 0.5
            combined_mask[mask_indices] = 0
            confidence_map[mask_indices] = 1.0
        elif label_id in detection_cfg.priority_label_ids:
            mask_indices = sam_mask > 0.5
            combined_mask[mask_indices] = label_id
            confidence_map[mask_indices] = 1.0
        else:
            mask_indices = (sam_mask > 0.5) & (confidence_map < confidence)
            combined_mask[mask_indices] = label_id
            confidence_map[mask_indices] = confidence


def autoLabel(sam_model, img_3D, orig_img_shape, img_name, detection_cfg: DetectionConfig, make_square=True,
              device="cpu", boxes="pred", gt_masks_dir=None, gt_mask_ext="nii.gz", pred_labels_dir=None):
    """Run SAM from per-slice boxes over a 3D volume (no in-loop YOLO).

    ``boxes='pred'``: YOLO .txt from ``pred_labels_dir`` (typically ``{run_dir}/label_pred``), paired with the
    volume's slices in slice-number order. ``boxes='gt'``: boxes from the GT mask volume in ``gt_masks_dir``.

    Returns a 3D numpy array (uint8) of the same shape as orig_img_shape with combined label IDs per voxel.
    """
    mask_3D_orig_size = np.zeros(orig_img_shape, dtype=np.uint8)
    sam_model = sam_model.to(device)

    gt_volume = None
    if boxes == "gt":
        gt_path = os.path.join(gt_masks_dir, f"{img_name}.{gt_mask_ext}")
        gt_volume = np.asarray(load_data(gt_path)) if os.path.isfile(gt_path) else None
        if gt_volume is None or gt_volume.shape != tuple(orig_img_shape):
            print(f"  WARNING {img_name}: no GT mask matching the image at {gt_path}; no boxes, empty prediction.")
            gt_volume = None

    ordered_stems: list[str] = []
    if boxes == "pred":
        ordered_stems = ordered_yolo_txt_stems(img_name, pred_labels_dir)
        if len(ordered_stems) != img_3D.shape[0]:
            raise ValueError(
                f"{img_name}: {len(ordered_stems)} YOLO label files vs {img_3D.shape[0]} volume slices; "
                "cannot align slices safely. Check that label_pred (or --pred-labels-dir) and "
                "inference.volume_dir come from the same ImagePrep extraction."
            )

    for slice_idx in range(img_3D.shape[0]):
        img_2D = np.squeeze(img_3D[slice_idx, :, :])
        img_2D_3c = np.repeat(img_2D[:, :, None], 3, axis=-1)
        img_2D_sam = transform_data(img_2D_3c, {"transpose": [2, 0, 1], "normalize": "1", "dtype": "np.float64"})
        img_2D_sam = torch.tensor(img_2D_sam).float().to(device)

        combined_mask = torch.zeros((orig_img_shape[1], orig_img_shape[2]), dtype=torch.float32).to(device)
        confidence_map = torch.zeros((orig_img_shape[1], orig_img_shape[2]), dtype=torch.float32).to(device)
        if boxes == "gt":
            det_items = [] if gt_volume is None else get_det_items_from_gt_slice(np.squeeze(gt_volume[slice_idx]), device)
        else:
            det_items = load_det_items_from_yolo_txt(os.path.join(pred_labels_dir, f"{ordered_stems[slice_idx]}.txt"),
                                                     orig_img_shape, device)
        run_sam_and_combine_for_slice(img_2D_sam, det_items, sam_model, orig_img_shape, detection_cfg, make_square,
                                      device, combined_mask, confidence_map)
        mask_3D_orig_size[slice_idx] = combined_mask.cpu().numpy().astype(np.uint8)

    return mask_3D_orig_size


def run_sam_from_boxes(cfg, run_dir, volume_dir, *, boxes="pred", pred_labels_dir=None, device=None, verbose=False,
                       base_dir=None):
    """Run SAM on volumes using boxes from ``label_pred`` (pred) or GT masks (gt). Writes ``seg_pred/``; volumes
    with a mask already there are skipped."""
    base_dir = base_dir or os.getcwd()
    infer_cfg = cfg.get("inference") or {}
    models = infer_cfg.get("models") or {}
    detection_config = detection_config_from_dict(models.get("obj_det") or {})
    segmentation_cfg = models.get("segmentation") or {}

    boxes = (boxes or "pred").lower()
    if boxes not in ("pred", "gt"):
        raise ValueError(f"boxes must be 'pred' or 'gt', got {boxes!r}")
    pred_labels_dir_resolved = gt_masks_dir = None
    if boxes == "pred":
        pred_labels_dir_resolved = resolve_maybe_path(base_dir, pred_labels_dir or os.path.join(run_dir, LABEL_PRED_SUBDIR))
        if not pred_labels_dir_resolved or not os.path.isdir(pred_labels_dir_resolved):
            raise ValueError(
                f"Predicted boxes directory not found: {pred_labels_dir_resolved}. "
                f"Run `yolo` into this run first, or pass --pred-labels-dir."
            )
        print("pred labels dir: ", pred_labels_dir_resolved)
    else:
        gt_masks_dir = resolve_maybe_path(
            base_dir, require_config_path((cfg.get("evaluation") or {}).get("gt_masks_dir"), "evaluation.gt_masks_dir"))
        print("gt masks dir: ", gt_masks_dir)
    output_ext = infer_cfg.get("output_ext", "nii.gz")

    img_cfg = cfg.get("image_config") or {}
    img_properties = img_cfg.get("data_properties") or {}
    no_dicom_ext_flag = cfg.get("no_dicom_extension_flag", False)
    make_square = (cfg.get("preprocessing_cfg") or {}).get("make_square", False)
    dataImagePrep = make_image_prep_from_cfg(cfg, verbose=verbose)

    sam_model_path = os.path.join(root, segmentation_cfg.get("model_path"), segmentation_cfg.get("model_weights"))
    print("sam model path: ", sam_model_path)
    device = device or infer_cfg.get("device") or cfg.get("device", "cpu")
    sam_model = make_predictor(
        model_type=segmentation_cfg.get("model_type"),
        comp_cfg={"prompt_encoder": False, "image_encoder": False, "mask_decoder": False},
        initial_weights=os.path.join(root, segmentation_cfg.get("base_model")),
        finetuned_weights=sam_model_path,
        device=device,
    )

    if not volume_dir:
        raise ValueError(
            "inference.volume_dir is required. Set it in the dataset YAML or pass --volumes. "
            "Inference does not default this path from general_data_dir."
        )
    inference_files = locate_files(volume_dir)
    valid_ids = load_valid_volume_ids(infer_cfg, base_dir)
    if valid_ids is not None:
        inference_files = [f for f in inference_files if extract_filename(f) in valid_ids]

    for image_path in tqdm(inference_files):
        img_name = extract_filename(image_path)
        if os.path.exists(extract_full_filename(run_dir, filename=img_name, output_ext=output_ext,
                                                subfolder=SEG_PRED_SUBDIR)):
            continue
        img_3D, orig_img_shape = image_data_prep(image_path, dataImagePrep, expected_properties=img_properties,
                                                 no_dicom_ext_flag=no_dicom_ext_flag)
        pred_volume = autoLabel(sam_model, img_3D, orig_img_shape, img_name, detection_config, make_square=make_square,
                                device=device, boxes=boxes, gt_masks_dir=gt_masks_dir, gt_mask_ext=output_ext,
                                pred_labels_dir=pred_labels_dir_resolved)
        # A preprocessed NIfTI input carries its grid geometry (src/preprocessing/spacing.py) in the affine.
        affine = None
        if str(image_path).endswith((".nii", ".nii.gz")):
            input_nii = nib.load(image_path)
            if tuple(input_nii.shape) == tuple(np.shape(pred_volume)):
                affine = input_nii.affine
        save_prediction(pred_volume, run_dir, filename=img_name, output_ext=output_ext, subfolder=SEG_PRED_SUBDIR,
                        affine=affine)
