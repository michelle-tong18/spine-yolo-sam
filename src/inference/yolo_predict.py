#!/usr/bin/env python3
"""
YOLO predict: slice images -> per volume relevel and gap fill (src/yolo_finetuning/postprocess_volume_3d) ->
``{run_dir}/label_pred/{volume}-{slice:03d}.txt`` (class x y w h conf, normalized; ``det_limit`` and the other
``inference.models.obj_det`` filters applied).

Called by ``run_inference.py`` (``yolo`` / ``both``): from volumes (slices extracted with the same ImagePrep as SAM,
numbered by the scan's slice number, src/preprocessing/spacing.py) or from existing PNG slices (``--slices``).
"""

import gc
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.inference.inference_config import LABEL_PRED_SUBDIR, DetectionConfig, detection_config_from_dict
from src.inference.sam_from_boxes import prepare_detections_for_txt
from src.inference.infer_helper import (
    extract_filename,
    image_data_prep,
    load_valid_volume_ids,
    locate_files,
    make_image_prep_from_cfg,
)
from src.preprocessing.data_transforms import transform_data
from src.preprocessing.spacing import slice_numbers
from src.yolo_finetuning.postprocess_volume_3d import (
    gap_fill_config_from_dict,
    gap_fill_volume,
    relevel_config_from_dict,
    relevel_results,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def get_volume_id_and_slice(stem: str) -> tuple[str, int]:
    """``{volume_id}-{slice}`` -> (volume_id, slice); a stem without a numeric suffix is slice 0."""
    idx = stem.rfind("-")
    if idx >= 0 and stem[idx + 1:].isdigit():
        return stem[:idx], int(stem[idx + 1:])
    return stem, 0


def group_image_paths_by_volume(paths: list) -> dict:
    """Group image paths by volume id; within each volume, sort by slice number."""
    by_vol = defaultdict(list)
    for pth in paths:
        by_vol[get_volume_id_and_slice(Path(pth).stem)[0]].append(Path(pth))
    for vid in by_vol:
        by_vol[vid].sort(key=lambda x: get_volume_id_and_slice(x.stem)[1])
    return dict(by_vol)


def yolo_settings(infer_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Predict settings from ``inference`` (device, imgsz) and ``inference.models.obj_det``."""
    obj_det = (infer_cfg.get("models") or {}).get("obj_det") or {}
    relevel = obj_det.get("relevel") or {}
    model_path, weights = str(obj_det.get("model_path", "")), str(obj_det.get("model_weights", "best.pt"))
    return {
        "model_path": os.path.join(model_path if os.path.isabs(model_path) else os.path.join(root, model_path), weights),
        "device": infer_cfg.get("device", "cpu"),
        "imgsz": int(infer_cfg.get("imgsz", 1024)),
        "conf": float(obj_det.get("conf", 0.25)),
        "iou": float(obj_det.get("iou", 0.5)),
        "max_det": int(obj_det.get("max_det", 50)),
        "relevel": relevel if relevel.get("enabled") else None,
        "gap_fill": gap_fill_config_from_dict(obj_det.get("postprocess")),
    }


def save_one_result_txt(result, label_out: Path, detection_cfg: DetectionConfig) -> None:
    """Write one slice's boxes (after the detection filters) to ``{label_out}/{stem}.txt``."""
    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}"
             for c, x, y, w, h, conf in prepare_detections_for_txt(result, detection_cfg)]
    (Path(label_out) / f"{Path(result.path).stem}.txt").write_text("\n".join(lines))


def predict_volumes(model, paths: list, label_out: Path, settings: Dict[str, Any], class_names: dict,
                    detection_cfg: DetectionConfig) -> dict:
    """YOLO on each slice; per volume relevel and gap fill (each when configured); write one .txt per slice.
    Returns detections written per class name."""
    relevel_cfg = relevel_config_from_dict(settings["relevel"]) if settings["relevel"] else None
    gap_fill_cfg = settings["gap_fill"]
    label_out.mkdir(parents=True, exist_ok=True)
    class_counts: dict = defaultdict(int)
    for volume_id, vol_paths in tqdm(sorted(group_image_paths_by_volume(paths).items()), desc="YOLO predict"):
        results = [next(model.predict(source=str(pth), stream=True, imgsz=settings["imgsz"], conf=settings["conf"],
                                      iou=settings["iou"], device=settings["device"], verbose=False,
                                      max_det=settings["max_det"]))
                   for pth in vol_paths]
        notes = []
        if relevel_cfg is not None:
            stats = relevel_results(results, relevel_cfg)
            notes.append(f"relevel {stats['relevel_tracks']} tracks, {stats['relevel_relabel']} boxes relabelled")
        if gap_fill_cfg is not None:
            notes.append(f"gap fill {gap_fill_volume(results, detection_cfg, gap_fill_cfg)['gap_filled']} boxes")
        if notes:
            print(f"  {volume_id}: {'; '.join(notes)}")
        for result in results:
            save_one_result_txt(result, label_out, detection_cfg)
            for c, *_ in prepare_detections_for_txt(result, detection_cfg):
                class_counts[class_names.get(c, f"class_{c}")] += 1
        del results
        gc.collect()
    return dict(class_counts)


def run_yolo_predict(cfg: Dict[str, Any], run_dir: str, img_dir: str, base_dir: Optional[str] = None) -> Optional[Path]:
    """YOLO on the slice images in ``img_dir`` (``inference.valid_list`` applied); returns ``{run_dir}/label_pred``."""
    base_dir = base_dir or os.getcwd()
    infer_cfg = cfg.get("inference") or {}
    settings = yolo_settings(infer_cfg)
    gap_fill = settings["gap_fill"]
    print(f"\n{'=' * 60}\nYOLO INFERENCE — PREDICT\n{'=' * 60}")
    print(f"  Model:   {settings['model_path']}")
    print(f"  Images:  {img_dir}")
    print(f"  Conf:    {settings['conf']}  IoU NMS: {settings['iou']}  ImgSz: {settings['imgsz']}  "
          f"Device: {settings['device']}")
    print(f"  Relevel: {'ON' if settings['relevel'] else 'OFF'}  Gap fill: "
          + (f"ON (max {gap_fill.max_gap_slices} slices, anchor {gap_fill.anchor})" if gap_fill else "OFF"))

    images = sorted(p for p in Path(img_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    valid_ids = load_valid_volume_ids(infer_cfg, base_dir)
    if valid_ids is not None:
        images = [p for p in images if get_volume_id_and_slice(p.stem)[0] in valid_ids]
    print(f"  Found {len(images)} images.")
    if not images:
        return None

    model = YOLO(settings["model_path"])
    mask_labels = infer_cfg.get("mask_labels") or cfg.get("mask_labels")
    class_names = {int(k): str(v) for k, v in mask_labels.items()} if mask_labels else dict(model.names)
    detection_cfg = detection_config_from_dict((infer_cfg.get("models") or {}).get("obj_det") or {})
    label_out = Path(run_dir) / LABEL_PRED_SUBDIR
    class_counts = predict_volumes(model, images, label_out, settings, class_names, detection_cfg)
    print(f"  Detections written: {sum(class_counts.values())} across {len(images)} images.")
    for name, count in sorted(class_counts.items()):
        print(f"    {name}: {count}")
    print(f"\n  Saved prediction .txt files to: {label_out}\n\nDone.")
    return label_out


def extract_volume_slices_for_yolo(cfg: Dict[str, Any], volume_dir: str, dest_dir: str,
                                   base_dir: Optional[str] = None, verbose: bool = False) -> list:
    """Extract 2D PNGs with the same ImagePrep as SAM. Writes ``{volume}-{slice:03d}.png``, numbered by the slice's
    index in the scan (``spacing.slice_numbers``), so predicted labels share the GT label names."""
    base_dir = base_dir or os.getcwd()
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    img_cfg = cfg.get("image_config") or {}
    data_prep = make_image_prep_from_cfg(cfg, verbose=verbose)

    files = locate_files(volume_dir)
    valid_ids = load_valid_volume_ids(cfg.get("inference") or {}, base_dir)
    if valid_ids is not None:
        files = [f for f in files if extract_filename(f) in valid_ids]

    png_paths = []
    for image_path in tqdm(files, desc="Extract slices for YOLO"):
        img_name = extract_filename(image_path)
        img_3D, _ = image_data_prep(image_path, data_prep, expected_properties=img_cfg.get("data_properties") or {},
                                    no_dicom_ext_flag=cfg.get("no_dicom_extension_flag", False))
        slices = [img_3D] if img_3D.ndim == 2 else [np.squeeze(img_3D[i]) for i in range(img_3D.shape[0])]
        numbers = slice_numbers(img_name, os.path.dirname(image_path), len(slices))
        for slice_number, img_2D in zip(numbers, slices):
            if img_2D.ndim == 3:
                img_2D = img_2D[:, :, 0]
            img_2D_3c = np.repeat(img_2D[:, :, None], 3, axis=-1)
            img_2D_det = transform_data(img_2D_3c, {"normalize": "np.uint8", "dtype": "np.uint8"})
            out_path = dest / f"{img_name}-{slice_number:03d}.png"
            Image.fromarray(img_2D_det, "RGB").save(out_path)
            png_paths.append(out_path)
    print(f"  Extracted {len(png_paths)} slice PNG(s) from {len(files)} volume(s) -> {dest}")
    return png_paths


def run_yolo_predict_from_volumes(cfg: Dict[str, Any], run_dir: str, volume_dir: str, *,
                                  base_dir: Optional[str] = None, verbose: bool = False) -> Optional[Path]:
    """Extract slices with ImagePrep, run YOLO, write ``{run_dir}/label_pred``, discard the temporary PNGs."""
    with tempfile.TemporaryDirectory(prefix="yolo_slices_") as tmp:
        if not extract_volume_slices_for_yolo(cfg, volume_dir, tmp, base_dir=base_dir, verbose=verbose):
            return None
        return run_yolo_predict(cfg, run_dir, tmp, base_dir=base_dir)


def run_yolo_predict_from_slices(cfg: Dict[str, Any], run_dir: str, slice_img_dir: str, *,
                                 base_dir: Optional[str] = None) -> Optional[Path]:
    """YOLO on existing PNG slices named ``{volume}-{slice}``; writes ``{run_dir}/label_pred`` with the same stems."""
    return run_yolo_predict(cfg, run_dir, slice_img_dir, base_dir=base_dir)
