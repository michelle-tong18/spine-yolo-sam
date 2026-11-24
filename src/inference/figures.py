"""
Verification figures for an inference run (``run_inference.py ... --figures``): per volume, one PNG with a few evenly
spaced slices in three rows (the image alone, YOLO boxes, SAM masks), written to ``{run}/figures/<volume>.png``.

Colors follow src/utils/visualizers.py (``ColorInfo``): the ``rainbow`` colormap sampled at the labels present, the
same label in the same color in both rows, masks at alpha 0.4, image window = 2nd-98th percentile, and a labelled
colorbar. The image is prepared with the same ImagePrep as SAM inference, so it lies on the masks' grid; the k-th label
file of a volume (in slice-number order) belongs to its k-th slice, as in SAM. Column titles give each slice's number
in the scan (src/preprocessing/spacing.py).
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.inference.infer_helper import extract_filename, image_data_prep, locate_files, make_image_prep_from_cfg  # noqa: E402
from src.inference.inference_config import LABEL_PRED_SUBDIR, SEG_PRED_SUBDIR  # noqa: E402
from src.inference.sam_from_boxes import ordered_yolo_txt_stems  # noqa: E402
from src.preprocessing.spacing import slice_numbers  # noqa: E402
from src.utils.visualizers import ColorInfo  # noqa: E402

FIGURES_SUBDIR = "figures"
ROW_NAMES = ("image", "YOLO boxes", "SAM masks")


def evenly_spaced_slices(n_total: int, n: int) -> list[int]:
    """``n`` slice positions spread evenly from the first to the last slice (all slices when there are fewer)."""
    if n_total <= n:
        return list(range(n_total))
    return sorted({int(round(i)) for i in np.linspace(0, n_total - 1, n)})


def read_boxes(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    """(class, x_center, y_center, width, height), normalized, from a YOLO label file."""
    if not txt_path.is_file() or txt_path.stat().st_size == 0:
        return []
    rows = np.loadtxt(txt_path, ndmin=2)
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]


def save_volume_figure(img: np.ndarray, seg: np.ndarray, boxes_by_slice: list[list[tuple]], numbers: list[int],
                       mask_labels: dict[int, str], save_path: Path, title: str, n_slices: int = 5) -> list[int]:
    """
    One PNG: rows image / YOLO boxes / SAM masks on ``n_slices`` evenly spaced slices of ``img`` (slice-first).
    ``boxes_by_slice``: per volume slice, normalized boxes ([] when YOLO was not run). Returns the slice positions
    shown.
    """
    colors = ColorInfo()
    labels = sorted((set(int(v) for v in np.unique(seg)) | {b[0] for s in boxes_by_slice for b in s}) - {0})
    cmap, norm = colors.adjust_cmap_for_labels("rainbow", labels, colors.set_mask_clim(np.array(labels)))
    image_clim = colors.set_image_clim(img)
    shown = evenly_spaced_slices(img.shape[0], n_slices)
    fig, axes = plt.subplots(len(ROW_NAMES), len(shown), figsize=(3 * len(shown), 3 * len(ROW_NAMES) + 1),
                             squeeze=False)
    for col, z in enumerate(shown):
        h, w = img[z].shape
        for ax in axes[:, col]:
            ax.imshow(img[z], cmap="gray", clim=image_clim, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
        for cls, xc, yc, bw, bh in boxes_by_slice[z]:
            axes[1, col].add_patch(Rectangle(((xc - bw / 2) * w, (yc - bh / 2) * h), bw * w, bh * h, fill=False,
                                             lw=1.2, edgecolor=cmap(norm(cls))))
        axes[2, col].imshow(np.ma.masked_where(seg[z] == 0, seg[z]), cmap=cmap, norm=norm, alpha=0.4,
                            interpolation="nearest")
        axes[0, col].set_title(f"slice {numbers[z]}", fontsize=12)
    for row, name in enumerate(ROW_NAMES):
        axes[row, 0].set_ylabel(name, fontsize=12)
    plt.suptitle(title, fontsize=16)
    if labels:
        colors.add_colorbar(fig, axes.ravel().tolist(), {lid: mask_labels.get(lid, str(lid)) for lid in labels}, cmap, norm, labels)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return shown


def save_run_figures(cfg: dict, run_dir: str, volume_dir: str, n_slices: int = 5) -> list[Path]:
    """One figure per segmented volume of a run (``{run}/seg_pred``); returns the saved paths."""
    mask_labels = {int(k): str(v) for k, v in cfg["mask_labels"].items()}
    img_cfg = cfg.get("image_config") or {}
    image_prep = make_image_prep_from_cfg(cfg)
    label_dir = Path(run_dir) / LABEL_PRED_SUBDIR
    saved = []
    for image_path in sorted(locate_files(volume_dir)):
        name = extract_filename(image_path)
        seg_path = Path(run_dir) / SEG_PRED_SUBDIR / f"{name}.nii.gz"
        if not seg_path.is_file():
            continue
        img, _ = image_data_prep(image_path, image_prep, expected_properties=img_cfg.get("data_properties") or {},
                                 no_dicom_ext_flag=cfg.get("no_dicom_extension_flag", False))
        seg = np.asanyarray(nib.load(seg_path).dataobj).astype(np.int64)
        if img.shape != seg.shape:
            print(f"  figures: skip {name}: image {img.shape} vs segmentation {seg.shape}")
            continue
        stems = ordered_yolo_txt_stems(name, str(label_dir)) if label_dir.is_dir() else []
        boxes = [read_boxes(label_dir / f"{s}.txt") for s in stems] if len(stems) == img.shape[0] else [[]] * img.shape[0]
        path = Path(run_dir) / FIGURES_SUBDIR / f"{name}.png"
        save_volume_figure(img, seg, boxes, slice_numbers(name, os.path.dirname(image_path), img.shape[0]),
                           mask_labels, path, name, n_slices)
        saved.append(path)
    print(f"  Saved {len(saved)} verification figure(s) to {Path(run_dir) / FIGURES_SUBDIR}")
    return saved
