"""
Segmentation metrics: load pred and GT masks from disk and compute Dice, IoU and HD95 per series / patient / overall
and per label / tissue group / all tissues (``compute_segmentation_metrics_by_scope``). Read-from-disk only.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt

from src.preprocessing.spacing import load_grid_spacing
from src.utils.file_management.file_handler import load_data
from src.utils.metrics import compute_binary_iou, compute_dice_coefficient


def load_volume(path):
    """Load 3D volume from nii.gz, npy, or npz."""
    if not os.path.isfile(path):
        return None
    return load_data(path)


def get_test_stems_from_gt_masks(gt_mask_dir, test_subject_ids, ext="nii.gz"):
    """
    Ordered volume stems for evaluation from the GT mask dir: ``{subject_id}.{ext}`` or, with one volume
    per accession / series, ``{subject_id}_{series}.{ext}``. Subjects keep their input order; series are
    sorted within each subject.
    """
    gt_dir = Path(gt_mask_dir)
    stems = []
    for sid in test_subject_ids:
        paths = list(gt_dir.glob(f"{sid}.{ext}")) + list(gt_dir.glob(f"{sid}_*.{ext}"))
        stems.extend(sorted(p.name.removesuffix(f".{ext}") for p in paths))
    return stems


def slice_surface_distances(gt2d: np.ndarray, pred2d: np.ndarray, row_mm: float, col_mm: float) -> np.ndarray | None:
    """In-plane surface distances (mm) between two 2D masks, both directions: every boundary pixel of each to the
    nearest boundary pixel of the other. None when either mask is empty (HD95 undefined on that slice)."""
    if not gt2d.any() or not pred2d.any():
        return None
    # Crop to both masks (+1 px) so the distance transforms stay small; distances between pixels inside are unchanged.
    ys, xs = np.nonzero(gt2d | pred2d)
    crop = (slice(max(ys.min() - 1, 0), ys.max() + 2), slice(max(xs.min() - 1, 0), xs.max() + 2))
    g, p = gt2d[crop], pred2d[crop]
    g_edge, p_edge = g & ~binary_erosion(g), p & ~binary_erosion(p)
    to_pred = distance_transform_edt(~p_edge, sampling=(row_mm, col_mm))
    to_gt = distance_transform_edt(~g_edge, sampling=(row_mm, col_mm))
    return np.concatenate([to_pred[g_edge], to_gt[p_edge]])


def volume_hd95(gt_bin: np.ndarray, pred_bin: np.ndarray, row_mm: float, col_mm: float) -> dict[str, float]:
    """
    HD95 (mm) of one class in one volume, in-plane per slice (preprocessing may keep non-adjacent slices, so
    distances across slices are not used): ``hd95`` mean of per-slice 95th percentiles, ``hd95_pooled`` 95th
    percentile of every slice's distances pooled, ``hd95_undefined_frac`` share of the slices holding the class in
    GT or prediction where only one of the two does. All nan without pixel spacing.
    """
    present = gt_bin.reshape(gt_bin.shape[0], -1).any(axis=1) | pred_bin.reshape(pred_bin.shape[0], -1).any(axis=1)
    if not (np.isfinite(row_mm) and np.isfinite(col_mm)):
        return {"hd95": np.nan, "hd95_pooled": np.nan, "hd95_undefined_frac": np.nan}
    defined = [d for z in np.flatnonzero(present)
               if (d := slice_surface_distances(gt_bin[z], pred_bin[z], row_mm, col_mm)) is not None]
    n_present = int(present.sum())
    return {
        "hd95": float(np.mean([np.percentile(d, 95) for d in defined])) if defined else np.nan,
        "hd95_pooled": float(np.percentile(np.concatenate(defined), 95)) if defined else np.nan,
        "hd95_undefined_frac": (n_present - len(defined)) / n_present if n_present else np.nan,
    }


def compute_segmentation_metrics_by_scope(
    pred_dir,
    gt_masks_dir,
    subject_ids,
    mask_labels: dict[int, str],
    tissue_groups: dict[str, list[int]],
    pred_ext="nii.gz",
) -> pd.DataFrame:
    """
    Dice / IoU and HD95 (mm, ``volume_hd95``) for every (scope, level) pair as a long-format DataFrame.

    level: ``label`` (each class), ``tissue`` (each ``tissue_groups`` entry as one binary mask),
        ``all_tissues`` (every label as one binary mask).
    scope: ``series`` rows score one volume; ``patient`` and ``overall`` rows are the mean over
        series rows (nan skipped), with ``n_series_scored`` counting the volumes with a Dice.
    Pixel spacing for HD95 is the one preprocessing recorded for the GT volume (src/preprocessing/spacing.py);
    HD95 is nan for volumes without it.
    """
    label_ids = sorted(k for k in mask_labels if k != 0)
    classes = ([("label", mask_labels[lid], [lid]) for lid in label_ids]
               + [("tissue", name, ids) for name, ids in tissue_groups.items()]
               + [("all_tissues", "all_tissues", label_ids)])
    rows, no_spacing = [], []
    for stem in get_test_stems_from_gt_masks(gt_masks_dir, subject_ids, pred_ext):
        gt = load_volume(os.path.join(gt_masks_dir, f"{stem}.{pred_ext}"))
        pred = load_volume(os.path.join(pred_dir, f"{stem}.{pred_ext}"))
        if gt is None or pred is None or np.shape(gt) != np.shape(pred):
            print(f"  Skipping {stem}: prediction missing or shape differs from GT.")
            continue
        gt, pred = np.asarray(gt).astype(np.int64), np.asarray(pred).astype(np.int64)
        spacing = load_grid_spacing(stem, gt_masks_dir)
        if spacing is None:
            no_spacing.append(stem)
        row_mm, col_mm = spacing[1:] if spacing is not None else (np.nan, np.nan)
        subject, _, series = stem.partition("_")
        for level, name, ids in classes:
            g, p = np.isin(gt, ids), np.isin(pred, ids)
            rows.append({
                "scope": "series", "patient_id": subject, "series": series, "level": level, "class": name,
                "dice": float(compute_dice_coefficient(g, p)),
                "iou": float(compute_binary_iou(g, p)),
                **volume_hd95(g, p, row_mm, col_mm),
                "volume_gt_voxels": int(g.sum()),
                "volume_pred_voxels": int(p.sum()),
            })
    if no_spacing:
        print(f"  HD95: no pixel spacing recorded for {len(no_spacing)} volume(s) (e.g. {no_spacing[0]}); their HD95 "
              "is nan. Preprocessing records spacing in nifti_preprocessed/spacing.csv (src/preprocessing/spacing.py).")

    series_df = pd.DataFrame(rows)
    if series_df.empty:
        return series_df
    agg = {"dice": ("dice", "mean"), "iou": ("iou", "mean"), "hd95": ("hd95", "mean"),
           "hd95_pooled": ("hd95_pooled", "mean"), "hd95_undefined_frac": ("hd95_undefined_frac", "mean"),
           "n_series_scored": ("dice", "count")}
    patient_df = (
        series_df.groupby(["patient_id", "level", "class"], sort=False).agg(**agg).reset_index()
        .assign(scope="patient", series="")
    )
    overall_df = (
        series_df.groupby(["level", "class"], sort=False).agg(**agg).reset_index()
        .assign(scope="overall", patient_id="", series="")
    )
    columns = ["scope", "patient_id", "series", "level", "class", "dice", "iou", "hd95", "hd95_pooled",
               "hd95_undefined_frac", "n_series_scored", "volume_gt_voxels", "volume_pred_voxels"]
    return pd.concat([overall_df, patient_df, series_df], ignore_index=True).reindex(columns=columns)
