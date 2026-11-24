"""
Evaluation entrypoint: score a finished inference run against GT, from disk only (no models).

    python src/evaluation/run_evaluation.py example_sagittal_vertdisc --run <run_dir> [--output_dir DIR]

Test volumes are the ``test`` subjects of ``split_id`` (config/splits/splits_registry.yaml); GT from
``evaluation.gt_labels_dir`` (YOLO .txt per slice) and ``evaluation.gt_masks_dir`` (3D masks). Writes to
``--output_dir`` (default ``{run}/evaluation``):
    detection_metrics_breakdown.csv     precision / recall / F1 / mAP per scope x level x class
    segmentation_metrics_breakdown.csv  Dice / IoU / HD95 per scope x level x class
Scopes: overall, patient, series (one volume). Levels: label, tissue (``evaluation.tissue_groups`` merged into one
class), all_tissues. The overall rows are printed.
"""
import os
import argparse

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.file_management.config_handler import get_config_name, load_main_config
from src.sam_finetuning.datamodule.dynamic_dataset_loader import get_splits
from src.inference.inference_config import LABEL_PRED_SUBDIR, SEG_PRED_SUBDIR
from src.evaluation.detection_metrics import (
    compute_detection_metrics_by_scope,
    get_test_stems_from_gt_labels,
    volumes_without_predictions,
)
from src.evaluation.segmentation_metrics import compute_segmentation_metrics_by_scope

DETECTION_CSV = "detection_metrics_breakdown.csv"
SEGMENTATION_CSV = "segmentation_metrics_breakdown.csv"


def _required(cfg, key):
    """``evaluation.<key>``; required, not defaulted from general_data_dir."""
    value = (cfg.get("evaluation") or {}).get(key)
    if not value:
        raise ValueError(f"evaluation.{key} is required to score this run. "
                         "Evaluation does not default this path from general_data_dir.")
    return value


def _class_names(cfg):
    """Label ID -> name from ``mask_labels``, without background (0)."""
    return {int(k): str(v) for k, v in cfg.get("mask_labels", {}).items() if int(k) != 0}


def _tissue_groups(cfg):
    """evaluation.tissue_groups: tissue name -> class IDs merged into one class for hierarchical metrics."""
    groups = (cfg.get("evaluation") or {}).get("tissue_groups") or {}
    if not groups:
        print("  Note: evaluation.tissue_groups not set; breakdown has label and all_tissues levels only.")
    return {str(name): [int(i) for i in ids] for name, ids in groups.items()}


def _save_and_print(df, summary, metric_cols, out_path):
    """Write the breakdown ``df`` and print the ``summary`` rows' ``metric_cols``."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    for _, row in summary.iterrows():
        values = "  ".join(f"{c} {row[c]:.4f}" for c in metric_cols if c in row)
        print(f"  {row['level']:<11} {row['class']:<15} {values}")
    print(f"  Breakdown (overall / patient / series) saved to {out_path}")


def run_detection_evaluation(cfg, pred_labels_dir, out_path):
    """Detection breakdown of ``pred_labels_dir`` (``{run}/label_pred``) against ``evaluation.gt_labels_dir``."""
    gt_labels_dir = _required(cfg, "gt_labels_dir")
    stems = get_test_stems_from_gt_labels(gt_labels_dir, get_splits(cfg)["test"].tolist())
    unmatched = volumes_without_predictions(pred_labels_dir, stems)
    if unmatched:
        print(f"  WARNING: {len(unmatched)} volume(s) have no prediction named like their GT slices "
              f"(e.g. {unmatched[0]}); they score as all misses. Predictions and GT labels must "
              "both be named {volume}-{scan slice number}: run inference on the volumes preprocessing wrote.")
    print(f"\n--- Detection metrics ({len(stems)} GT slices) ---")
    df = compute_detection_metrics_by_scope(pred_labels_dir, gt_labels_dir, stems, _class_names(cfg),
                                            _tissue_groups(cfg),
                                            iou_thresholds=(cfg.get("evaluation") or {}).get("iou_thresholds", [0.5]))
    summary = df[(df["scope"] == "overall") & (df["class"] == "__overall__")]
    _save_and_print(df, summary, ["micro_precision", "micro_recall", "micro_F1"]
                    + [c for c in df.columns if c.startswith("mAP@")], out_path)


def run_segmentation_evaluation(cfg, pred_dir, out_path):
    """Segmentation breakdown of ``pred_dir`` (``{run}/seg_pred``) against ``evaluation.gt_masks_dir``."""
    pred_ext = (cfg.get("inference") or {}).get("output_ext", "nii.gz") or "nii.gz"
    print("\n--- Segmentation metrics ---")
    df = compute_segmentation_metrics_by_scope(pred_dir=pred_dir, gt_masks_dir=_required(cfg, "gt_masks_dir"),
                                               subject_ids=get_splits(cfg)["test"].tolist(),
                                               mask_labels=_class_names(cfg), tissue_groups=_tissue_groups(cfg),
                                               pred_ext=pred_ext)
    if df.empty:
        print("  No volumes scored; breakdown not written.")
        return
    summary = df[(df["scope"] == "overall") & (df["level"] != "label")]
    _save_and_print(df, summary, ["dice", "iou", "hd95"], out_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions: detection and/or segmentation metrics on the test set.")
    parser.add_argument("config_name", help="Dataset config name (YAML without extension) or a YAML path.")
    parser.add_argument("--run", dest="run_dir", default=None,
                        help="Inference run directory. Reads {run}/label_pred/ and {run}/seg_pred/.")
    parser.add_argument("--detection_labels_dir", default=None,
                        help="Directory of prediction .txt files (label_pred/). Overrides --run for detection.")
    parser.add_argument("--segmentation_pred_dir", default=None,
                        help="Directory of predicted 3D masks (seg_pred/). Overrides --run for segmentation.")
    parser.add_argument("--output_dir", default=None, help="Where to write the breakdown CSVs (default: {run}/evaluation).")
    args = parser.parse_args()

    if not args.run_dir and not args.detection_labels_dir and not args.segmentation_pred_dir:
        parser.error("provide at least one of --run, --detection_labels_dir, or --segmentation_pred_dir")
    output_dir = args.output_dir or (os.path.join(args.run_dir, "evaluation") if args.run_dir else None)
    if output_dir is None:
        parser.error("--output_dir is required without --run")
    cfg = load_main_config(get_config_name(args.config_name), os.getcwd(), sub_dir="datasets")

    pred_labels = args.detection_labels_dir or (os.path.join(args.run_dir, LABEL_PRED_SUBDIR) if args.run_dir else None)
    if pred_labels and os.path.isdir(pred_labels):
        run_detection_evaluation(cfg, pred_labels, os.path.join(output_dir, DETECTION_CSV))
    elif pred_labels:
        print(f"  Skipping detection: labels dir not found: {pred_labels}")

    seg_pred = args.segmentation_pred_dir or (os.path.join(args.run_dir, SEG_PRED_SUBDIR) if args.run_dir else None)
    if seg_pred and os.path.isdir(seg_pred):
        run_segmentation_evaluation(cfg, seg_pred, os.path.join(output_dir, SEGMENTATION_CSV))
    elif seg_pred:
        print(f"  Skipping segmentation: pred dir not found: {seg_pred}")


if __name__ == "__main__":
    main()
