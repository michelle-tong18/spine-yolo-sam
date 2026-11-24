"""
Detection metrics: load pred/GT .txt labels and compute precision, recall, F1, mAP.
No inference code; read-from-disk only.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def load_txt_labels(label_dir, stems, has_conf=False, warn_missing=True):
    """
    Load YOLO .txt label files.
    Returns: {stem: [(class_id, x_c, y_c, w, h [, conf]), ...]}
    """
    label_dir = Path(label_dir)
    labels = {}
    missing = 0
    for stem in stems:
        txt_path = label_dir / f"{stem}.txt"
        boxes = []
        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    if has_conf and len(parts) >= 6:
                        conf = float(parts[5])
                        boxes.append((cls_id, x_c, y_c, w, h, conf))
                    else:
                        boxes.append((cls_id, x_c, y_c, w, h))
        else:
            missing += 1
        labels[stem] = boxes
    if missing > 0 and warn_missing:
        print(f"  Note: {missing}/{len(stems)} stems have no label file.")
    return labels


def compute_iou_box(box1, box2):
    """IoU between two (x_c, y_c, w, h) normalized boxes."""
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


def match_preds_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Greedy IoU matching.
    pred_boxes: [(x_c, y_c, w, h, conf), ...]
    gt_boxes:   [(x_c, y_c, w, h), ...]
    Returns: [(conf, is_tp), ...], n_gt
    """
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return [(p[4], False) for p in pred_boxes], 0

    preds_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    matched = set()
    results = []

    for pred in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched:
                continue
            iou = compute_iou_box(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou, best_idx = iou, gi

        if best_iou >= iou_threshold and best_idx >= 0:
            matched.add(best_idx)
            results.append((pred[4], True))
        else:
            results.append((pred[4], False))

    return results, n_gt


def compute_ap(matches, n_gt):
    """101-point interpolated AP."""
    if n_gt == 0:
        return 0.0

    matches_sorted = sorted(matches, key=lambda x: x[0], reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []

    for _, is_tp in matches_sorted:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p_at_r = max((p for p, r in zip(precisions, recalls) if r >= t), default=0.0)
        ap += p_at_r / 101
    return ap


def compute_detection_metrics_from_labels(preds, gt, stems, class_names, iou_thresholds=None):
    """``compute_detection_metrics`` on already-loaded ``load_txt_labels`` dicts."""
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    all_stems = sorted(set(stems))

    all_classes = sorted(k for k in class_names.keys() if k != 0)
    if not all_classes:
        all_classes = sorted(set().union(*(set(b[0] for b in boxes) for boxes in gt.values())))

    matches_by_class = {t: {c: [] for c in all_classes} for t in iou_thresholds}
    gt_count = {c: 0 for c in all_classes}

    for stem in all_stems:
        gt_by_cls = {}
        for box in gt.get(stem, []):
            gt_by_cls.setdefault(box[0], []).append(box[1:5])
        pred_by_cls = {}
        for box in preds.get(stem, []):
            pred_by_cls.setdefault(box[0], []).append(box[1:6])

        for cls_id in all_classes:
            g = gt_by_cls.get(cls_id, [])
            p = pred_by_cls.get(cls_id, [])
            gt_count[cls_id] += len(g)
            for t in iou_thresholds:
                m, _ = match_preds_to_gt(p, g, iou_threshold=t)
                matches_by_class[t][cls_id].extend(m)

    metrics = {}
    for cls_id in all_classes:
        name = class_names.get(cls_id, f"class_{cls_id}")
        n_gt = gt_count[cls_id]
        aps = {f"AP@{t:.2f}": compute_ap(matches_by_class[t][cls_id], n_gt) for t in iou_thresholds}

        primary = iou_thresholds[0]
        tp = sum(1 for _, hit in matches_by_class[primary][cls_id] if hit)
        fp = sum(1 for _, hit in matches_by_class[primary][cls_id] if not hit)
        fn = n_gt - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics[name] = {
            "class_id": cls_id,
            "n_gt": n_gt,
            "n_pred": tp + fp,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": prec,
            "recall": rec,
            "F1": f1,
            **aps,
        }

    active = [c for c in all_classes if gt_count[c] > 0]
    tot_tp = sum(metrics[class_names.get(c, f"class_{c}")]["TP"] for c in all_classes)
    tot_fp = sum(metrics[class_names.get(c, f"class_{c}")]["FP"] for c in all_classes)
    tot_fn = sum(metrics[class_names.get(c, f"class_{c}")]["FN"] for c in all_classes)
    tot_gt = sum(gt_count[c] for c in all_classes)

    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    if active:
        macro_p = np.mean([metrics[class_names.get(c, f"class_{c}")]["precision"] for c in active])
        macro_r = np.mean([metrics[class_names.get(c, f"class_{c}")]["recall"] for c in active])
        macro_f1 = np.mean([metrics[class_names.get(c, f"class_{c}")]["F1"] for c in active])
        map_d = {
            f"mAP@{t:.2f}": np.mean([metrics[class_names.get(c, f"class_{c}")][f"AP@{t:.2f}"] for c in active])
            for t in iou_thresholds
        }
    else:
        macro_p = macro_r = macro_f1 = 0.0
        map_d = {f"mAP@{t:.2f}": 0.0 for t in iou_thresholds}

    metrics["__overall__"] = {
        "n_gt": tot_gt,
        "n_pred": tot_tp + tot_fp,
        "TP": tot_tp,
        "FP": tot_fp,
        "FN": tot_fn,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_F1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_F1": macro_f1,
        **map_d,
    }
    return metrics

def get_test_stems_from_gt_labels(gt_labels_dir, test_subject_ids):
    """
    Ordered slice stems for evaluation from the GT labels dir and test subject IDs.
    A subject's labels are ``{subject_id}-{###}.txt`` or, with one volume per accession / series,
    ``{subject_id}_{series}-{###}.txt``. Subjects keep their input order; within a subject, stems are
    sorted by series, then slice index.
    """
    gt_dir = Path(gt_labels_dir)
    stems = []
    for sid in test_subject_ids:
        records = []
        for p in list(gt_dir.glob(f"{sid}-*.txt")) + list(gt_dir.glob(f"{sid}_*-*.txt")):
            volume, _, idx = p.stem.rpartition("-")
            series = volume[len(sid) + 1:]
            if not idx.isdigit():
                continue
            records.append((series, int(idx), p.stem))
        stems.extend(stem for _, _, stem in sorted(records))
    return stems


def volumes_without_predictions(pred_labels_dir, gt_stems) -> list[str]:
    """
    Volumes with GT slices but no prediction file of the same name. Predictions and GT labels are both named
    ``{volume}-{slice:03d}`` with the slice's index in the scan; a volume listed here was predicted with other names
    (e.g. from volumes that were not preprocessed) and would be scored as all misses.
    """
    pred_stems = {p.stem for p in Path(pred_labels_dir).glob("*.txt")}
    gt_volumes = {volume_id_from_stem(s) for s in gt_stems}
    matched = {volume_id_from_stem(s) for s in gt_stems if s in pred_stems}
    return sorted(gt_volumes - matched)


def volume_id_from_stem(stem: str) -> str:
    """Volume / patient id from a slice stem ``{volume_id}-{slice_idx}``."""
    stem = str(stem)
    idx = stem.rfind("-")
    if idx >= 0 and stem[idx + 1:].isdigit():
        return stem[:idx]
    return stem


def merge_label_classes(labels, class_to_group):
    """Relabel each box to its group id (e.g. every disc -> one class); boxes in no group are dropped."""
    return {
        stem: [(class_to_group[b[0]], *b[1:]) for b in boxes if b[0] in class_to_group]
        for stem, boxes in labels.items()
    }


def subject_and_series_from_stem(stem):
    """'{subject}_{series}-{slice}' -> (subject, series); '{subject}-{slice}' -> (subject, ''). Subject ids contain no '_'."""
    subject, _, series = volume_id_from_stem(stem).partition("_")
    return subject, series


def compute_detection_metrics_by_scope(
    pred_labels_dir,
    gt_labels_dir,
    stems,
    class_names,
    tissue_groups,
    iou_thresholds=None,
):
    """
    Detection metrics for every (scope, level) pair as a long-format DataFrame.

    scope: ``overall`` (all stems), ``patient`` (every series of one subject pooled),
        ``series`` (one volume).
    level: ``label`` (each class), ``tissue`` (classes in each ``tissue_groups`` entry merged into
        one class before matching, so an L3 box on the L4 body is a TP for ``vertebrae``),
        ``all_tissues`` (every class merged into one).
    Each (scope, level) block also carries the ``__overall__`` micro/macro row.
    """
    all_stems = sorted(set(stems))
    preds = load_txt_labels(Path(pred_labels_dir), all_stems, has_conf=True, warn_missing=False)
    gt = load_txt_labels(Path(gt_labels_dir), all_stems, has_conf=False, warn_missing=False)

    all_classes = sorted(k for k in class_names if k != 0)
    # Group ids start at 1: compute_detection_metrics_from_labels skips class 0 as background.
    levels = {
        "label": ({c: c for c in all_classes}, {c: class_names[c] for c in all_classes}),
        "tissue": (
            {c: gi for gi, ids in enumerate(tissue_groups.values(), start=1) for c in ids},
            {gi: name for gi, name in enumerate(tissue_groups, start=1)},
        ),
        "all_tissues": ({c: 1 for c in all_classes}, {1: "all_tissues"}),
    }

    stems_by_patient = defaultdict(list)
    stems_by_series = defaultdict(list)
    for stem in all_stems:
        subject, series = subject_and_series_from_stem(stem)
        stems_by_patient[subject].append(stem)
        stems_by_series[(subject, series)].append(stem)
    scopes = [("overall", "", "", all_stems)]
    scopes += [("patient", p, "", st) for p, st in sorted(stems_by_patient.items())]
    scopes += [("series", p, s, st) for (p, s), st in sorted(stems_by_series.items())]

    rows = []
    for level, (class_to_group, group_names) in levels.items():
        if not group_names:
            continue
        level_preds = merge_label_classes(preds, class_to_group)
        level_gt = merge_label_classes(gt, class_to_group)
        for scope, patient_id, series, scope_stems in scopes:
            metrics = compute_detection_metrics_from_labels(
                level_preds, level_gt, scope_stems, group_names, iou_thresholds
            )
            for cls_name, m in metrics.items():
                rows.append({
                    "scope": scope,
                    "patient_id": patient_id,
                    "series": series,
                    "level": level,
                    "class": cls_name,
                    **{k: _to_builtin(v) for k, v in m.items()},
                })
    return pd.DataFrame(rows)


def _to_builtin(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
