"""Detection settings (``inference.models.obj_det``) shared by YOLO predict, gap fill and SAM, and run subfolders."""

from __future__ import annotations

from dataclasses import dataclass, field

LABEL_PRED_SUBDIR = "label_pred"
SEG_PRED_SUBDIR = "seg_pred"


@dataclass
class DetectionConfig:
    """Object-detection and label-handling config for YOLO + SAM."""

    conf: float = 0.5
    iou: float = 0.7
    det_limit: dict[int, int] = field(default_factory=dict)
    remove_label_ids: list[int] = field(default_factory=list)
    ignore_label_ids: list[int] = field(default_factory=list)
    subtract_label_ids: list[int] = field(default_factory=list)
    priority_label_ids: list[int] = field(default_factory=list)
    single_cls: bool = False


def detection_config_from_dict(obj_det: dict) -> DetectionConfig:
    """DetectionConfig from ``inference.models.obj_det``."""
    return DetectionConfig(
        conf=float(obj_det.get("conf", 0.5)),
        iou=float(obj_det.get("iou", 0.7)),
        det_limit={int(k): int(v) for k, v in (obj_det.get("det_limit") or {}).items()},
        remove_label_ids=list(obj_det.get("remove_label_ids") or []),
        ignore_label_ids=list(obj_det.get("ignore_label_ids") or []),
        subtract_label_ids=list(obj_det.get("subtract_label_ids") or []),
        priority_label_ids=list(obj_det.get("priority_label_ids") or []),
        single_cls=bool(obj_det.get("single_cls", False)),
    )
