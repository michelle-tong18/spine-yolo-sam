"""
Gap fill for one volume of YOLO detections (slice order), applied after relevel (relevel.py):

1. Collect: each slice's boxes, filtered as for SAM (``remove_label_ids``, ``det_limit``, ...).
2. Per class, for two consecutive slices ``a`` < ``b`` that both hold the class with at most ``max_gap_slices``
   empty slices between them, copy an anchor box (largest in-plane area of the class on ``a`` and ``b``, or highest
   confidence) into every empty slice, with confidence ``SYNTHETIC_CONF``.
3. ``det_limit`` again, so filled boxes obey the per-slice caps, and the boxes are written back into each slice's
   Ultralytics ``Results`` in place.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from ultralytics.engine.results import Boxes

from src.inference.inference_config import DetectionConfig
from src.inference.sam_from_boxes import _filter_detections, prepare_detections_for_sam

from .config import GapFillConfig

SYNTHETIC_CONF = 0.05


@dataclass
class VolumeDet:
    """One box on one slice (``xyxy`` in pixels)."""

    slice_idx: int
    label_id: int
    confidence: float
    xyxy: torch.Tensor


def extract_volume_detections(results_batch: Sequence[Any], detection_cfg: DetectionConfig) -> List[VolumeDet]:
    """Every slice's boxes, filtered as for SAM (``prepare_detections_for_sam``)."""
    dets = []
    for slice_idx, result in enumerate(results_batch):
        if result is None or result.boxes is None or len(result.boxes) == 0:
            continue
        for label_id, bbox, conf in prepare_detections_for_sam(result, detection_cfg):
            dets.append(VolumeDet(slice_idx, int(label_id), float(conf), bbox.detach().float().cpu().contiguous()))
    return dets


def _apply_det_limit(dets: List[VolumeDet], detection_cfg: DetectionConfig) -> Tuple[List[VolumeDet], int]:
    """Per-slice caps (``det_limit`` and the other filters); returns (kept, n_dropped)."""
    by_slice: Dict[int, List[VolumeDet]] = defaultdict(list)
    for d in dets:
        by_slice[d.slice_idx].append(d)
    kept = []
    for s in sorted(by_slice):
        row = by_slice[s]
        items = sorted(((d.label_id, i, d.confidence) for i, d in enumerate(row)), key=lambda x: -x[2])
        kept += [row[i] for _, i, _ in _filter_detections(items, detection_cfg)]
    return kept, len(dets) - len(kept)


def _area(d: VolumeDet) -> float:
    return max(0.0, float(d.xyxy[2] - d.xyxy[0])) * max(0.0, float(d.xyxy[3] - d.xyxy[1]))


def fill_gaps(dets: List[VolumeDet], n_slices: int, cfg: GapFillConfig) -> int:
    """Append synthetic boxes into through-plane gaps (``dets`` is extended in place); returns how many."""
    by_label: Dict[int, List[VolumeDet]] = defaultdict(list)
    for d in dets:
        by_label[d.label_id].append(d)
    filled = 0
    for label_id, group in by_label.items():
        occupied = sorted({d.slice_idx for d in group})
        for a, b in zip(occupied[:-1], occupied[1:]):
            if b - a <= 1 or b - a - 1 > cfg.max_gap_slices:
                continue
            boundary = [d for d in group if d.slice_idx in (a, b)]
            key = (lambda d: d.confidence) if cfg.anchor == "highest_conf" else _area
            anchor = max(boundary, key=key)
            for s in range(a + 1, b):
                if s < n_slices:
                    dets.append(VolumeDet(s, label_id, SYNTHETIC_CONF, anchor.xyxy.clone()))
                    filled += 1
    return filled


def rebuild_result_boxes(result: Any, dets_on_slice: List[VolumeDet]) -> None:
    """Write ``dets_on_slice`` back into ``result.boxes`` (mutates Ultralytics ``Results`` in place)."""
    if not dets_on_slice:
        result.boxes = None
        return
    device = result.boxes.xyxy.device if result.boxes is not None else torch.device("cpu")
    xyxy = torch.stack([d.xyxy.to(device) for d in dets_on_slice])
    conf = torch.tensor([d.confidence for d in dets_on_slice], dtype=torch.float32, device=device)
    cls = torch.tensor([d.label_id for d in dets_on_slice], dtype=torch.float32, device=device)
    result.boxes = Boxes(torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1), orig_shape=result.orig_shape)


def gap_fill_volume(results_batch: Sequence[Any], detection_cfg: DetectionConfig, cfg: GapFillConfig) -> Dict[str, int]:
    """Gap fill one volume's ``Results`` (slice order) in place; returns counters."""
    dets, dropped_before = _apply_det_limit(extract_volume_detections(results_batch, detection_cfg), detection_cfg)
    filled = fill_gaps(dets, len(results_batch), cfg)
    dets, dropped_after = _apply_det_limit(dets, detection_cfg)
    by_slice: Dict[int, List[VolumeDet]] = defaultdict(list)
    for d in dets:
        by_slice[d.slice_idx].append(d)
    for s, result in enumerate(results_batch):
        if result is not None:
            rebuild_result_boxes(result, by_slice.get(s, []))
    return {"gap_filled": filled, "det_limit_dropped": dropped_before + dropped_after}
