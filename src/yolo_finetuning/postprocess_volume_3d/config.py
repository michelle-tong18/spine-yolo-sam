"""
Gap fill configuration, from ``inference.models.obj_det.postprocess`` in the dataset YAML (see README).

    postprocess:
      enabled: true
      type: axial_spine_muscle        # sagittal_spine | axial_spine_muscle
      gap_fill:
        enabled: true
        max_gap_slices: 4
        anchor: largest_area          # largest_area | highest_conf
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

POSTPROCESS_TYPES = ("sagittal_spine", "axial_spine_muscle")
_POSTPROCESS_KEYS = {"enabled", "type", "gap_fill"}
_GAP_FILL_KEYS = {"enabled", "max_gap_slices", "anchor"}


@dataclass
class GapFillConfig:
    """Copy an anchor box into the empty slices between two slices that hold the same class."""

    max_gap_slices: int = 8
    anchor: str = "largest_area"  # largest_area | highest_conf


def _check_keys(block: Mapping[str, Any], allowed: set, where: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)} "
                         "(src/yolo_finetuning/postprocess_volume_3d/README.md)")


def gap_fill_config_from_dict(postprocess: Optional[Mapping[str, Any]]) -> Optional[GapFillConfig]:
    """GapFillConfig when ``postprocess.enabled`` and ``postprocess.gap_fill.enabled``, else None."""
    if not postprocess:
        return None
    _check_keys(postprocess, _POSTPROCESS_KEYS, "inference.models.obj_det.postprocess")
    gap_fill = postprocess.get("gap_fill") or {}
    _check_keys(gap_fill, _GAP_FILL_KEYS, "inference.models.obj_det.postprocess.gap_fill")
    if not postprocess.get("enabled") or not gap_fill.get("enabled"):
        return None
    if postprocess.get("type") not in POSTPROCESS_TYPES:
        raise ValueError(f"inference.models.obj_det.postprocess.type={postprocess.get('type')!r}; "
                         f"use one of {POSTPROCESS_TYPES}")
    anchor = str(gap_fill.get("anchor", "largest_area"))
    if anchor not in ("largest_area", "highest_conf"):
        raise ValueError(f"postprocess.gap_fill.anchor={anchor!r}; use largest_area or highest_conf")
    return GapFillConfig(max_gap_slices=int(gap_fill.get("max_gap_slices", 8)), anchor=anchor)
