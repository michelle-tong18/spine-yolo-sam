"""
Relevel: give each physical level one label through the volume.

YOLO usually gets the tissue right (vertebra vs disc vs facet) but can be one level off on some slices.
For each chain (an ordered cranial→caudal list of class IDs, e.g. vertebrae+discs or facets):

0. One structure, one box: same-tissue boxes on a slice that overlap in y (``dup_y_iou``) are one detection;
   only the most confident is tracked and the others take its level afterwards (det_limit then drops them).
1. Track: boxes of one tissue are linked across neighbouring slices by box-centre ``y``, after removing the
   whole column's shift between slices (up to ``max_column_shift`` box heights; lateral slices drift), allowing
   gaps of up to ``max_link_gap`` slices. Each track is one physical level, whatever YOLO called it.
2. Merge: same-tissue tracks that never share a slice and sit at the same height (median ``y`` within
   ``merge_y_tol`` box heights) are the same level split by a longer gap (e.g. left / right facets) and
   are merged, closest pair first.
3. Assign: tracks are ordered by median ``y`` (cranial = smaller ``y``) and mapped to the chain's levels by
   a monotone DP that matches tissue types and maximizes confidence-weighted YOLO votes (anchor levels,
   e.g. L5, weighted by ``anchor_weight``) minus ``skip_penalty`` per level skipped between two tracks and
   ``unassigned_penalty`` x the most a track could gain (``anchor_weight`` x its weight) for any track left
   without a level: a well-supported level has to be explained, so a shift cannot drop the true S1, while a
   short low-confidence false positive is cheaper to leave alone than to shift the chain around. Every box in a
   track takes its track's level.

Pure Python + numpy; ``relevel_results`` adapts Ultralytics ``Results`` (src/inference/yolo_predict.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class ChainConfig:
    name: str
    order: list[int]  # cranial → caudal class IDs
    tissue_of: dict[int, str]  # class ID → tissue name (tracks never mix tissues)
    anchors: set[int] = field(default_factory=set)


@dataclass
class RelevelConfig:
    chains: list[ChainConfig]
    anchor_weight: float = 2.0
    max_link_gap: int = 2  # slices a track may skip between two boxes
    link_y_tol: float = 0.75  # max |Δy| between linked boxes, in mean box heights
    merge_y_tol: float = 0.5  # max |Δ median y| between merged tracks, in mean box heights
    skip_penalty: float = 0.5  # per skipped level, in units of the chain's median track weight
    unassigned_penalty: float = 1.0  # per track left without a level, in units of anchor_weight x its weight
    dup_y_iou: float = 0.5  # same-tissue boxes on one slice overlapping this much in y are one structure
    max_column_shift: float = 1.0  # largest whole-column shift between slices removed before linking, in box heights


def relevel_config_from_dict(raw: Mapping[str, Any]) -> RelevelConfig:
    """
    ``inference.models.obj_det.relevel`` block, e.g.::

        chains:
          vertebrae_discs:
            order: [19, 18, ..., 1, 12, 13]           # cranial → caudal
            tissues: {vertebrae: [19, ..., 13], discs: [18, ..., 12]}
            anchor_class_ids: [1, 12, 13]
    """
    chains = []
    for name, body in (raw.get("chains") or {}).items():
        order = [int(c) for c in body["order"]]
        tissue_of = {int(c): str(t) for t, ids in (body.get("tissues") or {name: order}).items() for c in ids}
        missing = [c for c in order if c not in tissue_of]
        if missing:
            raise ValueError(f"relevel chain {name!r}: classes {missing} have no tissue")
        chains.append(ChainConfig(str(name), order, tissue_of, {int(c) for c in body.get("anchor_class_ids") or []}))
    return RelevelConfig(
        chains=chains,
        anchor_weight=float(raw.get("anchor_weight", 2.0)),
        max_link_gap=int(raw.get("max_link_gap", 2)),
        link_y_tol=float(raw.get("link_y_tol", 0.75)),
        merge_y_tol=float(raw.get("merge_y_tol", 0.5)),
        skip_penalty=float(raw.get("skip_penalty", 0.5)),
        unassigned_penalty=float(raw.get("unassigned_penalty", 1.0)),
        dup_y_iou=float(raw.get("dup_y_iou", 0.5)),
        max_column_shift=float(raw.get("max_column_shift", 1.0)),
    )


@dataclass
class _Box:
    slice_pos: int
    index: int  # position in that slice's box list
    cls: int
    y: float
    h: float
    conf: float


@dataclass
class _Track:
    boxes: list[_Box]

    @property
    def slices(self) -> set[int]:
        return {b.slice_pos for b in self.boxes}

    @property
    def median_y(self) -> float:
        return float(np.median([b.y for b in self.boxes]))

    @property
    def median_h(self) -> float:
        return float(np.median([b.h for b in self.boxes]))

    @property
    def weight(self) -> float:
        return float(sum(b.conf for b in self.boxes))


def _y_iou(y1: float, h1: float, y2: float, h2: float) -> float:
    """1-D IoU of the vertical extents [y - h/2, y + h/2] of two boxes."""
    inter = min(y1 + h1 / 2, y2 + h2 / 2) - max(y1 - h1 / 2, y2 - h2 / 2)
    return max(0.0, inter) / (h1 + h2 - max(0.0, inter))


def _column_shift(previous: list[_Box], current: list[_Box], cfg: RelevelConfig) -> float:
    """
    Vertical shift of the whole column from the active tracks' last boxes to this slice (lateral sagittal slices
    drift with curvature / rotation). Tries each box-pair offset within max_column_shift box heights and keeps
    the one that lines up the most boxes with a same-class box within the link gate, then the most boxes, then
    the smallest leftover distance. YOLO's classes break the tie when the drift is about half a level spacing,
    where y alone fits a shift up and a shift down equally well.
    """
    h = float(np.median([b.h for b in current]))
    offsets = {0.0} | {b.y - p.y for p in previous for b in current if abs(b.y - p.y) <= cfg.max_column_shift * h}
    best_key, best = None, 0.0
    for d in offsets:
        nearest = [min(previous, key=lambda p: abs(b.y - d - p.y)) for b in current]
        residuals = [abs(b.y - d - p.y) for b, p in zip(current, nearest)]
        matched = [(r, p.cls == b.cls) for r, b, p in zip(residuals, current, nearest) if r <= cfg.link_y_tol * b.h]
        key = (-sum(same for _, same in matched), -len(matched), sum(r for r, _ in matched), abs(d))
        if best_key is None or key < best_key:
            best_key, best = key, d
    return best


def _link_tracks(boxes: list[_Box], cfg: RelevelConfig) -> list[_Track]:
    """
    Greedy one-to-one linking slice by slice: after removing the column shift, each box joins an active track
    within the gate, preferring one that ends in the same YOLO class, then the closest in y.
    """
    by_slice: dict[int, list[_Box]] = {}
    for b in boxes:
        by_slice.setdefault(b.slice_pos, []).append(b)
    tracks: list[_Track] = []
    for s in sorted(by_slice):
        active = [ti for ti, tr in enumerate(tracks) if 1 <= s - tr.boxes[-1].slice_pos <= cfg.max_link_gap + 1]
        shift = _column_shift([tracks[ti].boxes[-1] for ti in active], by_slice[s], cfg) if active else 0.0
        candidates = []
        for ti in active:
            last = tracks[ti].boxes[-1]
            gap = s - last.slice_pos
            for bi, b in enumerate(by_slice[s]):
                dy = abs(b.y - shift - last.y)
                if dy <= cfg.link_y_tol * (b.h + last.h) / 2:
                    candidates.append((b.cls != last.cls, dy, gap, ti, bi))
        used_tracks, used_boxes = set(), set()
        for _other_cls, _dy, _gap, ti, bi in sorted(candidates):
            if ti in used_tracks or bi in used_boxes:
                continue
            tracks[ti].boxes.append(by_slice[s][bi])
            used_tracks.add(ti)
            used_boxes.add(bi)
        tracks.extend(_Track([b]) for bi, b in enumerate(by_slice[s]) if bi not in used_boxes)
    return tracks


def _merge_tracks(tracks: list[_Track], cfg: RelevelConfig) -> tuple[list[_Track], int]:
    """
    Merge same-level tracks split by a gap: never on a shared slice, median y within merge_y_tol mean
    heights. Closest pair first, re-evaluated after every merge so a merged track is checked again.
    """
    tracks = list(tracks)
    n_merged = 0
    while True:
        best = None
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                if a.slices & b.slices:
                    continue
                dist = abs(a.median_y - b.median_y) / ((a.median_h + b.median_h) / 2)
                if dist <= cfg.merge_y_tol and (best is None or dist < best[0]):
                    best = (dist, i, j)
        if best is None:
            return tracks, n_merged
        _, i, j = best
        merged = _Track(sorted(tracks[i].boxes + tracks[j].boxes, key=lambda b: b.slice_pos))
        tracks = [t for k, t in enumerate(tracks) if k not in (i, j)] + [merged]
        n_merged += 1


def _assign_levels(tracks: list[_Track], chain: ChainConfig, cfg: RelevelConfig) -> list[int | None]:
    """
    Monotone DP from tracks (sorted by median y) to chain positions with matching tissue. A track left
    unassigned keeps its YOLO labels and costs unassigned_penalty x anchor_weight x its weight; leading /
    trailing levels are free.
    """
    order = chain.order
    m = len(order)
    weights = [cfg.anchor_weight if c in chain.anchors else 1.0 for c in order]
    penalty = cfg.skip_penalty * float(np.median([t.weight for t in tracks]))
    votes = []
    for t in tracks:
        v = np.zeros(m)
        for b in t.boxes:
            if b.cls in order:
                v[order.index(b.cls)] += b.conf
        votes.append(v)
    tissue_of_track = [chain.tissue_of[t.boxes[0].cls] for t in tracks]

    NONE = m  # sentinel index: nothing assigned yet
    best = np.full(m + 1, -np.inf)
    best[NONE] = 0.0
    back = []  # per track: for each state, (previous state, assigned position or None)
    for ti in range(len(tracks)):
        # Leave this track unassigned, keep the state.
        new = best - cfg.unassigned_penalty * cfg.anchor_weight * tracks[ti].weight
        choice = [(j, None) for j in range(m + 1)]
        for j in range(m):
            if chain.tissue_of[order[j]] != tissue_of_track[ti]:
                continue
            gain = weights[j] * votes[ti][j]
            cand, prev = (best[NONE], NONE) if best[NONE] > -np.inf else (-np.inf, None)
            for k in range(j):
                if best[k] > -np.inf and best[k] - penalty * (j - k - 1) > cand:
                    cand, prev = best[k] - penalty * (j - k - 1), k
            if prev is not None and cand + gain >= new[j]:
                new[j], choice[j] = cand + gain, (prev, j)
        best = new
        back.append(choice)

    state = int(np.argmax(best))
    assigned: list[int | None] = [None] * len(tracks)
    for ti in range(len(tracks) - 1, -1, -1):
        prev, pos = back[ti][state]
        assigned[ti] = pos
        state = prev
    return [order[p] if p is not None else None for p in assigned]


def relevel_volume(
    slices: Sequence[Sequence[tuple[int, float, float, float]]],
    cfg: RelevelConfig,
) -> tuple[list[list[int]], dict[str, int]]:
    """
    ``slices``: per slice, in volume order, a list of ``(cls, y_center, height, conf)`` (normalized y/height).
    Returns new class IDs in the same layout, plus counters. Classes in no chain are left unchanged.
    """
    new_cls = [[int(b[0]) for b in boxes] for boxes in slices]
    stats = {"relevel_tracks": 0, "relevel_merges": 0, "relevel_relabel": 0, "relevel_unassigned_tracks": 0,
             "relevel_duplicates_skipped": 0}
    for chain in cfg.chains:
        # One structure, one box: on a slice, a box of the same tissue whose vertical extent overlaps a more
        # confident box (1-D IoU >= dup_y_iou) is a second detection of that structure, whatever its class. Only
        # the most confident box is tracked; the duplicate takes the same level afterwards, so det_limit can drop
        # it instead of it forming an extra track that claims a level.
        chain_boxes, duplicate_of = [], {}
        for s, boxes in enumerate(slices):
            kept: list[_Box] = []
            for i in sorted((i for i, b in enumerate(boxes) if int(b[0]) in chain.tissue_of), key=lambda i: -boxes[i][3]):
                cls, y, h, conf = int(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
                same = next((k for k in kept if chain.tissue_of[k.cls] == chain.tissue_of[cls]
                             and _y_iou(k.y, k.h, y, h) >= cfg.dup_y_iou), None)
                if same is None:
                    kept.append(_Box(s, i, cls, y, h, conf))
                else:
                    duplicate_of[(s, i)] = (s, same.index)
            chain_boxes += kept
        stats["relevel_duplicates_skipped"] += len(duplicate_of)
        if not chain_boxes:
            continue
        tracks: list[_Track] = []
        for tissue in sorted(set(chain.tissue_of.values())):
            linked = _link_tracks([b for b in chain_boxes if chain.tissue_of[b.cls] == tissue], cfg)
            merged, n_merged = _merge_tracks(linked, cfg)
            tracks.extend(merged)
            stats["relevel_merges"] += n_merged
        tracks.sort(key=lambda t: t.median_y)
        stats["relevel_tracks"] += len(tracks)
        for track, level in zip(tracks, _assign_levels(tracks, chain, cfg)):
            if level is None:
                stats["relevel_unassigned_tracks"] += 1
                continue
            for b in track.boxes:
                if new_cls[b.slice_pos][b.index] != level:
                    stats["relevel_relabel"] += 1
                new_cls[b.slice_pos][b.index] = level
        for (s, i), (_, primary) in duplicate_of.items():
            if new_cls[s][i] != new_cls[s][primary]:
                stats["relevel_relabel"] += 1
            new_cls[s][i] = new_cls[s][primary]
    return new_cls, stats


def relevel_results(results: Sequence[Any], cfg: RelevelConfig) -> dict[str, int]:
    """Relevel one volume of Ultralytics ``Results`` (slice order) in place; returns counters."""
    slices, datas = [], []
    for r in results:
        data = r.boxes.data.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 6))
        oh = float(r.orig_shape[0])
        slices.append([(int(d[5]), (d[1] + d[3]) / 2 / oh, (d[3] - d[1]) / oh, float(d[4])) for d in data])
        datas.append(data)
    new_cls, stats = relevel_volume(slices, cfg)
    for r, data, cls in zip(results, datas, new_cls):
        if len(data) == 0:
            continue
        updated = data.copy()
        updated[:, 5] = cls
        r.update(boxes=r.boxes.data.new_tensor(updated))
    return stats
