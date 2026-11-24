"""Tests for src/yolo_finetuning/postprocess_volume_3d/relevel.py on synthetic sagittal volumes (pure numpy, no torch)."""
import numpy as np

from src.yolo_finetuning.postprocess_volume_3d.relevel import ChainConfig, RelevelConfig, relevel_results, relevel_volume

L3, L3L4, L4, L4L5, L5, L5S1, S1 = 5, 4, 3, 2, 1, 12, 13
CANAL = 20
SPINE = ChainConfig(
    "vertebrae_discs",
    order=[19, 18, 17, 16, 15, 14, 11, 10, 9, 8, 7, 6, L3, L3L4, L4, L4L5, L5, L5S1, S1],
    tissue_of={**{c: "vertebrae" for c in (19, 17, 15, 11, 9, 7, L3, L4, L5, S1)},
               **{c: "discs" for c in (18, 16, 14, 10, 8, 6, L3L4, L4L5, L5S1)}},
    anchors={L5, L5S1, S1},
)
FACETS = ChainConfig("facets", order=[29, 28, 27, 26, 25, 24, 23, 22, 21],
                     tissue_of={c: "facets" for c in range(21, 30)}, anchors={21})
CFG = RelevelConfig(chains=[SPINE, FACETS])
# Cranial → caudal: (class, y centre, height); vertebrae 0.07 tall, discs 0.025, ~0.05 apart.
TRUE_LEVELS = [(L3, 0.30, 0.07), (L3L4, 0.35, 0.025), (L4, 0.40, 0.07), (L4L5, 0.45, 0.025),
               (L5, 0.50, 0.07), (L5S1, 0.55, 0.025), (S1, 0.62, 0.08)]
SHIFT_UP = {L4: L3, L4L5: L3L4, L5: L4, L5S1: L4L5, S1: L5}  # YOLO one level too cranial


def make_volume(n_slices=12, mislabel=None, drop=(), extra=None, jitter=0.004, seed=0):
    """Per slice: [(cls, y, h, conf)]. mislabel: {slice: {true_cls: yolo_cls}}; drop: {(slice, true_cls)}."""
    rng = np.random.default_rng(seed)
    slices, truth = [], []
    for s in range(n_slices):
        boxes, labels = [], []
        for cls, y, h in TRUE_LEVELS:
            if (s, cls) in drop:
                continue
            yolo_cls = (mislabel or {}).get(s, {}).get(cls, cls)
            boxes.append((yolo_cls, y + rng.normal(0, jitter), h, 0.9))
            labels.append(cls)
        for cls, y, h in (extra or {}).get(s, []):
            boxes.append((cls, y, h, 0.8))
            labels.append(cls)
        slices.append(boxes)
        truth.append(labels)
    return slices, truth


def test_one_off_slices_are_corrected():
    slices, truth = make_volume(mislabel={s: SHIFT_UP for s in (3, 4, 5)})
    new, stats = relevel_volume(slices, CFG)
    assert new == truth
    assert stats["relevel_relabel"] == 3 * len(SHIFT_UP)


def test_box_missing_on_central_slice_keeps_track_and_neighbours():
    slices, truth = make_volume(drop={(6, L4)}, mislabel={7: SHIFT_UP})
    new, stats = relevel_volume(slices, CFG)
    assert new == truth
    assert stats["relevel_tracks"] == len(TRUE_LEVELS)  # L4 still one track across the gap
    assert stats["relevel_merges"] == 0  # bridged by gap linking, not by merging


def test_level_never_detected_does_not_shift_the_rest():
    slices, truth = make_volume(drop={(s, L4L5) for s in range(12)})
    new, _ = relevel_volume(slices, CFG)
    assert new == truth


def test_gap_split_track_merges_and_takes_one_level():
    # L4 missing on slices 4-8 (longer than max_link_gap), and YOLO calls the second half L3.
    drop = {(s, L4) for s in range(4, 9)}
    mislabel = {s: {L4: L3} for s in range(9, 12)}
    slices, truth = make_volume(drop=drop, mislabel=mislabel)
    new, stats = relevel_volume(slices, CFG)
    assert stats["relevel_merges"] >= 1
    assert new == truth


def test_adjacent_levels_on_disjoint_slices_are_not_merged():
    # L4 only on slices 0-3, L5 only on slices 8-11: same tissue, never on a shared slice, but a level apart.
    drop = {(s, L4) for s in range(4, 12)} | {(s, L5) for s in range(0, 8)}
    slices, truth = make_volume(drop=drop)
    new, stats = relevel_volume(slices, CFG)
    assert new == truth
    assert stats["relevel_merges"] == 0


def test_left_and_right_facets_merge_and_follow_the_majority():
    # Facets on lateral slices only (0-2 left, 9-11 right); one right-side slice is one level off.
    extra = {s: [(21, 0.53, 0.04), (22, 0.43, 0.04)] for s in (0, 1, 2, 9, 10, 11)}
    slices, truth = make_volume(extra=extra)
    slices[10] = [(22 if c == 21 else c, y, h, conf) for c, y, h, conf in slices[10]]
    new, stats = relevel_volume(slices, CFG)
    assert new == truth
    assert stats["relevel_merges"] == 2  # L5 and L4 facets: left + right each


def test_lumbosacral_anchor_breaks_a_tied_shift():
    # S1 / L5-S1 out of view; half the slices are right, half one level too cranial: a tie without anchors.
    L2, L2L3 = 7, 6
    drop = {(s, c) for s in range(12) for c in (L5S1, S1)}
    up = {L3: L2, L3L4: L2L3, L4: L3, L4L5: L3L4, L5: L4}
    slices, truth = make_volume(drop=drop, mislabel={s: up for s in range(6, 12)}, jitter=0.0)
    new, _ = relevel_volume(slices, CFG)
    assert new == truth


def test_every_track_is_explained():
    # The upper four levels are one too caudal on every slice; only L5 / L5-S1 / S1 are right. Shifting the
    # whole chain down would agree with more votes but leave the true L5-S1 and S1 tracks without a level.
    down = {L3: L4, L3L4: L4L5, L4: L5, L4L5: L5S1}
    slices, truth = make_volume(mislabel={s: down for s in range(12)})
    new, stats = relevel_volume(slices, CFG)
    assert new == truth
    assert stats["relevel_unassigned_tracks"] == 0


def test_short_false_positive_does_not_shift_the_chain():
    # One low-confidence vertebra box between L4 and L4-L5 on one slice: left alone, not given a level.
    slices, truth = make_volume(extra={5: [(L4, 0.425, 0.03)]})
    slices[5][-1] = (L4, 0.425, 0.03, 0.3)
    new, stats = relevel_volume(slices, CFG)
    assert new == truth
    assert stats["relevel_unassigned_tracks"] == 1


def test_second_box_on_the_same_structure_does_not_claim_a_level():
    # Raw YOLO (before det_limit) also boxes the L4 vertebra as "L3" (weaker, slightly offset) on every slice.
    # Tracked on its own it would form a second track at L4's height and shift the chain; as a duplicate it
    # takes L4's level, so det_limit can drop it.
    slices, truth = make_volume(extra={s: [(L3, 0.405, 0.068)] for s in range(12)})
    for s in range(12):
        slices[s][-1] = (L3, 0.405, 0.068, 0.5)
        truth[s][-1] = L4  # the duplicate follows the L4 box it sits on
    new, stats = relevel_volume(slices, CFG)
    assert new == truth
    assert stats["relevel_duplicates_skipped"] == 12


def test_column_shift_between_slices_does_not_mix_levels():
    # Lateral facets: the whole column moves down ~0.06 between slices 5 and 6 (about one facet height, more
    # than half the 0.1 level spacing). Linking on raw y would join L4 above the jump to L5 below it.
    levels = [(25, 0.33), (24, 0.43), (23, 0.53), (22, 0.63), (21, 0.73)]
    slices, truth = [], []
    for s in range(12):
        shift = 0.06 if s >= 6 else 0.0
        slices.append([(c, y + shift, 0.065, 0.8) for c, y in levels])
        truth.append([c for c, _ in levels])
    slices[2] = [(24 if c == 25 else c, y, h, conf) for c, y, h, conf in slices[2]]  # one-off on one slice
    new, stats = relevel_volume(slices, CFG)
    assert new == truth
    assert stats["relevel_tracks"] == len(levels)


def test_classes_outside_chains_are_untouched():
    extra = {s: [(CANAL, 0.45, 0.4)] for s in range(12)}
    slices, truth = make_volume(extra=extra, mislabel={2: SHIFT_UP})
    new, _ = relevel_volume(slices, CFG)
    assert new == truth


class _Tensor:
    """Minimal stand-in for a torch tensor (cpu / numpy / new_tensor)."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def new_tensor(self, arr):
        return _Tensor(arr)


class _Boxes:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.arr)


class _Result:
    def __init__(self, rows, size=1024):
        self.orig_shape = (size, size)
        self.boxes = _Boxes(_Tensor(rows))

    def update(self, boxes):
        self.boxes = _Boxes(boxes)


def test_relevel_results_updates_class_column_only():
    slices, truth = make_volume(mislabel={4: SHIFT_UP})
    size = 1024
    results = []
    for boxes in slices:
        rows = [[100, (y - h / 2) * size, 300, (y + h / 2) * size, conf, cls] for cls, y, h, conf in boxes]
        results.append(_Result(rows, size))
    before = [r.boxes.data.arr[:, :5].copy() for r in results]
    relevel_results(results, CFG)
    assert [r.boxes.data.arr[:, 5].astype(int).tolist() for r in results] == truth
    assert all(np.array_equal(b, r.boxes.data.arr[:, :5]) for b, r in zip(before, results))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
