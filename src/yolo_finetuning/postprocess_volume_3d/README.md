# `postprocess_volume_3d`

Per-volume refinement of the 2D YOLO detections of one scan, in slice order. Training does not use this module.
`run_inference.py yolo` / `both` run it inside YOLO predict, for each volume:

```text
raw YOLO boxes -> relevel (optional) -> gap fill (optional) -> det_limit and filters when writing label_pred/
```

The inference log names each step and whether it ran, e.g. `Relevel: ON  Gap fill: OFF`.

| Released model | relevel | gap fill |
|---|---|---|
| sagittal (`config/datasets/example_sagittal_vertdisc.yaml`) | on (vertebrae + discs) | off |
| axial (`config/datasets/example_axial_muscle.yaml`) | off | on, `max_gap_slices: 4` |

## Relevel (`relevel.py`, `inference.models.obj_det.relevel`)

One label per physical level through the volume. Boxes of one tissue are linked across neighbouring slices into
tracks (one physical level each), tracks are ordered cranial to caudal and matched to the chain's levels by a
monotone assignment that maximizes the confidence-weighted YOLO votes, and every box takes its track's level.
Keys and their units are documented in the example sagittal config and in the module docstring.

## Gap fill (`pipeline.py`, `inference.models.obj_det.postprocess`)

For each class, when two slices hold it with at most `max_gap_slices` empty slices between them, an anchor box from
those two slices is copied into every empty slice (confidence 0.05). `det_limit` is applied before and after, so filled
boxes obey the per-slice caps.

```yaml
postprocess:
  enabled: true
  type: axial_spine_muscle        # sagittal_spine | axial_spine_muscle
  gap_fill:
    enabled: true
    max_gap_slices: 4
    anchor: largest_area          # largest_area (in-plane area) | highest_conf
```

Gap fill runs only when both `postprocess.enabled` and `gap_fill.enabled` are true. Any other key in the block is an
error, so a misspelled key does not silently change nothing.
