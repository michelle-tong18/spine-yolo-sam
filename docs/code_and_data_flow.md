# Code and data flow

User commands: [README](../README.md). YAML keys: [config_reference.md](config_reference.md). Function-level diagrams: [pipeline_diagram.md](pipeline_diagram.md).

One dataset config name drives every step (`CLI` or `CONFIG_NAME`). Infer writes files; eval reads files. There is no live YOLO inside SAM.

## Disk layout

```text
{general_data_dir}/
  nifti_preprocessed/{img,mask}     # 3D volumes (preprocess); affine = grid spacing in mm
  nifti_preprocessed/spacing.csv    # voxel spacing per volume (src/preprocessing/spacing.py)
  slice_preprocessed/YOLO           # train/val/test PNGs + labels
  slice_preprocessed/SAM            # train/val/test NPY slices

{output_data_dir}/inference/{task_name}/Run_N/
  label_pred/                       # YOLO .txt  ({volume}-{slice:03d}.txt, slice = index in the scan)
  seg_pred/                         # SAM 3D NIfTI (input affine kept)
  figures/                          # optional: --figures (boxes + masks, evenly spaced slices)
  disc_measurements.csv             # optional: run_measurements.py --metrics discs (sagittal)
  label_volumes.csv                 # optional: run_measurements.py --metrics volumes
  label_volumes_per_level.csv       # optional, axial run: run_level_volumes.py (needs a sagittal run too)

work_dir/yolo/<yolo_project>/       # YOLO train (Ultralytics project; then name/, e.g. runs/)
work_dir/sam/<sam_project>/         # SAM train (then group_name/Run_N in example configs)
```

`general_data_dir` is the preprocess/train working root. `output_data_dir` is the predictions root. They may be the same tree (for example `${general_data_dir}/results`) but inference never invents either path.

## Who reads and writes

```text
CSV (data_dir)
    │
    ▼
full_standardization  ──writes──►  nifti_preprocessed/ (+ spacing.csv)  +  slice_preprocessed/{YOLO,SAM}
    │
    ▼
dataset_split         ──writes──►  YOLO data.yaml + train/val/test lists
    │
    ├── train_yolo      ──reads──►  slice_preprocessed/YOLO
    │                   ──writes─►  work_dir/yolo/<yolo_project>/
    │
    └── finetune_main   ──reads──►  slice_preprocessed/SAM  (same split_id)
                        ──writes─►  work_dir/sam/<sam_project>/
            │
            ▼
run_inference.py
    yolo | both  ──reads──►  inference.volume_dir
                 ──per volume─►  relevel → gap fill → det_limit  (relevel, gap fill optional)
                 ──writes─►  Run_N/label_pred
    sam  | both  ──reads──►  volume_dir  +  boxes (label_pred or GT masks)
                 ──writes─►  Run_N/seg_pred
            │
            ├──────────────────────────────────────────────┐
            ▼                                              ▼
run_evaluation.py --run Run_N  (needs GT)          run_measurements.py --run-dir Run_N  (no GT)
    ──reads──►  Run_N/label_pred  +  Run_N/seg_pred       ──reads──►  Run_N/seg_pred  +  volume_dir  +  spacing.csv
    ──reads──►  evaluation.gt_labels_dir               ──writes─►  Run_N/disc_measurements.csv,
                +  evaluation.gt_masks_dir                          Run_N/label_volumes.csv
                (+ spacing.csv for HD95)
```

Training, inference, evaluation and measurements are separate commands: inference writes segmentations only, and
evaluation (against GT) and measurements (no GT) each read a finished run, so either can be re-run without
re-running the models.

## Inference contract

Required: `inference.volume_dir`, `output_data_dir`, `inference.task_name`.

| Command | Input | Output |
|---------|-------|--------|
| `yolo` | volumes (or `--slices` PNGs) | `label_pred/` |
| `sam --boxes pred` | volumes + `label_pred/` | `seg_pred/` |
| `sam --boxes gt` | volumes + `evaluation.gt_masks_dir` | `seg_pred/` |
| `both` | volumes | `label_pred/` then `seg_pred/` (same `Run_N`) |

`both` is two disk steps (YOLO, then SAM from those txt files), not in-loop detection. For a new cohort, preprocess it (image-only when there are no labels), then point `volume_dir` at its `nifti_preprocessed/img` and set `output_data_dir`.
