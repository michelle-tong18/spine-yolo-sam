# Pipeline diagrams

User commands: [README](../README.md). Paths: [code_and_data_flow.md](code_and_data_flow.md). YAML: [config_reference.md](config_reference.md).


## User-facing commands

```
run_pipeline.py  [--build-dataset] [--finetune yolo|sam|both] [--inference]
        │
        ├── build     full_standardization → dataset_split
        ├── finetune  train_yolo and/or finetune_main
        └── infer     run_inference.py both
                      (yolo from volumes, then sam --boxes pred)

run_inference.py  yolo|sam|both  --config NAME
        │
        ├── yolo  →  {run}/label_pred
        ├── sam --boxes pred|gt  →  {run}/seg_pred
        └── both  →  label_pred then seg_pred (same Run_N)

run_evaluation.py  NAME  --run {run_dir}
        │
        └── reads label_pred/ and seg_pred/  (no model load)
```

Order when several `run_pipeline` flags are set: build → train → inference.

## Inference internals

Libraries under the CLI. Users do not run `yolo_predict.py` or `sam_from_boxes.py`.

```
run_inference.py
        │
        ├── resolve volume_dir, output_data_dir, task_name
        │         → {output_data_dir}/inference/{task_name}/Run_N
        │
        ├── yolo | both
        │         extract slices (same ImagePrep as SAM)
        │         YOLO predict  (+ optional per volume: relevel, then gap fill)
        │         write label_pred/   discard temp PNGs
        │
        └── sam | both
                  load boxes from label_pred  (--boxes pred)
                       or GT masks            (--boxes gt)
                  SAM per bbox, combine per slice
                  write seg_pred/
```

### SAM combine rules (per bbox)

After SAM produces a mask for one box, rules run **in this order**. The first match wins.

1. **`ignore_label_ids`** — do not add it to the combined volume.
2. **`subtract_label_ids`** — set that region to 0 in the combined mask.
3. **`priority_label_ids`** — overwrite overlapping voxels with this label.
4. **Otherwise** — write the label only where this box’s confidence is higher than the current `confidence_map`. Leave other pixels as they are.

## Preprocess and train (contributor)

**`full_standardization.py`**

```
data_standardization → process_data_paths
        ├── image_data_prep / mask_data_prep
        └── save_slice_data  →  slice_preprocessed/{YOLO,SAM}
```

**`dataset_split.py`** — `get_splits` / `get_yolo_path` / `get_sam_path` → train/val/test + YOLO `data.yaml`.

**YOLO train** — `train_yolo.py` → `model.train` → `work_dir/yolo/<yolo_project>/` (optional W&B sweep).

**SAM train** — `finetune_main.py` → dataloaders by split → `finetunedSAM` → Trainer → `work_dir/sam/<sam_project>/`.

**Gap fill** (optional, released axial model) — `inference.models.obj_det.postprocess.gap_fill`; see [`src/yolo_finetuning/postprocess_volume_3d/README.md`](../src/yolo_finetuning/postprocess_volume_3d/README.md).

**Relevel** (optional, released sagittal model) — `inference.models.obj_det.relevel`: one label per physical level through the volume, on the raw boxes before gap fill and `det_limit`; see [`relevel.py`](../src/yolo_finetuning/postprocess_volume_3d/relevel.py).
