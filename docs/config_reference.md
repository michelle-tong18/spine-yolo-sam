# Dataset YAML config reference

Configs live in `config/datasets/<name>.yaml`. Copy `example_axial_muscle.yaml` or `example_sagittal_vertdisc.yaml` and edit paths.

One file drives preprocess, train, infer, and eval. Interpolation such as `${general_data_dir}/nifti_preprocessed/img` is allowed. Inference still requires the resolved `volume_dir` and `output_data_dir` — it does not default those from `general_data_dir`.

## Keys by step

| Step | Command | Keys you set |
|------|---------|--------------|
| Build | `full_standardization.py`, `dataset_split.py` | `data_dir`, `image_column`, `mask_column`, `general_data_dir`, `preprocessing_cfg`, `mask_labels`, `split_id` |
| Train | `train_yolo.py`, `finetune_main.py` | `YOLO`, `SAM` |
| Infer | `run_inference.py {yolo\|sam\|both}` | `output_data_dir`, `inference.volume_dir`, `inference.task_name`, `inference.models` |
| Eval | `run_evaluation.py --run <run_dir>` | `evaluation.gt_labels_dir`, `evaluation.gt_masks_dir`, optional `evaluation.tissue_groups` |
| Measure | `run_measurements.py --run-dir <run_dir> --metrics discs,volumes` | optional `measurements` (discs, sagittal label map), `evaluation.tissue_groups` (volumes) |

`run_pipeline.py` is an orchestrator for build / train / optional `both` inference. It uses the same YAML.

## Top-level keys

| Key | Used by | Description |
|-----|---------|-------------|
| `general_data_dir` | Preprocess, train, interpolation | Working root. Preprocessing writes under this path. Optional for infer except as a string to interpolate. |
| `work_dir` | YOLO/SAM train | Training output root (relative to repo cwd). Example configs set `YOLO.yolo_output_dir: ${work_dir}/yolo` and `SAM.module.work_dir: ${work_dir}/sam`. |
| `output_data_dir` | Infer | Required predictions root. Keep this off the raw-data tree. |
| `data_dir` | Preprocess only | CSV of image and mask paths. Not an inference input. |
| `image_column`, `mask_column` | Preprocess | CSV column names. |
| `mask_labels` | All steps | Integer id → class name (`0` is background). |
| `preprocessing_cfg` | Preprocess, SAM train, infer | Image size, crop, `make_square`, `yolo_relabel`, etc. Infer uses the same `ImagePrep` for volumes. `image_only: True` (or an empty `mask_column`) preprocesses scans without labels. `pixel_spacing_mm: [slice, row, col]` is used only for scans whose header has no spacing (see [Voxel spacing](#voxel-spacing)). |
| `image_config`, `mask_config` | Preprocess, infer | Transforms and expected data properties. |
| `nifti_fig_cfg` | Preprocess | `fig_type: 2D_overlay` writes QC figures for scans with a mask: image + mask overlay (`nifti_preprocessed/fig/`) and YOLO boxes (`slice_preprocessed/SAM/figs/`). Empty to skip. |
| `split_id` | Split, train, eval | Key in `config/splits/splits_registry.yaml` (or in `splits_registry`). |
| `splits_registry` | Split, train, eval | Optional path to another splits registry YAML (same layout). |
| `YOLO` | YOLO train, split | Detector, `data_yaml`, training hyperparameters. Set `yolo_parameters.project` and `yolo_parameters.name` here (not in `YOLO_default_cfg`); relative `project` is saved under `yolo_output_dir`. |
| `SAM` | SAM train | Experiment, datamodule, module. Runs go under `module.work_dir` / `module.task_name`. |
| `inference` | Infer | Volumes, task name, weights, detection settings (relevel, gap fill). |
| `evaluation` | Eval, `sam --boxes gt` | GT label and mask dirs. Required to score; required for GT-box SAM. `tissue_groups` merges labels into tissues for the breakdown. |
| `measurements` | Disc measurements | `mid_slices` (default 3) and `levels` (default: every disc in `mask_labels`). |

## Train outputs

Training writes next to the repo (not under `general_data_dir`).

| Trainer | Config | Layout |
|---------|--------|--------|
| YOLO | `YOLO.yolo_output_dir`, `YOLO.yolo_parameters.project`, `YOLO.yolo_parameters.name` | `{yolo_output_dir}/<project>/<name>/` (example configs: `work_dir/yolo/.../runs`) |
| SAM | `SAM.module.work_dir`, `SAM.module.task_name` | `{SAM.module.work_dir}/<sam_project>/` then optional `group_name` and `Run_N` (example configs: `work_dir/sam/...`) |

Set `work_dir` at the top of the dataset YAML, then point `YOLO.yolo_output_dir` and `SAM.module.work_dir` at `${work_dir}/yolo` and `${work_dir}/sam`. `YOLO_WORK_DIR` can override the YOLO root if `yolo_output_dir` is unset. Missing keys raise an error rather than falling back to a hardcoded path. Point `inference.models.*.model_path` at the resulting checkpoint folders after training.

## Inference I/O

Public CLI:

```bash
python src/inference/run_inference.py {yolo|sam|both} --config <name>
```

Required (YAML or CLI override):

| Key | CLI | Meaning |
|-----|-----|---------|
| `inference.volume_dir` | `--volumes` | Folder of 3D volumes to label. |
| `output_data_dir` | `--output` | Predictions root. |
| `inference.task_name` | (YAML) | Name under that root. Example configs use `${SAM.experiment.name}`. |

Layout:

```text
{output_data_dir}/inference/{task_name}/Run_N/
  label_pred/    # YOLO .txt, stems {volume}-{slice:03d} (slice = index in the scan, as GT slice labels)
  seg_pred/      # SAM 3D NIfTI
```

`Run_N` increments. `--run-num N` or `--run-dir PATH` reuse a run (for example `yolo` then `sam --boxes pred`).

### Inputs by subcommand

| Command | Reads | Writes |
|---------|-------|--------|
| `yolo` | `volume_dir` (or `--slices` / `inference.slice_img_dir` if PNGs already exist) | `{run}/label_pred` |
| `sam --boxes pred` | `volume_dir` plus `{run}/label_pred` or `--pred-labels-dir` | `{run}/seg_pred` |
| `sam --boxes gt` | `volume_dir` plus `evaluation.gt_masks_dir` | `{run}/seg_pred` |
| `both` | `volume_dir` only | `{run}/label_pred` then `{run}/seg_pred` |

`yolo` / `both` extract slices with the same `ImagePrep` as SAM, write `{volume}-{slice:03d}.txt`, and discard temp PNGs. `both` does not take a slice folder. The slice number is the slice's index in the scan (after the YAML transforms), from `kept_slices` in `nifti_preprocessed/spacing.csv`: the same name preprocessing gives the GT slice labels, so predictions and GT match by file name even when a volume was cropped to its labelled slices. Without a recorded row (older preprocessing, or volumes that were not preprocessed) slices are numbered by position. SAM pairs label files with volume slices in slice order and names its QC / raw outputs with the same numbers; `seg_pred/` volumes are on the preprocessed grid, slice for slice like the GT masks. Evaluation warns about volumes whose predictions share no file name with their GT slices.

Optional: `inference.valid_list` (CSV column `id`) filters volume stems. Weights stay under `inference.models.*`.

### CLI flags

| Flag | Overrides |
|------|-----------|
| `--volumes` | `inference.volume_dir` |
| `--output` | `output_data_dir` |
| `--pred-labels-dir` | Predicted YOLO `.txt` dir for `sam --boxes pred` |
| `--slices` | Existing PNG folder (`yolo` only) |
| `--boxes pred\|gt` | SAM box source (`sam` only; `both` is always pred) |
| `--run-dir` / `--run-num` | Reuse `Run_N` |
| `--figures` | After SAM, `{run}/figures/<volume>.png`: the image, YOLO boxes and masks (three rows) on 5 evenly spaced slices (`sam` / `both`) |

### Detection / combine knobs

Under `inference.models.obj_det`:

| Key | Meaning |
|-----|---------|
| `conf` | Ultralytics predict confidence threshold. |
| `iou` | Ultralytics NMS IoU. |
| `remove_label_ids` | Drop these class IDs (never written, never passed to SAM). |
| `ignore_label_ids` | Run SAM but do not write the mask into the combined volume. |
| `subtract_label_ids` | Zero that SAM region in the combined mask. |
| `priority_label_ids` | Overwrite overlapping labels with this class. |
| `det_limit` | Max detections per class per slice, e.g. `{1: 2, 2: 2}`. |
| `relevel` | Optional per-volume level assignment (see below). The released sagittal model uses it. |
| `postprocess` | Optional gap fill (see below). The released axial model uses it. |
| `max_det` | Ultralytics max detections per slice before the filters (default 50). |

YOLO and SAM both use `inference.device` (do not set a separate detector device).

Each volume's boxes go through relevel, then gap fill (each when enabled), then `det_limit` and the other filters when the label files are written; the log names each step, e.g. `Relevel: ON  Gap fill: OFF` ([`src/yolo_finetuning/postprocess_volume_3d/README.md`](../src/yolo_finetuning/postprocess_volume_3d/README.md)).

Optional gap fill under `inference.models.obj_det.postprocess` (any other key is an error):

| Key | Meaning |
|-----|---------|
| `enabled` | Turn the block on. |
| `type` | `sagittal_spine` or `axial_spine_muscle`. Required when `enabled` is true. |
| `gap_fill.enabled` | Copy an anchor box into the empty slices between two slices that hold the same class. |
| `gap_fill.max_gap_slices` | Largest gap (empty slices) filled. Released axial model: 4. |
| `gap_fill.anchor` | `largest_area` (default) or `highest_conf`: which box of the two boundary slices is copied. |

Optional level assignment under `inference.models.obj_det.relevel` ([`relevel.py`](../src/yolo_finetuning/postprocess_volume_3d/relevel.py)). Per volume it links each tissue's boxes across slices into tracks (one physical level each), orders the tracks cranial → caudal and gives every box in a track one label, so a level is not called L4 on some slices and L5 on others. It runs on the raw YOLO boxes, before gap fill and `det_limit`.

| Key | Meaning |
|-----|---------|
| `enabled` | Run relevel after YOLO predict. |
| `chains` | Per chain: `order` (cranial → caudal class IDs), `tissues` (tissue → class IDs; tracks never mix tissues), `anchor_class_ids` (levels whose votes count `anchor_weight` x). |
| `anchor_weight`, `max_link_gap`, `link_y_tol`, `merge_y_tol`, `skip_penalty`, `unassigned_penalty`, `dup_y_iou`, `max_column_shift` | Tracking and assignment settings; defaults and units in the example sagittal config. |

## Evaluation

```bash
python src/evaluation/run_evaluation.py <name> --run {run_dir}
```

Reads `{run}/label_pred/` and `{run}/seg_pred/`. Point at those folders directly with `--detection_labels_dir` and `--segmentation_pred_dir`. `evaluation.gt_labels_dir` and `evaluation.gt_masks_dir` must be set to score (interpolation OK). Test volumes are the `test` subjects of `split_id`; GT files are `{subject}` or `{subject}_{series}` (with `include_subject_accession`).

Eval writes `detection_metrics_breakdown.csv` and `segmentation_metrics_breakdown.csv` to `--output_dir` (default `{run}/evaluation`): one row per scope (`overall`, `patient`, `series`) × level (`label`, `tissue`, `all_tissues`) × class. Segmentation rows carry Dice, IoU and HD95 in mm (`hd95`: mean of per-slice in-plane 95th percentiles; `hd95_pooled`; `hd95_undefined_frac`: share of slices where only GT or only the prediction has the class). HD95 needs the spacing preprocessing recorded for the GT volumes and is nan without it.

## Voxel spacing

Preprocessing records each volume's voxel spacing ([`spacing.py`](../src/preprocessing/spacing.py)) in two places: the NIfTI affine of `nifti_preprocessed/{img,mask}/<volume>.nii.gz`, and `nifti_preprocessed/spacing.csv` (source scan, `spacing_source`, shape and spacing after the YAML transforms, kept slices, `slice_mm` / `row_mm` / `col_mm` of the preprocessed grid, and `world_affine`). Inference copies the input affine onto `seg_pred/`.

`world_affine` maps a preprocessed voxel (slice, row, col) to patient coordinates in mm (LPS), from DICOM ImagePositionPatient / ImageOrientationPatient or the NIfTI / MetaImage header, followed through the YAML transforms, the slice crop and the pad / resize. The NIfTI affine is its RAS form, so viewers place a patient's sagittal and axial volumes together; without it the affine is diagonal (spacing only). It is left empty when the header has no position / orientation, the transforms include `rot90` / `drop_dim`, or the kept slices are not evenly spaced (image-only preprocessing keeps every slice).

Spacing comes from the scan header: DICOM PixelSpacing (else ImagerPixelSpacing) and the ImagePositionPatient step (else SpacingBetweenSlices, else SliceThickness); NIfTI zooms; MetaImage ElementSpacing. It follows the YAML axis transforms, the slice crop, padding and resize. A header without spacing is never read as 1 mm: `preprocessing_cfg.pixel_spacing_mm` is used if set, otherwise spacing is nan (`spacing_source: missing`), HD95 is nan, and disc measurements report only their scale-free columns.

## Measurements

Measurements read a finished inference run (no ground truth); each volume is loaded once for every requested metric.

```bash
python src/measurements/run_measurements.py <name> --run-dir {run_dir} --metrics discs,volumes [--volumes DIR]
```

`volumes` writes `{run}/label_volumes.csv`: voxels and cm³ per label and per `evaluation.tissue_groups` entry, from the recorded voxel spacing (nan without it). It assumes adjacent slices, as image-only preprocessing keeps.

### Axial volume per vertebral level

```bash
python src/measurements/run_level_volumes.py --pairs pairs.csv \
    --sagittal-config <sagittal> --sagittal-run {sagittal_run} --axial-config <axial> --axial-run {axial_run}
```

`pairs.csv` lists `patient_id, sagittal_volume, axial_volume`. Each disc in the sagittal segmentation defines a mid-plane (through its centroid, normal to its thinnest axis); level V is the space between the mid-planes of the discs above and below (L4: L3-L4 to L4-L5). Every axial muscle voxel is assigned to the level enclosing it, in patient coordinates from both volumes' `world_affine`, so the two scans must share one patient position (same session). Writes `{axial_run}/label_volumes_per_level.csv`: per pair, level and label / `evaluation.tissue_groups` entry, voxels, cm³, `n_axial_slices` and `coverage_frac` (share of the level's height the axial slices cover). Definitions: [`levels.py`](../src/measurements/levels.py).

`discs`, for each segmented volume of a sagittal run, writes one row per disc level to `{run}/disc_measurements.csv`: disc heights (anterior / mid / posterior thirds) and vertebral heights in mm, disc height indices, disc / canal signal ratio and posterior high-signal fraction (T2), posterior displacement and area beyond the line between the adjacent vertebral corners. Each is the median over `measurements.mid_slices` slices centred on the largest spinal canal area. Needs `spinal_canal` and discs named `<upper>-<lower>` in `mask_labels`, and the preprocessed volumes inference ran on (`inference.volume_dir`, or `--volumes`). Definitions: [`disc_measurements.py`](../src/measurements/disc_measurements.py).







