# spine-yolo-sam

A detection-then-segmentation pipeline for spine MRI that combines YOLO for localization with the Segment Anything Model (SAM) for segmentation. YOLO predicts bounding boxes for each label on individual MRI slices; postprocessing promotes anatomical consistency across the 3D volume; the detections prompt SAM in resulting 3D segmentation mask for evaluation and quantitative measurements.

The repository supports the full workflow (from preprocessing and model training to inference, evaluation, and anatomical measurements) with one dataset YAML configuration used across steps.

```text
CSV of image/mask paths
    -> preprocess (volumes + YOLO/SAM slices; voxel spacing recorded)
    -> train YOLO and/or SAM
    -> infer (boxes, one label per level, then masks)
    -> evaluate (detection, Dice / IoU / HD95)
    -> metrics (tissue volumes, disc measurements, muscle tissue volumes by spinal level)
```

One dataset YAML drives every step. Inference is one command with subcommands (`yolo`, `sam`, `both`). Metrics are a separate command that reads files from disk.

To label your own scans with the released models, skip training: see [Quickstart](#quickstart-label-new-scans).

## Requirements

- Python 3.10+ (`requires-python` in [`pyproject.toml`](pyproject.toml))
- A CUDA GPU is recommended; example configs set `device: cuda`
- Python packages: [`requirements.txt`](requirements.txt) (Conda wrapper: [`environment.yml`](environment.yml))

Core stack: PyTorch, torchvision, Ultralytics YOLO, Segment Anything, MONAI, plus imaging/IO (`nibabel`, `pydicom`, SimpleITK, OpenCV, `albumentations`). W&B is used by SAM training and YOLO sweeps.

## Install

**Conda** (from the repo root):

```bash
git clone git@github.com:michelle-tong18/spine-yolo-sam.git
cd spine-yolo-sam
conda env create -f environment.yml
conda activate spine-yolo-sam
```

**venv + pip:**

```bash
git clone git@github.com:michelle-tong18/spine-yolo-sam.git
cd spine-yolo-sam
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pip install torch` can land a CPU-only wheel. If `python -c "import torch; print(torch.cuda.is_available())"` is `False` and you have a GPU, install a CUDA build of `torch` and `torchvision` from [pytorch.org](https://pytorch.org/get-started/locally/) for your driver.

### Model weights

The released models (sagittal: vertebrae, discs, spinal canal, facets; axial: paraspinal muscles) are on Hugging Face at
[`mwtong/spine-yolo-sam`](https://huggingface.co/mwtong/spine-yolo-sam). The example configs read them from `weights/spine-yolo-sam/`:

```bash
pip install huggingface_hub
huggingface-cli download mwtong/spine-yolo-sam --local-dir weights/spine-yolo-sam
# -> weights/spine-yolo-sam/{yolo_sagittal_v1.pt, sam_sagittal_v1.pth, yolo_axial_muscle_v1.pt, sam_axial_muscle_v1.pth}
```

SAM inference and training also need the base [Segment Anything](https://github.com/facebookresearch/segment-anything) ViT-B checkpoint:

```bash
mkdir -p weights/SAM
wget -P weights/SAM https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Training YOLO from scratch starts from the Ultralytics `yolo26x.pt` checkpoint ([Ultralytics](https://docs.ultralytics.com/models/)), saved as `weights/YOLO/yolo26x.pt` (`YOLO.model`).

The models segment lumbar spine MRI: sagittal T1- or T2-weighted series (sagittal model) and axial T1- or T2-weighted series (axial model). A CUDA GPU is recommended.

## Data CSV

Preprocessing reads a CSV whose path is `data_dir` in the dataset config. Required columns:

| Column | Meaning |
|--------|---------|
| `subject_id` | Patient / case id |
| image column | Path to a volume (NIfTI) or DICOM directory. Name this in `image_column`. |
| mask column | Path to a label volume. Name this in `mask_column`. |
| `accession` | Optional visit id. Set `include_subject_accession: True` if one subject has multiple scans. |

Volume filenames after preprocessing are `{subject_id}` (or `{subject_id}_{accession}`). For training and evaluation, list the subject ids of each split under `split_id` in [`config/splits/splits_registry.yaml`](config/splits/splits_registry.yaml). For inference only, leave `mask_column` empty: scans without a mask are preprocessed image-only. [`config/datasets/example_scans.csv`](config/datasets/example_scans.csv) is a template for scans to label (one row per series, DICOM folders or NIfTI files).

Preprocessing records each volume's voxel spacing from its header (DICOM, NIfTI, MetaImage) in the output NIfTI affine and `nifti_preprocessed/spacing.csv`; HD95 and disc measurements use it. For scans whose header has no spacing, set `preprocessing_cfg.pixel_spacing_mm`; otherwise those mm values are left empty. Details: [docs/config_reference.md](docs/config_reference.md#voxel-spacing).

## Configure

Copy an example and edit paths:

- [`config/datasets/example_axial_muscle.yaml`](config/datasets/example_axial_muscle.yaml) — paraspinal muscles
- [`config/datasets/example_sagittal_vertdisc.yaml`](config/datasets/example_sagittal_vertdisc.yaml) — vertebrae, discs, spinal canal and facets (one label per level through the volume with `relevel`; optional disc measurements)

Both carry the label maps and detection settings of the released models.

Set at least:

- `general_data_dir` — working root for preprocess/train data (also used to interpolate other paths)
- `work_dir` — training output root (example configs use `work_dir/yolo` and `work_dir/sam` under this)
- `output_data_dir` — where inference writes (keep this off the raw-data tree)
- `data_dir` / `image_column` / `mask_column`
- `mask_labels`
- `split_id` (and the matching lists in `splits_registry.yaml`)
- `inference.volume_dir` and `inference.task_name`
- `inference.models.obj_det.model_path` and `inference.models.segmentation.model_path`

YOLO training writes to `{work_dir}/yolo/<YOLO.yolo_parameters.project>/` (`YOLO.yolo_output_dir`). SAM training writes to `{work_dir}/sam/<SAM.module.task_name>/` (`SAM.module.work_dir`; example configs then add `<group_name>/Run_N`).

For a new unlabeled cohort, point `inference.volume_dir` and `output_data_dir` at that data. Inference does not guess those paths from `general_data_dir`.

Full key map: [docs/config_reference.md](docs/config_reference.md).

## Try it on the sample scans

[`samples/data/`](samples/data/) holds de-identified lumbar MRIs of one patient, a T1 and a T2 per model:
`sample_{sagittal,axial}_{t1,t2}.nii.gz`. After [installing](#install) and downloading the [model weights](#model-weights):

```bash
python samples/run_samples.py            # all four samples; --planes sagittal or --planes axial for one plane
python samples/run_samples.py --no-levels   # skip the axial muscle volume per vertebral level
```

It preprocesses, segments and measures the scans of each plane together with that plane's example config, on the
GPU when there is one and otherwise on the CPU (slower). Everything it writes is under `samples/outputs/` (per plane:
preprocessed volumes, the generated config, and `run/`). The samples are one patient from one session, so it also
computes the axial muscle volume per vertebral level (T1 with T1, T2 with T2), between the mid-planes of the discs
found on the sagittal scan (`samples/outputs/axial/run/label_volumes_per_level.csv`; see [Measurements](#measurements)).
Open `samples/outputs/<plane>/run/figures/sample_<plane>_<contrast>.png` first: the image, the YOLO boxes and the masks
on 5 evenly spaced slices. The masks are in `run/seg_pred/` and the tables in `run/*.csv` (cm³ per label; per-disc
measurements for sagittal).

## Quickstart: label new scans

1. Download the released weights and the SAM base checkpoint (see [Model weights](#model-weights)).
2. List your scans in a CSV like [`config/datasets/example_scans.csv`](config/datasets/example_scans.csv) (sagittal series for the sagittal model, axial series for the axial model).
3. Copy `config/datasets/example_sagittal_vertdisc.yaml` (or the axial one) and set `general_data_dir` (a working folder), `data_dir` (your CSV), `image_column: image`, `mask_column: ""` and `output_data_dir`. The weight paths already point at `weights/spine-yolo-sam/`; keep `inference.volume_dir: ${general_data_dir}/nifti_preprocessed/img`.
4. Preprocess, detect and segment, then measure (`discs` for per-disc measurements on sagittal scans, `volumes` for cm³ per label):

```bash
cd scripts
cp train_yolo.env.example my_vars.env   # edit CONDA_ENV_NAME
bash sample_quickstart.sh my_sagittal_config /path/to/runs/new_scans discs,volumes
```

or without the script:

```bash
python src/preprocessing/full_standardization.py my_sagittal_config
python src/inference/run_inference.py both --config my_sagittal_config --run-dir /path/to/runs/new_scans --figures
python src/measurements/run_measurements.py my_sagittal_config --run-dir /path/to/runs/new_scans --metrics discs,volumes
```

Masks land in `<run_dir>/seg_pred/`, with `disc_measurements.csv` and `label_volumes.csv` next to them. `--figures` (the quickstart script sets it) saves `<run_dir>/figures/<volume>.png` per scan: the image, the YOLO boxes and the masks for every label (three rows) on 5 evenly spaced slices, to check the result at a glance.

## Run

Copy an example YAML, edit paths, and use that name (no `.yaml`) in every command below.

The `sample_*.sh` scripts run from `scripts/` and source `scripts/my_vars.env` (conda env, optional W&B keys):

```bash
cd scripts
cp train_yolo.env.example my_vars.env
# edit CONDA_ENV_NAME; add WANDB_* only if you use W&B
```

### Shell scripts

`#SBATCH` lines are ignored by `bash`; edit them for your site and submit with `sbatch` on a cluster.

```bash
cd scripts

# 1. Preprocess volumes and write train/val/test splits
bash sample_build_dataset.sh example_axial_muscle

# 2. Train (GPU). Afterward, point inference.models.*.model_path in the YAML at the run folders.
bash sample_train_yolo.sh example_axial_muscle
bash sample_train_sam.sh example_axial_muscle

# 3. Detect then segment (writes output_data_dir/inference/<task>/Run_N/)
bash sample_infer.sh example_axial_muscle both

# 4. Score a run (needs evaluation.gt_*_dir in the YAML); writes <run>/evaluation/
cd .. && bash scripts/evaluate.sh example_axial_muscle \
  /path/to/output_data_dir/inference/example_axial_muscle/Run_1
```

W&B sweep (needs a `sweep_cfg` block in the YAML): `bash sample_train_yolo_sweep.sh example_axial_muscle`.

To share a trained SAM model, keep only its weights (no optimizer state): `python src/sam_finetuning/export_model_weights.py <run>/<run_id>_finetuned_model_best.pth weights/my_sam.pth`. Ultralytics' `best.pt` is already stripped at the end of training.

### Same steps in Python

```bash
python src/preprocessing/full_standardization.py example_axial_muscle
python src/preprocessing/dataset_split.py example_axial_muscle
python src/yolo_finetuning/train_yolo.py example_axial_muscle
python src/sam_finetuning/finetune_main.py example_axial_muscle
python src/run_pipeline.py example_axial_muscle --build-dataset --finetune both
```

W&B hyperparameter sweeps: `python src/yolo_finetuning/train_yolo_sweep.py <config>` (requires a `sweep_cfg` block).

### Inference

One entry: `src/inference/run_inference.py`. Subcommands change behavior; they are not separate models.

```bash
# Usual path: detect on volumes, then segment from those boxes (same Run_N)
python src/inference/run_inference.py both --config example_axial_muscle

# Boxes only
python src/inference/run_inference.py yolo --config example_axial_muscle

# Masks from predicted boxes (reuse the yolo run)
python src/inference/run_inference.py sam --config example_axial_muscle --boxes pred --run-num 1

# Masks from ground-truth boxes
python src/inference/run_inference.py sam --config example_axial_muscle --boxes gt
```

With `inference.models.obj_det.relevel.enabled` (released sagittal model), each volume's YOLO boxes are relevelled (one label per physical level, cranial to caudal) before the optional gap fill (`postprocess.gap_fill`, released axial model) and `det_limit`; the log reports each, e.g. `Relevel: ON  Gap fill: OFF`.

CLI can override YAML: `--volumes`, `--output`, `--pred-labels-dir`. `yolo` can take existing PNGs with `--slices`. `both` always reads volumes. `--figures` (with `sam` or `both`) saves a verification figure per volume to `{run}/figures/`: the image alone, YOLO boxes and masks (one row each) on evenly spaced slices, one color per label (the `rainbow` scheme of `src/utils/visualizers.py`).

Writes:

```text
{output_data_dir}/inference/{task_name}/Run_N/
  label_pred/    # YOLO .txt  ({volume}-{slice:03d}.txt, slice = index in the scan, as GT slice labels)
  seg_pred/      # SAM 3D NIfTI (affine of the preprocessed input volume)
  figures/       # with --figures: boxes + masks on evenly spaced slices, per volume
```

`--run-num` / `--run-dir` reuse a run instead of creating `Run_{N+1}`.

### Measurements

Measurements read a finished run, without ground truth, so they can be re-run without re-running the models:

```bash
python src/measurements/run_measurements.py example_sagittal_vertdisc --run-dir /path/to/Run_1 --metrics discs,volumes
```

`discs` (sagittal) writes `disc_measurements.csv`: per disc level, heights and height indices, disc / canal signal ratio, posterior displacement and area beyond the posterior vertebral line. `volumes` writes `label_volumes.csv`: voxels and cm³ per label and per `evaluation.tissue_groups` entry. Both use the voxel spacing recorded at preprocessing.

For patients with a sagittal and an axial scan from the same session, axial muscle volume per vertebral level (between the mid-planes of the discs above and below):

```bash
python src/measurements/run_level_volumes.py --pairs pairs.csv \
    --sagittal-config example_sagittal_vertdisc --sagittal-run /path/to/sagittal_run \
    --axial-config example_axial_muscle --axial-run /path/to/axial_run
```

`pairs.csv` has columns `patient_id, sagittal_volume, axial_volume`; the result is `label_volumes_per_level.csv` in the axial run, with each level's axial coverage. It uses the scans' position and orientation recorded at preprocessing ([docs/config_reference.md](docs/config_reference.md#axial-volume-per-vertebral-level)).

### Evaluation

Infer writes files; eval only reads them. Set `evaluation.gt_labels_dir` and `evaluation.gt_masks_dir` in the YAML.

```bash
python src/evaluation/run_evaluation.py example_axial_muscle \
  --run /path/to/output_data_dir/inference/example_axial_muscle/Run_1
```

Folder overrides: `--detection_labels_dir` (`label_pred/`) and `--segmentation_pred_dir` (`seg_pred/`).

Eval writes `detection_metrics_breakdown.csv` and `segmentation_metrics_breakdown.csv` to `--output_dir` (default `<run>/evaluation`) and prints their overall rows: one row per scope (`overall`, `patient`, `series`) × level (`label`, `tissue`, `all_tissues`) × class. Tissue levels merge the labels listed in `evaluation.tissue_groups` (e.g. all discs) into one class; `all_tissues` merges every label. Segmentation rows carry Dice, IoU and HD95 in mm (in-plane per slice; needs the spacing recorded at preprocessing).

## Outputs

| Step | Location |
|------|----------|
| Preprocessed volumes | `$general_data_dir/nifti_preprocessed/{img,mask}`, voxel spacing in `nifti_preprocessed/spacing.csv` |
| YOLO slices | `$general_data_dir/slice_preprocessed/YOLO` |
| SAM slices | `$general_data_dir/slice_preprocessed/SAM` |
| YOLO training | `work_dir/yolo/<yolo_project>/` |
| SAM training | `work_dir/sam/<sam_project>/` |
| Predictions | `$output_data_dir/inference/$task_name/Run_N/{label_pred,seg_pred}` |
| Measurements | `<run>/disc_measurements.csv`, `<run>/label_volumes.csv`, `<axial run>/label_volumes_per_level.csv` |
| Metrics | `<run>/evaluation/` (`scripts/evaluate.sh`) |

## More documentation

| Doc | What it covers |
|-----|----------------|
| [docs/config_reference.md](docs/config_reference.md) | YAML keys and CLI flags |
| [docs/code_and_data_flow.md](docs/code_and_data_flow.md) | What each step reads and writes |
| [docs/pipeline_diagram.md](docs/pipeline_diagram.md) | Command flow and SAM combine rules |
| [src/yolo_finetuning/postprocess_volume_3d/README.md](src/yolo_finetuning/postprocess_volume_3d/README.md) | Relevel and gap fill (per-volume detection refinement) |
| [CHANGELOG.md](CHANGELOG.md) | Changes between releases |

## Tests

```bash
pip install pytest
python -m pytest
```

## Citation

Tong MW, et al. An automated detect-then-segment framework for scalable tissue segmentation and quantification of multi-site lumbar spine MRI. Sci Rep, (2026). doi:[10.1038/s41598-026-81955-3](https://doi.org/10.1038/s41598-026-81955-3)

## License

Copyright (c) 2026 spine-yolo-sam contributors

Code: MIT. Third-party models keep their own licenses: Segment Anything (Apache 2.0) and Ultralytics YOLO (AGPL-3.0). The released YOLO weights are fine-tuned from Ultralytics models and the SAM weights from Segment Anything; see the Hugging Face model card for their terms.
