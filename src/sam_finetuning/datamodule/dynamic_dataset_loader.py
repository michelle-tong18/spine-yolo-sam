"""SAM finetuning data: train / val / test slice datasets by the subjects of ``split_id`` (config/splits/splits_registry.yaml)."""

import os
import logging
import glob
import numpy as np
from typing import Dict

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import src.sam_finetuning.utils.gpu_setup as GPUSetup
from src.utils.file_management.config_loader import load_yaml
from src.sam_finetuning.datamodule.npy_dataset import SAMDataset

# Retrieve a logger for the module
logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def subject_slice_files(mask_dir: str, subject_id: str) -> list[str]:
    """A subject's mask slices: ``{subject_id}-{slice}.npy`` or, with accession / series, ``{subject_id}_*.npy``.
    Matching the separator keeps subject ``010`` from taking ``0101-99999``'s slices."""
    return sorted(glob.glob(os.path.join(mask_dir, f"{subject_id}-*")) + glob.glob(os.path.join(mask_dir, f"{subject_id}_*")))


def process_dataset_by_split(cfg):
    """SAMDataset per split from ``{slice_preprocessed/SAM}/{imgs,gts}``: the slices of the split's subjects
    (``subject_slice_files``). Augmentation: ``augmentation_pipeline.<split>`` (none when unset)."""
    rank = GPUSetup.get_rank()
    logger.info(f"Rank {rank}: Starting to process datasets")

    dataset_name = cfg.get('dataset', {}).get('name', '')
    augmentation = cfg.get('augmentation_pipeline') or {}
    instance_bbox = cfg.get('preprocessing_cfg').get('instance_bbox', False)
    remove_label_ids = cfg.get('preprocessing_cfg').get('remove_label_ids', [])
    bbox_shift = cfg.get('SAM').get('datamodule', {}).get("bbox_shift", 0)

    npy_base_dir = get_sam_path(cfg)
    npy_image_path = os.path.join(npy_base_dir, 'imgs')
    npy_mask_path = os.path.join(npy_base_dir, 'gts')
    splits = get_splits(cfg)

    datasets = []
    for split_name in SPLITS:
        logger.info(f"Rank {rank}: Processing {split_name} split")
        masks, imgs = [], []
        for pat_id in splits[split_name]:
            mask_f = subject_slice_files(npy_mask_path, pat_id)
            masks.extend(mask_f)
            imgs.extend(os.path.join(npy_image_path, os.path.split(f)[1]) for f in mask_f)
        logger.info(f"{split_name}: {len(imgs)} slices")
        datasets.append(SAMDataset(masks, imgs, bbox_shift, instance_bbox, remove_label_ids, dataset_name,
                                   augmentation_config=augmentation.get(split_name)))
    logger.info(f"Rank {rank}: Finished processing datasets")
    return tuple(datasets)


def create_dataloader(datasets, batch_size, num_workers):
    train_dataset, val_dataset, test_dataset = datasets

    num_tasks = GPUSetup.get_world_size()
    global_rank = GPUSetup.get_rank()

    # Use DistributedSampler for the training/val datasets in a distributed setup
    if GPUSetup.is_distributed():
        train_sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        shuffle = False  # Shuffle is handled by the DistributedSampler
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=train_sampler, shuffle=shuffle)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=val_sampler, shuffle=shuffle)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info(f"Rank {global_rank}: DataLoaders initialized with batch_size={batch_size}, num_workers={num_workers}")
    return train_loader, val_loader, test_loader


def get_splits(cfg) -> Dict[str, np.ndarray]:
    """train / val / test subject IDs of ``cfg['split_id']`` in the splits registry: ``cfg['splits_registry']`` (a YAML
    path) when set, else config/splits/splits_registry.yaml."""
    split_id = cfg.get('split_id')
    registry_path = cfg.get('splits_registry') or os.path.join(root, 'config', 'splits', 'splits_registry.yaml')
    registry = load_yaml(registry_path) if os.path.isfile(registry_path) else {}
    if not split_id or not isinstance(registry.get(split_id), dict):
        raise ValueError(f"split_id={split_id!r} not found in {registry_path}; list the subjects of each split there "
                         "and set split_id in the dataset config.")
    return {name: np.asarray(registry[split_id].get(name) or []).astype(str) for name in SPLITS}


def get_yolo_path(cfg):
    """YOLO slice data directory: ``{general_data_dir}/slice_preprocessed/YOLO``."""
    return os.path.join(cfg.get('general_data_dir', os.getcwd()), 'slice_preprocessed', 'YOLO')


def get_sam_path(cfg):
    """SAM slice data directory (NPY imgs / gts): ``{general_data_dir}/slice_preprocessed/SAM``."""
    return os.path.join(cfg.get('general_data_dir', os.getcwd()), 'slice_preprocessed', 'SAM')
