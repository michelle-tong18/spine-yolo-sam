""" Finetuning scripts functional for non instance-based finetuning.
    Config: use CLI config_name or env CONFIG_NAME (get_config_name) for SLURM/env consistency.
"""

import os
import logging
import argparse
from collections import OrderedDict
from monai.losses import DiceLoss

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.distributed as dist

from segment_anything import sam_model_registry

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import src.sam_finetuning.utils.gpu_setup as GPUSetup
from src.sam_finetuning.utils.utils import determine_run_directory
from src.utils.file_management.config_handler import get_config_name, load_main_config, summarize_config
from src.utils.file_management.args_handler import process_kwargs
from src.sam_finetuning.datamodule.dynamic_dataset_loader import create_dataloader, process_dataset_by_split
from src.sam_finetuning.engine.finetune_engine import Trainer
from src.sam_finetuning.engine.models.sam import finetunedSAM
from src.sam_finetuning.utils.logging import log_info

# Retrieve a logger for the module
logger = logging.getLogger(__name__)

def datamodule_by_splits(cfg, run_path=None):

    # get datasets by splits
    train_dataset, val_dataset, test_dataset = process_dataset_by_split(cfg)

    # -------------------- SAVE DATASET SUMMARY -------------------- #   
    log_info(f"train set: {len(train_dataset)}")
    log_info(f"val set:   {len(val_dataset)}")
    log_info(f"test set:  {len(test_dataset)}")

    # -------------------- EXTRACT DATALOADERS -------------------- #    
    train_loader, val_loader, test_loader = create_dataloader(
            datasets = (train_dataset, val_dataset, test_dataset), 
            batch_size = cfg.get('SAM',{}).get('datamodule', {}).get('batch_size', 2),
            num_workers = cfg.get('SAM',{}).get('datamodule', {}).get('num_workers', 1),
            )

    return train_loader, val_loader, test_loader


def resolve_sam_finetuned_path(module_cfg: dict) -> str:
    """SAM.module.model_path + model_weights: prior finetuned weights to continue from ("" = start from pretrain_model)."""
    model_path = module_cfg.get("model_path") or ""
    model_weights = module_cfg.get("model_weights") or ""
    if model_path and model_weights:
        return os.path.join(str(model_path), str(model_weights))
    return ""


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(str(k).startswith("module.") for k in state_dict):
        return OrderedDict(
            (k[7:] if str(k).startswith("module.") else k, v) for k, v in state_dict.items()
        )
    return state_dict


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return _strip_module_prefix(checkpoint["model"])
    return _strip_module_prefix(checkpoint)


def prepare_training_base(comp_cfg, model_type:str, initial_weights:str, optimizer_cfg:dict, resume_checkpoint_path:str, device):
    log_info(f"Preparing training base on {device}")

    start_epoch = 0  # Warm-start from prior weights; do not restore old epoch/optimizer

    sam_model = sam_model_registry[model_type](checkpoint=initial_weights)

    # Model finetuning setup
    finetuned_model = finetunedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        config=comp_cfg
    ).to(device)

    if resume_checkpoint_path:
        if not os.path.isfile(resume_checkpoint_path):
            raise FileNotFoundError(f"SAM finetuned weights not found: {resume_checkpoint_path}")
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
            finetuned_model.load_state_dict(_state_dict_from_checkpoint(checkpoint), strict=True)
            log_info(f"Loaded prior SAM weights (warm start, epoch 0) from {resume_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load SAM weights from {resume_checkpoint_path}: {e}")
            raise e

    # Prepare optimizer with parameters that require gradients
    img_mask_encdec_params = [param for param in finetuned_model.parameters() if param.requires_grad]
    
    optimizer = AdamW(img_mask_encdec_params, lr=optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'])

    scheduler = ReduceLROnPlateau(optimizer)  # stepped on the validation loss each epoch

    # Prepare loss functions
    loss_fn = (DiceLoss(sigmoid=True, squared_pred=True, reduction="mean"), nn.BCEWithLogitsLoss(reduction="mean"))

    return finetuned_model, optimizer, scheduler, loss_fn, start_epoch


def finetune_main(cfg):
    # --------------- SET UP ENVIRONMENT --------------- #  
    rank = GPUSetup.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    sam_cfg = cfg.get('SAM',{})

    # Detect if we have a GPU available and choose device accordingly
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    logger.info(f"Rank: {rank}")
    logger.info(f"Device Count: {ngpus_per_node}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Distributed: {GPUSetup.is_distributed()}")
    logger.info(f"Local Rank {local_rank}: Starting finetune_main")

    if GPUSetup.is_distributed():
        if rank % ngpus_per_node == 0:
            print("Before DDP initialization:", flush=True)
            os.system("nvidia-smi")

    # ------------- SET UP EXPERIMENT RUN  ------------- #
    module_cfg = sam_cfg.get('module',{})

    work_dir = module_cfg.get('work_dir')
    if not work_dir:
        raise ValueError(
            "Set SAM.module.work_dir in the dataset YAML "
            "(example configs use ${work_dir}/sam)."
        )
    task_name  = f"{module_cfg.get('task_name','unknown_task')}"
    group_name = f"{module_cfg.get('group_name','SAM')}"

    if GPUSetup.is_distributed():
        # If distributed training is enabled, synchronize creation of the run directory
        
        if GPUSetup.is_main_process(): 
            # Only main process determines the run directory
            run_path = determine_run_directory(work_dir, task_name, group_name)
            # Since run_path is a string, use broadcast_object_list
            dist.broadcast_object_list([run_path], src=0)  # src=0 denotes the main process
        else:
            # Receive broadcasted run_path
            run_path = [None]  # Placeholder for the received object
            dist.broadcast_object_list(run_path, src=0)
            run_path = run_path[0]  # Unpack the list to get the actual path
    else:
        # If not distributed, directly determine the run directory
        run_path = determine_run_directory(work_dir, task_name, group_name)

    print(f"Run path: {run_path}")

    # --------------- SET UP DATALOADERS --------------- #
    train_loader, val_loader, test_loader = datamodule_by_splits(cfg, run_path)

    # --------------- SET UP MODEL --------------- #
    # Initialize model, optimizer, loss functions, and potentially load checkpoint
    model, optimizer, scheduler, loss_fn, start_epoch = prepare_training_base(
            comp_cfg = sam_cfg.get('module').get('trainable', {}),
            model_type = sam_cfg.get('module').get('model_type'), 
            initial_weights = sam_cfg.get('module').get('pretrain_model'),
            optimizer_cfg = {'lr': sam_cfg.get('module').get('optimizer').get('lr'), 
                        'weight_decay': sam_cfg.get('module').get('optimizer').get('weight_decay')},
            resume_checkpoint_path = resolve_sam_finetuned_path(sam_cfg.get('module', {})),
            device = device, 
            )
    
    if GPUSetup.is_distributed():
        # Convert all BatchNorm layers to SyncBatchNorm layers
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], \
            broadcast_buffers=True, find_unused_parameters=True)

        torch.backends.cudnn.benchmark = True

    if GPUSetup.is_distributed():
        if rank % ngpus_per_node == 0:
            print("After DDP initialization:", flush=True)
            os.system("nvidia-smi")

    # --------------- TRAIN --------------- #
    print("Begin training...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler, 
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        module_cfg=sam_cfg.get('module'),
        datamodule_cfg=sam_cfg.get('datamodule'),
        experiment_cfg=sam_cfg.get('experiment'),
        run_path=run_path,
        device=device,
        start_epoch=start_epoch,
        verbose=False,
    )

    summarize_config(cfg, path=os.path.join(root, run_path, sam_cfg.get('output_configuration').get('save_path')))
    
    # Start training and validation phases
    logger.info(f"Local Rank {local_rank}: Starting training and validation phases...")
    trainer.train(num_epochs=sam_cfg.get('module',{}).get('num_epochs',10))

if __name__ == "__main__":
    
    try:
        # Argument parsing to allow for flexible execution of the script
        parser = argparse.ArgumentParser(description="Fine-tuning model setup.")
        parser.add_argument("config_name", help="Name of the YAML configuration file")
        parser.add_argument('--kwargs', nargs=argparse.REMAINDER, help="Additional command line key=value pairs")
        args = parser.parse_args()
        kwargs = process_kwargs(args.kwargs)
    
        # Config name from CLI or CONFIG_NAME env; do not add .yaml — load_main_config/extract_config_path add it
        config_name = get_config_name(args.config_name)
    
        cfg = load_main_config(config_name, root, sub_dir='datasets', **kwargs)
    
        # Setup logging for the application
        logger = GPUSetup.setup_logging(
                config_level = cfg.get('SAM',{}).get('output_configuration', {}).get('logging_level', 'INFO').upper(),
                logger = logging.getLogger(__name__))
    
        # Setup environment for distributed training, if applicable
        GPUSetup.setup(distributed = cfg.get('SAM',{}).get('distributed', False),
                       seed = cfg.get('SEED', 42))
    
        finetune_main(cfg)
            
    except Exception:
        # Log with the traceback, then fail the process so scripts and schedulers see the error
        logger.exception("SAM fine-tuning failed")
        raise
    finally:
        if 'GPUSetup' in locals():
            GPUSetup.cleanup()
            log_info("Cleanup completed.")






