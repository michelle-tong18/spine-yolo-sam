# train_yolo_sweep.py
# Config: use CLI config_name or env CONFIG_NAME (get_config_name) for SLURM/env consistency.
import os
import argparse
import torch
import wandb

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import sys
sys.path.append(root)

from src.utils.file_management.config_handler import get_config_name
from train_yolo import load_config, build_yolo_model, resolve_yolo_work_dir, run_yolo_train


def train_with_sweep_params(config, device):
    """Train function that will be called by wandb sweep agent."""
    # Initialize wandb run
    wandb.init()
    
    # Get sweep parameters from wandb.config
    sweep_params = dict(wandb.config)
    
    # Update train_cfg with sweep parameters
    train_cfg = config['train_cfg'].copy()
    for key, value in sweep_params.items():
        train_cfg[key] = value
    
    model = build_yolo_model(config, device)
    print(f'Training with sweep parameters: {sweep_params}')

    # Prepare training configuration
    if train_cfg['resume'] is True:
        train_cfg = {
            'resume': train_cfg['resume'],
            'save': train_cfg['save'],
            'save_period': train_cfg['save_period'],
            'batch': train_cfg['batch'],
            'project': train_cfg.get('project'),
            'name': train_cfg.get('name'),
        }

    # Raw project name for W&B (load_config shortens it for the local save_dir)
    train_cfg['project'] = config['yolo_parameters']['project']

    # cwd = work_dir/yolo so Ultralytics writes work_dir/yolo/<project>/
    output_base = resolve_yolo_work_dir(config)
    run_dir = os.path.abspath(output_base)
    os.makedirs(run_dir, exist_ok=True)
    orig_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        if torch.cuda.is_available():
            vis = os.environ.get('CUDA_VISIBLE_DEVICES')
            print(
                f'Number of devices: {torch.cuda.device_count()}, '
                f'Visible cuda devices: {vis}'
            )
        run_yolo_train(model, config['data_yaml'], train_cfg)
    finally:
        os.chdir(orig_cwd)
    
    return 

def main(config):
    """Main function to initialize and run wandb sweep."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Get sweep configuration
    sweep_cfg = config.get('sweep_cfg', {})
    if not sweep_cfg:
        raise ValueError("No sweep_cfg found in configuration. Please add sweep configuration to your YAML file.")

    # Join existing sweep
    sweep_id = config.get('sweep_id', None)

    entity = os.environ.get("WANDB_ENTITY")
    project = config['yolo_parameters']['project']

    # Create a new sweep when no valid sweep_id is provided
    if sweep_id is None:
        if not entity:
            raise ValueError("WANDB_ENTITY must be set when creating a new sweep (e.g. in train_yolo.env).")
        sweep_id = wandb.sweep(
            sweep=sweep_cfg,
            project=project,
            entity=entity
        )
        print(f"Created sweep with ID: {sweep_id}")
    else:
        print(f"Joining existing sweep: {sweep_id} (entity={entity}, project={project})")
    # import pdb; pdb.set_trace()
    print("Starting sweep agent...")
    # Run sweep agent; pass entity and project so W&B can resolve the sweep (avoids 404 project not found)
    wandb.agent(
        sweep_id,
        entity=entity,
        project=project,
        function=lambda: train_with_sweep_params(config, device),
        count=config.get('sweep_count', None)  # Run indefinitely until stopped or sweep completes
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Model with Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    # Assuming the script is run from the root directory
    config_name = get_config_name(args.config_name)
    base_dir = os.getcwd()
    config = load_config(config_name, base_dir)

    main(config)