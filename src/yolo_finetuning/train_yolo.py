"""
Train the YOLO detector from the dataset YAML's ``YOLO`` block (Ultralytics), merged over config/yolo/<train_cfg>.yaml.

    python src/yolo_finetuning/train_yolo.py <config>

Starting checkpoint (weights/YOLO/, repo-relative): ``YOLO.model`` with ``yolo_parameters.pretrained: True`` (released
models: yolo26x.pt), a checkpoint path in ``pretrained``, else ``YOLO.model`` or yolo26m.pt (architecture only).
"""
import os
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import sys
sys.path.append(root)

from src.utils.file_management.config_handler import load_main_config

ARCH_FALLBACK = 'weights/YOLO/yolo26m.pt'

def resolve_yolo_work_dir(config):
    """Parent of Ultralytics ``project`` folders. Set ``YOLO.yolo_output_dir`` in the dataset YAML."""
    output_base = config.get('yolo_output_dir') or os.environ.get('YOLO_WORK_DIR')
    if not output_base:
        raise ValueError(
            "Set YOLO.yolo_output_dir in the dataset YAML "
            "(example configs use ${work_dir}/yolo)."
        )
    return output_base


def load_config(config_file_name, base_dir):
    """The dataset YAML's ``YOLO`` block, with ``train_cfg`` = config/yolo/<train_cfg>.yaml overridden by ``yolo_parameters``."""
    data_config = load_main_config(config_file_name, base_dir, sub_dir='datasets')
    config = data_config.get('YOLO',{})

    config['train_cfg'] = load_main_config(config['train_cfg'], base_dir, sub_dir='yolo')

    if 'resume' not in config['train_cfg'].keys():
        config['train_cfg']['resume'] = False

    # overload default yolo configuration settings
    yolo_parameters = config.get('yolo_parameters') or {}
    for key in yolo_parameters.keys():
        config['train_cfg'][key] = yolo_parameters[key]

    # Place YOLO output under {yolo_output_dir}/{project}/{name}.
    # Increment name ourselves (runs -> runs2) when exist_ok is False; keep ``project``
    # short for W&B (max 128 chars). Setting save_dir after increment avoids Ultralytics
    # writing into a fixed .../runs folder on every relaunch.
    output_base = resolve_yolo_work_dir(config)
    project = config['train_cfg'].get('project')
    if not project:
        raise ValueError(
            "Set YOLO.yolo_parameters.project in the dataset YAML "
            "(Ultralytics name also belongs there, not in YOLO_default_cfg)."
        )
    run_name = config['train_cfg'].get('name') or 'runs'
    if not os.path.isabs(project):
        project_short = str(project).replace('/', '-')
        exist_ok = bool(config['train_cfg'].get('exist_ok', False))
        save_dir = increment_path(Path(output_base) / project_short / run_name, exist_ok=exist_ok)
        config['train_cfg']['save_dir'] = str(save_dir)
        config['train_cfg']['project'] = project_short
        config['train_cfg']['name'] = save_dir.name

    return config

def _resolve_init_checkpoint(config):
    """(kind, path) of the starting checkpoint: ``YOLO.model`` when ``pretrained`` is True, the ``pretrained`` path
    (loaded with task='detect'), else ``YOLO.model`` or the yolo26m architecture."""
    if config.get('model_type', 'YOLO').upper() != 'YOLO':
        raise ValueError(f"YOLO.model_type={config.get('model_type')!r}: only YOLO is supported")
    pretrained = (config.get('yolo_parameters') or {}).get('pretrained', None)
    model_path = config.get('model')
    if pretrained is True:
        if not model_path:
            raise ValueError("YOLO pretrained=True requires YOLO.model to be set to a checkpoint path (.pt or .yaml).")
        return 'standard', model_path
    if pretrained:
        return 'task_detect', pretrained
    return 'standard', model_path or ARCH_FALLBACK


def build_yolo_model(config, device):
    """
    Load an Ultralytics model from config. See _resolve_init_checkpoint for weight resolution.
    """
    kind, path = _resolve_init_checkpoint(config)
    model = YOLO(path).to(device) if kind == 'standard' else YOLO(path, task='detect').to(device)
    print(f'Model init weights: {path}')

    print(f'Model loaded with: {model.device.type}')
    return model


def ultralytics_train_device():
    """Device index/list for model.train(device=...). None means omit (CPU path)."""
    if not torch.cuda.is_available():
        return None
    num_devices = torch.cuda.device_count()
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f'Number of devices: {num_devices}, Visible cuda devices: {visible_devices}')
        
    if num_devices > 1:
        if visible_devices and visible_devices.strip():
            return [int(x.strip()) for x in visible_devices.split(',') if x.strip() != '']
        return list(range(num_devices))
    if visible_devices and visible_devices.strip():
        return int(visible_devices.split(',')[0].strip())
    return 0


def run_yolo_train(model, data_yaml, train_cfg, train_device=None):
    """Run Ultralytics training with consistent GPU device handling.

    If ``train_device`` is omitted, calls :func:`ultralytics_train_device` internally.
    """
    dev = ultralytics_train_device() if train_device is None else train_device
    if dev is None:
        model.train(data=data_yaml, **train_cfg)
    else:
        model.train(data=data_yaml, device=dev, **train_cfg)


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model = build_yolo_model(config, device)

    # Resume keeps only what Ultralytics needs to continue the last run
    if config['train_cfg']['resume'] is True:
        config['train_cfg'] = {
            'resume': config['train_cfg']['resume'],
            'save': config['train_cfg']['save'],
            'save_period': config['train_cfg']['save_period'],
            'batch': config['train_cfg']['batch'],
            'project': config['train_cfg'].get('project'),
            'name': config['train_cfg'].get('name'),
        }
        print(config['train_cfg'])

    run_yolo_train(model, config['data_yaml'], config['train_cfg'], ultralytics_train_device())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Model with Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    config = load_config(args.config_name, os.getcwd())

    main(config)

