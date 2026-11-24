# run_pipeline.py
"""
Orchestration entrypoint: build dataset → finetune YOLO and/or SAM → optional inference.
Config: use CLI config_name or env CONFIG_NAME (get_config_name) for SLURM/env consistency.
Runs from project root; changes directory to repo root if needed.
Calls existing scripts via subprocess so behavior and logs stay identical.
"""

import argparse
import os
import subprocess
import sys

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.file_management.config_handler import get_config_name


def _run(cmd: list[str], step_name: str) -> None:
    """Run a command from repo root; raise on non-zero exit."""
    try:
        rc = subprocess.run(cmd, cwd=root, check=False)
    except FileNotFoundError as e:
        print(f"[run_pipeline] {step_name}: command failed: {e}", file=sys.stderr)
        raise
    if rc.returncode != 0:
        raise SystemExit(rc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline: build dataset → finetune YOLO/SAM → optional inference.",
    )
    parser.add_argument(
        "config_name",
        nargs="?",
        default=None,
        help="Config name (e.g. example_axial_muscle). Optional if CONFIG_NAME env is set.",
    )
    parser.add_argument(
        "--finetune",
        choices=("yolo", "sam", "both"),
        default=None,
        help="Run finetuning: yolo, sam, or both. Default: none.",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Run preprocessing and split before finetuning.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference after finetuning when applicable.",
    )
    args = parser.parse_args()

    config_name = get_config_name(args.config_name)
    if config_name is None:
        parser.error("config_name is required (pass as first argument or set CONFIG_NAME)")

    python = sys.executable
    src = os.path.join(root, "src")

    if args.build_dataset:
        _run(
            [python, os.path.join(src, "preprocessing", "full_standardization.py"), config_name],
            "full_standardization",
        )
        _run(
            [python, os.path.join(src, "preprocessing", "dataset_split.py"), config_name],
            "dataset_split",
        )

    if args.finetune in ("yolo", "both"):
        _run(
            [python, os.path.join(src, "yolo_finetuning", "train_yolo.py"), config_name],
            "train_yolo",
        )
    if args.finetune in ("sam", "both"):
        _run(
            [python, os.path.join(src, "sam_finetuning", "finetune_main.py"), config_name],
            "finetune_main",
        )

    if args.inference:
        _run(
            [
                python,
                os.path.join(src, "inference", "run_inference.py"),
                "both",
                "--config",
                config_name,
            ],
            "run_inference",
        )


if __name__ == "__main__":
    main()
