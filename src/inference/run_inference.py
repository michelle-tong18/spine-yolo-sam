"""Public inference CLI: subcommands yolo | sam | both.

    python src/inference/run_inference.py both --config example_sagittal_vertdisc
    python src/inference/run_inference.py yolo --config example_sagittal_vertdisc
    python src/inference/run_inference.py sam  --config example_sagittal_vertdisc --boxes pred
    python src/inference/run_inference.py sam  --config example_sagittal_vertdisc --boxes gt
    python src/inference/run_inference.py both --config example_sagittal_vertdisc --figures   (+ {run}/figures/)

CLI overrides YAML: ``--volumes``, ``--output``, ``--pred-labels-dir``, ``--slices``,
``--run-dir`` / ``--run_num``.
"""
from __future__ import annotations

import argparse
import os

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import sys
sys.path.append(root)

from src.utils.file_management.config_handler import get_config_name, load_main_config, summarize_config
from src.inference.infer_helper import resolve_inference_io, resolve_maybe_path
from src.inference.yolo_predict import run_yolo_predict_from_slices, run_yolo_predict_from_volumes
from src.inference.figures import save_run_figures
from src.inference.sam_from_boxes import run_sam_from_boxes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inference: yolo writes label_pred/, sam writes seg_pred/, "
            "both runs yolo-from-volumes then sam --boxes pred in the same Run_N."
        )
    )
    parser.add_argument(
        "command",
        choices=("yolo", "sam", "both"),
        help="yolo: detect; sam: segment from boxes; both: yolo then sam on predicted boxes.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config name in config/datasets/ (or set CONFIG_NAME).",
    )
    parser.add_argument(
        "--volumes",
        default=None,
        help="Override inference.volume_dir (folder of 3D volumes).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override top-level output_data_dir (predictions root).",
    )
    parser.add_argument(
        "--pred-labels-dir",
        dest="pred_labels_dir",
        default=None,
        help="Override predicted YOLO .txt dir for sam --boxes pred (default: {run}/label_pred).",
    )
    parser.add_argument(
        "--slices",
        default=None,
        help="Existing PNG slice folder for standalone `yolo` only (skips volume extract).",
    )
    parser.add_argument(
        "--boxes",
        choices=("pred", "gt"),
        default=None,
        help="Box source for `sam`: pred (label_pred/) or gt (evaluation.gt_masks_dir). Required for sam.",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help="Reuse this run directory (e.g. yolo then sam --boxes pred).",
    )
    parser.add_argument(
        "--run_num",
        "--run-num",
        dest="run_num",
        type=int,
        default=None,
        help="Reuse Run_N under {output_data_dir}/inference/{task_name}/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose ImagePrep during volume load/extract.",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="After SAM, save {run}/figures/<volume>.png: image, YOLO boxes and masks on evenly spaced slices.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command == "sam" and args.boxes is None:
        parser.error("sam requires --boxes pred or --boxes gt")
    if args.command == "both" and args.boxes == "gt":
        parser.error("both always uses predicted boxes; use `sam --boxes gt` for GT boxes")
    if args.command == "both" and args.slices:
        parser.error("both does not take a slice folder; it reads volume_dir")
    if args.command == "yolo" and args.boxes:
        parser.error("yolo does not take --boxes")
    if args.command != "yolo" and args.slices:
        parser.error("--slices is only valid with the yolo subcommand")
    if args.command == "yolo" and args.pred_labels_dir:
        parser.error("--pred-labels-dir is only valid with sam (or both, to override label_pred)")
    if args.command == "yolo" and args.figures:
        parser.error("--figures needs masks: use both (or sam)")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    config_name = get_config_name(args.config)
    if config_name is None:
        parser.error("config is required (pass --config or set CONFIG_NAME)")

    base_dir = os.getcwd()
    cfg = load_main_config(config_name, base_dir, sub_dir="datasets")
    infer = cfg.get("inference") or {}

    slice_img = args.slices or infer.get("slice_img_dir")
    yolo_from_slices = args.command == "yolo" and bool(slice_img)
    io = resolve_inference_io(
        cfg,
        volumes=args.volumes,
        output=args.output,
        run_dir=args.run_dir,
        run_num=args.run_num,
        base_dir=base_dir,
        require_volume_dir=not yolo_from_slices,
    )
    run_dir = io["run_dir"]
    print(f"Inference output directory: {os.path.abspath(run_dir)}")
    summarize_config(cfg, path=os.path.join(root, run_dir, "Run_Summaries"))

    device = infer.get("device") or cfg.get("device", "cpu")

    if args.command in ("yolo", "both"):
        if yolo_from_slices:
            slice_dir = resolve_maybe_path(base_dir, slice_img)
            print(f"YOLO from existing slices: {slice_dir}")
            run_yolo_predict_from_slices(
                cfg,
                run_dir,
                slice_dir,
                base_dir=base_dir,
            )
        else:
            print(f"YOLO from volumes: {io['volume_dir']}")
            run_yolo_predict_from_volumes(
                cfg,
                run_dir,
                io["volume_dir"],
                base_dir=base_dir,
                verbose=args.verbose,
            )

    if args.command in ("sam", "both"):
        boxes = args.boxes or "pred"
        pred_labels_dir = args.pred_labels_dir or io["label_pred_dir"]
        print(f"SAM from boxes={boxes}  volumes={io['volume_dir']}")
        run_sam_from_boxes(
            cfg,
            run_dir,
            io["volume_dir"],
            boxes=boxes,
            pred_labels_dir=pred_labels_dir if boxes == "pred" else None,
            device=device,
            verbose=args.verbose,
            base_dir=base_dir,
        )

    if args.figures:
        save_run_figures(cfg, run_dir, io["volume_dir"])


if __name__ == "__main__":
    main()
