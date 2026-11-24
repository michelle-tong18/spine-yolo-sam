"""
Export a SAM fine-tuning checkpoint for inference / sharing: keep only the model weights (``{"model": state_dict}``,
without optimizer state or training counters, and without the ``module.`` prefix of multi-GPU training). Inference
(``inference.models.segmentation``) and continued training (``SAM.module.model_path`` + ``model_weights``) both load
this format.

    python src/sam_finetuning/export_model_weights.py <run>/<id>_finetuned_model_best.pth weights/my_sam.pth
"""
import argparse
from collections import OrderedDict

import torch


def model_only_checkpoint(checkpoint) -> dict:
    """``{"model": state_dict}`` of a training checkpoint (``{"model": ..., "optimizer": ...}``) or a bare state dict."""
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return {"model": OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())}


def main():
    parser = argparse.ArgumentParser(description="Keep only the model weights of a SAM fine-tuning checkpoint.")
    parser.add_argument("checkpoint", help="Training checkpoint (.pth) from SAM fine-tuning.")
    parser.add_argument("output", help="Output .pth with the model weights only.")
    args = parser.parse_args()
    torch.save(model_only_checkpoint(torch.load(args.checkpoint, map_location="cpu", weights_only=False)), args.output)
    print(f"Model weights -> {args.output}")


if __name__ == "__main__":
    main()
