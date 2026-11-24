"""Evaluation package: load predictions and GT from disk, compute metrics, report.

Submodules are imported explicitly (e.g. ``from src.evaluation.X import ...``).
Heavy optional dependencies (MONAI/torch/nibabel) live in segmentation helpers and
are not imported at package import time.
"""

__all__: list[str] = []
