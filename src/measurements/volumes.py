"""
Volume of each label and tissue group in one predicted segmentation (pure numpy, no I/O).

Volume = voxel count x voxel volume (slice_mm x row_mm x col_mm of the preprocessed grid, src/preprocessing/
spacing.py). This assumes the volume's slices are adjacent, which holds for image-only preprocessing (every slice
kept); volumes cropped to labelled slices for training may skip slices. Without spacing, ``volume_cm3`` is nan and
``voxels`` is still reported.
"""
import numpy as np

MM3_PER_CM3 = 1000.0


def label_volume_rows(seg: np.ndarray, mask_labels: dict[int, str], tissue_groups: dict[str, list[int]],
                      voxel_mm3: float) -> list[dict]:
    """One row per label (``level`` label) and per tissue group (``level`` tissue): voxels and volume_cm3."""
    counts = np.bincount(np.asarray(seg, dtype=np.int64).ravel(), minlength=max(mask_labels) + 1)
    rows = [{"level": "label", "class": name, "voxels": int(counts[i])}
            for i, name in sorted(mask_labels.items()) if i != 0]
    rows += [{"level": "tissue", "class": name, "voxels": int(sum(counts[i] for i in ids if i < len(counts)))}
             for name, ids in tissue_groups.items()]
    for row in rows:
        row["volume_cm3"] = row["voxels"] * voxel_mm3 / MM3_PER_CM3
    return rows
