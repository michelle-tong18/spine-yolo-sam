"""Dice and IoU of two binary masks."""
import numpy as np


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute Sørensen-Dice coefficient."""
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.nan
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def compute_binary_iou(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """Binary IoU between two boolean masks."""
    inter = int((mask_gt & mask_pred).sum())
    union = int((mask_gt | mask_pred).sum())
    if union == 0:
        return float("nan")
    return inter / union
