"""
Axial tissue volume per vertebral level, from one patient's sagittal and axial segmentations (pure numpy, no I/O).

Both volumes are placed in patient coordinates with their ``world_affine`` (src/preprocessing/spacing.py). In the
sagittal segmentation each disc defines a mid-plane: through the disc's centroid, normal to its thinnest principal
axis, oriented caudally (toward the vertebra below). Vertebral level V is the space between the mid-planes of the disc
above (U-V) and the disc below (V-W), e.g. L4 between L3-L4 and L4-L5; levels without both discs are not reported.
Every labelled axial voxel is assigned to the level whose two mid-planes enclose its centre.

Assumes the two scans share one patient position (same session, no movement in between; DICOM coordinates are
per frame of reference). ``coverage_frac`` reports how much of each level's height the axial slices cover, since an
axial series rarely spans whole levels.
"""
import numpy as np

from src.measurements.disc_measurements import disc_neighbours

MAX_PCA_POINTS = 20000
MM3_PER_CM3 = 1000.0


def voxel_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """(n, 3) voxel indices -> (n, 3) patient coordinates (mm)."""
    return indices @ affine[:3, :3].T + affine[:3, 3]


def disc_planes(seg: np.ndarray, affine: np.ndarray, mask_labels: dict[int, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Disc name -> (centroid, unit normal pointing caudally) of each disc present with at least one adjacent
    vertebra (needed to orient the normal)."""
    rng = np.random.default_rng(0)
    centroids = {}
    for lid in (int(v) for v in np.unique(seg) if v != 0):
        idx = np.argwhere(seg == lid)
        if len(idx) > MAX_PCA_POINTS:
            idx = idx[rng.choice(len(idx), MAX_PCA_POINTS, replace=False)]
        centroids[lid] = voxel_world(idx.astype(float), affine)
    planes = {}
    for disc_id, (upper_id, lower_id) in disc_neighbours(mask_labels).items():
        pts = centroids.get(disc_id)
        if pts is None or len(pts) < 3 or (upper_id not in centroids and lower_id not in centroids):
            continue
        centre = pts.mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
        normal = vecs[:, 0]  # smallest variance: across the disc
        caudal = (centroids[lower_id].mean(axis=0) - centre) if lower_id in centroids else \
            (centre - centroids[upper_id].mean(axis=0))
        planes[mask_labels[disc_id]] = (centre, normal if normal @ caudal > 0 else -normal)
    return planes


def vertebral_levels(mask_labels: dict[int, str], planes: dict) -> list[tuple[str, str, str]]:
    """(vertebra, disc above, disc below) for every vertebra whose two discs both have a mid-plane."""
    above, below = {}, {}
    for disc_id, (upper_id, lower_id) in disc_neighbours(mask_labels).items():
        below[mask_labels[upper_id]] = mask_labels[disc_id]
        above[mask_labels[lower_id]] = mask_labels[disc_id]
    return [(v, above[v], below[v]) for v in mask_labels.values()
            if v in above and v in below and above[v] in planes and below[v] in planes]


def axial_level_volume_rows(axial_seg: np.ndarray, axial_affine: np.ndarray, axial_labels: dict[int, str],
                            tissue_groups: dict[str, list[int]], planes: dict,
                            levels: list[tuple[str, str, str]]) -> list[dict]:
    """
    Per level, one row per axial label (``level_kind`` label) and per tissue group (tissue): voxels, volume_cm3,
    n_axial_slices (axial slices whose segmented centre lies in the level) and coverage_frac (those slices' extent
    along the spine / the level's height, at most 1).
    """
    voxel_mm3 = abs(float(np.linalg.det(axial_affine[:3, :3])))
    slice_step_mm = float(np.linalg.norm(axial_affine[:3, 0]))
    idx = np.argwhere(axial_seg > 0)
    labels = axial_seg[tuple(idx.T)].astype(np.int64)
    pts = voxel_world(idx.astype(float), axial_affine)
    # Each slice's centre: the centroid of its segmented pixels (the paraspinal region), else the image centre.
    h, w = axial_seg.shape[1:]
    slice_centres = []
    for z in range(axial_seg.shape[0]):
        on = idx[idx[:, 0] == z]
        rc = on[:, 1:].mean(axis=0) if len(on) else np.array([(h - 1) / 2, (w - 1) / 2])
        slice_centres.append(voxel_world(np.array([[z, *rc]], float), axial_affine)[0])
    slice_centres = np.array(slice_centres)
    n_bins = max(int(max(axial_labels)), int(labels.max()) if len(labels) else 0) + 1

    rows = []
    for vertebra, disc_above, disc_below in levels:
        (c_up, n_up), (c_down, n_down) = planes[disc_above], planes[disc_below]
        inside = ((pts - c_up) @ n_up >= 0) & ((pts - c_down) @ n_down < 0)
        counts = np.bincount(labels[inside], minlength=n_bins)
        in_slices = ((slice_centres - c_up) @ n_up >= 0) & ((slice_centres - c_down) @ n_down < 0)
        height = float((c_down - c_up) @ ((n_up + n_down) / np.linalg.norm(n_up + n_down)))
        n_slices = int(in_slices.sum())
        level = {"level": vertebra, "disc_above": disc_above, "disc_below": disc_below,
                 "n_axial_slices": n_slices,
                 "coverage_frac": min(1.0, n_slices * slice_step_mm / height) if height > 0 else np.nan}
        classes = [("label", name, [lid]) for lid, name in sorted(axial_labels.items()) if lid != 0]
        classes += [("tissue", name, ids) for name, ids in tissue_groups.items()]
        for kind, name, ids in classes:
            voxels = int(sum(counts[i] for i in ids if i < n_bins))
            rows.append({**level, "level_kind": kind, "class": name, "voxels": voxels,
                         "volume_cm3": voxels * voxel_mm3 / MM3_PER_CM3})
    return rows
