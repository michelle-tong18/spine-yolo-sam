"""
Per-disc-level measurements on one sagittal T2 slice of a predicted segmentation (pure geometry, no I/O).

Everything is measured in mm in the disc's own frame, so lordosis (e.g. a tilted L5-S1) does not bias heights:
    u  the disc's long axis (principal component of its pixels), pointing posterior (toward the canal)
    v  perpendicular to u, pointing caudal (toward the vertebra below)
Heights are extents along v within 1-pixel strips along u; widths are extents along u within strips along v.

Measurements (nan when a structure they need is missing on the slice):
    disc_height_{anterior,mid,posterior}_mm   median strip height in each third of the disc's AP extent
    vert_height_{upper,lower}_mm              median strip height of the middle third of each adjacent vertebra
    disc_height_index_{anterior,mid,posterior}  disc height / mean adjacent vertebral height (scale free)
    disc_canal_signal_ratio                   mean disc T2 / mean canal T2 within the disc's SI band (scale free)
    posterior_displacement_mm                 how far disc extends posterior to the line between the posterior-
                                              inferior corner of the upper and the posterior-superior corner of
                                              the lower vertebra
    area_beyond_line_mm2                      disc area posterior to that line
    posterior_high_signal_fraction            fraction of the posterior quarter of the disc brighter than the mean
                                              canal signal (high-intensity zone)

``measure_discs_in_volume`` takes the median of each over the mid-sagittal slices of a 3D volume;
``measurement_levels`` checks which disc levels a label map supports.
"""
import warnings

import numpy as np

NAN = float("nan")
CORNER_FRACTION = 0.2  # vertebral pixels within this fraction of its SI extent nearest the disc form the corner


def _points_mm(mask: np.ndarray, row_mm: float, col_mm: float) -> np.ndarray:
    return np.argwhere(mask).astype(float) * np.array([row_mm, col_mm])


def disc_frame(disc_pts: np.ndarray, canal_pts: np.ndarray, lower_pts: np.ndarray):
    """(centroid, u, v): u the disc's long axis pointing toward the canal, v perpendicular pointing caudal."""
    centroid = disc_pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((disc_pts - centroid).T))
    u = vecs[:, -1]
    if len(canal_pts) and (canal_pts.mean(axis=0) - centroid) @ u < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    if len(lower_pts) and (lower_pts.mean(axis=0) - centroid) @ v < 0:
        v = -v
    return centroid, u, v


def _strip_extents(along: np.ndarray, across: np.ndarray, strip_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Per strip of width ``strip_mm`` along ``along``: (strip centre, extent of ``across`` + one pixel)."""
    if not len(along):
        return np.array([]), np.array([])
    strips = np.round((along - along.min()) / strip_mm).astype(int)  # round: a slight tilt must not split pixel columns
    centres, extents = [], []
    for k in np.unique(strips):
        sel = strips == k
        centres.append(along[sel].mean())
        extents.append(across[sel].max() - across[sel].min() + strip_mm)
    return np.array(centres), np.array(extents)


def _third_medians(along: np.ndarray, across: np.ndarray, strip_mm: float) -> tuple[float, float, float]:
    """Median strip extent in the first, middle and last third of the ``along`` range."""
    centres, extents = _strip_extents(along, across, strip_mm)
    if not len(centres):
        return NAN, NAN, NAN
    lo, hi = along.min(), along.max()
    edges = lo + (hi - lo) * np.array([0, 1 / 3, 2 / 3, 1 + 1e-9])
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (centres >= a) & (centres < b)
        out.append(float(np.median(extents[sel])) if sel.any() else NAN)
    return tuple(out)


def _mean(values) -> float:
    values = [x for x in values if np.isfinite(x)]
    return float(np.mean(values)) if values else NAN


def disc_geometry(seg: np.ndarray, disc_id: int, upper_id: int, lower_id: int, canal_id: int, row_mm: float,
                  col_mm: float) -> dict | None:
    """
    The disc-frame geometry measure_disc_level measures from: pixel points in mm per
    structure ("disc", "upper", "lower", "canal"), the frame (centroid, u, v), every point's coordinates along u
    (``a``) and v (``s``), the vertebral corners nearest the disc (``corners``: (a, s) points) and the posterior
    line through them (``line``: point, direction, posterior normal; None without a corner). None when the disc has
    fewer than 3 pixels.
    """
    if (seg == disc_id).sum() < 3:
        return None
    pix = min(row_mm, col_mm)
    pts = {name: _points_mm(seg == i, row_mm, col_mm)
           for name, i in (("disc", disc_id), ("upper", upper_id), ("lower", lower_id), ("canal", canal_id))}
    centroid, u, v = disc_frame(pts["disc"], pts["canal"], pts["lower"])
    a = {k: (p - centroid) @ u for k, p in pts.items()}
    s = {k: (p - centroid) @ v for k, p in pts.items()}
    corners = []
    for k, near_disc_is_high_s in (("upper", True), ("lower", False)):
        if not len(pts[k]):
            continue
        lo, hi = s[k].min(), s[k].max()
        band = s[k] >= hi - CORNER_FRACTION * (hi - lo) if near_disc_is_high_s else s[k] <= lo + CORNER_FRACTION * (hi - lo)
        # Posterior-most pixel column (within half a pixel, for a slight tilt), then the pixel nearest the disc.
        candidates = np.flatnonzero(band & (a[k] >= a[k][band].max() - 0.5 * pix))
        i = candidates[np.lexsort((-a[k][candidates], np.abs(s[k][candidates])))[0]]
        corners.append(np.array([a[k][i], s[k][i]]))
    line = None
    if corners:
        p0 = corners[0]
        t = (corners[1] - corners[0]) if len(corners) == 2 else np.array([0.0, 1.0])
        t = t / np.linalg.norm(t)
        n = np.array([t[1], -t[0]])
        line = (p0, t, n if n[0] > 0 else -n)  # normal pointing posterior (+a)
    return {"pts": pts, "centroid": centroid, "u": u, "v": v, "a": a, "s": s, "corners": corners, "line": line,
            "pix": pix}


def measure_disc_level(seg: np.ndarray, img: np.ndarray, disc_id: int, upper_id: int, lower_id: int, canal_id: int,
                       row_mm: float, col_mm: float) -> dict[str, float]:
    """All measurements for one disc on one 2D sagittal slice (``seg`` labels, ``img`` T2 intensities)."""
    out = dict.fromkeys(MEASUREMENTS, NAN)
    g = disc_geometry(seg, disc_id, upper_id, lower_id, canal_id, row_mm, col_mm)
    if g is None:
        return out
    pts, a, s, pix = g["pts"], g["a"], g["s"], g["pix"]

    # Heights: disc by thirds of its AP extent; vertebrae from their middle third.
    ant, mid, post = _third_medians(a["disc"], s["disc"], pix)
    out.update(disc_height_anterior_mm=ant, disc_height_mid_mm=mid, disc_height_posterior_mm=post)
    vert = {k: _third_medians(a[k], s[k], pix)[1] if len(pts[k]) else NAN for k in ("upper", "lower")}
    out.update(vert_height_upper_mm=vert["upper"], vert_height_lower_mm=vert["lower"])
    vert_mean = _mean(vert.values())
    for third, height in (("anterior", ant), ("mid", mid), ("posterior", post)):
        out[f"disc_height_index_{third}"] = height / vert_mean if np.isfinite(vert_mean) else NAN

    # Signal: disc vs canal (CSF + cord) within the disc's SI band.
    disc_px = img[seg == disc_id].astype(float)
    canal_mask = seg == canal_id
    in_band = (s["canal"] >= s["disc"].min()) & (s["canal"] <= s["disc"].max())
    canal_px = img[canal_mask][in_band].astype(float) if canal_mask.any() else np.array([])
    canal_mean = float(canal_px.mean()) if len(canal_px) else NAN
    out["disc_canal_signal_ratio"] = float(disc_px.mean()) / canal_mean if canal_mean > 0 else NAN
    posterior = a["disc"] > a["disc"].min() + 0.75 * (a["disc"].max() - a["disc"].min())
    out["posterior_high_signal_fraction"] = (float((disc_px[posterior] > canal_mean).mean())
                                             if np.isfinite(canal_mean) and posterior.any() else NAN)

    # Posterior line between the vertebral corners nearest the disc.
    if g["line"] is not None:
        p0, _, n = g["line"]
        depth = (np.stack([a["disc"], s["disc"]], axis=1) - p0) @ n
        out["posterior_displacement_mm"] = float(max(depth.max(), 0.0))
        # A pixel centered on the line is not beyond it; the tolerance absorbs rounding in the disc frame.
        out["area_beyond_line_mm2"] = float((depth > 1e-6 * pix).sum() * row_mm * col_mm)
    return out


MEASUREMENTS = (
    "disc_height_anterior_mm", "disc_height_mid_mm", "disc_height_posterior_mm",
    "vert_height_upper_mm", "vert_height_lower_mm",
    "disc_height_index_anterior", "disc_height_index_mid", "disc_height_index_posterior",
    "disc_canal_signal_ratio", "posterior_high_signal_fraction",
    "posterior_displacement_mm", "area_beyond_line_mm2",
)


def disc_neighbours(mask_labels: dict[int, str]) -> dict[int, tuple[int, int]]:
    """Disc label id -> (upper, lower) vertebra label ids, from names like ``L4-L5`` (``L5-S1`` -> L5, S1)."""
    ids = {name: i for i, name in mask_labels.items()}
    out = {}
    for i, name in mask_labels.items():
        upper, sep, lower = str(name).partition("-")
        if sep and upper in ids and lower in ids:
            out[i] = (ids[upper], ids[lower])
    return out


def mid_sagittal_slices(seg: np.ndarray, canal_id: int, n: int) -> list[int]:
    """``n`` consecutive slices centred on the one with the largest canal area (the midline)."""
    canal_area = (seg == canal_id).reshape(seg.shape[0], -1).sum(axis=1)
    centre = int(np.argmax(canal_area))
    lo = max(0, min(centre - n // 2, seg.shape[0] - n))
    return list(range(lo, min(lo + n, seg.shape[0])))


MM_COLUMNS = [m for m in MEASUREMENTS if m.endswith(("_mm", "_mm2"))]


def measurement_levels(mask_labels: dict[int, str], measurements_cfg: dict) -> list[str]:
    """Disc levels to measure (``measurements.levels``, default every disc with both vertebrae labelled); raises
    when the label map cannot support them."""
    if "spinal_canal" not in mask_labels.values():
        raise ValueError("Disc measurements need a 'spinal_canal' label in mask_labels (sagittal label map, "
                         "e.g. config/datasets/example_sagittal_vertdisc.yaml).")
    neighbours = disc_neighbours(mask_labels)
    discs = [mask_labels[i] for i in sorted(neighbours)]
    levels = list(measurements_cfg.get("levels") or discs)
    unknown = [lvl for lvl in levels if lvl not in discs]
    if unknown:
        raise ValueError(f"measurements.levels {unknown} are not discs with both adjacent vertebrae in mask_labels "
                         f"(discs: {discs}).")
    return levels


def measure_discs_in_volume(seg: np.ndarray, img: np.ndarray, mask_labels: dict[int, str], levels: list[str],
                            n_mid: int, row_mm: float, col_mm: float) -> list[dict]:
    """One row per disc level in ``levels``: median of each measurement over the mid-sagittal slices."""
    ids = {name: i for i, name in mask_labels.items()}
    canal_id = ids["spinal_canal"]
    neighbours = disc_neighbours(mask_labels)
    slices = mid_sagittal_slices(seg, canal_id, n_mid)
    rows = []
    for level in levels:
        disc_id = ids[level]
        upper_id, lower_id = neighbours[disc_id]
        per_slice = [measure_disc_level(seg[z], img[z], disc_id, upper_id, lower_id, canal_id, row_mm, col_mm)
                     for z in slices]
        with warnings.catch_warnings():  # all-nan columns (structure missing on every slice) stay nan
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = {m: float(np.nanmedian([s[m] for s in per_slice])) for m in MEASUREMENTS}
        rows.append({"level": level, "slices": ";".join(map(str, slices)),
                     "n_slices_with_disc": int(sum((seg[z] == disc_id).any() for z in slices)),
                     "row_mm": row_mm, "col_mm": col_mm, **medians})
    return rows
