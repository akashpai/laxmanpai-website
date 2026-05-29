"""Metric scale recovery.

Monocular Structure-from-Motion is *scale ambiguous*: a reconstruction from a
phone video is only correct up to an unknown global scale factor. To hit a
sub-millimetre target you MUST inject a metric reference into the scene. Without
this, every downstream accuracy number is meaningless.

Recommended references, in order of robustness for the intraoral case:
  1. A ChArUco / ArUco target of precisely known size printed on the retractor
     (high-contrast, detected automatically, gives many correspondences).
  2. A calibrated photogrammetry scale bar in the same focal plane as the teeth
     (best practice: >= 3 bars for a statistical scale estimate).
  3. A known inter-landmark distance (e.g. a measured bracket width) as a
     last resort.

CRITICAL: the reference must sit in the SAME focal plane as the teeth, because
intraoral macro photography has a very shallow depth of field; an out-of-plane
reference biases the scale.

This module provides:
  * estimate_scale_from_aruco: detect an ArUco/ChArUco board and return the
    real-world-per-pixel or real-world-per-recon-unit scale.
  * metric_scale_from_reference_distance: scale factor from a single known
    distance between two reconstructed points.
  * apply_scale: rescale a point cloud / mesh about its centroid.

OpenCV (cv2) is required only for the ArUco detection helpers; the pure-geometry
helpers work with numpy alone so the rest of the pipeline and the tests do not
depend on OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ScaleEstimate:
    scale: float                 # multiply reconstruction coords by this -> mm
    method: str
    residual: float = 0.0        # spread of per-measurement scale (mm/unit), if any
    n_measurements: int = 1


def metric_scale_from_reference_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
    known_distance_mm: float,
) -> ScaleEstimate:
    """Scale factor from one known real-world distance.

    ``point_a``/``point_b`` are two points in the (unscaled) reconstruction whose
    true separation is ``known_distance_mm``. Returns the factor to multiply all
    reconstruction coordinates by to obtain millimetres.
    """
    a = np.asarray(point_a, dtype=np.float64)
    b = np.asarray(point_b, dtype=np.float64)
    recon_dist = float(np.linalg.norm(a - b))
    if recon_dist <= 0:
        raise ValueError("reference points coincide; cannot derive scale")
    if known_distance_mm <= 0:
        raise ValueError("known_distance_mm must be positive")
    return ScaleEstimate(
        scale=known_distance_mm / recon_dist,
        method="reference_distance",
    )


def metric_scale_from_many(
    recon_distances: np.ndarray,
    known_distances_mm: np.ndarray,
) -> ScaleEstimate:
    """Robust scale from several reference pairs (best practice: >= 3 scale bars).

    Reports the mean scale and the residual spread, which is a direct, honest
    proxy for how well-conditioned the metric scale is.
    """
    rd = np.asarray(recon_distances, dtype=np.float64)
    kd = np.asarray(known_distances_mm, dtype=np.float64)
    if rd.shape != kd.shape or rd.ndim != 1 or rd.size < 1:
        raise ValueError("recon_distances and known_distances_mm must be 1D, same length")
    if np.any(rd <= 0):
        raise ValueError("reconstruction distances must be positive")
    scales = kd / rd
    return ScaleEstimate(
        scale=float(np.mean(scales)),
        method="multi_reference",
        residual=float(np.std(scales)),
        n_measurements=int(rd.size),
    )


def apply_scale(points: np.ndarray, scale: float, about_centroid: bool = False) -> np.ndarray:
    """Rescale a point set (optionally about its centroid to keep it in place)."""
    pts = np.asarray(points, dtype=np.float64)
    if about_centroid:
        c = pts.mean(axis=0)
        return (pts - c) * scale + c
    return pts * scale


# --------------------------------------------------------------------------- #
# OpenCV-dependent ArUco / ChArUco detection
# --------------------------------------------------------------------------- #
def _import_cv2():
    try:
        import cv2  # noqa: F401

        return cv2
    except Exception as exc:  # pragma: no cover - exercised only without cv2
        raise ImportError(
            "OpenCV (opencv-python) is required for ArUco detection. Install "
            "with `pip install opencv-contrib-python`."
        ) from exc


def estimate_scale_from_aruco(
    image: np.ndarray,
    marker_length_mm: float,
    dictionary: str = "DICT_5X5_50",
) -> Optional[Tuple[float, int]]:
    """Detect ArUco markers and return (mm per pixel, num markers) or None.

    This gives an *image-plane* scale, useful for sanity checks and for scaling a
    reconstruction when marker corners are triangulated into the model. For a
    full metric reconstruction, feed the detected, known-size marker corners into
    the bundle adjustment / scale step (see reconstruction.colmap_runner).

    Returns None if no markers are found.
    """
    cv2 = _import_cv2()
    aruco = cv2.aruco
    dict_id = getattr(aruco, dictionary)
    aruco_dict = aruco.getPredefinedDictionary(dict_id)

    # OpenCV >= 4.7 uses ArucoDetector; fall back to the legacy API otherwise.
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # pragma: no cover - legacy OpenCV
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    if ids is None or len(corners) == 0:
        return None

    # average marker side length in pixels across all detected markers
    side_px = []
    for c in corners:
        pts = c.reshape(4, 2)
        sides = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0]),
        ]
        side_px.append(np.mean(sides))
    mean_side_px = float(np.mean(side_px))
    mm_per_px = marker_length_mm / mean_side_px
    return mm_per_px, int(len(corners))
