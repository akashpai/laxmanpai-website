"""Validation metrics for dental surface reconstruction.

Implements the standard dental-metrology workflow described in the literature:

  1. Best-fit (ICP) superimposition of a *test* mesh onto a *reference* mesh.
  2. Signed/unsigned nearest-neighbour surface deviation.
  3. Summary statistics: mean, RMS, 90th/95th percentile, Hausdorff (max).
  4. Trueness (vs ground truth) and precision (scan-to-scan), per ISO 5725.

Why these metrics: dental accuracy = trueness + precision (ISO 5725-1).
Trueness is closeness of a scan to the true geometry (needs an independent
reference such as a lab/industrial scanner or CBCT); precision is the spread
among repeated scans of the same object. Surface deviation is reported as RMS
and visualised as a colour map in tools like Geomagic Control X.

Clinically load-bearing thresholds (see docs/RESEARCH_REPORT.md for citations):
  * Restoration marginal gap acceptable: <= 0.120 mm (120 um).
  * Implant-supported tolerance: ~0.010 mm (10 um).
  * Clear-aligner manufacturing accepted error: < 0.250 mm.
  * Modern IOS full-arch trueness: ~0.020-0.115 mm (the benchmark to beat).

NOTE on RMS being software-dependent: alignment + deviation calculations differ
between metrology programs, so absolute RMS is only comparable within one tool.
This module fixes the algorithm (point-to-point nearest neighbour after ICP) so
results are reproducible and self-consistent.

All distances are in the units of the input point coordinates. The pipeline is
designed so reconstructions are metrically scaled to millimetres before this
module runs (see calibration.scale).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover - scipy is a hard dependency for this module
    _KDTree = None


def _as_points(arr) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) point array, got shape {pts.shape}")
    return pts


def _require_kdtree():
    if _KDTree is None:  # pragma: no cover
        raise ImportError(
            "scipy is required for nearest-neighbour queries. Install with "
            "`pip install scipy`."
        )


# --------------------------------------------------------------------------- #
# Rigid alignment (Kabsch / Procrustes + ICP)
# --------------------------------------------------------------------------- #
def best_fit_transform(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (R, t) mapping ``source`` onto ``target``.

    Uses the Kabsch/Umeyama SVD solution. ``source`` and ``target`` must be
    corresponding point sets of identical length. Returns (R 3x3, t 3,) such
    that ``target ~= source @ R.T + t``.
    """
    src = _as_points(source)
    tgt = _as_points(target)
    if src.shape != tgt.shape:
        raise ValueError("source and target must have identical shape for Kabsch")

    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    H = (src - src_c).T @ (tgt - tgt_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = tgt_c - R @ src_c
    return R, t


@dataclass
class ICPResult:
    R: np.ndarray
    t: np.ndarray
    aligned: np.ndarray
    rms: float
    iterations: int

    def transform(self, points) -> np.ndarray:
        return _as_points(points) @ self.R.T + self.t


def iterative_closest_point(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 60,
    tolerance: float = 1e-6,
    init: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> ICPResult:
    """Best-fit superimposition of ``source`` onto ``target`` (no correspondence).

    Standard point-to-point ICP: at each step, match every source point to its
    nearest target point, then solve the rigid transform that minimises the
    matched distances. This mirrors the automated best-fit alignment used in
    dental metrology before computing surface deviation.
    """
    _require_kdtree()
    src = _as_points(source)
    tgt = _as_points(target)
    tree = _KDTree(tgt)

    if init is not None:
        R, t = init
        cur = src @ np.asarray(R).T + np.asarray(t)
    else:
        R = np.eye(3)
        t = np.zeros(3)
        cur = src.copy()

    prev_err = np.inf
    iters = 0
    for iters in range(1, max_iterations + 1):
        dist, idx = tree.query(cur, k=1)
        matched = tgt[idx]
        Rd, td = best_fit_transform(cur, matched)
        cur = cur @ Rd.T + td
        # accumulate transform (compose Rd,td onto running R,t)
        R = Rd @ R
        t = Rd @ t + td
        err = float(np.sqrt(np.mean(dist ** 2)))
        if abs(prev_err - err) < tolerance:
            break
        prev_err = err

    dist, _ = tree.query(cur, k=1)
    rms = float(np.sqrt(np.mean(dist ** 2)))
    return ICPResult(R=R, t=t, aligned=cur, rms=rms, iterations=iters)


# --------------------------------------------------------------------------- #
# Surface deviation
# --------------------------------------------------------------------------- #
@dataclass
class DeviationResult:
    """Summary of nearest-neighbour distances from a test to a reference set."""

    mean: float
    rms: float
    p90: float
    p95: float
    hausdorff: float
    median: float
    std: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)

    def within(self, threshold: float) -> float:
        """Fraction of points within ``threshold`` (set after construction)."""
        raise NotImplementedError(
            "call surface_deviation(..., return_distances=True) for per-point data"
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"mean={self.mean:.4f} rms={self.rms:.4f} median={self.median:.4f} "
            f"p90={self.p90:.4f} p95={self.p95:.4f} hausdorff={self.hausdorff:.4f} "
            f"(n={self.n})"
        )


def surface_deviation(
    test: np.ndarray,
    reference: np.ndarray,
    align: bool = True,
    symmetric: bool = True,
    return_distances: bool = False,
):
    """Nearest-neighbour surface deviation of ``test`` against ``reference``.

    Args:
        test: (N, 3) reconstructed/predicted points (e.g. sampled from the mesh).
        reference: (M, 3) ground-truth points (e.g. IOS scan).
        align: run best-fit ICP first (set False if already registered).
        symmetric: report the symmetric distance (max of test->ref and
            ref->test nearest distances) which is the honest measure for
            reconstruction completeness; if False, only test->ref is used.
        return_distances: also return the per-point distance array.

    Returns:
        DeviationResult, or (DeviationResult, distances) if return_distances.
    """
    _require_kdtree()
    test_pts = _as_points(test)
    ref_pts = _as_points(reference)

    if align:
        icp = iterative_closest_point(test_pts, ref_pts)
        test_pts = icp.aligned

    ref_tree = _KDTree(ref_pts)
    d_tr, _ = ref_tree.query(test_pts, k=1)

    if symmetric:
        test_tree = _KDTree(test_pts)
        d_rt, _ = test_tree.query(ref_pts, k=1)
        distances = np.concatenate([d_tr, d_rt])
    else:
        distances = d_tr

    result = DeviationResult(
        mean=float(np.mean(distances)),
        rms=float(np.sqrt(np.mean(distances ** 2))),
        p90=float(np.percentile(distances, 90)),
        p95=float(np.percentile(distances, 95)),
        hausdorff=float(np.max(distances)),
        median=float(np.median(distances)),
        std=float(np.std(distances)),
        n=int(distances.size),
    )
    if return_distances:
        return result, distances
    return result


def fraction_within(distances: np.ndarray, threshold: float) -> float:
    """Fraction of per-point distances at or below ``threshold`` (e.g. 0.5 mm)."""
    d = np.asarray(distances, dtype=np.float64)
    if d.size == 0:
        return 0.0
    return float(np.mean(d <= threshold))


# --------------------------------------------------------------------------- #
# Trueness & precision (ISO 5725)
# --------------------------------------------------------------------------- #
def trueness(test: np.ndarray, ground_truth: np.ndarray, **kwargs) -> DeviationResult:
    """Trueness = deviation of a single reconstruction from the true geometry.

    Thin wrapper over :func:`surface_deviation` to make intent explicit.
    """
    return surface_deviation(test, ground_truth, **kwargs)


def precision(scans, align: bool = True) -> DeviationResult:
    """Precision = spread among repeated scans of the same object.

    Computes pairwise symmetric surface deviation between all repeated scans and
    aggregates them. ``scans`` is a list of (N, 3) point arrays. Lower is more
    reproducible. (No ground truth needed — that is the whole point of ISO
    precision: reproducibility, not closeness to truth.)
    """
    scans = [_as_points(s) for s in scans]
    if len(scans) < 2:
        raise ValueError("precision requires at least two repeated scans")

    all_d = []
    for i in range(len(scans)):
        for j in range(i + 1, len(scans)):
            _, d = surface_deviation(
                scans[i], scans[j], align=align, symmetric=True, return_distances=True
            )
            all_d.append(d)
    distances = np.concatenate(all_d)
    return DeviationResult(
        mean=float(np.mean(distances)),
        rms=float(np.sqrt(np.mean(distances ** 2))),
        p90=float(np.percentile(distances, 90)),
        p95=float(np.percentile(distances, 95)),
        hausdorff=float(np.max(distances)),
        median=float(np.median(distances)),
        std=float(np.std(distances)),
        n=int(distances.size),
    )
