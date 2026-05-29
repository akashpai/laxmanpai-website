"""End-to-end orchestration: video -> metric visible surface -> completion -> validation.

This wires the stages together with explicit, inspectable hand-offs. Stages that
require heavy external engines (COLMAP, a trained segmenter, a trained SSM) are
invoked through their wrappers/interfaces, so the orchestration logic itself is
importable and testable without those engines installed.

Stage map (see docs/ROADMAP.md):
    1. capture     : extract sharp keyframes from the video
    2. reconstruct : SfM+MVS -> point cloud of the VISIBLE buccal surfaces
    3. scale       : make the reconstruction metric (mm) via a known reference
    4. segment     : per-vertex FDI label + buccal/lingual/occlusal class
    5. complete    : fit the SSM to the buccal vertices, predict hidden surfaces
    6. validate    : surface deviation vs ground-truth IOS, per region
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PipelineConfig
from .completion.ssm import StatisticalShapeModel, PartialFitResult
from .validation.metrics import surface_deviation, DeviationResult, fraction_within


@dataclass
class CompletionReport:
    fit: PartialFitResult
    n_observed: int
    n_predicted: int
    mean_hidden_uncertainty_mm: float
    flagged_low_confidence: int


def complete_hidden_surfaces(
    model: StatisticalShapeModel,
    observed_indices: np.ndarray,
    observed_points: np.ndarray,
    config: PipelineConfig,
) -> CompletionReport:
    """Stage 5: predict the unseen lingual/occlusal vertices from the buccal ones.

    The returned report includes per-vertex predictive uncertainty and a count of
    vertices flagged as low-confidence (predictive std above the configured
    threshold) — these are the regions the camera never saw and that must NOT be
    presented as measured geometry.
    """
    fit = model.fit_to_partial(
        observed_indices=observed_indices,
        observed_points=observed_points,
        noise_sigma=config.completion.noise_sigma_mm,
    )
    hidden = fit.hidden_indices
    hidden_unc = fit.per_vertex_std[hidden] if hidden.size else np.array([0.0])
    flagged = int(np.sum(fit.per_vertex_std > config.completion.uncertainty_flag_mm))
    return CompletionReport(
        fit=fit,
        n_observed=int(observed_indices.size),
        n_predicted=int(hidden.size),
        mean_hidden_uncertainty_mm=float(np.mean(hidden_unc)),
        flagged_low_confidence=flagged,
    )


@dataclass
class RegionValidation:
    overall: DeviationResult
    buccal: Optional[DeviationResult]
    hidden: Optional[DeviationResult]
    frac_within_aligner: float
    frac_within_restoration: float


def validate_against_ground_truth(
    predicted_shape: np.ndarray,
    ground_truth: np.ndarray,
    config: PipelineConfig,
    buccal_mask: Optional[np.ndarray] = None,
    hidden_mask: Optional[np.ndarray] = None,
) -> RegionValidation:
    """Stage 6: validate the completed shape against a ground-truth IOS scan.

    Reports overall deviation plus, if masks are supplied, the *per-region*
    breakdown — which is essential because the honest story is "buccal is
    measured and accurate; lingual/occlusal is predicted and looser".
    """
    overall, dist = surface_deviation(
        predicted_shape,
        ground_truth,
        align=config.validation.align,
        symmetric=config.validation.symmetric,
        return_distances=True,
    )

    def region(mask):
        if mask is None or not np.any(mask):
            return None
        return surface_deviation(
            predicted_shape[mask], ground_truth, align=False, symmetric=False
        )

    # region breakdown uses test->ref distances on the already-aligned points;
    # re-run alignment once, then evaluate masks without re-aligning.
    return RegionValidation(
        overall=overall,
        buccal=region(buccal_mask),
        hidden=region(hidden_mask),
        frac_within_aligner=fraction_within(dist, config.validation.aligner_threshold_mm),
        frac_within_restoration=fraction_within(dist, config.validation.restoration_threshold_mm),
    )
