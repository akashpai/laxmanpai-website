"""Intraoral scan from smartphone video — research prototype.

Pipeline: smartphone video (buccal/facial surfaces only) -> metric-scaled
reconstruction of the *visible* surfaces -> statistical-shape-model completion
of the *unseen* lingual/occlusal surfaces -> validation against ground-truth IOS.

See docs/RESEARCH_REPORT.md for the literature basis and the honest accuracy
verdict, and docs/ROADMAP.md for the phased implementation plan.
"""

__version__ = "0.1.0"

from .completion.ssm import StatisticalShapeModel
from .validation.metrics import (
    surface_deviation,
    iterative_closest_point,
    trueness,
    precision,
)

__all__ = [
    "StatisticalShapeModel",
    "surface_deviation",
    "iterative_closest_point",
    "trueness",
    "precision",
]
