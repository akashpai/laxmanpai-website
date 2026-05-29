"""Stable interface for tooth segmentation models.

Implement ``ToothSegmenter`` with your chosen backbone (MeshSegNet / iMeshSegNet,
TeethGNN, DilatedToothSegNet, a Point Transformer, etc.) trained on Teeth3DS.
The rest of the pipeline only depends on this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class SegmentationResult:
    """Per-vertex tooth labels for a mesh/point cloud.

    Attributes:
        labels: (N,) FDI two-digit numbers per vertex; 0 = gingiva.
        instances: (N,) tooth instance ids (separates adjacent same-type teeth).
        surface_class: (N,) one of {0:buccal, 1:lingual, 2:occlusal, 3:other}.
            This is the key field for completion: buccal = observed by the
            camera, lingual/occlusal = must be predicted by the SSM.
    """

    labels: np.ndarray
    instances: np.ndarray
    surface_class: np.ndarray

    def buccal_mask(self) -> np.ndarray:
        return self.surface_class == 0

    def hidden_mask(self) -> np.ndarray:
        return np.isin(self.surface_class, [1, 2])


# FDI numbering helpers ------------------------------------------------------ #
FDI_QUADRANTS = {1: "upper-right", 2: "upper-left", 3: "lower-left", 4: "lower-right"}


def fdi_is_molar(fdi: int) -> bool:
    """Molars are positions 6,7,8 in each quadrant (e.g. 16,17,18,...)."""
    return fdi % 10 in (6, 7, 8) and fdi // 10 in (1, 2, 3, 4)


class ToothSegmenter(Protocol):
    def segment(self, vertices: np.ndarray, faces: np.ndarray | None = None) -> SegmentationResult:
        """Return per-vertex FDI labels, instances, and surface classes."""
        ...
