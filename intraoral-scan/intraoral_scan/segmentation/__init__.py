"""Tooth segmentation + FDI numbering (interface).

On clean intraoral meshes this problem is essentially solved in the literature
(Dice ~0.96-0.98 with iMeshSegNet / TeethGNN / transformer models trained on
Teeth3DS). We therefore define a stable interface here and treat the model as a
pluggable component, rather than reimplementing a mesh GNN.

The segmentation step assigns every vertex a tooth instance + FDI label, which
is what the SSM completion needs to (a) put each tooth in correspondence with
the per-tooth shape priors and (b) know which vertices are buccal (observed) vs
lingual/occlusal (to be predicted).
"""

from .interface import ToothSegmenter, SegmentationResult

__all__ = ["ToothSegmenter", "SegmentationResult"]
