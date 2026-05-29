"""Configuration objects for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CaptureConfig:
    sample_every: int = 5
    min_sharpness: float = 100.0
    max_frames: int = 200
    # metric reference
    aruco_marker_length_mm: float = 5.0
    aruco_dictionary: str = "DICT_5X5_50"


@dataclass
class ReconstructionConfig:
    engine: str = "colmap"          # "colmap" | "gaussian_splatting"
    dense: bool = True
    use_gpu: bool = True


@dataclass
class CompletionConfig:
    model_path: str = ""            # path to a trained SSM (.npz)
    # observation noise = your measured buccal reconstruction error, in mm.
    # Larger -> trust the prior more for hidden surfaces.
    noise_sigma_mm: float = 0.1
    # vertices whose predictive std exceeds this are flagged low-confidence
    uncertainty_flag_mm: float = 0.25


@dataclass
class ValidationConfig:
    align: bool = True
    symmetric: bool = True
    # clinical thresholds for pass/fail reporting (mm)
    aligner_threshold_mm: float = 0.25
    restoration_threshold_mm: float = 0.12


@dataclass
class PipelineConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    workspace: str = "./workspace"
