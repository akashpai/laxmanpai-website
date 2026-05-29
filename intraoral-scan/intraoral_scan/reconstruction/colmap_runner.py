"""Visible-surface reconstruction wrappers (COLMAP SfM+MVS, and a 3DGS hook).

This produces the metric-scaled reconstruction of the *visible buccal surfaces*
from the extracted keyframes. It wraps external engines rather than
reimplementing them:

  * COLMAP for classic Structure-from-Motion + Multi-View Stereo. Robust,
    well-understood, but fragile on textureless/specular enamel (COLMAP's own
    docs warn against shiny, low-texture surfaces).
  * 3D Gaussian Splatting / NeuS-style neural surfaces as an alternative path
    for the specular case (see docs/RESEARCH_REPORT.md: Dental3R, DentalSplat).

The reconstruction is UP TO SCALE; call calibration.scale to make it metric
before any accuracy evaluation.

This module shells out to the `colmap` binary if present. If it is not
installed, methods raise a clear error — the rest of the package (SSM,
validation, calibration geometry) does not depend on COLMAP being installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReconstructionResult:
    sparse_dir: str
    dense_ply: Optional[str]
    n_registered_images: int
    log: List[str] = field(default_factory=list)
    is_metric: bool = False


class ColmapReconstructor:
    """Thin, well-documented wrapper around the COLMAP CLI."""

    def __init__(self, colmap_bin: str = "colmap", use_gpu: bool = True):
        self.colmap_bin = colmap_bin
        self.use_gpu = use_gpu

    def available(self) -> bool:
        return shutil.which(self.colmap_bin) is not None

    def _run(self, args: List[str], log: List[str]) -> None:
        if not self.available():
            raise RuntimeError(
                f"COLMAP binary '{self.colmap_bin}' not found on PATH. Install "
                "COLMAP (https://colmap.github.io/) to run reconstruction, or "
                "supply a pre-computed point cloud to the pipeline."
            )
        cmd = [self.colmap_bin] + args
        log.append(" ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log.append(proc.stdout)
        if proc.returncode != 0:
            log.append(proc.stderr)
            raise RuntimeError(f"COLMAP step failed: {' '.join(args[:2])}\n{proc.stderr}")

    def reconstruct(
        self,
        image_dir: str,
        workspace: str,
        dense: bool = True,
    ) -> ReconstructionResult:
        """Run feature extraction -> matching -> mapping -> (optional) dense MVS.

        Returns a ReconstructionResult. The result is NOT metric yet.
        """
        os.makedirs(workspace, exist_ok=True)
        db_path = os.path.join(workspace, "database.db")
        sparse_dir = os.path.join(workspace, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        log: List[str] = []

        gpu = "1" if self.use_gpu else "0"

        # 1. feature extraction (SINGLE shared camera model for one phone)
        self._run([
            "feature_extractor",
            "--database_path", db_path,
            "--image_path", image_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV",
            "--SiftExtraction.use_gpu", gpu,
        ], log)

        # 2. exhaustive matching (fine for the few-hundred frames we keep)
        self._run([
            "exhaustive_matcher",
            "--database_path", db_path,
            "--SiftMatching.use_gpu", gpu,
        ], log)

        # 3. incremental mapping (sparse SfM)
        self._run([
            "mapper",
            "--database_path", db_path,
            "--image_path", image_dir,
            "--output_path", sparse_dir,
        ], log)

        dense_ply = None
        if dense:
            dense_dir = os.path.join(workspace, "dense")
            os.makedirs(dense_dir, exist_ok=True)
            model0 = os.path.join(sparse_dir, "0")
            self._run([
                "image_undistorter",
                "--image_path", image_dir,
                "--input_path", model0,
                "--output_path", dense_dir,
                "--output_type", "COLMAP",
            ], log)
            self._run([
                "patch_match_stereo",
                "--workspace_path", dense_dir,
                "--PatchMatchStereo.geom_consistency", "true",
            ], log)
            dense_ply = os.path.join(dense_dir, "fused.ply")
            self._run([
                "stereo_fusion",
                "--workspace_path", dense_dir,
                "--output_path", dense_ply,
            ], log)

        n_images = self._count_registered(os.path.join(sparse_dir, "0"))
        return ReconstructionResult(
            sparse_dir=sparse_dir,
            dense_ply=dense_ply,
            n_registered_images=n_images,
            log=log,
            is_metric=False,
        )

    @staticmethod
    def _count_registered(model_dir: str) -> int:
        images_txt = os.path.join(model_dir, "images.txt")
        if os.path.exists(images_txt):
            with open(images_txt) as fh:
                # each registered image = 2 non-comment lines
                lines = [ln for ln in fh if not ln.startswith("#") and ln.strip()]
            return len(lines) // 2
        return 0
