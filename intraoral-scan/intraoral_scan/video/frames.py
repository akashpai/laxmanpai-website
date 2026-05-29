"""Frame extraction and sharpness selection from smartphone video.

Photogrammetry / 3DGS quality is dominated by input image quality. Intraoral
video is plagued by motion blur, saliva glare, and shallow depth of field, so we
do NOT use every frame: we select sharp, well-spaced keyframes.

Sharpness is scored by the variance of the Laplacian (a standard blur metric):
sharp images have high-frequency content -> high Laplacian variance; blurred
frames are smooth -> low variance.

OpenCV is required for video decoding. The scoring function is exposed
separately so it can be unit-tested on synthetic images.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np


def sharpness_score(gray_image: np.ndarray) -> float:
    """Variance-of-Laplacian sharpness. Higher = sharper.

    Pure-numpy 3x3 Laplacian so this works without OpenCV (used by tests).
    """
    img = np.asarray(gray_image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("sharpness_score expects a 2D grayscale image")
    # discrete Laplacian via convolution with the 4-neighbour kernel
    lap = (
        -4.0 * img
        + np.roll(img, 1, axis=0)
        + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1)
        + np.roll(img, -1, axis=1)
    )
    # ignore the 1px border affected by wrap-around
    inner = lap[1:-1, 1:-1]
    return float(inner.var())


def _import_cv2():
    try:
        import cv2  # noqa: F401

        return cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV (opencv-python) is required for video decoding. Install with "
            "`pip install opencv-python`."
        ) from exc


def extract_sharp_frames(
    video_path: str,
    out_dir: str,
    sample_every: int = 5,
    min_sharpness: float = 100.0,
    max_frames: int = 200,
) -> List[str]:
    """Extract sharp keyframes from a video to ``out_dir``.

    Args:
        video_path: path to the smartphone video.
        out_dir: directory to write selected JPEG frames.
        sample_every: consider every Nth frame (decimation before scoring).
        min_sharpness: drop frames whose variance-of-Laplacian is below this.
        max_frames: cap on the number of saved frames (keeps the sharpest,
            evenly distributed across the clip).

    Returns:
        list of written file paths.
    """
    cv2 = _import_cv2()
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    candidates = []  # (frame_index, sharpness, bgr_frame)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                s = sharpness_score(gray)
                if s >= min_sharpness:
                    candidates.append((idx, s, frame))
            idx += 1
    finally:
        cap.release()

    if not candidates:
        return []

    # keep the sharpest max_frames, but preserve temporal coverage by binning
    candidates.sort(key=lambda c: c[0])
    if len(candidates) > max_frames:
        bins = np.array_split(candidates, max_frames)
        candidates = [max(b, key=lambda c: c[1]) for b in bins if len(b)]

    written = []
    for n, (frame_idx, _s, frame) in enumerate(candidates):
        path = os.path.join(out_dir, f"frame_{n:04d}_src{frame_idx:06d}.jpg")
        cv2.imwrite(path, frame)
        written.append(path)
    return written
