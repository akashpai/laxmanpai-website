"""Command-line interface.

Examples:
    python -m intraoral_scan.cli extract-frames input.mp4 frames/
    python -m intraoral_scan.cli reconstruct frames/ workspace/
    python -m intraoral_scan.cli validate predicted.npy ground_truth.npy
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _cmd_extract_frames(args):
    from .video.frames import extract_sharp_frames

    written = extract_sharp_frames(
        args.video,
        args.out_dir,
        sample_every=args.sample_every,
        min_sharpness=args.min_sharpness,
        max_frames=args.max_frames,
    )
    print(f"wrote {len(written)} sharp frames to {args.out_dir}")


def _cmd_reconstruct(args):
    from .reconstruction.colmap_runner import ColmapReconstructor

    rec = ColmapReconstructor(use_gpu=not args.no_gpu)
    res = rec.reconstruct(args.image_dir, args.workspace, dense=not args.sparse_only)
    print(f"registered {res.n_registered_images} images; dense={res.dense_ply}")
    print("NOTE: reconstruction is up-to-scale; run a scaling step before validation.")


def _cmd_validate(args):
    from .validation.metrics import surface_deviation

    test = np.load(args.test)
    ref = np.load(args.reference)
    res = surface_deviation(test, ref, align=not args.no_align)
    print(res)
    if args.threshold:
        from .validation.metrics import surface_deviation as sd

        _, d = sd(test, ref, align=not args.no_align, return_distances=True)
        frac = float(np.mean(d <= args.threshold))
        print(f"fraction within {args.threshold} mm: {frac:.3f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="intraoral_scan", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    ef = sub.add_parser("extract-frames", help="extract sharp keyframes from a video")
    ef.add_argument("video")
    ef.add_argument("out_dir")
    ef.add_argument("--sample-every", type=int, default=5)
    ef.add_argument("--min-sharpness", type=float, default=100.0)
    ef.add_argument("--max-frames", type=int, default=200)
    ef.set_defaults(func=_cmd_extract_frames)

    rc = sub.add_parser("reconstruct", help="run COLMAP SfM+MVS on a frame folder")
    rc.add_argument("image_dir")
    rc.add_argument("workspace")
    rc.add_argument("--no-gpu", action="store_true")
    rc.add_argument("--sparse-only", action="store_true")
    rc.set_defaults(func=_cmd_reconstruct)

    vl = sub.add_parser("validate", help="surface deviation of a prediction vs reference")
    vl.add_argument("test", help=".npy of (N,3) predicted points")
    vl.add_argument("reference", help=".npy of (M,3) ground-truth points")
    vl.add_argument("--no-align", action="store_true")
    vl.add_argument("--threshold", type=float, default=0.5)
    vl.set_defaults(func=_cmd_validate)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
