"""End-to-end demo on synthetic data (no COLMAP / OpenCV / GPU needed).

Simulates the core scientific claim of the project:
  * we build a statistical shape model of a (synthetic) tooth population,
  * "observe" only the buccal half of a held-out tooth (as a phone would),
  * predict the unseen lingual/occlusal half with the SSM,
  * validate the completed shape against the (known) ground truth, per region.

Run:  python scripts/demo_completion.py
"""

import numpy as np

from intraoral_scan.completion.ssm import StatisticalShapeModel
from intraoral_scan.config import PipelineConfig
from intraoral_scan.pipeline import (
    complete_hidden_surfaces,
    validate_against_ground_truth,
)

N = 80
K_TRUE = 6


def make_population(seed=0, n_shapes=300):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(N, 3)) * 3.0
    M = rng.normal(size=(K_TRUE, N * 3))
    Q, _ = np.linalg.qr(M.T)
    modes = Q.T[:K_TRUE]
    std = np.linspace(1.2, 0.2, K_TRUE)
    shapes = []
    for _ in range(n_shapes):
        c = rng.normal(size=K_TRUE) * std
        shapes.append((base.reshape(-1) + c @ modes).reshape(-1, 3))
    held_out_c = rng.normal(size=K_TRUE) * std
    held_out = (base.reshape(-1) + held_out_c @ modes).reshape(-1, 3)
    return shapes, held_out


def main():
    cfg = PipelineConfig()
    cfg.completion.noise_sigma_mm = 0.05
    cfg.completion.uncertainty_flag_mm = 0.20

    shapes, ground_truth = make_population()
    model = StatisticalShapeModel.train(shapes, variance_threshold=0.999)
    print(f"Trained SSM: {model.n_components} modes over {model.n_vertices} vertices")

    # the camera sees the buccal half only
    buccal = np.arange(0, N // 2)
    hidden = np.arange(N // 2, N)

    # Inject *patient-specific* high-frequency anatomy on the hidden
    # (lingual/occlusal) surfaces that the population prior cannot possibly know
    # — this is the irreducible per-individual variance the literature describes.
    detail = np.zeros((N, 3))
    detail[hidden] = np.random.default_rng(2).normal(scale=0.18, size=(hidden.size, 3))
    ground_truth = ground_truth + detail

    observed_points = ground_truth[buccal] + np.random.default_rng(1).normal(
        scale=cfg.completion.noise_sigma_mm, size=(buccal.size, 3)
    )

    report = complete_hidden_surfaces(model, buccal, observed_points, cfg)
    print(
        f"Completion: observed {report.n_observed} vertices, "
        f"predicted {report.n_predicted}; "
        f"mean hidden uncertainty {report.mean_hidden_uncertainty_mm:.3f} mm; "
        f"{report.flagged_low_confidence} vertices flagged low-confidence"
    )

    buccal_mask = np.zeros(N, bool); buccal_mask[buccal] = True
    hidden_mask = np.zeros(N, bool); hidden_mask[hidden] = True

    val = validate_against_ground_truth(
        report.fit.full_shape, ground_truth, cfg, buccal_mask, hidden_mask
    )
    print("\nValidation vs ground truth (synthetic, units ~ mm):")
    print(f"  overall : {val.overall}")
    print(f"  buccal  : {val.buccal}")
    print(f"  hidden  : {val.hidden}")
    print(f"  fraction within aligner thr ({cfg.validation.aligner_threshold_mm} mm): "
          f"{val.frac_within_aligner:.3f}")
    print(f"  fraction within restoration thr ({cfg.validation.restoration_threshold_mm} mm): "
          f"{val.frac_within_restoration:.3f}")

    print(
        "\nTakeaway: buccal (measured) error << hidden (predicted) error — exactly "
        "the asymmetry to expect on real data. The hidden surface is a prediction, "
        "not a measurement, and is flagged as such."
    )


if __name__ == "__main__":
    main()
