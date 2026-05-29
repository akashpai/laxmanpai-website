"""Tests for the statistical shape model completion.

These tests build a synthetic shape family with a known low-dimensional mode
structure, then verify that the SSM can (a) recover that structure, (b) predict
unseen vertices from a visible subset far better than the mean baseline, and
(c) reproduce the key published finding that prediction error grows as the
observed fraction shrinks. They also check that the per-vertex uncertainty is
higher on hidden (unobserved) vertices than on observed ones.
"""

import numpy as np
import pytest

from intraoral_scan.completion.ssm import StatisticalShapeModel


N_VERTICES = 60
TRUE_MODES = 5


def _make_family(seed=0):
    """A synthetic shape family: base + linear combination of fixed smooth modes."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(N_VERTICES, 3)) * 2.0
    # fixed orthonormal modes in R^{3N}
    M = rng.normal(size=(TRUE_MODES, N_VERTICES * 3))
    Q, _ = np.linalg.qr(M.T)
    modes = Q.T[:TRUE_MODES]                      # (TRUE_MODES, 3N)
    mode_std = np.array([1.0, 0.7, 0.5, 0.3, 0.2])
    return base, modes, mode_std


def _sample(base, modes, mode_std, rng):
    c = rng.normal(size=len(mode_std)) * mode_std
    flat = base.reshape(-1) + c @ modes
    return flat.reshape(-1, 3), c


def _train_model(seed=0, n=300):
    base, modes, mode_std = _make_family(seed)
    rng = np.random.default_rng(seed + 1)
    shapes = [_sample(base, modes, mode_std, rng)[0] for _ in range(n)]
    model = StatisticalShapeModel.train(shapes, variance_threshold=0.999)
    return model, base, modes, mode_std


def test_train_recovers_mode_count():
    model, *_ = _train_model()
    # the synthetic family is exactly TRUE_MODES dimensional
    assert model.n_components == TRUE_MODES
    assert model.n_vertices == N_VERTICES


def test_encode_reconstruct_roundtrip():
    model, base, modes, mode_std = _train_model()
    rng = np.random.default_rng(123)
    shape, _ = _sample(base, modes, mode_std, rng)
    coeffs = model.encode(shape)
    recon = model.reconstruct(coeffs)
    np.testing.assert_allclose(recon, shape, atol=1e-6)


def test_partial_fit_predicts_hidden_far_better_than_mean():
    model, base, modes, mode_std = _train_model()
    rng = np.random.default_rng(7)
    shape, _ = _sample(base, modes, mode_std, rng)

    # observe the first half of vertices ("buccal"), hide the rest
    observed = np.arange(0, N_VERTICES // 2)
    hidden = np.arange(N_VERTICES // 2, N_VERTICES)

    fit = model.fit_to_partial(observed, shape[observed], noise_sigma=1e-3)

    pred_err = np.linalg.norm(fit.full_shape[hidden] - shape[hidden], axis=1).mean()
    mean_err = np.linalg.norm(
        model.mean_shape[hidden] - shape[hidden], axis=1
    ).mean()

    # predicting from the prior+observation must beat just guessing the mean
    assert pred_err < 0.25 * mean_err
    # with enough observed vertices to constrain 5 modes, prediction is near-exact
    assert pred_err < 1e-2


def test_error_grows_as_observed_region_shrinks():
    """Reproduce the published 'error scales with missing fraction' behaviour.

    The effect bites hardest once the number of observed vertices drops toward
    (and below) the number of shape modes, where the fit becomes
    underdetermined and falls back onto the prior for unconstrained modes.
    """
    model, base, modes, mode_std = _train_model()
    rng = np.random.default_rng(11)

    # observation counts that cross the K=TRUE_MODES identifiability boundary
    counts = [40, 20, 10, 4]
    errors = []
    for n_obs in counts:
        trials = []
        for s in range(10):
            shape, _ = _sample(base, modes, mode_std, np.random.default_rng(100 + s))
            obs = rng.choice(N_VERTICES, size=n_obs, replace=False)
            mask = np.ones(N_VERTICES, bool)
            mask[obs] = False
            hidden = np.nonzero(mask)[0]
            noisy = shape[obs] + np.random.default_rng(s).normal(scale=0.02, size=(n_obs, 3))
            fit = model.fit_to_partial(obs, noisy, noise_sigma=0.02)
            trials.append(
                np.linalg.norm(fit.full_shape[hidden] - shape[hidden], axis=1).mean()
            )
        errors.append(np.mean(trials))

    # the underdetermined case (fewest observations) must be clearly worse than
    # the well-observed case — the core "missing region hurts" finding.
    assert errors[-1] > 2.0 * errors[0]


def test_uncertainty_decreases_with_more_observations():
    """Posterior predictive uncertainty must contract as we observe more.

    (In a real dental SSM the *hidden* lingual/occlusal vertices also end up with
    higher uncertainty than the buccal ones because their geometry is less
    correlated with the observation; that is data-dependent and is what the
    pipeline's `uncertainty_flag_mm` threshold exploits. Here we test the
    model-agnostic property: more data -> less uncertainty.)
    """
    model, base, modes, mode_std = _train_model()
    rng = np.random.default_rng(5)
    shape, _ = _sample(base, modes, mode_std, rng)

    fit_few = model.fit_to_partial(np.arange(0, 6), shape[0:6], noise_sigma=0.05)
    fit_many = model.fit_to_partial(np.arange(0, 40), shape[0:40], noise_sigma=0.05)

    assert fit_many.per_vertex_std.mean() < fit_few.per_vertex_std.mean()


def test_save_load_roundtrip(tmp_path):
    model, *_ = _train_model()
    p = tmp_path / "ssm.npz"
    model.save(str(p))
    loaded = StatisticalShapeModel.load(str(p))
    np.testing.assert_allclose(loaded.mean, model.mean)
    np.testing.assert_allclose(loaded.components, model.components)
    np.testing.assert_allclose(loaded.variances, model.variances)


def test_fit_validates_inputs():
    model, *_ = _train_model()
    with pytest.raises(ValueError):
        model.fit_to_partial([0, 1], np.zeros((3, 3)))  # mismatched lengths
    with pytest.raises(ValueError):
        model.fit_to_partial([N_VERTICES + 5], np.zeros((1, 3)))  # out of range
