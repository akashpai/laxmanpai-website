import numpy as np
import pytest

from intraoral_scan.validation.metrics import (
    best_fit_transform,
    iterative_closest_point,
    surface_deviation,
    fraction_within,
    trueness,
    precision,
)


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_best_fit_transform_recovers_known_rigid():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    R = _random_rotation(rng)
    t = np.array([1.0, -2.0, 0.5])
    tgt = src @ R.T + t
    R_est, t_est = best_fit_transform(src, tgt)
    np.testing.assert_allclose(R_est, R, atol=1e-9)
    np.testing.assert_allclose(t_est, t, atol=1e-9)


def _small_rotation(angle_rad, axis):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = np.cos(angle_rad), np.sin(angle_rad), 1 - np.cos(angle_rad)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def test_icp_aligns_with_modest_misalignment():
    # ICP is a local optimiser; the realistic dental case is a small residual
    # misalignment after a coarse pre-registration, which it resolves exactly.
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(200, 3))
    R = _small_rotation(np.deg2rad(8), np.array([0.2, 1.0, -0.3]))
    t = np.array([0.05, 0.1, -0.02])
    moved = pts @ R.T + t
    res = iterative_closest_point(moved, pts)
    assert res.rms < 1e-6


def test_surface_deviation_zero_for_identical():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(100, 3))
    res = surface_deviation(pts, pts, align=True)
    assert res.mean < 1e-9
    assert res.rms < 1e-9
    assert res.hausdorff < 1e-9


def test_surface_deviation_constant_offset():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(150, 3))
    # a pure rigid offset should be removed by alignment -> ~0 deviation
    shifted = pts + np.array([5.0, 0.0, 0.0])
    res = surface_deviation(shifted, pts, align=True)
    assert res.mean < 1e-6


def test_fraction_within():
    d = np.array([0.1, 0.2, 0.6, 0.7])
    assert fraction_within(d, 0.5) == pytest.approx(0.5)


def test_precision_requires_two_scans():
    with pytest.raises(ValueError):
        precision([np.zeros((10, 3))])


def test_precision_zero_for_identical_scans():
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(80, 3))
    res = precision([pts, pts.copy(), pts.copy()])
    assert res.rms < 1e-9


def test_trueness_is_surface_deviation():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(60, 3))
    r1 = trueness(a, a)
    r2 = surface_deviation(a, a)
    assert r1.rms == pytest.approx(r2.rms)
