"""Statistical Shape Model (SSM) for completing the *unseen* tooth surfaces.

This is the core scientific component of the project. A smartphone video, even
with a retractor, only ever sees the **buccal/facial** surfaces of the teeth.
The **lingual/palatal** surfaces and the **occlusal** anatomy of molars are
never imaged. They cannot be measured, so they must be *predicted* from a prior
learned over a population of real full-arch intraoral scans.

The principled tool for this is a linear statistical shape model (PCA over a
set of shapes in dense correspondence), fit to the partial (visible) observation
under a Gaussian prior. This is the dental analogue of 3D Morphable Models for
faces, and it is exactly how published work reconstructs unseen tooth roots from
visible crowns (Buchaillard 2007; root-from-crown SSMs).

Honest accuracy expectation (see docs/RESEARCH_REPORT.md for citations):
  * The prediction of an unseen surface is bounded by the *irreducible
    population variance* of that surface conditioned on the visible part.
  * Published evidence: reconstruction error scales roughly *linearly* with the
    fraction of missing area, and jumps ~6x when the missing region includes the
    surface boundary (which the buccal-only case does for the occlusal/lingual).
  * Realistic floor for predicted occlusal/lingual surfaces: ~0.1-0.25 mm,
    i.e. 2-5x the ~0.05 mm clinical IOS bar. Good enough for visualisation /
    orthodontic monitoring; NOT good enough for restoration margins.
This module therefore also reports a *per-vertex predictive uncertainty* so the
pipeline can flag low-confidence (hidden) regions rather than pretend they are
measured.

Model
-----
A shape is N vertices in dense correspondence, flattened to a vector
x in R^{3N} ordered [x0,y0,z0, x1,y1,z1, ...].

    x ~= mean + Phi @ b

where Phi (3N x K) are the principal components (unit eigenvectors of the
training covariance) and b (K,) are shape coefficients with prior b_k ~ N(0,
lambda_k), lambda_k the k-th eigenvalue (variance along that mode).

Given a partial observation y of a subset S of vertices, the MAP estimate of b
solves a ridge-regularised least squares problem:

    b* = argmin_b  ||A b - (y - mean_S)||^2 / sigma^2  +  sum_k b_k^2 / lambda_k

with A = Phi restricted to the observed coordinate rows, and sigma the
observation noise (the buccal reconstruction error). The closed-form solution is

    b* = (A^T A / sigma^2 + diag(1/lambda))^{-1}  A^T (y - mean_S) / sigma^2

The completed shape is mean + Phi b*, from which the unseen vertices are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class PartialFitResult:
    """Result of fitting the SSM to a partial (visible-only) observation."""

    coefficients: np.ndarray          # (K,) shape coefficients b*
    full_shape: np.ndarray            # (N, 3) completed shape
    per_vertex_std: np.ndarray        # (N,) predictive std-dev per vertex (mm)
    observed_indices: np.ndarray      # vertex indices that were observed
    residual_rms: float               # fit residual on observed vertices

    @property
    def hidden_indices(self) -> np.ndarray:
        mask = np.ones(self.full_shape.shape[0], dtype=bool)
        mask[self.observed_indices] = False
        return np.nonzero(mask)[0]


class StatisticalShapeModel:
    """Linear (PCA) statistical shape model over shapes in dense correspondence."""

    def __init__(self, mean: np.ndarray, components: np.ndarray, variances: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)          # (3N,)
        self.components = np.asarray(components, dtype=np.float64)  # (K, 3N)
        self.variances = np.asarray(variances, dtype=np.float64)   # (K,)
        if self.mean.ndim != 1 or self.mean.size % 3 != 0:
            raise ValueError("mean must be a flat vector of length 3N")
        if self.components.shape[1] != self.mean.size:
            raise ValueError("components must have shape (K, 3N)")
        if self.variances.shape[0] != self.components.shape[0]:
            raise ValueError("variances must have one entry per component")

    # ------------------------------------------------------------------ #
    @property
    def n_vertices(self) -> int:
        return self.mean.size // 3

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    @property
    def mean_shape(self) -> np.ndarray:
        return self.mean.reshape(-1, 3)

    # ------------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        shapes: Sequence[np.ndarray],
        n_components: Optional[int] = None,
        variance_threshold: float = 0.99,
    ) -> "StatisticalShapeModel":
        """Build an SSM from training shapes in dense correspondence.

        Args:
            shapes: sequence of (N, 3) arrays, all with the same N and vertex
                ordering (i.e. already in correspondence — e.g. produced by
                segmenting + remeshing Teeth3DS scans to a common template).
            n_components: keep at most this many modes (None = as many as the
                data supports).
            variance_threshold: keep the smallest set of modes explaining at
                least this fraction of total variance.
        """
        X = np.stack([np.asarray(s, dtype=np.float64).reshape(-1) for s in shapes], axis=0)
        m, dim = X.shape
        if m < 2:
            raise ValueError("need at least two training shapes")

        mean = X.mean(axis=0)
        Xc = X - mean

        # PCA via SVD of the centred data matrix (rows = samples).
        # Xc = U S Vt ; principal directions are rows of Vt; eigenvalues of the
        # covariance are S^2 / (m - 1).
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        variances = (S ** 2) / (m - 1)

        # discard numerically-zero modes
        keep = variances > 1e-12
        Vt = Vt[keep]
        variances = variances[keep]

        # select by cumulative variance
        total = variances.sum()
        if total > 0:
            cum = np.cumsum(variances) / total
            k_var = int(np.searchsorted(cum, variance_threshold) + 1)
        else:
            k_var = len(variances)
        k = min(k_var, len(variances))
        if n_components is not None:
            k = min(k, n_components)

        return cls(mean=mean, components=Vt[:k], variances=variances[:k])

    # ------------------------------------------------------------------ #
    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Map shape coefficients b -> (N, 3) shape."""
        b = np.asarray(coefficients, dtype=np.float64)
        x = self.mean + b @ self.components
        return x.reshape(-1, 3)

    def encode(self, shape: np.ndarray) -> np.ndarray:
        """Project a full (N, 3) shape onto the model -> coefficients."""
        x = np.asarray(shape, dtype=np.float64).reshape(-1)
        return self.components @ (x - self.mean)

    # ------------------------------------------------------------------ #
    def _coord_indices(self, vertex_indices: np.ndarray) -> np.ndarray:
        vi = np.asarray(vertex_indices, dtype=np.int64).reshape(-1)
        return (vi[:, None] * 3 + np.array([0, 1, 2])).reshape(-1)

    def fit_to_partial(
        self,
        observed_indices: Sequence[int],
        observed_points: np.ndarray,
        noise_sigma: float = 0.1,
    ) -> PartialFitResult:
        """Complete an unseen shape from a visible subset of its vertices.

        Args:
            observed_indices: indices (into the template's N vertices) that the
                smartphone reconstruction actually observed — i.e. the buccal
                vertices.
            observed_points: (len(observed_indices), 3) measured positions,
                already registered into the model's coordinate frame.
            noise_sigma: standard deviation of the observation noise in mm
                (set this to your measured buccal reconstruction error; larger
                sigma trusts the prior more, smaller sigma trusts the data more).

        Returns:
            PartialFitResult with the completed shape and per-vertex uncertainty.
        """
        obs_idx = np.asarray(observed_indices, dtype=np.int64).reshape(-1)
        y = np.asarray(observed_points, dtype=np.float64)
        if y.shape != (obs_idx.size, 3):
            raise ValueError("observed_points must be (len(observed_indices), 3)")
        if obs_idx.max(initial=-1) >= self.n_vertices or obs_idx.min(initial=0) < 0:
            raise ValueError("observed_indices out of range for this model")

        coord_idx = self._coord_indices(obs_idx)
        A = self.components[:, coord_idx].T          # (3*|S|, K)
        mean_obs = self.mean[coord_idx]              # (3*|S|,)
        target = y.reshape(-1) - mean_obs            # (3*|S|,)

        inv_var = np.diag(1.0 / self.variances)      # prior precision (K, K)
        sigma2 = float(noise_sigma) ** 2
        # posterior precision and mean of the coefficients
        precision_mat = (A.T @ A) / sigma2 + inv_var
        cov_b = np.linalg.inv(precision_mat)         # posterior covariance (K, K)
        b = cov_b @ (A.T @ target) / sigma2          # MAP / posterior mean

        full = self.reconstruct(b)

        # per-vertex predictive std from the posterior over coefficients:
        # cov(x) = Phi cov_b Phi^T ; we want the per-vertex positional std, i.e.
        # sqrt of the trace of each vertex's 3x3 block.
        per_vertex_std = self._per_vertex_std(cov_b)

        residual = A @ b - target
        residual_rms = float(np.sqrt(np.mean(residual ** 2))) if residual.size else 0.0

        return PartialFitResult(
            coefficients=b,
            full_shape=full,
            per_vertex_std=per_vertex_std,
            observed_indices=obs_idx,
            residual_rms=residual_rms,
        )

    def _per_vertex_std(self, cov_b: np.ndarray) -> np.ndarray:
        """Positional predictive std per vertex from coefficient covariance.

        For vertex v with the 3xK component block Phi_v, the positional
        covariance is Phi_v cov_b Phi_v^T (3x3); we return sqrt(trace/3) as a
        scalar uncertainty in mm. Vertices strongly constrained by the
        observation get small values; unseen lingual/occlusal vertices get
        large values — exactly the flag the pipeline needs.
        """
        K = self.n_components
        comp = self.components.reshape(K, self.n_vertices, 3)  # (K, N, 3)
        # For each vertex: trace(Phi_v cov_b Phi_v^T) = sum_{a} (Phi_v[:,a]^T cov_b Phi_v[:,a])
        # Compute efficiently: M = cov_b @ comp_axis ; element-wise with comp_axis.
        var = np.zeros(self.n_vertices, dtype=np.float64)
        for axis in range(3):
            P = comp[:, :, axis]              # (K, N)
            M = cov_b @ P                     # (K, N)
            var += np.einsum("kn,kn->n", P, M)
        return np.sqrt(var / 3.0)

    # ------------------------------------------------------------------ #
    def sample(self, rng: Optional[np.random.Generator] = None, scale: float = 1.0) -> np.ndarray:
        """Draw a random plausible shape from the prior (for testing/augmentation)."""
        rng = rng or np.random.default_rng()
        b = rng.normal(0.0, np.sqrt(self.variances)) * scale
        return self.reconstruct(b)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, mean=self.mean, components=self.components, variances=self.variances
        )

    @classmethod
    def load(cls, path: str) -> "StatisticalShapeModel":
        d = np.load(path)
        return cls(mean=d["mean"], components=d["components"], variances=d["variances"])
