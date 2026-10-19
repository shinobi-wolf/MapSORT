"""Landmark association: Gaussian fusion and Mahalanobis (or Euclidean) cost."""

from __future__ import annotations

import numpy as np

from .schemas import LandmarkObservation

# A single-keypoint observation has near-zero scatter, so its covariance can be singular.
_RIDGE = 1e-12
_EYE = np.eye(3)


def _inv(cov: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(cov, dtype=float) + _RIDGE * _EYE)


class LandmarkState:
    """A track's map-frame position, fused from its observations in information form."""

    def __init__(self) -> None:
        self.information = np.zeros((3, 3))  # Lambda = sum of Sigma_i^-1
        self.information_mean = np.zeros(3)  # eta = sum of Sigma_i^-1 x_i
        self.count = 0

    def fuse(self, observation: LandmarkObservation) -> None:
        """Add one observation. `extent` is physical spread and is never fused."""
        precision = _inv(observation.covariance)
        self.information = self.information + precision
        self.information_mean = self.information_mean + precision @ np.asarray(
            observation.position, dtype=float
        )
        self.count += 1

    @property
    def observed(self) -> bool:
        return self.count > 0

    @property
    def gaussian(self) -> tuple[np.ndarray, np.ndarray]:
        """(mu_k, Sigma_k) = (Sigma_k eta, Lambda^-1)."""
        covariance = _inv(self.information)
        return covariance @ self.information_mean, covariance


DISTANCES = ("mahalanobis", "euclidean")


def cost_matrix(
    states: list[LandmarkState],
    observations: list[LandmarkObservation | None],
    infinity: float,
    distance: str = "mahalanobis",
) -> np.ndarray:
    """Squared Mahalanobis distances, or Euclidean distances between centres in map units,
    `infinity` where either side has no landmark."""
    costs = np.full((len(states), len(observations)), float(infinity))
    for row, state in enumerate(states):
        if state is None or not state.observed:
            continue
        mean, covariance = state.gaussian
        for col, observation in enumerate(observations):
            if observation is None:
                continue
            delta = np.asarray(observation.position, dtype=float) - mean
            if distance == "euclidean":
                costs[row, col] = np.linalg.norm(delta)
                continue
            total = covariance + np.asarray(observation.covariance, dtype=float)
            costs[row, col] = delta @ _inv(total) @ delta
    return costs
