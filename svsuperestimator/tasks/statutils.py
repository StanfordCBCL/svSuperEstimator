"""This module holds task helper function related to statistics."""
from __future__ import annotations

import numpy as np


def particle_wmean(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Calculate weighted mean of particles.

    Args:
        particles: Coordinates of particles.
        weights: Weights of particles.
    """
    return np.average(particles, weights=weights, axis=0)


def particle_map(particles: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    """Calculate maximum a posteriori (MAP) of particles.

    Args:
        particles: Coordinates of particles.
        posterior: Posterior or log-posterior of particles.
    """
    return np.array(particles[np.argmax(posterior)])


def particle_covmat(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Calculate covariance matrix of particles.

    Args:
        particles: Coordinates of particles.
        weights: Weights of particles.
    """
    return np.cov(particles.T, aweights=weights)
