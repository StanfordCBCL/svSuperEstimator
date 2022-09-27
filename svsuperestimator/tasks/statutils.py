"""This module holds task helper function related to statistics."""
from __future__ import annotations

import numpy as np
from pqueens.utils.pdf_estimation import (
    estimate_bandwidth_for_kde,
    estimate_pdf,
)
from scipy.stats import gaussian_kde


def particle_wmean(particles: np.ndarray, weights: np.ndarray):
    """Calculate weighted mean of particles.

    Args:
        particles: Coordinates of particles.
        weights: Weights of particles.
    """
    return np.average(particles, weights=weights, axis=0)


def particle_map(particles: np.ndarray, weights: np.ndarray):
    """Calculate maximum a posteriori (MAP) of particles.

    Args:
        particles: Coordinates of particles.
        weights: Weights of particles.
    """
    return np.array(particles[np.argmax(weights)])


def particle_covmat(particles: np.ndarray, weights: np.ndarray):
    """Calculate covariance matrix of particles.

    Args:
        particles: Coordinates of particles.
        weights: Weights of particles.
    """
    return np.cov(particles.T, aweights=weights)


def gaussian_kde_1d(
    x: np.ndarray,
    weights: np.ndarray,
    bounds: tuple[float, float],
    num: int = 100,
):
    """Calculate 1d kernel-density estimate.

    Args:
        x: Coordinates of particles.
        weights: Weights of particles.
        bounds: Bounds for kernel-density estimation.
        num: Number of points for kernel density estimate

    Returns:
        kde_x: Coordinates of kernel density estimate.
        kde: Kernel density estimate.
        bandwidth: Optimized bandwidth.
    """
    lin_x = np.linspace(bounds[0], bounds[1], num)
    bandwidth = estimate_bandwidth_for_kde(
        x,
        weights=weights,
        kernel="gaussian",
    )
    kde, kde_x = estimate_pdf(
        x,
        kernel_bandwidth=bandwidth,
        weights=weights,
        support_points=lin_x,
        kernel="gaussian",
    )
    return kde_x.flatten(), kde.flatten(), bandwidth


def gaussian_kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    num=100,
):
    """Calculate 2d kernel-density estimate.

    Args:
        x: X-coordinates of particles.
        y: Y-coordinates of particles.
        weights: Weights of particles.
        bounds: Bounds for kernel-density estimation.
        num: Number of points for kernel density estimate

    Returns:
        kde_x: X-coordinates of kernel density estimate.
        kde_y: Y-coordinates of kernel density estimate.
        kde: Kernel density estimate.
        bandwidth: Optimized bandwidth.
    """
    linspace_x = np.linspace(bounds[0][0], bounds[0][1], num)
    linspace_y = np.linspace(bounds[1][0], bounds[1][1], num)
    grid_x, grid_y = np.meshgrid(linspace_x, linspace_y)
    grid_points = np.array([grid_x.ravel(), grid_y.ravel()])

    kernel = gaussian_kde([x, y], weights=weights, bw_method="scott")
    kde = kernel(grid_points).reshape(num, -1)
    bandwidth = kernel.factor

    return linspace_x, linspace_y, kde, bandwidth
