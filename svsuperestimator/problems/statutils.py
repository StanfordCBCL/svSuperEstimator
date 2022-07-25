import numpy as np

from scipy.stats import gaussian_kde

from pqueens.utils.pdf_estimation import (
    estimate_bandwidth_for_kde,
    estimate_pdf,
)


def kernel_density_estimation_2d(
    x, y, bounds, weights, num_points_per_axis=1000, bw_method="scott"
):

    linspace_x = np.linspace(bounds[0][0], bounds[0][1], num_points_per_axis)
    linspace_y = np.linspace(bounds[1][0], bounds[1][1], num_points_per_axis)
    grid_x, grid_y = np.meshgrid(linspace_x, linspace_y)
    grid_points = np.array([grid_x.ravel(), grid_y.ravel()])

    kernel = gaussian_kde([x, y], weights=weights, bw_method=bw_method)
    kde = kernel(grid_points).reshape(num_points_per_axis, -1)
    bandwidth = kernel.factor

    return linspace_x, linspace_y, kde, bandwidth


def kernel_density_estimation_1d(x, weights, bounds):
    bandwidth = estimate_bandwidth_for_kde(x, weights=weights, min_samples=bounds[0], max_samples=bounds[1])
    kde, support_points = estimate_pdf(x, weights=weights)
    return support_points.flatten(), kde.flatten(), bandwidth
