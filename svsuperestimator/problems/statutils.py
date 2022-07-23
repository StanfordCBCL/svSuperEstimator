import numpy as np

from scipy.stats import gaussian_kde


def kernel_density_estimation_2d(
    x, y, weights, num_points_per_axis=1000, bw_method="scott"
):

    linspace_x = np.linspace(x.min(), x.max(), num_points_per_axis)
    linspace_y = np.linspace(y.min(), y.max(), num_points_per_axis)
    grid_x, grid_y = np.meshgrid(linspace_x, linspace_y)
    grid_points = np.array([grid_x.ravel(), grid_y.ravel()])

    kernel = gaussian_kde([x, y], weights=weights, bw_method=bw_method)
    kde = kernel(grid_points).reshape(num_points_per_axis, -1)
    bandwidth = kernel.factor

    return linspace_x, linspace_y, kde, bandwidth


def kernel_density_estimation_1d(x, weights, num_points, bw_method="scott"):
    linspace_x = np.linspace(x.min(), x.max(), num_points)
    kernel = gaussian_kde(x, weights=weights, bw_method=bw_method)
    kde = kernel(linspace_x)
    bandwidth = kernel.factor

    return linspace_x, kde.flatten(), bandwidth
