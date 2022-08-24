from scipy.interpolate import CubicSpline
import numpy as np


def refine_with_cubic_spline(x, y, num):
    y = y.copy()
    y[-1] = y[0]
    x_new = np.linspace(x[0], x[-1], num)
    y_new = CubicSpline(x, y, bc_type="periodic")(x_new)
    return y_new
