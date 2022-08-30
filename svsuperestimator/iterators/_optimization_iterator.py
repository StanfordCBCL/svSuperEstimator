"""This module holds the NelderMeadIterator class."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..forward_models import ForwardModel
from ._iterator import Iterator


class OptimizationIterator(Iterator):
    def __init__(
        self,
        forward_model: ForwardModel,
        y_obs: np.ndarray,
        x0: np.ndarray,
        bounds: list[tuple] = None,
        method="Nelder-Mead",
        max_iter=200,
        **kwargs,
    ) -> None:

        super().__init__()
        self._objective_function = lambda **kwargs: np.linalg.norm(
            forward_model.evaluate(**kwargs) - y_obs
        )
        self._x0 = x0
        self._bounds = bounds
        self._method = method
        self._max_iter = max_iter

    def run(self):
        return minimize(
            fun=self._objective_function,
            x0=self._x0,
            method=self.method,
            bounds=self._bounds,
            options={"maxiter": self._max_iter},
        ).x
