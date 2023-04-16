"""This module holds the SvSolverInputHandler class."""
from __future__ import annotations

import re
from typing import Any

from ._plain_handler import PlainHandler


class SvSolverInputHandler(PlainHandler):
    """Handler for svSolver input data."""

    @property
    def time_step_size(self) -> float:
        """Time step size of the simulation."""
        return float(self._get_configuration("Time Step Size"))

    @property
    def num_time_steps(self) -> float:
        """Number of time steps of the simulation."""
        return int(self._get_configuration("Number of Timesteps"))

    @num_time_steps.setter
    def num_time_steps(self, value: int) -> None:
        """Number of time steps of the simulation."""
        self._set_configuration("Number of Timesteps", value)

    @property
    def rcr_surface_ids(self) -> list[int]:
        """Surface ID sequence of the RCR boundary surfaces."""
        surface_list = self._get_configuration("List of RCR Surfaces")
        return [int(num) for num in surface_list.split()]

    @property
    def r_surface_ids(self) -> list[int]:
        """Surface ID sequence of the R boundary surfaces."""
        surface_list = self._get_configuration("List of Resistance Surfaces")
        return [int(num) for num in surface_list.split()]

    def set_tolerances(self, tolerance: float) -> None:
        """Set the solver tolerances.

        Args:
            tolerance: Tolerance.
        """
        self._set_configuration("Residual Criteria", f"{tolerance:f}")
        self._set_configuration(
            "Tolerance on Momentum Equations", f"{tolerance:f}"
        )
        self._set_configuration(
            "Tolerance on Continuity Equations", f"{tolerance:f}"
        )
        self._set_configuration(
            "Tolerance on svLS NS Solver", f"{tolerance:f}"
        )

    def _get_configuration(self, label: str) -> str:
        """Find a configuration by a label.

        Args:
            label: Label of the parameter.

        Returns:
            value: Value of the option in the data.
        """
        return re.search(
            label + ":" + r".*$", self.data, re.MULTILINE
        ).group()[  # type: ignore
            len(label) + 2 :
        ]

    def _set_configuration(self, label: str, value: Any) -> None:
        """Set a configuration by a label.

        Args:
            label: Label of the parameter.
            value: Value to set.
        """
        match = re.search(label + ":" + r".*$", self.data, re.MULTILINE)
        self.data = (
            self.data[: match.start()]  # type: ignore
            + f"{label}: {value}"
            + self.data[match.end() :]  # type: ignore
        )
