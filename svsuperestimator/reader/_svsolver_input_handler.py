"""This module holds the SvSolverInputHandler class."""
from __future__ import annotations

import re

from ._plain_handler import PlainHandler


class SvSolverInputHandler(PlainHandler):
    """Handler for svSolver input data."""

    @property
    def time_step_size(self) -> float:
        """Time step size of the simulation."""
        return float(self._find_configuration("Time Step Size"))
    
    @property
    def num_time_steps(self) -> float:
        """Number of time steps of the simulation."""
        return int(self._find_configuration("Number of Timesteps"))

    @property
    def rcr_surface_ids(self) -> list[int]:
        """Surface ID sequence of the RCR boundary surfaces."""
        surface_list = self._find_configuration("List of RCR Surfaces")
        return [int(num) for num in surface_list.split()]

    def _find_configuration(self, label: str) -> str:
        """Find a configuration by a label.

        Args:
            label: Label of the parameter.

        Returns:
            value: Value of the option in the data.
        """
        return re.search(
            label + ":" + r".*$", self.data, re.MULTILINE
        ).group()[len(label) + 2 :]
