"""This module holds the SvZeroDSolverInputHandler."""
from __future__ import annotations

import json
import os

from ._data_handler import DataHandler


class SvZeroDSolverInputHandler(DataHandler):
    """Handler for svZeroDSolver input data."""

    @classmethod
    def from_file(cls, filename: str) -> SvZeroDSolverInputHandler:
        """Create a new SvZeroDSolverInputHandler instance from file.

        Args:
            filename: Path to the file to read data from.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        with open(filename) as ff:
            data = json.load(ff)
        return cls(data)

    def to_file(self, filename: str) -> None:
        """Write the data to a file.

        Args:
            filename: Path to the file to read data from.
        """
        with open(filename, "w") as ff:
            json.dump(self.data, ff)

    @property
    def boundary_conditions(self) -> dict[str, dict]:
        """Get the boundary conditions of the configuration."""
        return {bc["bc_name"]: bc for bc in self.data["boundary_conditions"]}

    @property
    def vessels(self) -> dict[str, dict]:
        """Get the vessels of the configuration."""
        return self.data["vessels"]

    @property
    def outlet_boundary_conditions(self) -> dict[str, dict]:
        """Get the outlet boundary conditions of the configuration."""
        return {
            k: v
            for k, v in self.boundary_conditions.items()
            if not k == "INFLOW"
        }

    @property
    def num_pts_per_cyle(self) -> int:
        """Number of time steps per cardiac cycle."""
        return self.data["simulation_parameters"][
            "number_of_time_pts_per_cardiac_cycle"
        ]

    def copy(self) -> SvZeroDSolverInputHandler:
        """Create and return a copy of the handler.

        Returns:
            handler: Copy of the data handler.
        """
        return SvZeroDSolverInputHandler(self.data.copy())

    def update_simparams(
        self,
        abs_tol: float = None,
        max_nliter: int = None,
        steady_initial: bool = None,
        mean_only: bool = None,
        output_interval=None,
    ) -> None:
        """Update the simulation parameters.

        Args:
            abs_tol: Absolute tolerance for simulation.
            max_nliter: Maximum number of non-linear iterations per time step
                in time integration.
            steady_initial: Solve steady solution first and use as intitial
                condition.
            mean_only: Return only mean values over time steps.
            output_interval: Interval for writing a timestep to the output.
        """
        simparams = self.data["simulation_parameters"]
        if abs_tol is not None:
            simparams["absolute_tolerance"] = abs_tol
        if max_nliter is not None:
            simparams["maximum_nonlinear_iterations"] = max_nliter
        if steady_initial is not None:
            simparams["steady_initial"] = steady_initial
        if mean_only is not None:
            simparams["output_mean_only"] = mean_only
        if output_interval is not None:
            simparams["output_interval"] = output_interval
