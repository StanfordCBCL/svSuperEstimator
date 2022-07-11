"""This module holds the ZeroDModel and auxiliary classes."""
import json
import os
from dataclasses import dataclass

import numpy as np

from ..io import SimVascularProject


class ZeroDModel:
    """0D model.

    This class contains attributes and methods to save and modify a 0D model
    for the svZeroDSolver. It is aimed at facilitating the variation of
    input parameters.
    """

    def __init__(self, project: SimVascularProject):
        """Create a new ZeroDModel instance.

        Args:
            project: Project object to extract the model parameters from.
        """

        self._project = project
        self._config = project["rom_simulation_config"]

        # Not needed for evaluation
        if "description" in self._config:
            del self._config["description"]

        # Create python object representations for each boundary condition
        self.boundary_conditions: dict[str, "_BoundaryCondition"] = {}
        for bc in self._config["boundary_conditions"]:
            if bc["bc_type"] == "FLOW":
                self.boundary_conditions[
                    bc["bc_name"]
                ] = _FlowBoundaryCondition(
                    flow=np.array(bc["bc_values"]["Q"]),
                    time=np.array(bc["bc_values"]["t"]),
                )
            elif bc["bc_type"] == "RCR":
                self.boundary_conditions[
                    bc["bc_name"]
                ] = _RCRBoundaryCondition(
                    capacity=bc["bc_values"]["C"],
                    pressure_distal=bc["bc_values"]["Pd"],
                    resistance_distal=bc["bc_values"]["Rd"],
                    resistance_proximal=bc["bc_values"]["Rp"],
                )
            else:
                raise ValueError(
                    f"Unknown boundary condition type {bc['bc_type']}."
                )

    def make_configuration(
        self,
        target: str = None,
        num_cardiac_cycles=None,
        pts_per_cycle=None,
        absolute_tolerance=1e-8,
        maximum_nonlinear_iterations=30,
        output_interval=1,
        steady_initial=True,
        output_mean_only=False,
    ) -> None:
        """Make the configuration at a specified target.

        Creates all configuration files that are needed for a svZeroDSolver
        session.

        Args:
            target: Target folder for the configuration files.
        """
        config = self._config.copy()
        if num_cardiac_cycles is not None:
            config["simulation_parameters"][
                "number_of_cardiac_cycles"
            ] = num_cardiac_cycles
        if pts_per_cycle is not None:
            config["simulation_parameters"][
                "number_of_time_pts_per_cardiac_cycle"
            ] = pts_per_cycle
        config["simulation_parameters"].update(
            {
                "absolute_tolerance": absolute_tolerance,
                "maximum_nonlinear_iterations": maximum_nonlinear_iterations,
                "output_interval": output_interval,
                "steady_initial": steady_initial,
                "output_mean_only": output_mean_only,
            }
        )
        for bc in config["boundary_conditions"]:
            bc_obj = self.boundary_conditions[bc["bc_name"]]
            if isinstance(bc_obj, _FlowBoundaryCondition):
                bc["bc_values"].update(
                    {"Q": bc_obj.flow.tolist(), "t": bc_obj.time.tolist()}
                )
            elif isinstance(bc_obj, _RCRBoundaryCondition):
                bc["bc_values"].update(
                    {
                        "C": bc_obj.capacity,
                        "Pd": bc_obj.pressure_distal,
                        "Rd": bc_obj.resistance_distal,
                        "Rp": bc_obj.resistance_proximal,
                    }
                )
            else:
                raise ValueError(
                    f"Unknown boundary condition type {bc['bc_type']}."
                )
        if target is not None:
            config_file = os.path.join(target, "solver_0d.in")
            with open(config_file, "w") as ff:
                json.dump(config, ff)
        return config


class _BoundaryCondition:
    """Auxiliary class for boundary conditions."""

    pass


@dataclass
class _FlowBoundaryCondition(_BoundaryCondition):
    """Class representing a flow boundary condition."""

    flow: np.ndarray
    time: np.ndarray


@dataclass
class _RCRBoundaryCondition(_BoundaryCondition):
    """Class representing an RCR boundary condition."""

    capacity: float
    pressure_distal: float
    resistance_distal: float
    resistance_proximal: float
