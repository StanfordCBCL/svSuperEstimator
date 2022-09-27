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
            json.dump(self.data, ff, indent=4)

    @property
    def boundary_conditions(self) -> dict[str, dict]:
        """Get the boundary conditions of the configuration."""
        return {bc["bc_name"]: bc for bc in self.data["boundary_conditions"]}

    @property
    def vessels(self) -> dict[str, dict]:
        """Get the vessels of the configuration."""
        return {
            vessel["vessel_name"]: vessel for vessel in self.data["vessels"]
        }

    @property
    def junctions(self) -> dict[str, dict]:
        """Get the vessels of the configuration."""
        return {
            junction["junction_name"]: junction
            for junction in self.data["junctions"]
        }

    @property
    def outlet_boundary_conditions(self) -> dict[str, dict]:
        """Get the outlet boundary conditions of the configuration."""
        return {
            k: v
            for k, v in self.boundary_conditions.items()
            if not k == "INFLOW"
        }

    @property
    def vessel_to_bc_map(self) -> dict:
        """Boundary condition to vessel name map."""
        bc_map = {}
        for vessel_data in self.vessels.values():
            if "boundary_conditions" in vessel_data:
                for bc_type, bc_name in vessel_data[
                    "boundary_conditions"
                ].items():
                    bc_map[bc_name] = {"name": vessel_data["vessel_name"]}
                    if bc_type == "inlet":
                        bc_map[bc_name]["flow"] = "flow_in"
                        bc_map[bc_name]["pressure"] = "pressure_in"
                    else:
                        bc_map[bc_name]["flow"] = "flow_out"
                        bc_map[bc_name]["pressure"] = "pressure_out"
        return bc_map

    @property
    def vessel_id_to_name_map(self) -> dict:
        """Vessel ID to vessel name map."""
        id_map = {}
        for vessel_data in self.vessels.values():
            id_map[vessel_data["vessel_id"]] = vessel_data["vessel_name"]
        return id_map

    @property
    def num_pts_per_cycle(self) -> int:
        """Number of time steps per cardiac cycle."""
        return self.data["simulation_parameters"][
            "number_of_time_pts_per_cardiac_cycle"
        ]

    @property
    def nodes(self) -> list:
        """Nodes of the 0D model."""
        id_map = self.vessel_id_to_name_map
        connections = []
        for junction in self.junctions.values():
            for ivessel in junction["inlet_vessels"]:
                connections.append(
                    (id_map[ivessel], junction["junction_name"])
                )
            for ovessel in junction["outlet_vessels"]:
                connections.append(
                    (junction["junction_name"], id_map[ovessel])
                )
        for vessel in self.vessels.values():
            try:
                connections.append(
                    (
                        vessel["boundary_conditions"]["inlet"],
                        vessel["vessel_name"],
                    )
                )
            except KeyError:
                pass
            try:
                connections.append(
                    (
                        vessel["vessel_name"],
                        vessel["boundary_conditions"]["outlet"],
                    )
                )
            except KeyError:
                pass

        return connections

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
        num_cycles: int = None,
        steady_initial: bool = None,
        mean_only: bool = None,
        output_interval: bool = None,
        last_cycle_only: bool = None,
        variable_based: bool = None,
    ) -> None:
        """Update the simulation parameters.

        Args:
            abs_tol: Absolute tolerance for simulation.
            max_nliter: Maximum number of non-linear iterations per time step
                in time integration.
            num_cycles: Number of cardiac cycles to simulate.
            steady_initial: Solve steady solution first and use as intitial
                condition.
            mean_only: Return only mean values over time steps.
            output_interval: Interval for writing a timestep to the output.
            last_cycle_only: Output only last cycle.
            variable_based: Node based output.
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
        if last_cycle_only is not None:
            simparams["output_last_cycle_only"] = last_cycle_only
        if num_cycles is not None:
            simparams["number_of_cardiac_cycles"] = num_cycles
        if variable_based is not None:
            simparams["output_variable_based"] = variable_based
