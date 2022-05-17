"""This module holds the ZeroDSolver class."""
import json
import os
from tempfile import TemporaryDirectory
import re

import numpy as np
from svzerodsolver import solver

from ..model import ZeroDModel


class ZeroDSolver:
    """0D solver.

    This class contains attributes and methods to start new 0D simulations
    using the svZeroDSolver.
    """

    def __init__(self, model: ZeroDModel) -> None:
        """Create a new ZeroDSolver instance.

        Args:
            model: 0D model to simulate.
        """
        self._model = model

    def run_simulation(self) -> tuple[dict, np.ndarray]:
        """Run a new 0D solver session using the provided model.

        Returns:
            bc_result: The resulting pressure and flow values at the boundary
                conditions. The results for each boundary conditions are saved
                under the name of the boundary condition.
            time: Time steps corresponding to the pressure and flow values.
        """

        # Extract configuration from model
        config = self._model.get_svzerodsolver_config()

        # Create a temporary directory to perform the simulation in
        with TemporaryDirectory() as tmpdirname:

            # Save configuration file in tempdir and start simulation
            config_file = os.path.join(tmpdirname, "config.json")
            with open(config_file, "w") as ff:
                json.dump(config, ff)
            solver.set_up_and_run_0d_simulation(config_file)

            # Load in branch result (branch result has format
            # [field][branch][branch_node, time_step] with the fiels: flow,
            # pressure, distance, and time.
            branch_result = np.load(
                os.path.join(tmpdirname, "config_branch_results.npy"),
                allow_pickle=True,
            ).item()

        # Format output nicely according to boundary conditions
        bc_result = {}
        for vessel_data in config["vessels"]:
            if "boundary_conditions" in vessel_data:
                bc_type, bc_name = list(
                    vessel_data["boundary_conditions"].items()
                )[0]
                vessel_name = vessel_data["vessel_name"]

                # Get branch ID (it is not the same as the vessel ID; a branch
                # can have mutliple vessels)
                branch_id = int(
                    (
                        re.match(
                            r"([a-z]+)([0-9]+)",
                            vessel_name.split("_")[0],
                            re.I,
                        )
                    ).groups()[1]
                )

                # Extract result at boundary condition. Use inlet result for
                # inlet BCs and outlet result for outlet BCs, respectively.
                if bc_type == "inlet":
                    bc_result[bc_name] = {
                        "flow": np.array(
                            branch_result["flow"][branch_id][0, :]
                        ),
                        "pressure": np.array(
                            branch_result["pressure"][branch_id][0, :]
                        ),
                    }
                else:
                    bc_result[bc_name] = {
                        "flow": np.array(
                            branch_result["flow"][branch_id][-1, :]
                        ),
                        "pressure": np.array(
                            branch_result["pressure"][branch_id][-1, :]
                        ),
                    }

        return bc_result, np.array(branch_result["time"])
