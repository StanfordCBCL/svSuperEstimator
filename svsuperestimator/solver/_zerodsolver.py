"""This module holds the ZeroDSolver and auxiliary classes."""
import json
import os
from tempfile import TemporaryDirectory

import numpy as np
from svzerodsolver import solver

from ..model import ZeroDModel


class ZeroDSolver:
    """0D solver.

    This class contains attributes and methods to start new 0D simulations
    using the svZeroDSolver.
    """

    def __init__(self) -> None:
        """Create a new ZeroDSolver instance."""

    def run_simulation(self, model: ZeroDModel) -> dict:
        """Run a new 0D solver session using the provided model.

        Args:
            model: 0D model to simulate.
        """

        # Extract configuration from model
        config = model.get_svzerodsolver_config()

        # Create a temporary directory to perform the simulation in
        with TemporaryDirectory() as tmpdirname:
            config_file = os.path.join(tmpdirname, "config.json")
            with open(config_file, "w") as ff:
                json.dump(config, ff)
            solver.set_up_and_run_0d_simulation(config_file)
            branch_result = np.load(
                os.path.join(tmpdirname, "config_branch_results.npy"),
                allow_pickle=True,
            ).item()

        # Format output nicely
        output = {}
        for vessel_data in config["vessels"]:
            vid = int(vessel_data["vessel_id"])

            try:
                output[vid] = {
                    "length": vessel_data["vessel_length"],
                    "name": vessel_data["vessel_name"],
                    "flow_in": branch_result["flow"][vid][0, -1],
                    "flow_out": branch_result["flow"][vid][-1, -1],
                    "pressure_in": branch_result["pressure"][vid][0, -1],
                    "pressure_out": branch_result["pressure"][vid][-1, -1],
                }
            except KeyError:
                pass
        return output
