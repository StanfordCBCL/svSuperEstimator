"""This module holds the ZeroDSolver class."""
import orjson  # Faster than json
import os
from tempfile import TemporaryDirectory
from subprocess import call

import numpy as np
import pandas as pd
from svzerodsolver import run_simulation_from_config, svzerodsolvercpp

from ..model import ZeroDModel


class ZeroDSolver:
    """0D solver.

    This class contains attributes and methods to start new 0D simulations
    using the svZeroDSolver.
    """

    def __init__(self, cpp=True) -> None:
        """Create a new ZeroDSolver instance."""
        if cpp:
            self._solver_call = self._call_solver_cpp
        else:
            self._solver_call = self._call_solver_python

    @staticmethod
    def _call_solver_python(config):

        branch_result = run_simulation_from_config(config)
        # Format output nicely according to boundary conditions
        result = pd.DataFrame(
            columns=[
                "name",
                "time",
                "flow_in",
                "flow_out",
                "pressure_in",
                "pressure_out",
            ]
        )
        for branch_id, name in enumerate(branch_result["names"]):
            result = pd.concat(
                [
                    result,
                    pd.DataFrame.from_dict(
                        {
                            "name": [name] * len(branch_result["time"]),
                            "time": np.array(branch_result["time"]),
                            "flow_in": np.array(
                                branch_result["flow_in"][branch_id]
                            ),
                            "flow_out": np.array(
                                branch_result["flow_out"][branch_id]
                            ),
                            "pressure_in": np.array(
                                branch_result["pressure_in"][branch_id]
                            ),
                            "pressure_out": np.array(
                                branch_result["pressure_out"][branch_id]
                            ),
                        }
                    ),
                ]
            )
        return result

    @staticmethod
    def _call_solver_cpp(config):
        return svzerodsolvercpp.run(config)

    def run_simulation(self, model: ZeroDModel, mean=False) -> pd.DataFrame:
        """Run a new 0D solver session using the provided model.

        Args:
            model: The model to simulate.

        Returns:
            bc_result: The resulting pressure and flow values at the boundary
                conditions with corresponding timestamps in a pandas dataframe.
        """

        # Start simulation
        config = model.make_configuration(
            pts_per_cycle=100, output_mean_only=mean
        )
        return self._solver_call(config)
