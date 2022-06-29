"""This module holds the ZeroDSolver class."""
import orjson  # Faster than json
import os
from tempfile import TemporaryDirectory
from subprocess import call

import numpy as np
import pandas as pd
from svzerodsolver import run_simulation_from_config

from ..model import ZeroDModel


class ZeroDSolver:
    """0D solver.

    This class contains attributes and methods to start new 0D simulations
    using the svZeroDSolver.
    """

    def __init__(self) -> None:
        """Create a new ZeroDSolver instance."""
        pass

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
    def _call_solver_cpp(config, exec, mean=False):
        with TemporaryDirectory() as tmpdir:
            infile = os.path.join(tmpdir, "input.json")
            outfile = os.path.join(tmpdir, "output.csv")
            with open(infile, "wb") as ff:
                ff.write(
                    orjson.dumps(
                        config,
                        option=orjson.OPT_NAIVE_UTC
                        | orjson.OPT_SERIALIZE_NUMPY,
                    )
                )
            if mean:
                call(
                    args=[exec, infile, outfile, "mean"],
                    cwd=tmpdir,
                )
            else:
                call(
                    args=[exec, infile, outfile],
                    cwd=tmpdir,
                )
            branch_result = pd.read_csv(outfile, engine="pyarrow")
        return branch_result

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
            num_cardiac_cycles=5, pts_per_cycle=25
        )
        branch_result = self._call_solver_cpp(
            config,
            "/Users/stanford/svZeroDSolver/Release/svzerodsolver",
            mean,
        )
        # branch_result = self._call_solver_python(config)

        return branch_result
