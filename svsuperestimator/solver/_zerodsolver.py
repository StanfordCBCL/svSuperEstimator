"""This module holds the ZeroDSolver class."""
import json
import os
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from svzerodsolver import solver

from ..io import LinePlot
from ..model import ZeroDModel


class ZeroDSolver:
    """0D solver.

    This class contains attributes and methods to start new 0D simulations
    using the svZeroDSolver.
    """

    def __init__(self) -> None:
        """Create a new ZeroDSolver instance."""
        pass

    def run_simulation(self, model: ZeroDModel) -> pd.DataFrame:
        """Run a new 0D solver session using the provided model.

        Args:
            model: The model to simulate.

        Returns:
            bc_result: The resulting pressure and flow values at the boundary
                conditions with corresponding timestamps in a pandas dataframe.
        """

        # Create a temporary directory to perform the simulation in
        with TemporaryDirectory() as tmpdirname:

            # Make configuration in tempdir and start simulation
            model.make_configuration(tmpdirname)
            solver.set_up_and_run_0d_simulation(
                os.path.join(tmpdirname, "solver_0d.in")
            )

            # Load in branch result (branch result has format
            # [field][branch][branch_node, time_step] with the fiels: flow,
            # pressure, distance, and time.
            branch_result = np.load(
                os.path.join(tmpdirname, "solver_0d_branch_results.npy"),
                allow_pickle=True,
            ).item()
            with open(os.path.join(tmpdirname, "solver_0d.in")) as ff:
                config = json.load(ff)

        # Format output nicely according to boundary conditions
        result = pd.DataFrame(columns=["time", "name", "pressure", "flow"])
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
                    ).groups()[  # type: ignore
                        1
                    ]
                )

                # Extract result at boundary condition. Use inlet result for
                # inlet BCs and outlet result for outlet BCs, respectively.
                if bc_type == "inlet":
                    result = pd.concat(
                        [
                            result,
                            pd.DataFrame.from_dict(
                                {
                                    "name": [bc_name]
                                    * len(branch_result["time"]),
                                    "time": np.array(branch_result["time"]),
                                    "flow": np.array(
                                        branch_result["flow"][branch_id][0, :]
                                    ),
                                    "pressure": np.array(
                                        branch_result["pressure"][branch_id][
                                            0, :
                                        ]
                                    ),
                                }
                            ),
                        ]
                    )
                else:
                    result = pd.concat(
                        [
                            result,
                            pd.DataFrame.from_dict(
                                {
                                    "name": [bc_name]
                                    * len(branch_result["time"]),
                                    "time": np.array(branch_result["time"]),
                                    "flow": np.array(
                                        branch_result["flow"][branch_id][-1, :]
                                    ),
                                    "pressure": np.array(
                                        branch_result["pressure"][branch_id][
                                            -1, :
                                        ]
                                    ),
                                }
                            ),
                        ]
                    )

        return result

    @staticmethod
    def get_result_plots(result: pd.DataFrame) -> tuple[LinePlot, LinePlot]:
        """Create plots for flow and pressure result.

        Args:
            result: Result dataframe.

        Returns:
            flow_plot: Line plot for the flow over time.
            pres_plot: Line plot for pressure over time.
        """
        plot_result = result.copy()
        plot_result.flow *= 3.6  # Convert cm^3/s to l/h
        plot_result.pressure *= (
            0.00075006156130264  # Convert g/(cm s^2) to mmHg
        )
        flow_plot = LinePlot(
            plot_result,
            x="time",
            y="flow",
            color="name",
            title="Flow over time",
            xlabel=r"$s$",
            ylabel=r"$\frac{l}{h}$",
            legend_title="BC Name",
        )
        pres_plot = LinePlot(
            plot_result,
            x="time",
            y="pressure",
            color="name",
            title="Pressure over time",
            xlabel=r"$s$",
            ylabel=r"$mmHg$",
            legend_title="BC Name",
        )
        return flow_plot, pres_plot
