"""This module holds the ModelCalibrationLeastSquares task."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any
from time import time

import numpy as np
import pandas as pd
import pysvzerod
from rich.progress import BarColumn, Progress

from .. import reader, visualizer
from ..reader import CenterlineHandler
from ..reader import utils as readutils
from . import plotutils, taskutils
from .task import Task

NUM_REFINED = 100


class ModelCalibrationLeastSquares(Task):
    """Least Squares model calibration task.

    Calibrates blood vessels and junctions elements in a 0D model to an
    existing 3D solution.
    """

    # Task name to configure task in config file
    TASKNAME = "model_calibration_least_squares"

    # Configuration options for the task and their defaults
    DEFAULTS = {
        "zerod_config_file": None,
        "threed_solution_file": None,
        "centerline_padding": False,
        "calibrate_stenosis_coefficient": True,
        "initial_damping_factor": 1.0,
        "maximum_iterations": 100,
        **Task.DEFAULTS,
    }

    def core_run(self) -> None:
        """Core routine of the task."""

        # Loading data from project
        self.log(
            "Loading 0D simulation input from "
            f"{self.config['zerod_config_file']}"
        )
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

        # Get 3D simulation time step size from 3d simulation input file
        self.log(
            "Loading 3D simulation input from "
            f"{self.project['3d_simulation_input_path']}"
        )
        threed_config_handler = self.project["3d_simulation_input"]

        # Load 3D result file
        self.log(
            f"Loading 3D result from {self.config['threed_solution_file']}"
        )
        threed_result_handler = CenterlineHandler.from_file(
            self.config["threed_solution_file"], padding=True
        )

        # Load centerline
        self.log(f"Loading centerline from {self.project['centerline_path']}")
        cl_handler = self.project["centerline"]

        # Map centerline result to 0D element nodes
        self.log("Map 3D centerline result to 0D elements")
        branch_data, times = taskutils.map_centerline_result_to_0d_3(
            zerod_config_handler,
            cl_handler,
            threed_config_handler,
            threed_result_handler,
        )
        times_new = np.linspace(times[0], times[-1], NUM_REFINED)

        self.log("Create calibrator input file")
        connections = []
        vessel_id_map = {}
        for vessel_name, vessel_config in zerod_config_handler.vessels.items():
            vessel_id_map[vessel_config["vessel_id"]] = vessel_name
            if "boundary_conditions" not in vessel_config:
                continue
            bc_config = vessel_config["boundary_conditions"]
            if "inlet" in bc_config:
                connections.append((bc_config["inlet"], vessel_name))
            if "outlet" in bc_config:
                connections.append((vessel_name, bc_config["outlet"]))

        for (
            junction_name,
            junction_config,
        ) in zerod_config_handler.junctions.items():
            for inlet_vessel in junction_config["inlet_vessels"]:
                connections.append(
                    (vessel_id_map[inlet_vessel], junction_name)
                )

            ovessels = []
            for outlet_vessel in junction_config["outlet_vessels"]:
                connections.append(
                    (junction_name, vessel_id_map[outlet_vessel])
                )
                ovessels.append(vessel_id_map[outlet_vessel])

        y = {}
        dy = {}

        for connection in connections:
            if connection[0].startswith("branch"):
                branch_name = connection[0]
                branch_id, seg_id = branch_name.split("_")
                branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

                (
                    pressure,
                    dpressure,
                ) = taskutils.refine_with_cubic_spline_derivative(
                    times,
                    branch_data[branch_id][seg_id]["pressure_out"],
                    NUM_REFINED,
                )
                flow, dflow = taskutils.refine_with_cubic_spline_derivative(
                    times,
                    branch_data[branch_id][seg_id]["flow_out"],
                    NUM_REFINED,
                )

                y[
                    f"pressure:{branch_name}:{connection[1]}"
                ] = pressure.tolist()
                y[f"flow:{branch_name}:{connection[1]}"] = flow.tolist()
                dy[
                    f"pressure:{branch_name}:{connection[1]}"
                ] = dpressure.tolist()
                dy[f"flow:{branch_name}:{connection[1]}"] = dflow.tolist()

            if connection[1].startswith("branch"):
                branch_name = connection[1]
                branch_id, seg_id = branch_name.split("_")
                branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

                (
                    pressure,
                    dpressure,
                ) = taskutils.refine_with_cubic_spline_derivative(
                    times,
                    branch_data[branch_id][seg_id]["pressure_in"],
                    NUM_REFINED,
                )
                flow, dflow = taskutils.refine_with_cubic_spline_derivative(
                    times,
                    branch_data[branch_id][seg_id]["flow_in"],
                    NUM_REFINED,
                )

                y[
                    f"pressure:{connection[0]}:{branch_name}"
                ] = pressure.tolist()
                y[f"flow:{connection[0]}:{branch_name}"] = flow.tolist()
                dy[
                    f"pressure:{connection[0]}:{branch_name}"
                ] = dpressure.tolist()
                dy[f"flow:{connection[0]}:{branch_name}"] = dflow.tolist()

        zerod_config_handler.data["y"] = y
        zerod_config_handler.data["dy"] = dy

        zerod_config_handler.data["calibration_parameters"] = {
            "tolerance_gradient": 1e-6,
            "tolerance_increment": 1e-10,
            "initial_damping_factor": 1.0,
            "calibrate_stenosis_coefficient": self.config[
                "calibrate_stenosis_coefficient"
            ],
            "initial_damping_factor": self.config["initial_damping_factor"],
            "maximum_iterations": self.config["maximum_iterations"],
        }

        # Create debug plots
        if self.config["debug"]:
            self.log("Create debug plots")
            debug_folder = os.path.join(self.output_folder, "debug")
            os.makedirs(debug_folder, exist_ok=True)
            zerod_config_handler_test = (
                reader.SvZeroDSolverInputHandler.from_file(
                    self.config["zerod_config_file"]
                )
            )
            zerod_config_handler_test.update_simparams(
                last_cycle_only=True,
                variable_based=True,
                output_derivative=True,
            )
            zerod_result = pysvzerod.simulate(zerod_config_handler_test.data)
            with Progress(
                " " * 20 + "Creating plots... ",
                BarColumn(),
                "{task.completed}/{task.total} completed | "
                "{task.speed} plots/s",
                console=self.console,
            ) as progress:
                for key in progress.track(y):
                    selection = zerod_result[zerod_result.name == key]
                    plot = visualizer.Plot2D()
                    plot.add_line_trace(
                        times_new, y[key], name="3D", showlegend=True
                    )
                    plot.add_line_trace(
                        np.array(selection["time"]),
                        np.array(selection["y"]),
                        name="0D",
                        showlegend=True,
                    )
                    plot.to_image(os.path.join(debug_folder, f"{key}_y.png"))
                    plot = visualizer.Plot2D()
                    plot.add_line_trace(
                        times_new, dy[key], name="3D", showlegend=True
                    )
                    plot.add_line_trace(
                        np.array(selection["time"]),
                        np.array(selection["ydot"]),
                        name="0D",
                        showlegend=True,
                    )
                    plot.to_image(os.path.join(debug_folder, f"{key}_dy.png"))

        # Run calibration
        self.log("Start calibration 1")
        output_file = os.path.join(self.output_folder, "solver_0d.in")
        input_file = os.path.join(self.output_folder, "calibrator_0d.in")
        zerod_config_handler.to_file(input_file)
        start = time()
        calibrated_config = pysvzerod.calibrate(zerod_config_handler.data)
        end = time()
        with open(output_file, "w") as ff:
            json.dump(calibrated_config, ff, indent=4)
        self.log(f"Completed calibration in {end-start} s")

    def post_run(self) -> None:
        """Postprocessing routine of the task."""

        # Read data
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )
        zerod_opt_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            os.path.join(self.output_folder, "solver_0d.in")
        )
        threed_config_handler = self.project["3d_simulation_input"]
        threed_result_handler = CenterlineHandler.from_file(
            self.config["threed_solution_file"], padding=True
        )

        # Map centerline data to 0D elements
        branch_data, times = taskutils.map_centerline_result_to_0d_3(
            zerod_config_handler,
            self.project["centerline"],
            threed_config_handler,
            threed_result_handler,
        )

        # Run simulation for both configurations
        zerod_config_handler.update_simparams(last_cycle_only=True)
        zerod_opt_config_handler.update_simparams(last_cycle_only=True)
        zerod_result = pysvzerod.simulate(zerod_config_handler.data)
        zerod_opt_result = pysvzerod.simulate(zerod_opt_config_handler.data)

        # Extract time steps of last cardiac cycle
        pts_per_cycle = zerod_config_handler.num_pts_per_cycle
        sim_times = np.array(
            zerod_result[zerod_result.name == "branch0_seg0"]["time"][
                -pts_per_cycle:
            ]
        )
        sim_times -= np.amin(sim_times)

        # Extract results for each branch
        filter = {
            "pressure_in": taskutils.cgs_pressure_to_mmgh,
            "pressure_out": taskutils.cgs_pressure_to_mmgh,
            "flow_in": taskutils.cgs_flow_to_lmin,
            "flow_out": taskutils.cgs_flow_to_lmin,
        }
        results = pd.DataFrame()
        for branch_id, branch in branch_data.items():
            for seg_id, segment in branch.items():
                vessel_name = f"branch{branch_id}_seg{seg_id}"

                # Append 3d result
                results_new = pd.DataFrame(
                    dict(
                        name=[vessel_name + "_3d"] * len(times),
                        time=times,
                        **{n: f(segment[n]) for n, f in filter.items()},
                    )
                )
                results = pd.concat([results, results_new])

                # Append 0d result
                vessel_result = zerod_result[zerod_result.name == vessel_name]
                results_new = pd.DataFrame(
                    dict(
                        name=[vessel_name + "_0d"] * pts_per_cycle,
                        time=sim_times,
                        **{n: f(vessel_result[n]) for n, f in filter.items()},
                    )
                )
                results = pd.concat([results, results_new])

                # Append 0d optimized result
                vessel_result = zerod_opt_result[
                    zerod_opt_result.name == vessel_name
                ]
                results_new = pd.DataFrame(
                    dict(
                        name=[vessel_name + "_0d_opt"] * pts_per_cycle,
                        time=sim_times,
                        **{n: f(vessel_result[n]) for n, f in filter.items()},
                    )
                )
                results = pd.concat([results, results_new])

        # Save parameters to file
        self.database["timestamp"] = datetime.now().strftime(
            "%m/%d/%Y, %H:%M:%S"
        )

        results.to_csv(os.path.join(self.output_folder, "results.csv"))

    def generate_report(self) -> visualizer.Report:
        """Generate the task report."""

        results = pd.read_csv(os.path.join(self.output_folder, "results.csv"))

        # Add 3D plot of mesh with 0D elements
        report = visualizer.Report()
        report.add("Overview")
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )
        branch_data = readutils.get_0d_element_coordinates(
            self.project, zerod_config_handler
        )
        model_plot = plotutils.create_3d_geometry_plot_with_vessels(
            self.project, branch_data
        )
        report.add([model_plot])

        # Options for all plots
        common_plot_opts: dict[str, Any] = {
            "static": True,
            "width": 750,
            "height": 400,
        }

        # Options for pressure plots
        pres_plot_opts: dict[str, Any] = {
            "xaxis_title": r"$s$",
            "yaxis_title": r"$mmHg$",
            **common_plot_opts,
        }

        # Options for flow plots
        flow_plot_opts: dict[str, Any] = {
            "xaxis_title": r"$s$",
            "yaxis_title": r"$\frac{l}{min}$",
            **common_plot_opts,
        }

        # Sequence of plot titles options and labels
        plot_title_sequence = [
            "Inlet pressure",
            "Outlet pressure",
            "Inlet flow",
            "Outlet flow",
        ]
        plot_opts_sequence = [pres_plot_opts] * 2 + [flow_plot_opts] * 2
        plot_label_sequence = [
            "pressure_in",
            "pressure_out",
            "flow_in",
            "flow_out",
        ]

        # Options for 3d, 0d and 0d optimized
        threed_opts: dict[str, Any] = {
            "name": "3D",
            "showlegend": True,
            "color": "white",
            "dash": "dot",
            "width": 4,
        }
        zerod_opts: dict[str, Any] = {
            "name": "0D",
            "showlegend": True,
            "color": "#EF553B",
            "width": 3,
        }
        zerod_opt_opts: dict[str, Any] = {
            "name": "0D optimized",
            "showlegend": True,
            "color": "#636efa",
            "width": 3,
        }

        # Filter for results
        def result_fiter(name: str, label: str) -> pd.DataFrame:
            return results[results["name"] == name][label]

        # Trace sequence
        trace_suffix = ["_3d", "_0d", "_0d_opt"]
        trace_opts = [threed_opts, zerod_opts, zerod_opt_opts]

        for name in branch_data.keys():
            if not name.startswith("branch"):
                continue

            report.add("Results for " + name)
            plots = []
            for plot_title, plot_opts, plot_label in zip(
                plot_title_sequence,
                plot_opts_sequence,
                plot_label_sequence,
            ):
                # Create and append plot
                plots.append(
                    visualizer.Plot2D(
                        title=plot_title, **plot_opts  # type: ignore
                    )
                )
                for sfx, opt in zip(trace_suffix, trace_opts):
                    plots[-1].add_line_trace(
                        x=result_fiter(name + sfx, "time"),
                        y=result_fiter(name + sfx, plot_label),
                        **opt,
                    )
            report.add(plots)

        return report
