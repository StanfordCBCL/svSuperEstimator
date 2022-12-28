"""This module holds the BloodVesselTuning task."""
from __future__ import annotations

import os
from datetime import datetime
from multiprocessing import get_context
from typing import Any

import numpy as np
import orjson
import pandas as pd
from rich import box
from rich.table import Table
from scipy import optimize
from svzerodsolver import runnercpp

from .. import reader, visualizer
from ..reader import CenterlineHandler
from ..reader import utils as readutils
from . import plotutils, taskutils
from .task import Task


class BloodVesselTuning(Task):
    """Blood vessel tuning task.

    Tunes all blood vessels in a 0D simulation file to a 3D simulation result.
    """

    TASKNAME = "blood_vessel_tuning"
    DEFAULTS = {
        "zerod_config_file": None,
        "threed_solution_file": None,
        "num_procs": 1,
        "centerline_padding": False,
        **Task.DEFAULTS,
    }

    # The sequence that the blood vessel parameters are saved in arrays
    _PARAMETER_SEQUENCE = ["R_poiseuille", "C", "L", "stenosis_coefficient"]

    # Number of cycles to simulate blood vessels during optimization
    _OPT_NUM_CYCLES = 5

    # Parameter bounds for optimization
    _OPT_BOUNDS = [(0.0, None), (1.0e-8, None), (1e-12, None), (0.0, None)]

    # Optimization method used by scipy.optimize.minimize
    _OPT_METHOD = "Nelder-Mead"

    def core_run(self) -> None:
        """Core routine of the task."""

        # Loading data from project
        self.log("Loading 0D simulation input file")
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

        # Get 3D simulation time step size from 3d simulation input file
        threed_config_handler = self.project["3d_simulation_input"]
        threed_time_step_size = threed_config_handler.time_step_size
        threed_result_handler = CenterlineHandler.from_file(
            self.config["threed_solution_file"]
        )
        cl_handler = self.project["centerline"]
        self.log("Found 3D simulation time step size:", threed_time_step_size)

        # Map centerline result to 3D simulation
        self.log("Map 3D centerline result to 0D elements")
        branch_data, times = taskutils.map_centerline_result_to_0d_2(
            zerod_config_handler,
            cl_handler,
            threed_config_handler,
            threed_result_handler,
            padding=self.config["centerline_padding"],
        )

        for vessel in zerod_config_handler.vessels.values():
            name = vessel["vessel_name"]
            branch_id, seg_id = name.split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])
            start_values = vessel["zero_d_element_values"]
            branch_data[branch_id][seg_id]["theta_start"] = [
                start_values[n] for n in self._PARAMETER_SEQUENCE
            ]

        # Start optimizing the branches
        results = []
        num_pts = zerod_config_handler.num_pts_per_cycle
        result_labels = ["pressure_in", "pressure_out", "flow_in", "flow_out"]
        with get_context("spawn").Pool(
            processes=self.config["num_procs"]
        ) as pool:
            for branch_id, branch in branch_data.items():
                for seg_id, segment in branch.items():

                    results_data: dict[str, np.ndarray] = {
                        n: taskutils.refine_with_cubic_spline(
                            segment[n], num_pts
                        ).tolist()
                        for n in result_labels
                    }

                    segment_data = {
                        "branch_id": branch_id,
                        "seg_id": seg_id,
                        "times": np.linspace(
                            times[0], times[-1], num_pts
                        ).tolist(),
                        "maxfev": 2000,
                        "num_pts_per_cycle": num_pts,
                        "theta_start": np.array(segment["theta_start"]),
                        "debug": self.config["debug"],
                        "debug_folder": os.path.join(
                            self.output_folder, "debug"
                        ),
                        "name": f"branch{branch_id}_seg{seg_id}",
                        **results_data,
                    }
                    self.log(
                        f"Optimization for branch {branch_id} segment "
                        f"{seg_id} [bold #ff9100]started[/bold #ff9100]"
                    )
                    r = pool.apply_async(
                        self._optimize_blood_vessel,
                        (segment_data,),
                        callback=self._optimize_blood_vessel_callback,
                    )
                    results.append(r)

            # Collect results when processes are complete
            for r in results:
                r.wait()

            # Write results to respective branch in branch data
            for result in [r.get() for r in results]:
                branch_data[result["branch_id"]][result["seg_id"]][
                    "theta_opt"
                ] = {
                    n: result["theta_opt"][j]
                    for j, n in enumerate(self._PARAMETER_SEQUENCE)
                }

        # Update 0D simulation config with optimized parameters
        for vessel_config in zerod_config_handler.vessels.values():
            name = vessel_config["vessel_name"]
            branch_id, seg_id = name.split("_")
            branch_id = int(branch_id[6:])
            seg_id = int(seg_id[3:])
            vessel_config["zero_d_element_values"] = branch_data[branch_id][
                seg_id
            ]["theta_opt"]

        # Improve junctions
        self._make_resistive_junctions(
            zerod_config_handler, branch_data, times
        )

        # Writing data to project
        self.log("Save optimized 0D simulation file")
        zerod_config_handler.to_file(
            os.path.join(self.output_folder, "solver_0d.in")
        )

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
            self.config["threed_solution_file"]
        )

        # Map centerline data to 0D elements
        branch_data, times = taskutils.map_centerline_result_to_0d_2(
            zerod_config_handler,
            self.project["centerline"],
            threed_config_handler,
            threed_result_handler,
            padding=self.config["centerline_padding"],
        )

        # Run simulation for both configurations
        zerod_config_handler.update_simparams(last_cycle_only=True)
        zerod_opt_config_handler.update_simparams(last_cycle_only=True)
        for junction in zerod_config_handler.junctions.values():
            if junction["junction_type"] == "BloodVesselJunction":
                junction["junction_type"] = "NORMAL_JUNCTION"
        zerod_result = runnercpp.run_from_config(zerod_config_handler.data)
        zerod_opt_result = runnercpp.run_from_config(
            zerod_opt_config_handler.data
        )

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
            "flow_in": taskutils.cgs_flow_to_lh,
            "flow_out": taskutils.cgs_flow_to_lh,
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
        branch_data = readutils.get_0d_element_coordinates(self.project)
        model_plot = plotutils.create_3d_geometry_plot_with_vessels(
            self.project, branch_data
        )
        report.add([model_plot])

        report = visualizer.Report()
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
            "yaxis_title": r"$\frac{l}{h}$",
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

    @classmethod
    def _optimize_blood_vessel(cls, segment_data: dict) -> dict:
        """Optimization routine for one blood vessel."""

        # Determine normalization factor for pressure and flow
        pres_norm_factor = np.amax(segment_data["pressure_in"]) - np.amin(
            segment_data["pressure_in"]
        )
        flow_norm_factor = np.amax(segment_data["flow_out"]) - np.amin(
            segment_data["flow_out"]
        )

        bc_inflow = np.array(segment_data["flow_in"])
        outflow = np.array(segment_data["flow_out"])
        inpres = np.array(segment_data["pressure_in"])
        bc_outpres = np.array(segment_data["pressure_out"])

        # Define objective function
        def _objective_function(theta: np.ndarray) -> float:
            num_pts_per_cycle = segment_data["num_pts_per_cycle"]
            (
                inpres_sim,
                outflow_sim,
            ) = BloodVesselTuning._simulate_blood_vessel(
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                bc_times=segment_data["times"],
                bc_inflow=bc_inflow,
                bc_outpres=bc_outpres,
                num_pts_per_cycle=num_pts_per_cycle,
            )

            offset_pres = (np.abs(inpres_sim - inpres) / pres_norm_factor) ** 2
            offset_flow = (
                np.abs(outflow_sim - outflow) / flow_norm_factor
            ) ** 2

            return np.mean(np.concatenate([offset_pres, offset_flow]))

        # Start optimization
        x0 = segment_data["theta_start"].copy()
        x0[1] = 1e-8  # Only good when 3d wall is stiff
        result = optimize.minimize(
            fun=_objective_function,
            x0=x0,
            method="Nelder-Mead",
            options={"maxfev": segment_data["maxfev"], "adaptive": True},
            bounds=cls._OPT_BOUNDS,
        )

        if segment_data["debug"]:
            import matplotlib.pyplot as plt

            os.makedirs(segment_data["debug_folder"], exist_ok=True)
            num_pts_per_cycle = segment_data["num_pts_per_cycle"]
            (
                inpres_sim,
                outflow_sim,
            ) = BloodVesselTuning._simulate_blood_vessel(
                result.x[0],
                result.x[1],
                result.x[2],
                result.x[3],
                bc_times=segment_data["times"],
                bc_inflow=bc_inflow,
                bc_outpres=bc_outpres,
                num_pts_per_cycle=num_pts_per_cycle,
            )
            (
                inpres_sim_start,
                outflow_sim_start,
            ) = BloodVesselTuning._simulate_blood_vessel(
                segment_data["theta_start"][0],
                segment_data["theta_start"][1],
                segment_data["theta_start"][2],
                segment_data["theta_start"][3],
                bc_times=segment_data["times"],
                bc_inflow=bc_inflow,
                bc_outpres=bc_outpres,
                num_pts_per_cycle=num_pts_per_cycle,
            )
            fig, axs = plt.subplots(2, 2, figsize=[10, 10])
            fig.suptitle(segment_data["name"])
            axs[1, 1].remove()

            axs[0, 0].plot(
                segment_data["times"],
                taskutils.cgs_pressure_to_mmgh(inpres_sim_start),
                label="Inlet pressure (before)",
                color="grey",
            )
            axs[0, 0].plot(
                segment_data["times"],
                taskutils.cgs_pressure_to_mmgh(bc_outpres),
                "--",
                label="Outlet pressure (prescribed)",
                color="darkorange",
                dashes=(5, 5),
            )
            axs[0, 0].plot(
                segment_data["times"],
                taskutils.cgs_pressure_to_mmgh(inpres_sim),
                label="Inlet pressure (after)",
                color="black",
            )
            axs[0, 0].plot(
                segment_data["times"],
                taskutils.cgs_pressure_to_mmgh(inpres),
                "--",
                label="Inlet pressure (target)",
                color="red",
                dashes=(5, 5),
            )
            axs[0, 0].set_xlabel("Time [s]")
            axs[0, 0].set_ylabel("Pressure [mmHg]")
            axs[0, 0].legend(loc="upper right", bbox_to_anchor=(1, -0.13))
            axs[0, 0].set_title("Pressure")

            axs[0, 1].plot(
                segment_data["times"],
                taskutils.cgs_flow_to_lmin(outflow_sim_start),
                label="Outlet flow (before)",
                color="grey",
            )
            axs[0, 1].plot(
                segment_data["times"],
                taskutils.cgs_flow_to_lmin(bc_inflow),
                "--",
                label="Inlet flow (prescribed)",
                color="darkorange",
                dashes=(5, 5),
            )
            axs[0, 1].plot(
                segment_data["times"],
                taskutils.cgs_flow_to_lmin(outflow_sim),
                label="Outlet flow (after)",
                color="black",
            )
            axs[0, 1].plot(
                segment_data["times"],
                taskutils.cgs_flow_to_lmin(outflow),
                "--",
                label="Outlet flow (target)",
                color="red",
                dashes=(5, 5),
            )
            axs[0, 1].set_xlabel("Time [s]")
            axs[0, 1].set_ylabel("Flow [l/min]")
            axs[0, 1].legend(loc="upper right", bbox_to_anchor=(1, -0.13))
            axs[0, 1].set_title("Flow")

            axs[1, 0].axis("off")
            x0 = segment_data["theta_start"]
            text = f"Status: {result.success}\n"
            text += f"Evaluations: {result.nfev}\n"
            text += f"Resistance: {x0[0]:.1e} -> {result.x[0]:.1e}\n"
            text += f"Capacitance: {x0[1]:.1e} -> {result.x[1]:.1e}\n"
            text += f"Inductance: {x0[2]:.1e} -> {result.x[2]:.1e}\n"
            text += f"Stenosis coefficient: {x0[3]:.1e} -> {result.x[3]:.1e}"
            axs[1, 0].text(
                0.05,
                0.3,
                text,
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=axs[1, 0].transAxes,
            )

            fig.savefig(
                os.path.join(
                    segment_data["debug_folder"],
                    f"{segment_data['name']}.png",
                )
            )
            plt.close(fig)

        return {
            "theta_opt": result.x,
            "nfev": result.nfev,
            "success": result.success,
            "message": result.message,
            "error": result.fun,
            "error_before": _objective_function(segment_data["theta_start"]),
            **segment_data,
        }

    def _optimize_blood_vessel_callback(self, output: dict) -> None:
        """Callback after optimization to log results."""
        if output["success"]:
            self.log(
                f"Optimization for branch {output['branch_id']} segment "
                f"{output['seg_id']} [bold green]successful[/bold green] "
            )
            table = Table(box=box.HORIZONTALS, show_header=False)
            table.add_column()
            table.add_column(style="cyan")
            table.add_row("number of evaluations", str(output["nfev"]))
            table.add_row(
                "relative error",
                f"{output['error_before']*100:.3f} -> "
                f"{output['error']*100:.3f} %",
            )
            table.add_row(
                "resistance",
                f"{output['theta_start'][0]:.1e} -> "
                f"{output['theta_opt'][0]:.1e}",
            )
            table.add_row(
                "capacitance",
                f"{output['theta_start'][1]:.1e} -> "
                f"{output['theta_opt'][1]:.1e}",
            )
            table.add_row(
                "inductance",
                f"{output['theta_start'][2]:.1e} -> "
                f"{output['theta_opt'][2]:.1e}",
            )
            table.add_row(
                "stenosis coefficient",
                f"{output['theta_start'][3]:.1e} -> "
                f"{output['theta_opt'][3]:.1e}",
            )
            self.log(table)
        else:
            self.log(
                f"Optimization for branch {output['branch_id']} segment "
                f"{output['seg_id']} [bold red]failed[/bold red] with "
                f"message {output['message']} after {output['nfev']} "
                "evaluations."
            )

    @classmethod
    def _simulate_blood_vessel(
        cls,
        R: float,
        C: float,
        L: float,
        stenosis_coefficient: float,
        bc_times: np.ndarray,
        bc_inflow: np.ndarray,
        bc_outpres: np.ndarray,
        num_pts_per_cycle: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run a single-vessel simulation."""

        config = {
            "boundary_conditions": [
                {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {
                        "Q": bc_inflow,
                        "t": bc_times,
                    },
                },
                {
                    "bc_name": "OUTPRES",
                    "bc_type": "PRESSURE",
                    "bc_values": {
                        "P": bc_outpres,
                        "t": bc_times,
                    },
                },
            ],
            "junctions": [],
            "simulation_parameters": {
                "number_of_cardiac_cycles": cls._OPT_NUM_CYCLES,
                "number_of_time_pts_per_cardiac_cycle": num_pts_per_cycle,
                "steady_initial": False,
                "output_last_cycle_only": True,
            },
            "vessels": [
                {
                    "boundary_conditions": {
                        "inlet": "INFLOW",
                        "outlet": "OUTPRES",
                    },
                    "vessel_id": 0,
                    "vessel_name": "branch0_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "C": C,
                        "L": L,
                        "R_poiseuille": R,
                        "stenosis_coefficient": stenosis_coefficient,
                    },
                }
            ],
        }
        try:
            result = np.genfromtxt(
                runnercpp.svzerodsolvercpp.run(
                    orjson.dumps(
                        config,
                        option=orjson.OPT_NAIVE_UTC
                        | orjson.OPT_SERIALIZE_NUMPY,
                    )
                ).splitlines(),
                delimiter=",",
            )
        except RuntimeError:
            return (
                np.full(num_pts_per_cycle, np.nan),
                np.full(num_pts_per_cycle, np.nan),
            )

        # Extract quantities of interest
        sim_inpres = result[1:, 4]
        sim_outflow = result[1:, 3]

        return (
            sim_inpres,
            sim_outflow,
        )

    def _make_resistive_junctions(
        self,
        zerod_handler: reader.SvZeroDSolverInputHandler,
        mapped_data: dict,
        times: np.ndarray,
    ) -> None:
        """Convert normal junctions to resistive junctions."""

        vessel_id_map = zerod_handler.vessel_id_to_name_map

        nodes = zerod_handler.nodes

        num_pts = zerod_handler.num_pts_per_cycle

        junction_nodes = {
            n for n in nodes if n[0].startswith("J") or n[1].startswith("J")
        }

        ele1s = [node[0] for node in junction_nodes]
        target_junctions = set(
            [x for i, x in enumerate(ele1s) if i != ele1s.index(x)]
        )

        junctions = zerod_handler.junctions

        for junction_name in target_junctions:
            junction_data = junctions[junction_name]

            inlet_vessels = junction_data["inlet_vessels"]
            outlet_vessels = junction_data["outlet_vessels"]

            if len(inlet_vessels) > 1:
                raise NotImplementedError(
                    "Multiple inlets are currently not supported."
                )

            junction_values: dict[str, Any] = {
                n: [] for n in self._PARAMETER_SEQUENCE
            }

            inlet_branch_name = vessel_id_map[inlet_vessels[0]]
            branch_id, seg_id = inlet_branch_name.split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

            pressure_in = taskutils.refine_with_cubic_spline(
                mapped_data[branch_id][seg_id]["pressure_out"], num_pts
            ).tolist()

            for ovessel in outlet_vessels:

                outlet_branch_name = vessel_id_map[ovessel]
                branch_id, seg_id = outlet_branch_name.split("_")
                branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

                pressure_out = taskutils.refine_with_cubic_spline(
                    mapped_data[branch_id][seg_id]["pressure_in"], num_pts
                ).tolist()
                flow_out = taskutils.refine_with_cubic_spline(
                    mapped_data[branch_id][seg_id]["flow_in"], num_pts
                ).tolist()

                segment_data = {
                    "times": np.linspace(
                        times[0], times[-1], num_pts
                    ).tolist(),
                    "maxfev": 2000,
                    "num_pts_per_cycle": num_pts,
                    "theta_start": np.array([0.0, 1e-8, 1e-12, 0.0]),
                    "debug": self.config["debug"],
                    "debug_folder": os.path.join(self.output_folder, "debug"),
                    "pressure_in": pressure_in,
                    "pressure_out": pressure_out,
                    "flow_in": flow_out,
                    "flow_out": flow_out,
                    "name": f"{inlet_branch_name}_{vessel_id_map[ovessel]}",
                }

                result = self._optimize_blood_vessel(segment_data)

                result = {
                    n: result["theta_opt"][j]
                    for j, n in enumerate(self._PARAMETER_SEQUENCE)
                }

                for key, value in result.items():
                    junction_values[key].append(value)

            junction_data["junction_type"] = "BloodVesselJunction"
            junction_values_bef = junction_data.get("junction_values", {})
            junction_data["junction_values"] = junction_values

            self.log(
                f"Optimization for junction [cyan]{junction_name}[/cyan] "
                "[bold green]successful[/bold green]"
            )
            table = Table(box=box.HORIZONTALS, show_header=False)
            table.add_column()
            table.add_column(style="cyan")

            if junction_values_bef:

                for i, ovessel in enumerate(outlet_vessels):
                    table.add_row(
                        f"resistance {inlet_branch_name} -> "
                        f"{vessel_id_map[ovessel]}",
                        f"{junction_values_bef['R_poiseuille'][i]:.1e} -> "
                        f"{junction_values['R_poiseuille'][i]:.1e}",
                    )
                    table.add_row(
                        f"capacitance {inlet_branch_name} -> "
                        f"{vessel_id_map[ovessel]}",
                        f"{junction_values_bef['C'][i]:.1e} -> "
                        f"{junction_values['C'][i]:.1e}",
                    )
                    table.add_row(
                        f"inductance {inlet_branch_name} -> "
                        f"{vessel_id_map[ovessel]}",
                        f"{junction_values_bef['L'][i]:.1e} -> "
                        f"{junction_values['L'][i]:.1e}",
                    )
                    table.add_row(
                        f"stenosis_coefficient {inlet_branch_name} -> "
                        f"{vessel_id_map[ovessel]}",
                        f"{junction_values_bef['stenosis_coefficient'][i]:.1e}"
                        f"-> {junction_values['stenosis_coefficient'][i]:.1e}",
                    )
            else:
                for i, ovessel in enumerate(outlet_vessels):
                    table.add_row(
                        f"resistance {inlet_branch_name} -> "
                        f"{vessel_id_map[ovessel]}",
                        f"0.0 -> {junction_values['R_poiseuille'][i]:.1e}",
                    )
                    table.add_row(
                        f"capacitance {inlet_branch_name} -> "
                        f"{vessel_id_map[ovessel]}",
                        f"0.0 -> {junction_values['C'][i]:.1e}",
                    )
                    table.add_row(
                        f"inductance {inlet_branch_name} -> "
                        f"{vessel_id_map[ovessel]}",
                        f"0.0 -> {junction_values['L'][i]:.1e}",
                    )
                    table.add_row(
                        f"stenosis_coefficient {inlet_branch_name} -> "
                        f"{vessel_id_map[ovessel]}",
                        "0.0 -> "
                        f"{junction_values['stenosis_coefficient'][i]:.1e}",
                    )

            self.log(table)
