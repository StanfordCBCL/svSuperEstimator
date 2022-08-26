from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rich import box
from rich.table import Table
from scipy import optimize
from svzerodsolver import runnercpp

from .. import visualizer
from . import plotutils
from . import utils
from .task import Task


class BloodVesselTuning(Task):
    """Blood vessel tuning task.

    Tunes all blood vessels in a 0D simulation file to a 3D simualtion result.
    """

    TASKNAME = "BloodVesselTuning"
    DEFAULTS = {
        "threed_solution_file": None,
        "num_procs": 1,
        "maxfev": 2000,
    }

    # The sequence that the blood vessel parameters are saved in arrays
    _PARAMETER_SEQUENCE = ["R_poiseuille", "C", "L", "stenosis_coefficient"]

    # Number of cycles to simulate blood vessels during optimization
    _OPT_NUM_CYCLES = 5

    # Parameter bounds for optimization
    _OPT_BOUNDS = [(None, None), (1.0e-8, None), (1e-12, None), (None, None)]

    # Optimization method used by scipy.optimize.minimize
    _OPT_METHOD = "Nelder-Mead"

    def core_run(self):
        """Core routine of the task."""

        # Loading data from project
        self.log("Loading 0D simulation input file")
        zerod_config = self.project["rom_simulation_config"]

        # Get 3D simulation time step size from 3d simulation input file
        threed_time_step_size = self.project.time_step_size_3d
        self.log("Found 3D simulation time step size:", threed_time_step_size)

        # Map centerline result to 3D simulation
        self.log("Map 3D centerline result to 0D elements")
        branch_data, times = utils.map_centerline_result_to_0d(
            self.config["threed_solution_file"],
            zerod_config,
            threed_time_step_size,
        )
        for vessel in zerod_config["vessels"]:
            name = vessel["vessel_name"]
            branch_id, seg_id = name.split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])
            start_values = vessel["zero_d_element_values"]
            branch_data[branch_id][seg_id]["theta_start"] = [
                start_values[n] for n in self._PARAMETER_SEQUENCE
            ]

        # Start optimizing the branches
        pool = Pool(processes=self.config["num_procs"])
        results = []
        num_pts = zerod_config["simulation_parameters"][
            "number_of_time_pts_per_cardiac_cycle"
        ]
        result_labels = ["pressure_in", "pressure_out", "flow_in", "flow_out"]
        for branch_id, branch in branch_data.items():
            for seg_id, segment in branch.items():

                segment_data = {
                    "branch_id": branch_id,
                    "seg_id": seg_id,
                    "times": np.linspace(times[0], times[-1], num_pts),
                    "maxfev": self.config["maxfev"],
                    "num_pts_per_cycle": num_pts,
                    "theta_start": np.array(segment["theta_start"]),
                    **{
                        n: utils.refine_with_cubic_spline(segment[n], num_pts)
                        for n in result_labels
                    },
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
        results = [r.get() for r in results]

        # Write results to respective branch in branch data
        for result in results:
            branch_data[result["branch_id"]][result["seg_id"]]["theta_opt"] = {
                n: result["theta_opt"][j]
                for j, n in enumerate(self._PARAMETER_SEQUENCE)
            }

        # Update 0D simulation config with optimized parameters
        for i, vessel_config in enumerate(zerod_config["vessels"]):
            name = vessel_config["vessel_name"]
            branch_id, seg_id = name.split("_")
            branch_id = int(branch_id[6:])
            seg_id = int(seg_id[3:])
            zerod_config["vessels"][i]["zero_d_element_values"] = branch_data[
                branch_id
            ][seg_id]["theta_opt"]

        # Writing data to project
        self.log("Save optimized 0D simulation file")
        self.project["rom_simulation_config_optimized"] = zerod_config

    def post_run(self):

        zerod_config = self.project["rom_simulation_config"]
        zerod_opt_config = self.project["rom_simulation_config_optimized"]

        centerline_file = self.config["threed_solution_file"]

        branch_data, times = utils.map_centerline_result_to_0d(
            centerline_file, zerod_config, self.project.time_step_size_3d
        )

        zerod_result = runnercpp.run_from_config(zerod_config)
        zerod_opt_result = runnercpp.run_from_config(zerod_opt_config)

        pts_per_cycle = zerod_config["simulation_parameters"][
            "number_of_time_pts_per_cardiac_cycle"
        ]
        columns = [
            "name",
            "time",
            "pressure_in",
            "pressure_out",
            "flow_in",
            "flow_out",
        ]

        results = pd.DataFrame(columns=columns)

        sim_times = np.array(
            zerod_result[zerod_result.name == f"V0"]["time"][-pts_per_cycle:]
        )
        sim_times -= np.amin(sim_times)

        filter = {
            "pressure_in": utils.cgs_pressure_to_mmgh,
            "pressure_out": utils.cgs_pressure_to_mmgh,
            "flow_in": utils.cgs_flow_to_lh,
            "flow_out": utils.cgs_flow_to_lh,
        }

        for branch_id, branch in branch_data.items():

            for seg_id, segment in branch.items():
                vessel_id = segment["vessel_id"]
                vessel_name = f"branch{branch_id}_seg{seg_id}"

                # Append 3d result
                results_new = pd.DataFrame(
                    dict(
                        name=[vessel_name + "_3d"] * len(times),
                        time=times,
                        **{n: f(segment[n]) for n, f in filter.items()},
                    )
                )
                results = results.append(results_new)

                # Append 0d result
                vessel_result = zerod_result[
                    zerod_result.name == f"V{vessel_id}"
                ]
                results_new = pd.DataFrame(
                    dict(
                        name=[vessel_name + "_0d"] * len(sim_times),
                        time=sim_times,
                        **{
                            n: f(vessel_result[n][-pts_per_cycle:])
                            for n, f in filter.items()
                        },
                    )
                )
                results = results.append(results_new)

                # Append 0d optimized result
                vessel_result = zerod_opt_result[
                    zerod_opt_result.name == f"V{vessel_id}"
                ]
                results_new = pd.DataFrame(
                    dict(
                        name=[vessel_name + "_0d_opt"] * len(sim_times),
                        time=sim_times,
                        **{
                            n: f(vessel_result[n][-pts_per_cycle:])
                            for n, f in filter.items()
                        },
                    )
                )
                results = results.append(results_new)

        results.to_csv(os.path.join(self.output_folder, "results.csv"))

    def generate_report(self):

        results = pd.read_csv(os.path.join(self.output_folder, "results.csv"))

        branch_data, _ = utils.map_centerline_result_to_0d(
            self.config["threed_solution_file"],
            self.project["rom_simulation_config"],
            self.project.time_step_size_3d,
        )

        model_plot = plotutils.create_3d_geometry_plot_with_vessels(
            self.project, branch_data
        )

        report = visualizer.Report()
        report.add([model_plot])

        common_plot_opts = {
            "static": True,
            "width": 750,
            "height": 400,
        }
        pres_plot_opts = {
            "xaxis_title": r"$s$",
            "yaxis_title": r"$mmHg$",
            **common_plot_opts,
        }

        flow_plot_opts = {
            "xaxis_title": r"$s$",
            "yaxis_title": r"$\frac{l}{h}$",
            **common_plot_opts,
        }

        plot_title_sequence = [
            "Inlet pressure branch {} segment {}",
            "Outlet pressure branch {} segment {}",
            "Inlet flow branch {} segment {}",
            "Outlet flow branch {} segment {}",
        ]
        plot_opts_sequence = [pres_plot_opts] * 2 + [flow_plot_opts] * 2
        plot_label_sequence = [
            "pressure_in",
            "pressure_out",
            "flow_in",
            "flow_out",
        ]

        threed_opts = {
            "name": "3D",
            "showlegend": True,
            "color": "white",
            "dash": "dot",
            "width": 4,
        }
        zerod_opts = {
            "name": "0D",
            "showlegend": True,
            "color": "red",
            "width": 2,
        }
        zerod_opt_opts = {
            "name": "0D optimized",
            "showlegend": True,
            "color": "blue",
            "width": 2,
        }
        result_fiter = lambda name, label: results[results["name"] == name][
            label
        ]
        trace_suffix = ["_3d", "_0d", "_0d_opt"]
        trace_opts = [threed_opts, zerod_opts, zerod_opt_opts]

        for branch_id, branch in branch_data.items():

            for seg_id, segment in branch.items():

                name = f"branch{branch_id}_seg{seg_id}"

                report.add("Results for " + name)
                plots = []
                for plot_title, plot_opts, plot_label in zip(
                    plot_title_sequence,
                    plot_opts_sequence,
                    plot_label_sequence,
                ):

                    plots.append(
                        visualizer.Plot2D(
                            title=plot_title.format(branch_id, seg_id),
                            **plot_opts,
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
    def _optimize_blood_vessel(cls, segment_data):

        pres_norm_factor = np.mean(segment_data["pressure_in"])
        flow_norm_factor = np.amax(segment_data["flow_out"]) - np.amin(
            segment_data["flow_out"]
        )

        def objective_function(theta):
            (
                _,
                inpres_sim,
                outflow_sim,
            ) = BloodVesselTuning._simulate_blood_vessel(
                *theta,
                bc_times=segment_data["times"],
                bc_inflow=segment_data["flow_in"],
                bc_outpres=segment_data["pressure_out"],
                num_pts_per_cycle=segment_data["num_pts_per_cycle"],
            )

            offset_pres = (
                np.abs(inpres_sim - segment_data["pressure_in"])
                / pres_norm_factor
            )
            offset_flow = (
                np.abs(outflow_sim - segment_data["flow_out"])
                / flow_norm_factor
            )

            return np.mean(np.concatenate([offset_pres, offset_flow]))

        x0 = segment_data["theta_start"].copy()
        x0[1] = 1e-8  # Only good when 3d wall is stiff
        result = optimize.minimize(
            fun=objective_function,
            x0=x0,
            method=cls._OPT_METHOD,
            options={"maxfev": segment_data["maxfev"], "adaptive": True},
            bounds=cls._OPT_BOUNDS,
        )

        return {
            "theta_opt": result.x,
            "nfev": result.nfev,
            "success": result.success,
            "message": result.message,
            "error": result.fun,
            "error_before": objective_function(segment_data["theta_start"]),
            **segment_data,
        }

    def _optimize_blood_vessel_callback(self, output):
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
                f"{output['error_before']*100:.1f} -> {output['error']*100:.1f} %",
            )
            table.add_row(
                "resistance",
                f"{output['theta_start'][0]:.1e} -> {output['theta_opt'][0]:.1e}",
            )
            table.add_row(
                "capacitance",
                f"{output['theta_start'][1]:.1e} -> {output['theta_opt'][1]:.1e}",
            )
            table.add_row(
                "inductance",
                f"{output['theta_start'][2]:.1e} -> {output['theta_opt'][2]:.1e}",
            )
            table.add_row(
                "stenosis coefficient",
                f"{output['theta_start'][3]:.1e} -> {output['theta_opt'][3]:.1e}",
            )
            self.log(table)
        else:
            self.log(
                f"Optimization for branch {output['branch_id']} segment "
                f"{output['seg_id']} [bold red]failed[/bold red] with "
                f"message: {output['message']}"
            )

    @classmethod
    def _simulate_blood_vessel(
        cls,
        R,
        C,
        L,
        stenosis_coefficient,
        bc_times,
        bc_inflow,
        bc_outpres,
        num_pts_per_cycle,
    ):

        config = {
            "boundary_conditions": [
                {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {
                        "Q": bc_inflow.tolist(),
                        "t": bc_times.tolist(),
                    },
                },
                {
                    "bc_name": "OUTPRES",
                    "bc_type": "PRESSURE",
                    "bc_values": {
                        "P": bc_outpres.tolist(),
                        "t": bc_times.tolist(),
                    },
                },
            ],
            "junctions": [],
            "simulation_parameters": {
                "number_of_cardiac_cycles": cls._OPT_NUM_CYCLES,
                "number_of_time_pts_per_cardiac_cycle": num_pts_per_cycle,
                "steady_initial": False,
            },
            "vessels": [
                {
                    "boundary_conditions": {
                        "inlet": "INFLOW",
                        "outlet": "OUTPRES",
                    },
                    "vessel_id": 0,
                    "vessel_length": 10.0,
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

        result = runnercpp.run_from_config(config)

        sim_times = np.array(result["time"])
        sim_inpres = np.array(result["pressure_in"])
        sim_outflow = np.array(result["flow_out"])

        # Extract last cardiac cycle
        sim_times = sim_times[-num_pts_per_cycle:]
        sim_times = sim_times - np.amin(sim_times)
        sim_inpres = sim_inpres[-num_pts_per_cycle:]
        sim_outflow = sim_outflow[-num_pts_per_cycle:]

        return sim_times, sim_inpres, sim_outflow
