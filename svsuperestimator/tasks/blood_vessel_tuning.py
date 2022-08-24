from __future__ import annotations

import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rich import box
from rich.table import Table
from scipy import optimize
from scipy.interpolate import CubicSpline
from svzerodsolver import runnercpp

from svsuperestimator.problems._plotutils import (
    cgs_flow_to_lh,
    cgs_pressure_to_mmgh,
)

from .. import visualizer
from ..problems import plotutils
from ..reader.centerline import map_centerline_result_to_0d
from .task import Task

from . import utils


class BloodVesselTuning(Task):

    TASKNAME = "BloodVesselTuning"
    DEFAULTS = {
        "threed_solution_file": None,
        "num_procs": 1,
        "maxfev": 2000,
    }

    def core_run(self):
        """Core routine of the task."""

        # Loading data from project
        self.log("Loading 0D simulation input file")
        zerod_config = self.project["rom_simulation_config"]
        self.log("Reading 3D simualtion input file")
        threed_input = self.project["3d_simulation_input"]

        # Get 3D simulation time step size from 3d simulation input file
        threed_time_step_size = float(
            re.findall(r"Time Step Size: -?\d+\.?\d*", threed_input)[
                0
            ].split()[-1]
        )
        self.log(
            "Extracted 3D simulation time step size:", threed_time_step_size
        )

        # Map centerline result to 3D simulation
        self.log("Map 3D centerline result to 0D elements")
        parameters = ["R_poiseuille", "C", "L", "stenosis_coefficient"]
        branch_data, times = map_centerline_result_to_0d(
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
                start_values[n] for n in parameters
            ]

        # Start optimizing the branches
        pool = Pool(processes=self.config["num_procs"])
        results = []
        num_cycles = 5
        num_pts = zerod_config["simulation_parameters"][
            "number_of_time_pts_per_cardiac_cycle"
        ]
        for branch_id, branch in branch_data.items():
            for seg_id, segment in branch.items():

                segment_data = {
                    "branch_id": branch_id,
                    "seg_id": seg_id,
                    "times": np.linspace(times[0], times[-1], num_pts),
                    "inflow": utils.refine_with_cubic_spline(
                        times, segment["flow_0"], num_pts
                    ),
                    "outpres": utils.refine_with_cubic_spline(
                        times, segment["pressure_1"], num_pts
                    ),
                    "inpres": utils.refine_with_cubic_spline(
                        times, segment["pressure_0"], num_pts
                    ),
                    "outflow": utils.refine_with_cubic_spline(
                        times, segment["flow_1"], num_pts
                    ),
                    "maxfev": self.config["maxfev"],
                    "num_cycles": num_cycles,
                    "num_pts_per_cycle": num_pts,
                    "theta_start": segment["theta_start"],
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
        for r in results:
            r.wait()
        results = [r.get() for r in results]

        for result in results:
            branch_data[result["branch_id"]][result["seg_id"]]["theta_opt"] = {
                n: result["theta_opt"][j] for j, n in enumerate(parameters)
            }

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

        threed_input = self.project["3d_simulation_input"]
        threed_time_step_size = float(
            re.findall(r"Time Step Size: -?\d+\.?\d*", threed_input)[
                0
            ].split()[-1]
        )

        branch_data, times = map_centerline_result_to_0d(
            centerline_file, zerod_config, threed_time_step_size
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

        for branch_id, branch in branch_data.items():

            for seg_id, segment in branch.items():
                vessel_id = segment["vessel_id"]

                # Append 3d result
                results_new = pd.DataFrame()
                results_new["name"] = [
                    f"branch{branch_id}_seg{seg_id}_3d"
                ] * len(times)
                results_new["time"] = times
                results_new["pressure_in"] = cgs_pressure_to_mmgh(
                    segment["pressure_0"]
                )
                results_new["pressure_out"] = cgs_pressure_to_mmgh(
                    segment["pressure_1"]
                )
                results_new["flow_in"] = cgs_flow_to_lh(segment["flow_0"])
                results_new["flow_out"] = cgs_flow_to_lh(segment["flow_1"])
                results = results.append(results_new)

                # Append 0d result
                results_new = pd.DataFrame()
                results_new["name"] = [
                    f"branch{branch_id}_seg{seg_id}_0d"
                ] * len(sim_times)
                results_new["time"] = sim_times
                results_new["pressure_in"] = cgs_pressure_to_mmgh(
                    zerod_result[zerod_result.name == f"V{vessel_id}"][
                        "pressure_in"
                    ][-pts_per_cycle:]
                )
                results_new["pressure_out"] = cgs_pressure_to_mmgh(
                    zerod_result[zerod_result.name == f"V{vessel_id}"][
                        "pressure_out"
                    ][-pts_per_cycle:]
                )
                results_new["flow_in"] = cgs_flow_to_lh(
                    zerod_result[zerod_result.name == f"V{vessel_id}"][
                        "flow_in"
                    ][-pts_per_cycle:]
                )
                results_new["flow_out"] = cgs_flow_to_lh(
                    zerod_result[zerod_result.name == f"V{vessel_id}"][
                        "flow_out"
                    ][-pts_per_cycle:]
                )
                results = results.append(results_new)

                # Append 0d optimized result
                results_new = pd.DataFrame()
                results_new["name"] = [
                    f"branch{branch_id}_seg{seg_id}_0d_opt"
                ] * len(sim_times)
                results_new["time"] = sim_times
                results_new["pressure_in"] = cgs_pressure_to_mmgh(
                    zerod_opt_result[zerod_opt_result.name == f"V{vessel_id}"][
                        "pressure_in"
                    ][-pts_per_cycle:]
                )
                results_new["pressure_out"] = cgs_pressure_to_mmgh(
                    zerod_opt_result[zerod_opt_result.name == f"V{vessel_id}"][
                        "pressure_out"
                    ][-pts_per_cycle:]
                )
                results_new["flow_in"] = cgs_flow_to_lh(
                    zerod_opt_result[zerod_opt_result.name == f"V{vessel_id}"][
                        "flow_in"
                    ][-pts_per_cycle:]
                )
                results_new["flow_out"] = cgs_flow_to_lh(
                    zerod_opt_result[zerod_opt_result.name == f"V{vessel_id}"][
                        "flow_out"
                    ][-pts_per_cycle:]
                )
                results = results.append(results_new)

        results.to_csv(os.path.join(self.output_folder, "results.csv"))

    def generate_report(self):

        results = pd.read_csv(os.path.join(self.output_folder, "results.csv"))

        zerod_config = self.project["rom_simulation_config"]

        centerline_file = self.config["threed_solution_file"]

        threed_input = self.project["3d_simulation_input"]
        threed_time_step_size = float(
            re.findall(r"Time Step Size: -?\d+\.?\d*", threed_input)[
                0
            ].split()[-1]
        )

        branch_data, _ = map_centerline_result_to_0d(
            centerline_file, zerod_config, threed_time_step_size
        )

        bc_plot = plotutils.create_3d_geometry_plot_with_vessels(
            self.project, branch_data
        )

        report = visualizer.Report()
        report.add([bc_plot])
        plot_opts = {
            "static": True,
            "width": 750,
            "height": 400,
        }
        for branch_id, branch in branch_data.items():

            for seg_id, segment in branch.items():

                vessel_name = f"branch{branch_id}_seg{seg_id}"

                inpres_plot = visualizer.Plot2D(
                    title=f"Inlet pressure branch {branch_id} segment {seg_id}",
                    xaxis_title=r"$s$",
                    yaxis_title=r"$mmHg$",
                    **plot_opts,
                )
                inpres_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_3d"]["time"],
                    y=results[results["name"] == vessel_name + "_3d"][
                        "pressure_in"
                    ],
                    name="3D",
                    showlegend=True,
                    dash="dot",
                    width=3,
                )
                inpres_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_0d"]["time"],
                    y=results[results["name"] == vessel_name + "_0d"][
                        "pressure_in"
                    ],
                    name="0D",
                    showlegend=True,
                    width=2,
                )
                inpres_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_0d_opt"][
                        "time"
                    ],
                    y=results[results["name"] == vessel_name + "_0d_opt"][
                        "pressure_in"
                    ],
                    name="0D optimized",
                    showlegend=True,
                    width=2,
                )

                outpres_plot = visualizer.Plot2D(
                    title=f"Outlet pressure branch {branch_id} segment {seg_id}",
                    xaxis_title=r"$s$",
                    yaxis_title=r"$mmHg$",
                    **plot_opts,
                )
                outpres_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_3d"]["time"],
                    y=results[results["name"] == vessel_name + "_3d"][
                        "pressure_out"
                    ],
                    name="3D",
                    showlegend=True,
                    dash="dot",
                    width=3,
                )
                outpres_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_0d"]["time"],
                    y=results[results["name"] == vessel_name + "_0d"][
                        "pressure_out"
                    ],
                    name="0D",
                    showlegend=True,
                    width=2,
                )
                outpres_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_0d_opt"][
                        "time"
                    ],
                    y=results[results["name"] == vessel_name + "_0d_opt"][
                        "pressure_out"
                    ],
                    name="0D optimized",
                    showlegend=True,
                    width=2,
                )

                inflow_plot = visualizer.Plot2D(
                    title=f"Inlet flow branch {branch_id} segment {seg_id}",
                    xaxis_title=r"$s$",
                    yaxis_title=r"$\frac{l}{h}$",
                    **plot_opts,
                )
                inflow_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_3d"]["time"],
                    y=results[results["name"] == vessel_name + "_3d"][
                        "flow_in"
                    ],
                    name="3D",
                    showlegend=True,
                    dash="dot",
                    width=3,
                )
                inflow_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_0d"]["time"],
                    y=results[results["name"] == vessel_name + "_0d"][
                        "flow_in"
                    ],
                    name="0D",
                    showlegend=True,
                    width=2,
                )
                inflow_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_0d_opt"][
                        "time"
                    ],
                    y=results[results["name"] == vessel_name + "_0d_opt"][
                        "flow_in"
                    ],
                    name="0D optimized",
                    showlegend=True,
                    width=2,
                )

                outflow_plot = visualizer.Plot2D(
                    title=f"Outlet flow branch {branch_id} segment {seg_id}",
                    xaxis_title=r"$s$",
                    yaxis_title=r"$\frac{l}{h}$",
                    **plot_opts,
                )
                outflow_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_3d"]["time"],
                    y=results[results["name"] == vessel_name + "_3d"][
                        "flow_out"
                    ],
                    name="3D",
                    showlegend=True,
                    dash="dot",
                    width=3,
                )
                outflow_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_0d"]["time"],
                    y=results[results["name"] == vessel_name + "_0d"][
                        "flow_out"
                    ],
                    name="0D",
                    showlegend=True,
                    width=2,
                )
                outflow_plot.add_line_trace(
                    x=results[results["name"] == vessel_name + "_0d_opt"][
                        "time"
                    ],
                    y=results[results["name"] == vessel_name + "_0d_opt"][
                        "flow_out"
                    ],
                    name="0D optimized",
                    showlegend=True,
                    width=2,
                )
                report.add([inpres_plot, inflow_plot])
                report.add([outpres_plot, outflow_plot])

        return report

    @staticmethod
    def _optimize_blood_vessel(segment_data):

        pres_norm_factor = np.amax(segment_data["inpres"]) - np.amin(
            segment_data["inpres"]
        )
        flow_norm_factor = np.amax(segment_data["outflow"]) - np.amin(
            segment_data["outflow"]
        )

        def objective_function(theta):
            (
                _,
                inpres_sim,
                outflow_sim,
            ) = BloodVesselTuning._simulate_blood_vessel(
                *theta,
                bc_times=segment_data["times"],
                bc_inflow=segment_data["inflow"],
                bc_outpres=segment_data["outpres"],
                num_cycles=segment_data["num_cycles"],
                num_pts_per_cycle=segment_data["num_pts_per_cycle"],
            )

            mse = np.linalg.norm(
                (inpres_sim - segment_data["inpres"]) / pres_norm_factor
            ) + np.linalg.norm(
                (outflow_sim - segment_data["outflow"]) / flow_norm_factor
            )

            return mse

        result = optimize.minimize(
            fun=objective_function,
            x0=segment_data["theta_start"],
            method="Nelder-Mead",
            options={"maxfev": segment_data["maxfev"], "adaptive": True},
            bounds=[(None, None), (1e-12, None), (1e-12, None), (None, None)],
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

    @staticmethod
    def _simulate_blood_vessel(
        R,
        C,
        L,
        stenosis_coefficient,
        bc_times,
        bc_inflow,
        bc_outpres,
        num_cycles,
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
                "number_of_cardiac_cycles": num_cycles,
                "number_of_time_pts_per_cardiac_cycle": num_pts_per_cycle,
                "steady_initial": True,
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

        sim_times = np.array(result["time"])[-num_pts_per_cycle:]
        sim_times = sim_times - np.amin(sim_times)
        sim_inpres = np.array(result["pressure_in"])[-num_pts_per_cycle:]
        sim_outflow = np.array(result["flow_out"])[-num_pts_per_cycle:]

        return sim_times, sim_inpres, sim_outflow
