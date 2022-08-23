from __future__ import annotations
from itertools import cycle
from time import time

from .. import (
    model as mdl,
    solver as slv,
    visualizer,
)
from ..problems import plotutils
import pickle
import numpy as np
from .. import visualizer


from ..reader import SimVascularProject

from .task import Task

from scipy import interpolate

from svzerodsolver import runnercpp
from scipy import optimize

from multiprocessing import Pool

from ..reader.centerline import map_centerline_result_to_0d


from svsuperestimator.problems._plotutils import (
    cgs_flow_to_lh,
    cgs_pressure_to_mmgh,
)


class BloodVesselTuning(Task):

    TASKNAME = "Blood Vessel Tuning"

    def __init__(self, project: SimVascularProject, console=None):
        super().__init__(project, console=console)
        self.config = {
            "threed_solution_file": None,
            "threed_time_step_size": None,
            "num_procs": 1,
            "maxfev": 2000,
        }
        self.model = mdl.ZeroDModel(self.project)
        self.solver = slv.ZeroDSolver(cpp=True)

    def run(self, config):
        super().run(config)

        start = time()

        # Read 0d data
        zerod_config = self.project["rom_simulation_config"]

        mapped_data = map_centerline_result_to_0d(
            self.config["threed_solution_file"],
            zerod_config,
            self.config["threed_time_step_size"],
        )
        branch_data = mapped_data["branchdata"]

        parameters = ["R_poiseuille", "C", "L", "stenosis_coefficient"]

        for vessel in zerod_config["vessels"]:

            name = vessel["vessel_name"]
            branch_id, seg_id = name.split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

            start_values = vessel["zero_d_element_values"]
            branch_data[branch_id][seg_id]["theta_start"] = [
                start_values[n] for n in parameters
            ]

        # Start optimizing the branches
        with self.console.status(
            "[bold #ff9100]Waiting for optimizations to complete",
            spinner="simpleDotsScrolling",
            spinner_style="bold #ff9100",
        ) as _:
            pool = Pool(processes=self.config["num_procs"])
            results = []
            for branch_id, branch in branch_data.items():
                for seg_id in range(len(branch)):
                    segment_data = {
                        "branch_id": branch_id,
                        "seg_id": seg_id,
                        "timesteps": mapped_data["timesteps"],
                        "maxfev": self.config["maxfev"],
                        **branch_data[branch_id][seg_id],
                    }
                    self.print(
                        f"Starting optimization for branch {branch_id} segment {seg_id}"
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

        self.project["rom_simulation_config_optimized"] = zerod_config

        self.console.rule(f"Completed in {time()-start:.1f} seconds")

    def postprocess(self):
        report = self.generate_report()
        report.to_html(self.output_folder)

    def generate_report(self):

        zerod_config = self.project["rom_simulation_config"]
        zerod_opt_config = self.project["rom_simulation_config_optimized"]

        centerline_file = "/Users/stanford/data/3d_centerline/0069_0001.vtp"

        mapped_data = map_centerline_result_to_0d(
            centerline_file, zerod_config, 0.000339
        )
        times = mapped_data["timesteps"]

        bc_plot = plotutils.create_3d_geometry_plot_with_vessels(
            self.project, mapped_data
        )

        zerod_result = runnercpp.run_from_config(zerod_config)
        zerod_opt_result = runnercpp.run_from_config(zerod_opt_config)

        pts_per_cycle = zerod_config["simulation_parameters"][
            "number_of_time_pts_per_cardiac_cycle"
        ]

        report = visualizer.Report()
        report.add([bc_plot])
        for branch_id, branch in mapped_data["branchdata"].items():

            for seg_id in range(len(branch)):
                segment = branch[seg_id]
                report.add(f"Branch {branch_id} segment {seg_id}")
                vessel_id = segment["vessel_id"]

                sim_times = np.array(
                    zerod_result[zerod_result.name == f"V{vessel_id}"]["time"][
                        -pts_per_cycle:
                    ]
                )
                sim_times -= np.amin(sim_times)

                inpres_plot = visualizer.Plot2D(
                    title=f"Inlet pressure branch {branch_id} segment {seg_id}",
                    xaxis_title=r"$s$",
                    yaxis_title=r"$mmHg$",
                )
                inpres_plot.add_line_trace(
                    x=times,
                    y=cgs_pressure_to_mmgh(
                        segment["pressure_0"][-pts_per_cycle:]
                    ),
                    name="3D",
                    showlegend=True,
                    dash="dot",
                    width=3,
                )
                inpres_plot.add_line_trace(
                    x=sim_times,
                    y=cgs_pressure_to_mmgh(
                        zerod_result[zerod_result.name == f"V{vessel_id}"][
                            "pressure_in"
                        ][-pts_per_cycle:]
                    ),
                    name="0D",
                    showlegend=True,
                    width=2,
                )
                inpres_plot.add_line_trace(
                    x=sim_times,
                    y=cgs_pressure_to_mmgh(
                        zerod_opt_result[
                            zerod_opt_result.name == f"V{vessel_id}"
                        ]["pressure_in"][-pts_per_cycle:]
                    ),
                    name="0D optimized",
                    showlegend=True,
                    width=2,
                )

                outpres_plot = visualizer.Plot2D(
                    title=f"Outlet pressure branch {branch_id} segment {seg_id}",
                    xaxis_title=r"$s$",
                    yaxis_title=r"$mmHg$",
                )
                outpres_plot.add_line_trace(
                    x=times,
                    y=cgs_pressure_to_mmgh(segment["pressure_1"]),
                    name="3D",
                    showlegend=True,
                    dash="dot",
                    width=3,
                )
                outpres_plot.add_line_trace(
                    x=sim_times,
                    y=cgs_pressure_to_mmgh(
                        zerod_result[zerod_result.name == f"V{vessel_id}"][
                            "pressure_out"
                        ][-pts_per_cycle:]
                    ),
                    name="0D",
                    showlegend=True,
                    width=2,
                )
                outpres_plot.add_line_trace(
                    x=sim_times,
                    y=cgs_pressure_to_mmgh(
                        zerod_opt_result[
                            zerod_opt_result.name == f"V{vessel_id}"
                        ]["pressure_out"][-pts_per_cycle:]
                    ),
                    name="0D optimized",
                    showlegend=True,
                    width=2,
                )

                inflow_plot = visualizer.Plot2D(
                    title=f"Inlet flow branch {branch_id} segment {seg_id}",
                    xaxis_title=r"$s$",
                    yaxis_title=r"$\frac{l}{h}$",
                )
                inflow_plot.add_line_trace(
                    x=times,
                    y=cgs_flow_to_lh(segment["flow_0"]),
                    name="3D",
                    showlegend=True,
                    dash="dot",
                    width=3,
                )
                inflow_plot.add_line_trace(
                    x=sim_times,
                    y=cgs_flow_to_lh(
                        zerod_result[zerod_result.name == f"V{vessel_id}"][
                            "flow_in"
                        ][-pts_per_cycle:]
                    ),
                    name="0D",
                    showlegend=True,
                    width=2,
                )
                inflow_plot.add_line_trace(
                    x=sim_times,
                    y=cgs_flow_to_lh(
                        zerod_opt_result[
                            zerod_opt_result.name == f"V{vessel_id}"
                        ]["flow_in"][-pts_per_cycle:]
                    ),
                    name="0D optimized",
                    showlegend=True,
                    width=2,
                )

                outflow_plot = visualizer.Plot2D(
                    title=f"Outlet flow branch {branch_id} segment {seg_id}",
                    xaxis_title=r"$s$",
                    yaxis_title=r"$\frac{l}{h}$",
                )
                outflow_plot.add_line_trace(
                    x=times,
                    y=cgs_flow_to_lh(segment["flow_1"]),
                    name="3D",
                    showlegend=True,
                    dash="dot",
                    width=3,
                )
                outflow_plot.add_line_trace(
                    x=sim_times,
                    y=cgs_flow_to_lh(
                        zerod_result[zerod_result.name == f"V{vessel_id}"][
                            "flow_out"
                        ][-pts_per_cycle:]
                    ),
                    name="0D",
                    showlegend=True,
                    width=2,
                )
                outflow_plot.add_line_trace(
                    x=sim_times,
                    y=cgs_flow_to_lh(
                        zerod_opt_result[
                            zerod_opt_result.name == f"V{vessel_id}"
                        ]["flow_out"][-pts_per_cycle:]
                    ),
                    name="0D optimized",
                    showlegend=True,
                    width=2,
                )
                report.add([inpres_plot, inflow_plot])
                report.add([outpres_plot, outflow_plot])

        return report

    @staticmethod
    def _optimize_blood_vessel(segment_data):

        pres_norm_factor = np.amax(segment_data["pressure_0"]) - np.amin(
            segment_data["pressure_0"]
        )
        flow_norm_factor = np.amax(segment_data["flow_1"]) - np.amin(
            segment_data["flow_1"]
        )

        def objective_function(theta_log):
            theta = np.exp(theta_log)
            inpres_sim, outflow_sim = BloodVesselTuning._simulate_blood_vessel(
                *theta,
                times=segment_data["timesteps"],
                bc_inflow=segment_data["flow_0"],
                bc_outpres=segment_data["pressure_1"],
            )

            mse = np.linalg.norm(
                (inpres_sim - segment_data["pressure_0"]) / pres_norm_factor
            ) + np.linalg.norm(
                (outflow_sim - segment_data["flow_1"]) / flow_norm_factor
            )

            return mse

        result = optimize.minimize(
            fun=objective_function,
            x0=np.log(segment_data["theta_start"]),
            method="Nelder-Mead",
            options={"maxfev": segment_data["maxfev"]},
        )
        return {
            "theta_opt": np.exp(result.x),
            "nfev": result.nfev,
            "success": result.success,
            "message": result.message,
            **segment_data,
        }

    def _optimize_blood_vessel_callback(self, output):
        if output["success"]:
            self.print(
                f"Optimization for branch {output['branch_id']} segment "
                f"{output['seg_id']} [bold green]successful[/bold green] "
                f"after {output['nfev']} evaluations"
            )
        else:
            self.print(
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
        times,
        bc_inflow,
        bc_outpres,
    ):

        bc_inflow[-1] = bc_inflow[0]
        bc_outpres[-1] = bc_outpres[0]

        config = {
            "boundary_conditions": [
                {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {
                        "Q": bc_inflow.tolist(),
                        "t": times.tolist(),
                    },
                },
                {
                    "bc_name": "OUTPRES",
                    "bc_type": "PRESSURE",
                    "bc_values": {
                        "P": bc_outpres.tolist(),
                        "t": times.tolist(),
                    },
                },
            ],
            "junctions": [],
            "simulation_parameters": {
                "number_of_cardiac_cycles": 5,
                "number_of_time_pts_per_cardiac_cycle": 300,
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

        sim_times = np.array(result["time"])[-300:]
        sim_times = sim_times - np.amin(sim_times)
        sim_inpres = np.array(result["pressure_in"])[-300:]
        sim_outflow = np.array(result["flow_out"])[-300:]

        inpres = interpolate.interp1d(sim_times, sim_inpres)(times)
        outflow = interpolate.interp1d(sim_times, sim_outflow)(times)

        return inpres, outflow
