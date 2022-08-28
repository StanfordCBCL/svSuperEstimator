from __future__ import annotations

import os
import pickle
from datetime import datetime
from time import time

import numpy as np
import orjson
import pandas as pd
from rich.console import Console

from . import plotutils

from .. import forward_models, iterators
from .. import model as mdl

from .. import solver as slv
from .. import visualizer
from . import statutils
from ..reader import SimVascularProject
from .task import Task
from rich import print

CONSOLE = Console()


class WindkesselTuning(Task):

    TASKNAME = "WindkesselTuning"

    DEFAULTS = {
        "num_procs": 1,
        "num_particles": 100,
        "num_rejuvenation_steps": 2,
        "resampling_threshold": 0.5,
        "noise_factor": 0.05,
        "noise_type": "fixed_variance",
    }

    def core_run(self):

        # Create the forward model
        model = mdl.ZeroDModel(self.project)
        forward_model = _Forward_Model(model)

        # Get ground truth distal to proximal ratio
        theta_obs = np.log(
            [
                bc.resistance_distal + bc.resistance_proximal
                for bc in forward_model.outlet_bcs.values()
            ]
        )
        self.log("Setting target parameters to:", theta_obs)
        y_obs = forward_model.evaluate(
            **{f"k{i}": val for i, val in enumerate(theta_obs)}
        )
        self.log("Setting target observation to:", y_obs)

        self.database["theta_obs"] = theta_obs.tolist()
        self.database["y_obs"] = y_obs.tolist()

        self.log("Setup tuning process")
        iterator = iterators.SmcIterator(
            forward_model=forward_model,
            y_obs=y_obs,
            output_dir=self.output_folder,
            num_procs=int(self.config.get("num_procs")),
            num_particles=int(self.config.get("num_particles")),
            num_rejuvenation_steps=int(
                self.config.get("num_rejuvenation_steps")
            ),
            resampling_threshold=float(
                self.config.get("resampling_threshold")
            ),
            noise_value=float(self.config["noise_factor"])
            * np.mean(y_obs.ravel()),
            noise_type=str(self.config["noise_type"]),
        )

        for i in range(len(theta_obs)):
            iterator.add_random_variable(
                f"k{i}", "uniform", lower_bound=5.5, upper_bound=11.5
            )

        # Run the iterator
        self.log("Starting tuning process")
        iterator.run()

        # Save parameters to file
        self.database["timestamp"] = datetime.now().strftime(
            "%m/%d/%Y, %H:%M:%S"
        )
        self.set_parameters(self.database)

    def post_run(self):

        results_pp = {}

        self.log("Read raw result")
        parameters = self.get_parameters()
        particles, weights = self.get_raw_results()
        plotrange = [5.5, 11.5]

        # Calculate metrics
        self.log("Calculate metrics")
        results_pp["metrics"] = {
            "ground_truth": parameters["theta_obs"],
            "weighted_mean": statutils.particle_wmean(
                particles=particles, weights=weights
            ),
            "map": statutils.particle_map(
                particles=particles, weights=weights
            ),
            "coveriance_matrix": statutils.particle_covmat(
                particles=particles, weights=weights
            ),
        }

        # Calculate 1D marginal kernel density estimate
        results_pp["kernel_density"] = []
        for i in range(particles.shape[1]):
            self.log(f"Calculate kernel density estimates for parameter {i}")
            x, kde, bandwidth = statutils.gaussian_kde_1d(
                x=particles[:, i],
                weights=weights,
                bounds=plotrange,
                num=100,
            )
            results_pp["kernel_density"].append(
                {
                    "x": x,
                    "kernel_density": kde,
                    "bandwidth": bandwidth,
                    "opt_method": "30-fold cross-validation",
                }
            )

        self.log("Save postprocessed results")
        self.set_results(results_pp)

    def generate_report(self, project_overview=False):

        report = visualizer.Report()

        # Read results
        parameters = self.get_parameters()
        particles, weights = self.get_raw_results()
        results = self.get_results()
        metrics = results["metrics"]
        kernel_density = results["kernel_density"]
        plotrange = [5.5, 11.5]

        # Create project information section
        if project_overview:
            print("Create project overview")
            plot3d = plotutils.create_3d_geometry_plot_with_bcs(self.project)
            model = mdl.ZeroDModel(self.project)
            report.add("Project overview")
            report.add([model.get_boundary_condition_info(), plot3d])

        for i in range(particles.shape[1]):

            report.add(f"Results for {i}")

            # Calculate histogram data
            bins = int(
                (plotrange[1] - plotrange[0]) / kernel_density[i]["bandwidth"]
            )
            counts, bin_edges = np.histogram(
                particles[:, i],
                bins=bins,
                weights=weights,
                density=True,
                range=plotrange,
            )

            # Create kernel density estimation plot for k0
            distplot = visualizer.Plot2D(
                title="Weighted histogram and kernel density estimation of k0",
                xaxis_title="k0",
                yaxis_title="Kernel density",
                xaxis_range=plotrange,
            )
            distplot.add_bar_trace(
                x=bin_edges,
                y=counts,
                name="Weighted histogram",
            )
            distplot.add_line_trace(
                x=kernel_density[i]["x"],
                y=kernel_density[i]["kernel_density"],
                name="Kernel density estimate",
            )
            distplot.add_vline_trace(
                x=metrics["ground_truth"][i], text="Ground Truth"
            )

            report.add([distplot])

        return report

    def set_parameters(self, parameters) -> dict:
        """Set problem parameters.

        Args:
            parameters: Parameters of the problem.
        """

        with open(
            os.path.join(self.output_folder, "parameters.json"), "wb"
        ) as ff:
            ff.write(
                orjson.dumps(
                    parameters,
                    option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY,
                )
            )

    def get_parameters(self) -> dict:
        """Return problem parameters.

        Returns:
            parameters: Parameters of the problem.
        """
        with open(
            os.path.join(self.output_folder, "parameters.json"), "rb"
        ) as ff:
            parameters = orjson.loads(ff.read())
        return parameters

    def get_raw_results(
        self, frame: int = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw queens result.

        Args:
            frame: Specify smc iteration to read results from. If None, the
                final result will be returned.

        Returns:
            particles_x: X-coordinate of particles.
            particles_y: Y-coordinate of particles.
            weights: Weights of particles.
        """

        filename = (
            "results.pickle" if frame is None else f"results{frame}.pickle"
        )
        with open(
            os.path.join(self.output_folder, filename),
            "rb",
        ) as ff:
            results = pickle.load(ff)

        particles = np.array(results["raw_output_data"]["particles"])
        weights = np.array(results["raw_output_data"]["weights"])

        return particles, weights.flatten()

    def set_results(self, results):
        with open(
            os.path.join(self.output_folder, "results.json"), "wb"
        ) as ff:
            ff.write(
                orjson.dumps(
                    results,
                    option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY,
                )
            )

    def get_results(self):
        with open(
            os.path.join(self.output_folder, "results.json"), "rb"
        ) as ff:
            results = orjson.loads(ff.read())
        return results


class _Forward_Model:
    def __init__(self, model: mdl.ZeroDModel) -> None:
        self.model = model
        self.solver = slv.ZeroDSolver(cpp=True)

        self.outlet_bcs = {
            name: bc
            for name, bc in model.boundary_conditions.items()
            if name != "INFLOW"
        }

        self.num_pts_per_cyle = model._config["simulation_parameters"][
            "number_of_time_pts_per_cardiac_cycle"
        ]

        # Distal to proximal resistance ratio at each outlet
        self._distal_to_proximal = [
            bc.resistance_distal / bc.resistance_proximal
            for bc in self.outlet_bcs.values()
        ]

        # Time constants for each outlet
        self._time_constants = [
            bc.resistance_distal * bc.capacity
            for bc in self.outlet_bcs.values()
        ]

        self.bc_map = {}
        for branch_id, vessel_data in enumerate(model._config["vessels"]):
            if "boundary_conditions" in vessel_data:
                for bc_type, bc_name in vessel_data[
                    "boundary_conditions"
                ].items():
                    self.bc_map[bc_name] = {"name": f"V{branch_id}"}
                    if bc_type == "inlet":
                        self.bc_map[bc_name]["flow"] = "flow_in"
                        self.bc_map[bc_name]["pressure"] = "pressure_in"
                    else:
                        self.bc_map[bc_name]["flow"] = "flow_out"
                        self.bc_map[bc_name]["pressure"] = "pressure_out"

        self.bc_names = [self.bc_map[bc]["name"] for bc in self.outlet_bcs]
        self.bc_pressure = [
            self.bc_map[bc]["pressure"] for bc in self.outlet_bcs
        ]
        self.bc_flow = [self.bc_map[bc]["flow"] for bc in self.outlet_bcs]
        self.inflow_name = self.bc_map["INFLOW"]["name"]
        self.inflow_pressure = self.bc_map["INFLOW"]["pressure"]

    def evaluate(self, **kwargs):
        """Objective function for the optimization.

        Evaluates the sum of the offsets for the input output pressure relation
        for each outlet.
        """
        # Set new total resistance at each outlet
        for i, bc in enumerate(self.outlet_bcs.values()):
            ki = np.exp(kwargs[f"k{i}"])
            bc.resistance_proximal = (1.0 + self._distal_to_proximal[i]) / ki
            bc.resistance_distal = ki - bc.resistance_proximal

        try:
            result = self.solver.run_simulation(self.model)
        except RuntimeError:
            print(kwargs)
            raise SystemExit
            return np.array([np.nan] * (len(self.outlet_bcs) + 2))

        p_inlet = result[result.name == self.inflow_name][
            self.inflow_pressure
        ][-self.num_pts_per_cyle :]

        p_inlet_min, p_inlet_max = p_inlet.min(), p_inlet.max()

        q_outlet_mean = [
            result[result.name == name][flow_id][
                -self.num_pts_per_cyle :
            ].mean()
            for name, flow_id in zip(self.bc_names, self.bc_flow)
        ]
        return np.array([p_inlet_min, p_inlet_max, *q_outlet_mean])
