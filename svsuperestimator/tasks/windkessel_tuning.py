from __future__ import annotations

import os
import pickle
from datetime import datetime

import numpy as np
import orjson
from rich.console import Console
from svzerodsolver import runnercpp

from .. import iterators, visualizer
from ..reader import utils as readutils
from . import plotutils, statutils, taskutils
from .task import Task

CONSOLE = Console()


class WindkesselTuning(Task):
    """Windkessel tuning task.

    Tunes absolute resistance of Windkessel outles to mean outlet flow and
    minimum and maximum pressure at inlet targets.
    """

    TASKNAME = "WindkesselTuning"

    DEFAULTS = {
        "num_procs": 1,
        "num_particles": 100,
        "num_rejuvenation_steps": 2,
        "resampling_threshold": 0.5,
        "noise_factor": 0.05,
    }

    _THETA_RANGE = [7.0, 13.0]

    def core_run(self):
        """Core routine of the task."""

        # Load the 0D simulation configuration
        zerod_config_handler = self.project["0d_simulation_input"]

        # Refine inflow boundary using cubic splines
        inflow_bc = zerod_config_handler.boundary_conditions["INFLOW"][
            "bc_values"
        ]
        inflow_bc["Q"] = taskutils.refine_with_cubic_spline(
            inflow_bc["Q"], zerod_config_handler.num_pts_per_cycle
        )
        inflow_bc["t"] = np.linspace(
            inflow_bc["t"][0],
            inflow_bc["t"][-1],
            zerod_config_handler.num_pts_per_cycle,
        )

        # Get ground truth distal to proximal ratio
        theta_obs = np.log(
            [
                bc["bc_values"]["Rd"] + bc["bc_values"]["Rp"]
                for bc in zerod_config_handler.outlet_boundary_conditions.values()
            ]
        )
        self.log("Setting target parameters to:", theta_obs)
        self.database["theta_obs"] = theta_obs.tolist()

        # Setup forward model
        forward_model = _Forward_Model(zerod_config_handler)

        # Determine target observations through one forward evaluation
        y_obs = forward_model.evaluate(
            **{f"k{i}": val for i, val in enumerate(theta_obs)}
        )
        self.log("Setting target observation to:", y_obs)
        self.database["y_obs"] = y_obs.tolist()

        # Determine noise covariance
        noise_vector = (self.config["noise_factor"] * y_obs) ** 2.0
        self.log("Setting noise vector to:", noise_vector)
        self.database["noise"] = noise_vector.tolist()

        # Setup the iteratior
        self.log("Setup tuning process")
        iterator = iterators.SmcIterator(
            forward_model=forward_model,
            y_obs=y_obs,
            output_dir=self.output_folder,
            num_procs=self.config["num_procs"],
            num_particles=self.config["num_particles"],
            num_rejuvenation_steps=self.config["num_rejuvenation_steps"],
            resampling_threshold=self.config["resampling_threshold"],
            noise_value=noise_vector,
            noise_type="fixed_variance_vector",
        )

        # Add random variables for each parameter
        for i in range(len(theta_obs)):
            iterator.add_random_variable(
                f"k{i}",
                "uniform",
                lower_bound=self._THETA_RANGE[0],
                upper_bound=self._THETA_RANGE[1],
            )

        # Run the iterator
        self.log("Starting tuning process")
        iterator.run()

        # Save parameters to file
        self.database["timestamp"] = datetime.now().strftime(
            "%m/%d/%Y, %H:%M:%S"
        )

    def post_run(self):
        """Postprocessing routine of the task."""

        results = {}

        # Read raw results
        self.log("Read raw result")
        particles, weights = self._get_raw_results()

        # Calculate metrics
        self.log("Calculate metrics")
        results["metrics"] = {
            "ground_truth": self.database["theta_obs"],
            "weighted_mean": statutils.particle_wmean(
                particles=particles, weights=weights
            ),
            "map": statutils.particle_map(
                particles=particles, weights=weights
            ),
            "covariance_matrix": statutils.particle_covmat(
                particles=particles, weights=weights
            ),
        }

        # Calculate 1D marginal kernel density estimate
        results["kernel_density"] = []
        for i in range(particles.shape[1]):
            self.log(f"Calculate kernel density estimates for parameter {i}")
            x, kde, bandwidth = statutils.gaussian_kde_1d(
                x=particles[:, i],
                weights=weights,
                bounds=self._THETA_RANGE,
                num=1000,
            )
            results["kernel_density"].append(
                {
                    "x": x,
                    "kernel_density": kde,
                    "bandwidth": bandwidth,
                    "opt_method": "30-fold cross-validation",
                }
            )

        # Save the postprocessed result to a file
        self.log("Save postprocessed results")
        with open(
            os.path.join(self.output_folder, "results.json"), "wb"
        ) as ff:
            ff.write(
                orjson.dumps(
                    results,
                    option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY,
                )
            )

    def generate_report(self):
        """Generate the task report."""

        # Add 3D plot of mesh with 0D elements
        report = visualizer.Report()
        report.add("Overview")
        branch_data = readutils.get_0d_element_coordinates(self.project)
        model_plot = plotutils.create_3d_geometry_plot_with_vessels(
            self.project, branch_data
        )
        report.add([model_plot])

        # Read raw and postprocessed results
        with open(
            os.path.join(self.output_folder, "results.json"), "rb"
        ) as ff:
            results = orjson.loads(ff.read())
        particles, weights = self._get_raw_results()
        zerod_config = self.project["0d_simulation_input"]

        # Format the labels
        outlet_bcs = zerod_config.outlet_boundary_conditions
        bc_names = list(outlet_bcs.keys())
        theta_names = [rf"$\theta_{i}$" for i in range(len(bc_names))]

        # Create parallel coordinates plot
        paracoords = visualizer.Plot2D()
        paracoords.add_parallel_coordinates_plots(
            particles.T,
            bc_names,
            color_by=weights,
            plotrange=self._THETA_RANGE,
        )

        # Add heatmap for the covariance
        cov_plot = visualizer.Plot2D(title="Covariance")
        cov_plot.add_heatmap_trace(
            x=bc_names,
            y=bc_names,
            z=results["metrics"]["covariance_matrix"],
            name="Covariance",
        )
        report.add([paracoords, cov_plot])

        # Create distribition plots for all boundary conditions
        for i, bc_name in enumerate(bc_names):

            report.add(f"Results for {bc_name}")

            # Calculate histogram data
            bins = int(
                (self._THETA_RANGE[1] - self._THETA_RANGE[0])
                / results["kernel_density"][i]["bandwidth"]
            )
            counts, bin_edges = np.histogram(
                particles[:, i],
                bins=bins,
                weights=weights,
                density=True,
                range=self._THETA_RANGE,
            )

            # Create kernel density estimation plot for BC
            distplot = visualizer.Plot2D(
                title="Weighted histogram and kernel density estimation of "
                + bc_name,
                xaxis_title=theta_names[i],
                yaxis_title="Kernel density",
                xaxis_range=self._THETA_RANGE,
            )
            distplot.add_bar_trace(
                x=bin_edges,
                y=counts,
                name="Weighted histogram",
            )
            distplot.add_line_trace(
                x=results["kernel_density"][i]["x"],
                y=results["kernel_density"][i]["kernel_density"],
                name="Kernel density estimate",
            )
            distplot.add_vline_trace(
                x=results["metrics"]["ground_truth"][i], text="Ground Truth"
            )

            report.add([distplot])

        return report

    def _get_raw_results(
        self, frame: int = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw queens result.

        Args:
            frame: Specify smc iteration to read results from. If None, the
                final result will be returned.

        Returns:
            particles: Coordinates of the particles.
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


class _Forward_Model:
    """Windkessel tuning forward model.

    This forward model performs evaluations of a 0D model based on a
    given total resistance.
    """

    def __init__(self, zerod_config) -> None:
        """Construct the forward model.

        Args:
            zerod_config: 0D simulation config handler.
        """

        self.base_config = zerod_config
        self.outlet_bcs = self.base_config.outlet_boundary_conditions
        self.num_pts_per_cycle = self.base_config.num_pts_per_cycle

        # Distal to proximal resistance ratio at each outlet
        self._distal_to_proximal = [
            bc["bc_values"]["Rd"] / bc["bc_values"]["Rp"]
            for bc in self.outlet_bcs.values()
        ]

        # Time constants for each outlet
        self._time_constants = [
            bc["bc_values"]["Rd"] * bc["bc_values"]["C"]
            for bc in self.outlet_bcs.values()
        ]

        # Map boundary conditions to vessel names
        bc_map = {}
        for branch_id, vessel_data in enumerate(zerod_config.vessels):
            if "boundary_conditions" in vessel_data:
                for bc_type, bc_name in vessel_data[
                    "boundary_conditions"
                ].items():
                    bc_map[bc_name] = {"name": f"V{branch_id}"}
                    if bc_type == "inlet":
                        bc_map[bc_name]["flow"] = "flow_in"
                        bc_map[bc_name]["pressure"] = "pressure_in"
                    else:
                        bc_map[bc_name]["flow"] = "flow_out"
                        bc_map[bc_name]["pressure"] = "pressure_out"

        self.bc_names = [bc_map[bc]["name"] for bc in self.outlet_bcs]
        self.bc_flow = [bc_map[bc]["flow"] for bc in self.outlet_bcs]
        self.inflow_name = bc_map["INFLOW"]["name"]
        self.inflow_pressure = bc_map["INFLOW"]["pressure"]

    def evaluate(self, **kwargs):
        """Objective function for the optimization.

        Evaluates the sum of the offsets for the input output pressure relation
        for each outlet.
        """
        config = self.base_config.copy()

        # Set new total resistance at each outlet
        for i, bc in enumerate(config.outlet_boundary_conditions.values()):
            ki = np.exp(kwargs[f"k{i}"])
            bc["bc_values"]["Rp"] = ki / (1.0 + self._distal_to_proximal[i])
            bc["bc_values"]["Rd"] = ki - bc["bc_values"]["Rp"]
            bc["bc_values"]["C"] = (
                self._time_constants[i] / bc["bc_values"]["Rd"]
            )

        # Run simulation
        try:
            result = runnercpp.run_from_config(config.data)
        except RuntimeError:
            return np.array([9e99] * (len(self.bc_names) + 2))

        # Extract minimum and mixum inlet pressure for last cardiac cycle
        p_inlet = result[result.name == self.inflow_name][
            self.inflow_pressure
        ][-self.num_pts_per_cycle :]
        p_inlet_min, p_inlet_max = p_inlet.min(), p_inlet.max()

        # Extract mean outlet pressure for last cardiac cycle at each BC
        q_outlet_mean = [
            result[result.name == name][flow_id][
                -self.num_pts_per_cycle :
            ].mean()
            for name, flow_id in zip(self.bc_names, self.bc_flow)
        ]

        return np.array([p_inlet_min, p_inlet_max, *q_outlet_mean])
