"""This module holds the WindkesselTuning task."""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
import orjson
import pandas as pd
from pqueens.main import run
from rich.logging import RichHandler
from svzerodsolver import runnercpp

from .. import reader, visualizer
from ..reader import utils as readutils
from . import plotutils, statutils, taskutils
from .task import Task


class WindkesselTuning(Task):
    """Windkessel tuning task.

    Tunes absolute resistance of Windkessel outles to mean outlet flow and
    minimum and maximum pressure at inlet targets.
    """

    TASKNAME = "windkessel_tuning"

    DEFAULTS = {
        "zerod_config_file": None,
        "num_procs": 1,
        "theta_obs": None,
        "y_obs": None,
        "ground_truth_centerline": None,
        "num_particles": 100,
        "num_rejuvenation_steps": 2,
        "resampling_threshold": 0.5,
        "noise_factor": 0.05,
        "waste_free": True,
        **Task.DEFAULTS,
    }

    _THETA_RANGE = [7.0, 13.0]

    def core_run(self):
        """Core routine of the task."""

        # Load the 0D simulation configuration
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

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
        theta_obs = np.array(self.config["theta_obs"])
        self.log("Setting target parameters to:", theta_obs)
        self.database["theta_obs"] = theta_obs.tolist()

        # Setup forward model
        forward_model = _Forward_Model(zerod_config_handler)

        # Determine target observations through one forward evaluation
        y_obs = np.array(self.config["y_obs"])
        self.log("Setting target observation to:", y_obs)
        self.database["y_obs"] = y_obs.tolist()

        # Determine noise covariance
        noise_vector = (self.config["noise_factor"] * y_obs) ** 2.0
        self.log("Setting covariance to:", noise_vector)
        self.database["noise"] = noise_vector.tolist()

        # Setup the iteratior
        self.log("Setup tuning process")
        smc_runner = SMCRunner(
            forward_model=forward_model,
            y_obs=y_obs,
            output_dir=self.output_folder,
            num_procs=self.config["num_procs"],
            num_particles=self.config["num_particles"],
            num_rejuvenation_steps=self.config["num_rejuvenation_steps"],
            resampling_threshold=self.config["resampling_threshold"],
            noise_value=noise_vector,
            noise_type="fixed_variance_vector",
            waste_free=self.config["waste_free"],
        )

        # Add random variables for each parameter
        for i in range(len(theta_obs)):
            smc_runner.add_random_variable(
                f"k{i}",
                "uniform",
                lower_bound=self._THETA_RANGE[0],
                upper_bound=self._THETA_RANGE[1],
            )

        # Run the iterator
        self.log("Starting tuning process")
        smc_runner.run(
            loghandler=RichHandler(console=self.console, show_level=False)
        )

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
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

        # Calculate metrics
        self.log("Calculate metrics")
        ground_truth = self.database["theta_obs"]
        wmean = statutils.particle_wmean(particles=particles, weights=weights)
        maxap = statutils.particle_map(particles=particles, weights=weights)
        cov = statutils.particle_covmat(particles=particles, weights=weights)
        std = [cov[i][i] ** 0.5 for i in range(cov.shape[0])]
        wmean_error = [abs(m - gt) / gt for m, gt in zip(wmean, ground_truth)]
        map_error = [abs(m - gt) / gt for m, gt in zip(maxap, ground_truth)]
        results["metrics"] = {
            "ground_truth": ground_truth,
            "wmean": wmean,
            "wmean_error": wmean_error,
            "wstd": std,
            "map": maxap,
            "map_error": map_error,
            "covariance_matrix": cov,
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

        runnercpp.run_from_config(zerod_config_handler.data).to_csv(
            os.path.join(self.output_folder, "solution_gt.csv")
        )
        outlet_bcs = zerod_config_handler.outlet_boundary_conditions.values()
        distal_to_proximals = [
            bc["bc_values"]["Rd"] / bc["bc_values"]["Rp"] for bc in outlet_bcs
        ]
        time_constants = [
            bc["bc_values"]["Rd"] * bc["bc_values"]["C"] for bc in outlet_bcs
        ]
        for i, bc in enumerate(outlet_bcs):
            ki = np.exp(wmean[i])
            bc["bc_values"]["Rp"] = ki / (1.0 + distal_to_proximals[i])
            bc["bc_values"]["Rd"] = ki - bc["bc_values"]["Rp"]
            bc["bc_values"]["C"] = time_constants[i] / bc["bc_values"]["Rd"]
        zerod_config_handler.to_file(
            os.path.join(self.output_folder, "solver_0d_mean.in")
        )
        runnercpp.run_from_config(zerod_config_handler.data).to_csv(
            os.path.join(self.output_folder, "solution_mean.csv")
        )
        for i, bc in enumerate(outlet_bcs):
            ki = np.exp(maxap[i])
            bc["bc_values"]["Rp"] = ki / (1.0 + distal_to_proximals[i])
            bc["bc_values"]["Rd"] = ki - bc["bc_values"]["Rp"]
            bc["bc_values"]["C"] = time_constants[i] / bc["bc_values"]["Rd"]
        zerod_config_handler.to_file(
            os.path.join(self.output_folder, "solver_0d_map.in")
        )
        runnercpp.run_from_config(zerod_config_handler.data).to_csv(
            os.path.join(self.output_folder, "solution_map.csv")
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
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )
        num_pts_per_cycle = zerod_config_handler.num_pts_per_cycle
        bc_map = zerod_config_handler.vessel_to_bc_map

        result_gt = pd.read_csv(
            os.path.join(self.output_folder, "solution_gt.csv")
        )
        result_map = pd.read_csv(
            os.path.join(self.output_folder, "solution_map.csv")
        )
        result_mean = pd.read_csv(
            os.path.join(self.output_folder, "solution_mean.csv")
        )

        # Read raw and postprocessed results
        with open(
            os.path.join(self.output_folder, "results.json"), "rb"
        ) as ff:
            results = orjson.loads(ff.read())
        particles, weights = self._get_raw_results()
        zerod_config = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

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

        gt_opts = {
            "name": "Ground Truth",
            "showlegend": True,
            "color": "white",
            "dash": "dot",
            "width": 4,
        }
        map_opts = {
            "name": "MAP estimate",
            "showlegend": True,
            "color": "#EF553B",
            "width": 3,
        }
        mean_opts = {
            "name": "Mean estimate",
            "showlegend": True,
            "color": "#636efa",
            "width": 3,
        }

        # Create distribition plots for all boundary conditions
        for i, bc_name in enumerate(bc_names):

            report.add(f"Results for {bc_name}")

            # Calculate histogram data
            bandwidth = results["kernel_density"][i]["bandwidth"]
            bins = int(
                (self._THETA_RANGE[1] - self._THETA_RANGE[0]) / bandwidth
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
                title="Weighted histogram and kernel density estimation",
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
            gt = results["metrics"]["ground_truth"][i]
            wmean = results["metrics"]["wmean"][i]
            std = results["metrics"]["wstd"][i]
            wmean_error = results["metrics"]["wmean_error"][i] * 100
            map = results["metrics"]["map"][i]
            map_error = results["metrics"]["map_error"][i] * 100

            distplot._fig.add_annotation(
                text=(
                    f"ground truth [&#952;]: {gt:.2f}<br>"
                    f"mean &#177; std [&#952;]: {wmean:.2f} &#177; "
                    f"{std:.2f}<br>"
                    f"map [&#952;]: {map:.2f}<br>"
                    f"mean error [%]: {wmean_error:.2f}<br>"
                    f"map error [%]: {map_error:.2f}<br>"
                    f"bandwidth [&#952;]: {bandwidth:.2f}"
                ),
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.0,
                bordercolor="white",
                xanchor="right",
                yanchor="top",
                borderpad=7,
            )
            report.add([distplot])

            bc_result = result_gt[result_gt.name == bc_map[bc_name]["name"]]
            times = np.array(bc_result["time"])[-num_pts_per_cycle:]
            times -= times[0]
            pressure_plot = visualizer.Plot2D(
                title="Pressure",
                xaxis_title=r"$s$",
                yaxis_title=r"$mmHg$",
            )
            pressure_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_pressure_to_mmgh(
                    bc_result[bc_map[bc_name]["pressure"]][-num_pts_per_cycle:]
                ),
                **gt_opts,
            )
            bc_result = result_map[result_map.name == bc_map[bc_name]["name"]]
            pressure_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_pressure_to_mmgh(
                    bc_result[bc_map[bc_name]["pressure"]][-num_pts_per_cycle:]
                ),
                **map_opts,
            )
            bc_result = result_mean[result_map.name == bc_map[bc_name]["name"]]
            pressure_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_pressure_to_mmgh(
                    bc_result[bc_map[bc_name]["pressure"]][-num_pts_per_cycle:]
                ),
                **mean_opts,
            )

            flow_plot = visualizer.Plot2D(
                title="Flow",
                xaxis_title=r"$s$",
                yaxis_title=r"$\frac{l}{h}$",
            )
            flow_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_flow_to_lh(
                    bc_result[bc_map[bc_name]["flow"]][-num_pts_per_cycle:]
                ),
                **gt_opts,
            )
            bc_result = result_map[result_map.name == bc_map[bc_name]["name"]]
            flow_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_flow_to_lh(
                    bc_result[bc_map[bc_name]["flow"]][-num_pts_per_cycle:]
                ),
                **map_opts,
            )
            bc_result = result_mean[result_map.name == bc_map[bc_name]["name"]]
            flow_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_flow_to_lh(
                    bc_result[bc_map[bc_name]["flow"]][-num_pts_per_cycle:]
                ),
                **mean_opts,
            )

            report.add([pressure_plot, flow_plot])

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

        self.base_config = zerod_config.copy()
        self.outlet_bcs = self.base_config.outlet_boundary_conditions
        self.num_pts_per_cycle = self.base_config.num_pts_per_cycle
        self.base_config.data["simulation_parameters"].update(
            {"output_last_cycle_only": True, "output_interval": 10}
        )

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
        bc_map = zerod_config.vessel_to_bc_map

        self.bc_names = [bc_map[bc]["name"] for bc in self.outlet_bcs]
        self.bc_flow = [bc_map[bc]["flow"] for bc in self.outlet_bcs]
        self.inflow_name = bc_map["INFLOW"]["name"]
        self.inflow_pressure = bc_map["INFLOW"]["pressure"]

    def evaluate(self, **kwargs):
        """Objective function for the optimization.

        Evaluates the sum of the offsets for the input output pressure relation
        for each outlet.
        """

        # Set new total resistance at each outlet
        for i, bc in enumerate(
            self.base_config.outlet_boundary_conditions.values()
        ):
            ki = np.exp(kwargs[f"k{i}"])
            bc["bc_values"]["Rp"] = ki / (1.0 + self._distal_to_proximal[i])
            bc["bc_values"]["Rd"] = ki - bc["bc_values"]["Rp"]
            bc["bc_values"]["C"] = (
                self._time_constants[i] / bc["bc_values"]["Rd"]
            )

        # Run simulation
        try:
            result = runnercpp.run_from_config(self.base_config.data)
        except RuntimeError:
            return np.array([9e99] * (len(self.bc_names) + 2))

        # Extract minimum and maximum inlet pressure for last cardiac cycle
        p_inlet = result[result.name == self.inflow_name][self.inflow_pressure]

        # Extract mean outlet pressure for last cardiac cycle at each BC
        q_outlet_mean = [
            result.loc[result.name == name, flow_id].mean()
            for name, flow_id in zip(self.bc_names, self.bc_flow)
        ]

        return np.expand_dims(np.array([p_inlet.min(), p_inlet.max(), *q_outlet_mean]), axis=0)


class SMCRunner:
    """Sequentia-Monte-Carlo iterator for static models."""

    def __init__(
        self,
        forward_model,
        y_obs: np.ndarray,
        output_dir=None,
        num_procs=1,
        **kwargs,
    ):
        """Create a new SmcIterator instance.

        Args:
            forward_model: Forward model.
            y_obs: Matrix with row-wise observation vectors.
            output_dir: Output directory.
            num_procs: Number of parallel processes.
            kwargs: Optional parameters
                * `database_address`: Address to the database for QUEENS.
                * `num_particles`: Number of particles for SMC.
                * `resampling_threshold`: Resampling threshold for SMC.
                * `num_rejuvenation_steps`: Number of rejuvenation steps SMC.

        """
        self._y_obs = y_obs
        self._config = {
            "global_settings": {
                "output_dir": output_dir,
                "experiment_name": "results",
            },
            "database": {
                "name": "database",
                "type": "sqlite",
            },
            "forward_model": {
                "type": "simulation_model",
                "interface": "interface",
                "parameters": "parameters",
            },
            "interface": {
                "type": "direct_python_interface",
                "external_python_module_function": forward_model.evaluate,
                "num_workers": num_procs,
            },
            "parameters": {"random_variables": {}},
            "method": {
                "method_name": "smc_chopin",
                "method_options": {
                    "seed": 42,
                    "num_particles": kwargs["num_particles"],
                    "resampling_threshold": kwargs["resampling_threshold"],
                    "resampling_method": "systematic",
                    "feynman_kac_model": "adaptive_tempering",
                    "waste_free": kwargs["waste_free"],
                    "num_rejuvenation_steps": kwargs["num_rejuvenation_steps"],
                    "model": "model",
                    "max_feval": 9e99,
                    "result_description": {
                        "write_results": True,
                        "plot_results": False,
                    },
                },
            },
            "model": {
                "type": "gaussian",
                "forward_model": "forward_model",
                "output_label": "y_obs",
                "coordinate_labels": [],
                "noise_type": kwargs["noise_type"],
                "noise_value": kwargs["noise_value"],
                "experimental_file_name_identifier": "*.csv",
                "experimental_csv_data_base_dir": None,
                "parameters": "parameters",
            },
        }

    def add_random_variable(
        self,
        name: str,
        dist_type: str,
        **kwargs: dict,
    ):
        """Add a new random variable to the iterator configuration.

        Args:
            name: Name of the variable.
            dist_type: Distribution type of the variable (`normal`, `uniform`,
                `lognormal`, `beta`)
            options: Parameters of the distribution. For `uniform` distribution
                specify `lower_bound` and `upper_bound`. For `normal
                distribution specify `mean` and `covariance`. For `beta`
                distribution specify `lower_bound`, `upper_bound`, `a`, and
                `b`. For `lognormal` specify `normal_mean` and
                `normal_covariance`.
        """
        var_config = {
            "distribution": dist_type,
            "type": "FLOAT",
            "size": 1,
            "dimension": 1,
        }

        if dist_type == "uniform":
            var_config.update(
                {
                    "lower_bound": kwargs["lower_bound"],
                    "upper_bound": kwargs["upper_bound"],
                }
            )
        elif dist_type == "normal":
            var_config.update(
                {
                    "mean": kwargs["mean"],
                    "covariance": kwargs["covariance"],
                }
            )
        elif dist_type == "beta":
            var_config.update(
                {
                    "lower_bound": kwargs["lower_bound"],
                    "upper_bound": kwargs["upper_bound"],
                    "a": kwargs["a"],
                    "b": kwargs["b"],
                }
            )
        elif dist_type == "lognormal":
            var_config.update(
                {
                    "normal_mean": kwargs["normal_mean"],
                    "normal_covariance": kwargs["normal_covariance"],
                }
            )
        else:
            raise ValueError(f"Unknown distribution type {dist_type}")

        self._config["parameters"]["random_variables"][name] = var_config

    def run(self, loghandler):
        """Run the iterator."""

        with TemporaryDirectory() as tmpdir:

            # Make file for y_obs
            target_file = os.path.join(tmpdir, "targets.csv")
            with open(target_file, "w") as ff:
                ff.write("y_obs\n" + "\n".join([str(t) for t in self._y_obs]))
            self._config["model"]["experimental_csv_data_base_dir"] = tmpdir

            # Set output directory to temporary directory if not set
            if self._config["global_settings"]["output_dir"] is None:
                self._config["global_settings"]["output_dir"] = tmpdir

            with open(
                os.path.join(
                    self._config["global_settings"]["output_dir"],
                    "queens_input.json",
                ),
                "w",
            ) as ff:
                json.dump(
                    self._config,
                    ff,
                    indent=4,
                    default=lambda o: "<not serializable>",
                )

            # Run queens
            run(
                self._config,
                self._config["global_settings"]["output_dir"],
                handler=loghandler,
            )
