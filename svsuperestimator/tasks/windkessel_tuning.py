"""This module holds the WindkesselTuning task."""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from multiprocessing import get_context
from typing import Any, Dict, Optional

import numpy as np
import orjson
import pandas as pd
import particles
from particles import distributions as dists
from particles import smc_samplers as ssp
from rich.progress import BarColumn, Progress
from scipy import stats
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
        "num_particles": 100,
        "num_rejuvenation_steps": 2,
        "resampling_threshold": 0.5,
        "noise_factor": 0.05,
        **Task.DEFAULTS,
    }

    _THETA_RANGE = (7.0, 13.0)

    def core_run(self) -> None:
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
        ).tolist()
        inflow_bc["t"] = np.linspace(
            inflow_bc["t"][0],
            inflow_bc["t"][-1],
            zerod_config_handler.num_pts_per_cycle,
        ).tolist()

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
        std_vector = self.config["noise_factor"] * y_obs
        self.log("Setting std vector to:", std_vector)
        self.database["y_obs_std"] = std_vector.tolist()

        # Setup the iterator
        self.log("Setup tuning process")
        smc_runner = _SMCRunner(
            forward_model=forward_model,
            y_obs=y_obs,
            len_theta=len(theta_obs),
            likelihood_std_vector=std_vector,
            prior_bounds=self._THETA_RANGE,
            num_procs=self.config["num_procs"],
            num_particles=self.config["num_particles"],
            resampling_strategy="systematic",
            resampling_threshold=self.config["resampling_threshold"],
            num_rejuvenation_steps=self.config["num_rejuvenation_steps"],
            console=self.console,
        )

        # Run the iterator
        self.log("Starting tuning process")
        all_particles, all_weights, all_logpost = smc_runner.run()
        self.database["particles"] = all_particles
        self.database["weights"] = all_weights
        self.database["logpost"] = all_logpost

        # Save parameters to file
        self.database["timestamp"] = datetime.now().strftime(
            "%m/%d/%Y, %H:%M:%S"
        )

    def post_run(self) -> None:
        """Postprocessing routine of the task."""

        results: Dict[str, Any] = {}

        # Read raw results
        self.log("Read raw result")
        particles = np.array(self.database["particles"][-1])
        weights = np.array(self.database["weights"][-1])
        log_post = np.array(self.database["logpost"][-1])
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

        # Calculate metrics
        self.log("Calculate metrics")
        ground_truth = self.database["theta_obs"]
        wmean = statutils.particle_wmean(particles=particles, weights=weights)
        cov = statutils.particle_covmat(particles=particles, weights=weights)
        std = [cov[i][i] ** 0.5 for i in range(cov.shape[0])]
        wmean_error = [abs(m - gt) / gt for m, gt in zip(wmean, ground_truth)]

        max_post = statutils.particle_map(
            particles=particles, posterior=log_post
        )
        map_error = [abs(m - gt) / gt for m, gt in zip(max_post, ground_truth)]
        results["metrics"] = {
            "ground_truth": ground_truth,
            "weighted_mean": wmean,
            "weighted_mean_error": wmean_error,
            "weighted_std": std,
            "maximum_a_posteriori": max_post,
            "maximum_a_posteriori_error": map_error,
            "covariance_matrix": cov,
        }

        # Calculate exponential metrics
        particles_exp = np.exp(particles)
        ground_truth_exp = np.exp(self.database["theta_obs"])
        wmean_exp = statutils.particle_wmean(
            particles=particles_exp, weights=weights
        )
        maxap_exp = statutils.particle_map(
            particles=particles_exp, posterior=log_post
        )
        cov_exp = statutils.particle_covmat(
            particles=particles_exp, weights=weights
        )
        std_exp = [cov_exp[i][i] ** 0.5 for i in range(cov_exp.shape[0])]
        wmean_exp_error = [
            abs(m - gt) / gt for m, gt in zip(wmean_exp, ground_truth_exp)
        ]
        map_exp_error = [
            abs(m - gt) / gt for m, gt in zip(maxap_exp, ground_truth_exp)
        ]
        results["metrics"].update(
            {
                "exp_ground_truth": ground_truth_exp,
                "exp_weighted_mean": wmean_exp,
                "exp_weighted_mean_error": wmean_exp_error,
                "exp_weighted_std": std_exp,
                "exp_maximum_a_posteriori": maxap_exp,
                "exp_maximum_a_posteriori_error": map_exp_error,
                "exp_covariance_matrix": cov_exp,
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
            ki = np.exp(max_post[i])
            bc["bc_values"]["Rp"] = ki / (1.0 + distal_to_proximals[i])
            bc["bc_values"]["Rd"] = ki - bc["bc_values"]["Rp"]
            bc["bc_values"]["C"] = time_constants[i] / bc["bc_values"]["Rd"]
        zerod_config_handler.to_file(
            os.path.join(self.output_folder, "solver_0d_map.in")
        )
        runnercpp.run_from_config(zerod_config_handler.data).to_csv(
            os.path.join(self.output_folder, "solution_map.csv")
        )

    def generate_report(self) -> visualizer.Report:
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
        particles = np.array(self.database["particles"][-1])
        weights = np.array(self.database["weights"][-1])
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

        map_opts: Dict[str, Any] = {
            "name": "MAP estimate",
            "showlegend": True,
            "color": "#EF553B",
            "width": 3,
        }
        mean_opts: Dict[str, Any] = {
            "name": "Mean estimate",
            "showlegend": True,
            "color": "#636efa",
            "width": 3,
            "dash": "dash",
        }

        # Create distribition plots for all boundary conditions
        for i, bc_name in enumerate(bc_names):
            report.add(f"Results for {bc_name}")

            # Calculate histogram data
            bandwidth = 0.02
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
            distplot.add_vline_trace(
                x=results["metrics"]["ground_truth"][i], text="Ground Truth"
            )
            gt = results["metrics"]["ground_truth"][i]
            wmean = results["metrics"]["weighted_mean"][i]
            std = results["metrics"]["weighted_std"][i]
            wmean_error = results["metrics"]["weighted_mean_error"][i] * 100
            map = results["metrics"]["maximum_a_posteriori"][i]
            map_error = (
                results["metrics"]["maximum_a_posteriori_error"][i] * 100
            )

            gt_exp = results["metrics"]["exp_ground_truth"][i]
            wmean_exp = results["metrics"]["exp_weighted_mean"][i]
            std_exp = results["metrics"]["exp_weighted_std"][i]
            wmean_exp_error = (
                results["metrics"]["exp_weighted_mean_error"][i] * 100
            )
            map_exp = results["metrics"]["exp_maximum_a_posteriori"][i]
            map_exp_error = (
                results["metrics"]["exp_maximum_a_posteriori_error"][i] * 100
            )

            distplot._fig.add_annotation(
                text=(
                    f"ground truth [&#952;]: {gt:.2f}<br>"
                    f"mean &#177; std [&#952;]: {wmean:.2f} &#177; "
                    f"{std:.2f}<br>"
                    f"map [&#952;]: {map:.2f}<br>"
                    f"mean error [%]: {wmean_error:.2f}<br>"
                    f"map error [%]: {map_error:.2f}<br>"
                    f"bandwidth [&#952;]: {bandwidth:.2f}<br>"
                    f"exp. ground truth [&#952;]: {gt_exp:.2f}<br>"
                    f"exp. mean &#177; std [&#952;]: {wmean_exp:.2f} &#177; "
                    f"{std_exp:.2f}<br>"
                    f"exp. map [&#952;]: {map_exp:.2f}<br>"
                    f"exp. mean error [%]: {wmean_exp_error:.2f}<br>"
                    f"exp. map error [%]: {map_exp_error:.2f}"
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

            pressure_plot = visualizer.Plot2D(
                title="Pressure",
                xaxis_title=r"$s$",
                yaxis_title=r"$mmHg$",
            )
            bc_result = result_map[result_map.name == bc_map[bc_name]["name"]]
            times = np.array(bc_result["time"])[-num_pts_per_cycle:]
            times -= times[0]
            pressure_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_pressure_to_mmgh(
                    bc_result[bc_map[bc_name]["pressure"]].iloc[
                        -num_pts_per_cycle:
                    ]
                ),
                **map_opts,
            )
            bc_result = result_mean[result_map.name == bc_map[bc_name]["name"]]
            pressure_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_pressure_to_mmgh(
                    bc_result[bc_map[bc_name]["pressure"]].iloc[
                        -num_pts_per_cycle:
                    ]
                ),
                **mean_opts,
            )

            flow_plot = visualizer.Plot2D(
                title="Flow",
                xaxis_title=r"$s$",
                yaxis_title=r"$\frac{l}{min}$",
            )
            bc_result = result_map[result_map.name == bc_map[bc_name]["name"]]
            flow_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_flow_to_lmin(
                    bc_result[bc_map[bc_name]["flow"]].iloc[
                        -num_pts_per_cycle:
                    ]
                ),
                **map_opts,
            )
            bc_result = result_mean[result_map.name == bc_map[bc_name]["name"]]
            flow_plot.add_line_trace(
                x=times,
                y=taskutils.cgs_flow_to_lmin(
                    bc_result[bc_map[bc_name]["flow"]].iloc[
                        -num_pts_per_cycle:
                    ]
                ),
                **mean_opts,
            )

            report.add([pressure_plot, flow_plot])

        return report

    def _get_raw_results(
        self, frame: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw queens result.

        Args:
            frame: Specify smc iteration to read results from. If None, the
                final result will be returned.

        Returns:
            particles: Coordinates of the particles.
            weights: Weights of particles.
            log_post: Log posterior of particles.
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
        log_post = np.array(results["raw_output_data"]["log_posterior"])

        return particles, weights.flatten(), log_post


class _Forward_Model:
    """Windkessel tuning forward model.

    This forward model performs evaluations of a 0D model based on a
    given total resistance.
    """

    def __init__(self, zerod_config: reader.SvZeroDSolverInputHandler) -> None:
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

    def evaluate(self, sample: np.ndarray) -> np.ndarray:
        """Objective function for the optimization.

        Evaluates the sum of the offsets for the input output pressure relation
        for each outlet.
        """

        # Set new total resistance at each outlet
        for i, bc in enumerate(
            self.base_config.outlet_boundary_conditions.values()
        ):
            ki = np.exp(sample[f"k{i}"])
            bc["bc_values"]["Rp"] = ki / (1.0 + self._distal_to_proximal[i])
            bc["bc_values"]["Rd"] = ki - bc["bc_values"]["Rp"]
            bc["bc_values"]["C"] = (
                self._time_constants[i] / bc["bc_values"]["Rd"]
            )

        # Run simulation
        try:
            result = runnercpp.run_from_config(self.base_config.data)
        except RuntimeError:
            print("WARNING: Forward model evaluation failed.")
            return np.array([9e99] * (len(self.bc_names) + 2))

        # Extract minimum and maximum inlet pressure for last cardiac cycle
        p_inlet = result[result.name == self.inflow_name][self.inflow_pressure]

        # Extract mean outlet pressure for last cardiac cycle at each BC
        q_outlet_mean = [
            result.loc[result.name == name, flow_id].mean()
            for name, flow_id in zip(self.bc_names, self.bc_flow)
        ]

        return np.array([p_inlet.min(), p_inlet.max(), *q_outlet_mean])


class _SMCRunner:
    def __init__(
        self,
        forward_model: _Forward_Model,
        y_obs: np.ndarray,
        len_theta: int,
        likelihood_std_vector: np.ndarray,
        prior_bounds: tuple,
        num_particles: int,
        resampling_strategy: str,
        resampling_threshold: float,
        num_rejuvenation_steps: int,
        num_procs: int,
        console: Any,
    ):
        likelihood = stats.multivariate_normal(mean=np.zeros(len(y_obs)))

        prior = dists.StructDist(
            {
                f"k{i}": dists.Uniform(a=prior_bounds[0], b=prior_bounds[1])
                for i in range(len_theta)
            }
        )
        self.console = console
        self.len_theta = len_theta

        class StaticModel(ssp.StaticModel):
            def __init__(
                self, prior: dists.StructDist, len_theta: int
            ) -> None:
                super().__init__(None, prior)
                self.len_theta = len_theta

            def loglik(
                self, theta: np.ndarray, t: Optional[int] = None
            ) -> np.ndarray:
                results = []
                with get_context("fork").Pool(num_procs) as pool:
                    with Progress(
                        " " * 20 + "Evaluating samples... ",
                        BarColumn(),
                        "{task.completed}/{task.total} completed | "
                        "{task.speed} samples/s",
                        console=console,
                    ) as progress:
                        for res in progress.track(
                            pool.imap(forward_model.evaluate, theta, 1),
                            total=len(theta),
                        ):
                            results.append(res)
                return likelihood.logpdf(
                    (np.array(results) - y_obs) / likelihood_std_vector
                )

        static_model = StaticModel(prior, self.len_theta)

        fk_model = ssp.AdaptiveTempering(
            model=static_model, len_chain=1 + num_rejuvenation_steps
        )

        self.runner = particles.SMC(
            fk=fk_model,
            N=num_particles,
            resampling=resampling_strategy,
            ESSrmin=resampling_threshold,
            verbose=False,
        )

    def run(self) -> tuple[list, list, list]:
        all_particles = []
        all_weights = []
        all_logpost = []

        for _ in self.runner:
            particles = np.array(
                [
                    self.runner.X.theta[f"k{i}"].flatten()
                    for i in range(self.len_theta)
                ]
            ).T.tolist()
            weights = np.array(self.runner.W).tolist()
            logpost = np.array(self.runner.X.lpost).tolist()
            self.console.log(
                f"Completed SMC step {self.runner.t} | "
                f"[yellow]ESS[/yellow]: {self.runner.wgts.ESS:.2f} | "
                "[yellow]tempering exponent[/yellow]: "
                f"{self.runner.X.shared['exponents'][-1]:.2e}"
            )
            all_particles.append(particles)
            all_weights.append(weights)
            all_logpost.append(logpost)

        return all_particles, all_weights, all_logpost
