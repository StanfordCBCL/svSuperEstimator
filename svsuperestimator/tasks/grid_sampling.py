"""This module holds the GridSampling task."""
from __future__ import annotations

import json
import os
from datetime import datetime
from multiprocessing import get_context
from typing import Any, Optional

import numpy as np
import pysvzerod as svzerodplus
from pysvzerod import Solver
from rich.progress import BarColumn, Progress
from scipy import stats

from .. import reader, visualizer
from . import taskutils
from .plotutils import joint_plot
from .task import Task


class GridSampling(Task):
    """GridSamlping task"""

    TASKNAME = "grid_sampling"

    DEFAULTS = {
        "zerod_config_file": None,
        "num_procs": 1,
        "theta_range": None,
        "y_obs": None,
        "noise_factor": 0.05,
        "num_samples": 100,
        **Task.DEFAULTS,
    }

    def core_run(self) -> None:
        """Core routine of the task."""

        self.theta_range = self.config["theta_range"]

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

        # Setup forward model
        self.forward_model = _Forward_ModelRpRd(
            zerod_config_handler,
            os.path.join(self.output_folder, "sample_configs"),
        )

        # Set target observations
        y_obs = np.array(self.config["y_obs"])
        self.log("Setting target observation to:", y_obs)
        self.database["y_obs"] = y_obs.tolist()

        # Determine noise covariance
        std_vector = self.config["noise_factor"] * y_obs
        self.log("Setting std vector to:", std_vector)
        self.database["y_obs_std"] = std_vector.tolist()

        # Setup the iterator
        self.log("Setup tuning process")
        smc_runner = _GridRunner(
            forward_model=self.forward_model,
            y_obs=y_obs,
            len_theta=self.forward_model.num_params,
            likelihood_std_vector=std_vector,
            prior_bounds=self.config["theta_range"],
            num_procs=self.config["num_procs"],
            num_samples=int(np.sqrt(self.config["num_samples"])),
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

    def generate_report(self) -> visualizer.Report:
        particles = np.array(self.database["particles"][-1])
        weights = np.array(self.database["weights"][-1])
        assert particles.shape[1] == 2, "Expected 2D particle space"
        joint_plot(
            particles[:, 0],
            particles[:, 1],
            weights,
            self.config["theta_range"],
            os.path.join(self.output_folder, "joint_plot.png"),
        )


class _GridRunner:
    def __init__(
        self,
        forward_model: _Forward_Model,
        y_obs: np.ndarray,
        len_theta: int,
        likelihood_std_vector: np.ndarray,
        prior_bounds: tuple,
        num_samples: int,
        num_procs: int,
        console: Any,
    ):
        # print(prior_bounds)
        self.likelihood = stats.multivariate_normal(mean=np.zeros(len(y_obs)))
        self.y_obs = y_obs
        self.forward_model = forward_model
        self.likelihood_std_vector = likelihood_std_vector
        self.prior_bounds = prior_bounds
        self.num_samples = num_samples

        self.console = console
        self.len_theta = len_theta
        self.num_procs = num_procs

    def loglik(self, theta: np.ndarray, t: Optional[int] = None) -> np.ndarray:
        results = []
        with get_context("fork").Pool(self.num_procs) as pool:
            with Progress(
                " " * 20 + "Evaluating samples... ",
                BarColumn(),
                "{task.completed}/{task.total} completed | "
                "{task.speed} samples/s",
                console=self.console,
            ) as progress:
                for res in progress.track(
                    pool.imap(
                        self.forward_model.evaluate,
                        zip(range(len(theta)), theta),
                        1,
                    ),
                    total=len(theta),
                ):
                    results.append(res)

        return self.likelihood.logpdf(
            (np.array(results) - self.y_obs) / self.likelihood_std_vector
        )

    def run(self) -> tuple[list, list, list]:
        ranges = [
            np.linspace(
                self.prior_bounds[i][0],
                self.prior_bounds[i][1],
                self.num_samples,
            )
            for i in range(self.len_theta)
        ]

        all_particles = np.array(np.meshgrid(*ranges)).T.reshape(
            -1, self.len_theta
        )

        all_logpost = self.loglik(all_particles)
        all_logpost -= np.max(all_logpost)
        all_weights = np.exp(all_logpost)
        all_weights /= np.sum(all_weights)

        return [all_particles], [all_weights], [all_logpost]


class _Forward_Model:
    """Windkessel tuning forward model.

    This forward model performs evaluations of a 0D model based on a
    given total resistance.
    """

    def __init__(
        self,
        zerod_config: reader.SvZeroDSolverInputHandler,
        output_folder: str = None,
    ) -> None:
        """Construct the forward model.

        Args:
            zerod_config: 0D simulation config handler.
            output_folder: Optional output folder to write the the 0D configs to.
        """
        self.output_folder = output_folder
        if self.output_folder is not None:
            os.makedirs(self.output_folder, exist_ok=True)
        self.based_zerod = zerod_config
        self.base_config = zerod_config.data.copy()
        self.outlet_bcs = zerod_config.outlet_boundary_conditions
        self.outlet_bc_ids = []
        for i, bc in enumerate(zerod_config.data["boundary_conditions"]):
            if bc["bc_name"] in self.outlet_bcs:
                self.outlet_bc_ids.append(i)

        bc_node_names = zerod_config.get_bc_node_names()
        self.inlet_dof_name = [
            f"pressure:{n}" for n in bc_node_names if "INFLOW" in n
        ][0]
        self.outlet_dof_names = [
            f"flow:{n}" for n in bc_node_names if "INFLOW" not in n
        ]

    def to_file(self, filename: str):
        """Write configuration to 0D input file"""
        self.based_zerod.to_file(filename)

    def simulate_csv(self, filename: str):
        """Run forward simulation with base configuration and save results to csv"""
        svzerodplus.simulate(self.base_config).to_csv(filename)

    def simulate(self, sample: np.ndarray, num: int) -> Solver:
        """Run forward simulation with sample and return the solver object"""
        config = self.base_config.copy()

        # Change boundary conditions (set in derived class)
        self.change_boundary_conditions(config["boundary_conditions"], sample)

        if self.output_folder is not None:
            with open(
                os.path.join(self.output_folder, f"sample_{num}.json"), "w"
            ) as f:
                json.dump(config, f, indent=4)

        # Run simulation
        try:
            solver = Solver(config)
            solver.run()
            return solver
        except RuntimeError:
            return None

    def evaluate(self, sample: np.ndarray) -> np.ndarray:
        """Objective function for the optimization"""
        raise NotImplementedError

    def change_boundary_conditions(self, boundary_conditions, sample):
        """Specify how boundary conditions are set with parameters"""
        raise NotImplementedError


class _Forward_ModelRpRd(_Forward_Model):
    def __init__(
        self,
        zerod_config: reader.SvZeroDSolverInputHandler,
        output_folder: str = None,
    ) -> None:
        super().__init__(zerod_config, output_folder)

        self.base_config = zerod_config.data.copy()
        self.outlet_bcs = zerod_config.outlet_boundary_conditions
        self.outlet_bc_ids = []
        for i, bc in enumerate(zerod_config.data["boundary_conditions"]):
            if bc["bc_name"] in self.outlet_bcs:
                self.outlet_bc_ids.append(i)
        self.base_config["simulation_parameters"].update(
            {"output_last_cycle_only": True, "output_interval": 10}
        )

        bc_node_names = zerod_config.get_bc_node_names()
        self.inlet_dof_name = [
            f"pressure:{n}" for n in bc_node_names if "INFLOW" in n
        ][0]
        self.outlet_dof_names = [
            f"flow:{n}" for n in bc_node_names if "INFLOW" not in n
        ]

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

        self.num_params = 2

    def change_boundary_conditions(self, boundary_conditions, sample):
        # Set new total resistance at each outlet
        for i, bc_id in enumerate(self.outlet_bc_ids):
            if i == 2:
                ki = np.exp(sample[0])
            else:
                ki = np.exp(sample[1])
            bc_values = boundary_conditions[bc_id]["bc_values"]
            bc_values["Rp"] = ki / (1.0 + self._distal_to_proximal[i])
            bc_values["Rd"] = ki - bc_values["Rp"]
            bc_values["C"] = self._time_constants[i] / bc_values["Rd"]

    def evaluate(self, sample: np.ndarray) -> np.ndarray:
        """Get the pressure curve at the inlet"""
        num, sample = sample
        solver = self.simulate(sample, num)
        if solver is None:
            return np.array([9e99] * (len(self.outlet_dof_names) + 2))

        # Extract minimum and maximum inlet pressure for last cardiac cycle
        p_inlet = solver.get_single_result(self.inlet_dof_name)

        # Extract mean outlet pressure for last cardiac cycle at each BC
        q_outlet_mean = [
            solver.get_single_result_avg(dof) for dof in self.outlet_dof_names
        ]

        return np.array([p_inlet.min(), p_inlet.max(), *q_outlet_mean])
