from __future__ import annotations
import orjson
import os

from .. import (
    model as mdl,
    solver as slv,
    forward_models,
    iterators,
    visualizer,
)
from . import plotutils, statutils
import pickle
import numpy as np
from .. import visualizer
import pandas as pd
from datetime import datetime

from rich import print

from ..reader import SimVascularProject

from ._problem import Problem


class BivariantWindkesselSMCChopin(Problem):

    PROBLEM_NAME = "Bivariant-Windkessel-SMC-Chopin"

    def __init__(self, project: SimVascularProject, case_name: str = None):
        super().__init__(project, case_name)

        self.options = {
            "case_name": case_name,
            "num_procs": 1,
            "num_particles": 100,
            "num_rejuvenation_steps": 2,
            "resampling_threshold": 0.5,
            "noise_factor": 0.05,
            "noise_type": "fixed_variance",
        }

        for bc_name in mdl.ZeroDModel(project).get_outlet_bcs():
            self.options[bc_name + "_group"] = 0

    def run(self, config):

        print("Start [bold magenta]run[/bold magenta]")

        # Parameters for saving
        parameters = {
            "configuration": config,
            "problem_type": self.PROBLEM_NAME,
        }

        # Update case name and output folder based on config
        if "case_name" in config:
            self.case_name = config["case_name"]
        os.makedirs(self.output_folder, exist_ok=True)

        # Setup project model and solver
        model = mdl.ZeroDModel(self.project)
        solver = slv.ZeroDSolver(cpp=True)

        # Extract boundary condition groups
        bc_group_0 = []
        bc_group_1 = []
        for bc_name in model.get_outlet_bcs():
            group_id = int(config[bc_name + "_group"])
            if group_id == 0:
                bc_group_0.append(bc_name)
            elif group_id == 1:
                bc_group_1.append(bc_name)
            else:
                raise ValueError(
                    f"Invalid boundary condition group {group_id}."
                )

        # Create the forward model
        forward_model = (
            forward_models.BivariantWindkesselDistalToProximalResistance0D(
                model, solver, bc_group_0, bc_group_1
            )
        )

        # Get ground truth distal to proximal ratio
        k0 = np.log(
            np.mean(
                [
                    forward_model.outlet_bcs[name].resistance_distal
                    for name in bc_group_0
                ]
            )
            / np.mean(
                [
                    forward_model.outlet_bcs[name].resistance_proximal
                    for name in bc_group_0
                ]
            )
        )
        k1 = np.log(
            np.mean(
                [
                    forward_model.outlet_bcs[name].resistance_distal
                    for name in bc_group_1
                ]
            )
            / np.mean(
                [
                    forward_model.outlet_bcs[name].resistance_proximal
                    for name in bc_group_1
                ]
            )
        )
        parameters["x_obs"] = {"k0": k0, "k1": k1}

        # Running one simulation to determine targets
        y_obs = forward_model.evaluate(k0=k0, k1=k1)
        parameters["y_obs"] = list(y_obs)

        iterator = iterators.SmcIterator(
            forward_model=forward_model,
            y_obs=y_obs,
            output_dir=self.output_folder,
            num_procs=int(config.get("num_procs")),
            num_particles=int(config.get("num_particles")),
            num_rejuvenation_steps=int(config.get("num_rejuvenation_steps")),
            resampling_threshold=float(config.get("resampling_threshold")),
            noise_value=float(config["noise_factor"]) * np.mean(y_obs.ravel()),
            noise_type=str(config["noise_type"]),
        )

        for i in range(2):
            iterator.add_random_variable(
                f"k{i}", "uniform", lower_bound=-2, upper_bound=4.5
            )

        # Run the iterator
        iterator.run()

        # Save parameters to file
        parameters["timestamp"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.set_parameters(parameters)

    def postprocess(self):

        print("Start [bold magenta]postprocessing[/bold magenta]")
        results_pp = {}

        print("Read raw result")
        parameters = self.get_parameters()
        x, y, weights = self.get_raw_results()
        plotrange = [[-2.0, 4.5], [-2.0, 4.5]]

        # Calculate metrics
        print("Calculate metrics")
        results_pp["metrics"] = {
            "ground_truth": [
                parameters["x_obs"]["k0"],
                parameters["x_obs"]["k1"],
            ],
            "weighted_mean": statutils.particle_wmean(
                x=x, y=y, weights=weights
            ),
            "map": statutils.particle_map(x=x, y=y, weights=weights),
            "coveriance_matrix": statutils.particle_covmat(
                x=x, y=y, weights=weights
            ),
        }

        # Calculate 1D marginal kernel density estimate
        print("Calculate 1D marginal kernel density estimates")
        lin_x, kde_x, bandwidth_x = statutils.gaussian_kde_1d(
            x=x, weights=weights, bounds=plotrange[0], num=100
        )
        lin_y, kde_y, bandwidth_y = statutils.gaussian_kde_1d(
            x=y, weights=weights, bounds=plotrange[1], num=100
        )
        results_pp["kernel_density_1d"] = {
            "x": lin_x,
            "y": lin_y,
            "kernel_density_x": kde_x,
            "kernel_density_y": kde_y,
            "bandwidth_x": bandwidth_x,
            "bandwidth_y": bandwidth_y,
            "opt_method": "30-fold cross-validation",
        }

        # Calculate 2D kernel density estimate
        print("Calculate 2D kernel density estimate")
        bw_method = "scott"
        lin_x, lin_y, kde, bandwidth = statutils.gaussian_kde_2d(
            x=x, y=y, weights=weights, bounds=plotrange, num=100
        )
        results_pp["kernel_density_2d"] = {
            "x": lin_x,
            "y": lin_y,
            "kernel_density": kde,
            "bandwidth": bandwidth,
            "bw_method": bw_method,
        }

        print("Save postprocessed results")
        self.set_results(results_pp)

    def generate_report(self, project_overview=False):

        print("Start [bold magenta]visualization[/bold magenta]")
        report = visualizer.Report()

        # Read results
        print("Read results")
        parameters = self.get_parameters()
        x, y, weights = self.get_raw_results()
        results = self.get_results()
        metrics = results["metrics"]
        kernel_density_2d = results["kernel_density_2d"]
        kernel_density_1d = results["kernel_density_1d"]
        plotrange = [[-2.0, 4.5], [-2.0, 4.5]]

        # Create project information section
        if project_overview:
            print("Create project overview")
            plot3d = plotutils.create_3d_geometry_plot_with_bcs(self.project)
            model = mdl.ZeroDModel(self.project)
            report.add("Project overview")
            report.add([model.get_boundary_condition_info(), plot3d])

        report.add(f"Results")

        # Calculate histogram data
        bins_x = int(
            (plotrange[0][1] - plotrange[0][0])
            / kernel_density_1d["bandwidth_x"]
        )
        bins_y = int(
            (plotrange[1][1] - plotrange[1][0])
            / kernel_density_1d["bandwidth_y"]
        )
        counts_x, bin_edges_x = np.histogram(
            x, bins=bins_x, weights=weights, density=True, range=plotrange[0]
        )
        counts_y, bin_edges_y = np.histogram(
            y, bins=bins_y, weights=weights, density=True, range=plotrange[1]
        )

        # Create the 3d kernel density estimate plot
        print("Create 3d kernel density estimate plot")
        plot_posterior_3d = visualizer.Plot3D(
            title="Kernel density estimate",
            scene=dict(
                xaxis=dict(showbackground=False, title="k0"),
                yaxis=dict(showbackground=False, title="k1"),
                zaxis=dict(showbackground=False, title="", visible=False),
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            width=750,
            height=750,
            autosize=True,
        )
        plot_posterior_3d.add_surface_trace(
            x=kernel_density_2d["x"],
            y=kernel_density_2d["y"],
            z=kernel_density_2d["kernel_density"],
            name="KDE",
        )
        plot_posterior_3d.add_annotated_point_trace(
            x=metrics["ground_truth"][0],
            y=metrics["ground_truth"][1],
            z=1.2 * np.amax(kernel_density_2d["kernel_density"]),
            text="Ground Truth",
        )

        # Create the 3d kernel density estimate heatmap
        print("Create 2d kernel density estimate heatmap")
        heatmap_plot = visualizer.Plot2D(
            title="Heatmap of kernel density estimate with marginals",
            xaxis_title="$k_0$",
            yaxis_title="$k_1$",
            width=750,
            height=750,
            autosize=True,
            xaxis_range=plotrange[0],
            yaxis_range=plotrange[1],
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
            ),
        )
        heatmap_plot.add_heatmap_trace(
            x=kernel_density_2d["x"],
            y=kernel_density_2d["y"],
            z=kernel_density_2d["kernel_density"],
            name="Weighted particle density",
        )
        heatmap_plot.add_point_trace(x=x, y=y, color=weights, name="Particles")
        heatmap_plot.add_xy_bar_trace(
            x=bin_edges_x,
            y=bin_edges_y,
            z_x=counts_x,
            z_y=counts_y,
            name_x="Weighted histogram of k0",
            name_y="Weighted histogram of k1",
        )
        heatmap_plot.add_annotated_point_trace(
            x=[
                metrics["ground_truth"][0],
                metrics["map"][0],
                metrics["weighted_mean"][0],
            ],
            y=[
                metrics["ground_truth"][1],
                metrics["map"][1],
                metrics["weighted_mean"][1],
            ],
            name=["Ground Truth", "MAP", "Mean"],
            color=["black", "magenta", "blue"],
            symbol=["triangle-down", "x", "cross"],
            size=12,
        )

        report.add([heatmap_plot, plot_posterior_3d])

        # Create kernel density estimation plot for k0
        print("Create 1d histogram for k0")
        distplot_x = visualizer.Plot2D(
            title="Weighted histogram and kernel density estimation of k0",
            xaxis_title="k0",
            yaxis_title="Kernel density",
            xaxis_range=plotrange,
        )
        distplot_x.add_bar_trace(
            x=bin_edges_x,
            y=counts_x,
            name="Weighted histogram",
        )
        distplot_x.add_line_trace(
            x=kernel_density_1d["x"],
            y=kernel_density_1d["kernel_density_x"],
            name="Kernel density estimate",
        )
        distplot_x.add_vline_trace(
            x=metrics["ground_truth"][0], text="Ground Truth"
        )

        # Create kernel density estimation plot for k1
        print("Create 1d histogram for k1")
        distplot_y = visualizer.Plot2D(
            title="Weighted histogram and kernel density estimation of k1",
            xaxis_title="k1",
            yaxis_title="Kernel density",
            xaxis_range=plotrange,
        )
        distplot_y.add_bar_trace(
            x=bin_edges_y,
            y=counts_y,
            name="Weighted histogram",
        )
        distplot_y.add_line_trace(
            x=kernel_density_1d["y"],
            y=kernel_density_1d["kernel_density_y"],
            name="Kernel density estimate",
        )
        distplot_y.add_vline_trace(
            x=metrics["ground_truth"][1], text="Ground Truth"
        )

        report.add([distplot_x, distplot_y])

        # Add parameter table
        report.add(f"Parameters")
        param_data = {"Problem"}
        param_data = {
            key: str(value)
            for key, value in parameters.items()
            if key != "configuration"
        }
        param_data.update(
            {
                key: str(value)
                for key, value in parameters["configuration"].items()
            }
        )
        report.add(
            [
                pd.DataFrame(
                    {
                        "Name": list(param_data.keys()),
                        "Value": list(param_data.values()),
                    }
                )
            ]
        )

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

        return (
            particles[:, 0].flatten(),
            particles[:, 1].flatten(),
            weights.flatten(),
        )

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
