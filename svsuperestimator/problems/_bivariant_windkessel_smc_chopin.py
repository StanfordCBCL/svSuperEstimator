from __future__ import annotations
import json
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


class BivariantWindkesselSMCChopin:

    PROBLEM_NAME = "Bivariant-Windkessel-SMC-Chopin"

    def __init__(self, project, case_name=None):
        self.project = project
        self.case_name = case_name

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

    @property
    def output_folder(self):
        return os.path.join(
            self.project["rom_optimization_folder"], self.case_name
        )

    def run(self, config):

        # Paramaters for saving
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
        with open(
            os.path.join(self.output_folder, "parameters.json"), "w"
        ) as ff:
            json.dump(parameters, ff, indent=4)

        # Generate static report
        report = self.generate_report(project_overview=True)
        report_folder = os.path.join(self.output_folder, "report")
        report.to_html(report_folder)
        report.to_files(report_folder)

    def generate_report(self, project_overview=False):

        report = visualizer.Report()

        # Create project information section
        if project_overview:
            plot3d = plotutils.create_3d_geometry_plot_with_bcs(self.project)
            model = mdl.ZeroDModel(self.project)
            report.add_title("Project overview")
            report.add_plots([model.get_boundary_condition_info(), plot3d])

        report.add_title(f"Results")

        parameters = self._read_parameters()
        x, y, weights = self._read_results()
        plotrange = [[-2.0, 4.5], [-2.0, 4.5]]

        # Calculate 2d posterior
        bw_method = "scott"
        lin_x, lin_y, kde, bandwidth = statutils.kernel_density_estimation_2d(
            x, y, plotrange, weights, 1000, bw_method
        )

        ground_truth = [
            parameters["x_obs"]["k0"],
            parameters["x_obs"]["k1"],
        ]

        # Create the 3d kernel density estimate plot
        plot_posterior_3d = visualizer.Plot3D(
            title="Kernel density estimate",
            scene=dict(
                xaxis=dict(showbackground=False, title="k0"),
                yaxis=dict(showbackground=False, title="k1"),
                zaxis=dict(showbackground=False, title="Kernel density"),
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            width=750,
            height=750,
            autosize=True,
        )
        plot_posterior_3d.add_surface_trace(
            x=lin_x, y=lin_y, z=kde, name="KDE"
        )
        plot_posterior_3d.add_annotated_point_trace(
            x=ground_truth[0],
            y=ground_truth[1],
            z=1.1 * np.amax(kde),
            text="Ground Truth",
        )
        plot_posterior_3d.add_footnote(
            text=f"Kernel: Gaussian | Optimized Bandwith: {bandwidth:.3f} | Method: {bw_method}"
        )

        heatmap_plot = visualizer.Plot2D(
            title="Heatmap of kernel density estimate with marginals",
            xaxis_title="k0",
            yaxis_title="k1",
            width=750,
            height=750,
            autosize=True,
            xaxis_range=plotrange[0],
            yaxis_range=plotrange[1],
        )

        heatmap_plot.add_heatmap_trace(
            x=lin_x, y=lin_y, z=kde, name="Weighted particle density"
        )
        heatmap_plot.add_point_trace(x=x, y=y, color=weights, name="Particles")
        heatmap_plot.add_footnote(
            text=f"Kernel: Gaussian | Optimized Bandwith: {bandwidth:.3f} | Method: {bw_method}"
        )

        counts_x, bin_edges_x = np.histogram(
            x, bins=50, weights=weights, density=True, range=plotrange[0]
        )
        counts_y, bin_edges_y = np.histogram(
            y, bins=50, weights=weights, density=True, range=plotrange[1]
        )

        heatmap_plot.add_xy_bar_trace(
            x=bin_edges_x,
            y=bin_edges_y,
            z_x=counts_x,
            z_y=counts_y,
            name_x="Weighted histogram of k0",
            name_y="Weighted histogram of k1",
        )

        heatmap_plot.add_annotated_point_trace(
            x=ground_truth[0],
            y=ground_truth[1],
            text="Ground Truth",
            textcolor="white",
        )

        distplot_x = plotutils.create_kde_plot(
            x=x,
            weights=weights,
            plotrange=plotrange[0],
            ground_truth=ground_truth[0],
            param_name="k0",
        )
        distplot_y = plotutils.create_kde_plot(
            x=y,
            weights=weights,
            plotrange=plotrange[1],
            ground_truth=ground_truth[1],
            param_name="k1",
        )

        report.add_plots([heatmap_plot, plot_posterior_3d])
        report.add_plots([distplot_x, distplot_y])

        report.add_title(f"Parameters")
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
        report.add_table(
            pd.DataFrame(
                {
                    "Name": list(param_data.keys()),
                    "Value": list(param_data.values()),
                }
            )
        )

        return report

    def _read_parameters(self):
        with open(os.path.join(self.output_folder, "parameters.json")) as ff:
            parameters = json.load(ff)
        return parameters

    def _read_results(self):

        with open(
            os.path.join(self.output_folder, "results.pickle"),
            "rb",
        ) as ff:
            raw_results = pickle.load(ff)

        raw_output_data = raw_results["raw_output_data"]

        particles = raw_output_data["particles"]
        weights = raw_output_data["weights"]

        return (
            particles[:, 0].flatten(),
            particles[:, 1].flatten(),
            weights.flatten(),
        )
