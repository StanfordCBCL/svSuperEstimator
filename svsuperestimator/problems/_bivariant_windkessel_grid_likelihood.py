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
from . import plotutils
import pickle
import numpy as np
from .. import visualizer
import pandas as pd
from datetime import datetime
from scipy.interpolate import griddata

from rich import print


class BivariantWindkesselGridLikelihood:

    PROBLEM_NAME = "Bivariant-Windkessel-Grid-Likelihood"

    def __init__(self, project, case_name=None):
        self.project = project
        self.case_name = case_name

        self.options = {
            "case_name": case_name,
            "num_procs": 1,
            "num_grid_points": 100,
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

        iterator = iterators.GridLikelihoodIterator(
            forward_model=forward_model,
            y_obs=y_obs,
            num_grid_points=int(config.get("num_grid_points")),
            output_dir=self.output_folder,
            num_procs=int(config.get("num_procs")),
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
        x, y, z = self._read_results()

        ground_truth = [
            parameters["x_obs"]["k0"],
            parameters["x_obs"]["k1"],
        ]

        # Create the 3d kernel density estimate plot
        plot_likelihood_3d = visualizer.Plot3D(
            title="Bivariate kernel density estimate",
            scene=dict(
                xaxis=dict(showbackground=False, title="k0"),
                yaxis=dict(showbackground=False, title="k1"),
                zaxis=dict(showbackground=False, title="KDE"),
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            width=750,
            height=750,
            autosize=True,
        )
        plot_likelihood_3d.add_surface_trace(x=x, y=y, z=z, name="KDE")
        plot_likelihood_3d.add_annotated_point_trace(
            x=ground_truth[0],
            y=ground_truth[1],
            z=1.1 * np.amax(z),
            text="Ground Truth",
        )

        report.add_plots([plot_likelihood_3d])

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

        print(raw_results)

        input_data = raw_results["input_data"]
        values = np.exp(raw_results["raw_output_data"])
        x = input_data[:, 0]
        y = input_data[:, 1]
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata(
            (x, y), np.squeeze(values), (X, Y), method="linear", rescale=True
        )
        return xi, yi, Z
