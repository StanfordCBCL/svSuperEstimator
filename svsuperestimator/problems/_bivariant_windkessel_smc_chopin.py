from __future__ import annotations
import json
import os

from scipy.fftpack import fft
from .. import (
    model as mdl,
    solver as slv,
    forward_models,
    iterators,
    visualizer,
)
import pickle
import numpy as np
from .. import visualizer
import pandas as pd
from datetime import datetime
from ..app.helpers import create_table


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
        }

        for bc_name in mdl.ZeroDModel(project).get_outlet_bcs():
            self.options[bc_name + "_group"] = 0

    def run(self, config):

        # Paramaters for saving
        parameters = {
            "configuration": config,
            "problem_type": self.PROBLEM_NAME,
        }

        # Update case name and output folder based on config
        self.case_name = config["case_name"]
        self.output_folder = os.path.join(
            self.project["rom_optimization_folder"], self.case_name
        )
        os.makedirs(self.output_folder, exist_ok=True)

        # Setup project model and solver
        model = mdl.ZeroDModel(self.project)
        solver = slv.ZeroDSolver(cpp=True)

        # Save case name to file
        parameters["case_id"] = self.PROBLEM_NAME

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
        k0 = np.mean(
            [
                forward_model.outlet_bcs[name].resistance_distal
                for name in bc_group_0
            ]
        ) / np.mean(
            [
                forward_model.outlet_bcs[name].resistance_proximal
                for name in bc_group_0
            ]
        )
        k1 = np.mean(
            [
                forward_model.outlet_bcs[name].resistance_distal
                for name in bc_group_1
            ]
        ) / np.mean(
            [
                forward_model.outlet_bcs[name].resistance_proximal
                for name in bc_group_1
            ]
        )
        parameters["x_obs"] = {"k0": k0, "k1": k1}

        # Running one simulation to determine targets
        y_obs = forward_model.evaluate(k0=k0, k1=k1)
        parameters["y_obs"] = list(y_obs)

        iterator = iterators.SmcIterator(
            forward_model=forward_model,
            y_obs=y_obs,
            output_dir=self.output_folder,
            num_procs=config.get("num_procs", 1),
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
        report = self.generate_report()
        report_folder = os.path.join(self.output_folder, "report")
        report.to_html(report_folder)
        report.to_pngs(report_folder)

    def generate_report(self):

        report = visualizer.Report()
        report.add_title(f"Results")

        output_dir = os.path.join(
            self.project["rom_optimization_folder"], self.case_name
        )

        with open(
            os.path.join(output_dir, "results.pickle"),
            "rb",
        ) as ff:
            raw_results = pickle.load(ff)

        with open(os.path.join(output_dir, "parameters.json")) as ff:
            parameters = json.load(ff)

        mean = raw_results["mean"]
        var = raw_results["var"]
        raw_output_data = raw_results["raw_output_data"]

        particles = raw_output_data["particles"]
        weights = raw_output_data["weights"]
        log_posterior = raw_output_data["log_posterior"]
        mean = raw_output_data["mean"]
        var = raw_output_data["var"]

        x = np.exp(particles[:, 0])
        y = np.exp(particles[:, 1])
        z = np.exp(log_posterior - log_posterior.max())
        z = z / np.mean(z)

        particle_plot3d = visualizer.ParticlePlot3d(
            x,
            y,
            z,
            xlabel=r"k1",
            ylabel=r"k2",
            title="Bivariate posterior",
        )

        k0_plot = visualizer.ViolinPlot(
            pd.DataFrame(
                x,
                columns=[
                    "k0",
                ],
            ),
            title="Particle distribution for k0",
            ylabel="",
        )
        k0_gt = parameters["x_obs"]["k0"]
        k0_plot.add_lines(["k0"], [k0_gt], name="Ground Truth")

        k1_plot = visualizer.ViolinPlot(
            pd.DataFrame(
                y,
                columns=[
                    "k1",
                ],
            ),
            title="Particle distribution for k1",
            ylabel="",
        )
        k1_gt = parameters["x_obs"]["k1"]
        k1_plot.add_lines(["k1"], [k1_gt], name="Ground Truth")

        report.add_plots([k0_plot, k1_plot, particle_plot3d])

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
