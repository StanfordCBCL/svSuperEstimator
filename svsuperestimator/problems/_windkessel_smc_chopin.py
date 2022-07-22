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
import pickle
import numpy as np


class WindkesselSMCChopin:

    PROBLEM_NAME = "Windkessel-SMC-Chopin"

    def __init__(self, project, case_name=None):
        self.project = project
        self.case_name = case_name

        self.options = {
            "num_procs": 1,
            "num_particles": 100,
            "num_rejuvenation_steps": 2,
            "resampling_threshold": 0.5,
        }

        optim_folder = self.project["rom_optimization_folder"]
        if case_name is None:
            self.output_folder = optim_folder
        else:
            self.output_folder = os.path.join(optim_folder, case_name)

    def run(self, config):

        # Setup project model and solver
        model = mdl.MultiFidelityModel(self.project)
        solver = slv.ZeroDSolver(cpp=True)

        os.makedirs(self.output_folder, exist_ok=True)

        # Save parameters to file
        with open(
            os.path.join(self.output_folder, "parameters.json"), "w"
        ) as ff:
            json.dump(config, ff, indent=4)

        # Save case name to file
        with open(os.path.join(self.output_folder, "case.txt"), "w") as ff:
            ff.write(self.PROBLEM_NAME)

        # Create the foward model
        forward_model = forward_models.WindkesselDistalToProximalResistance0D(
            model.zerodmodel, solver
        )

        # Get ground truth distal to proximal ratio
        target_k = {
            f"k{i}": bc.resistance_distal / bc.resistance_proximal
            for i, bc in enumerate(forward_model.outlet_bcs.values())
        }

        # Running one simulation to determine targets
        y_obs = forward_model.evaluate(**target_k)

        iterator = iterators.SmcIterator(
            forward_model=forward_model,
            y_obs=y_obs,
            output_dir=self.output_folder,
            num_procs=config.get("num_procs", 1),
        )

        for i in range(len(target_k)):
            iterator.add_random_variable(
                f"k{i}", "uniform", lower_bound=-2, upper_bound=3
            )

        # Run the iterator
        iterator.run()

    def generate_report(self):

        report = visualizer.Report()

        output_dir = os.path.join(
            self.project["rom_optimization_folder"], self.case_name
        )

        with open(
            os.path.join(output_dir, "results.pickle"),
            "rb",
        ) as ff:
            raw_results = pickle.load(ff)

        mean = raw_results["mean"]
        var = raw_results["var"]
        raw_output_data = raw_results["raw_output_data"]

        particles = raw_output_data["particles"]
        weights = raw_output_data["weights"]
        log_posterior = raw_output_data["log_posterior"]
        mean = raw_output_data["mean"]
        var = raw_output_data["var"]

        x = particles[:, 0]
        y = particles[:, 1]
        z = np.exp(log_posterior - log_posterior.max())
        z = z / np.mean(z)

        particle_plot3d = visualizer.ParticlePlot3d(
            x,
            y,
            z,
            xlabel=r"Rp",
            ylabel=r"Rd",
        )
        report.add_plots(particle_plot3d)

        return report