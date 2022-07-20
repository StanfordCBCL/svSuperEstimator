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

    OPTIONS_WITH_DEFAULT = {
        "num_procs": 1,
        "num_particles": 100,
        "num_rejuvenation_steps": 2,
        "resampling_threshold": 0.5,
    }

    PROBLEM_NAME = "Windkessel-SMC-Chopin"

    @classmethod
    def run(cls, project, config, case_name=None):

        # Setup project model and solver
        model = mdl.MultiFidelityModel(project)
        solver = slv.ZeroDSolver(cpp=True)

        optim_folder = project["rom_optimization_folder"]
        if case_name is None:
            output_folder = optim_folder
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        else:
            output_folder = os.path.join(optim_folder, case_name)
            os.makedirs(output_folder, exist_ok=True)

        # Save parameters to file
        with open(os.path.join(output_folder, "parameters.json"), "w") as ff:
            json.dump(config, ff, indent=4)

        # Save case name to file
        with open(os.path.join(output_folder, "case.txt"), "w") as ff:
            ff.write(cls.PROBLEM_NAME)

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
            output_dir=output_folder,
            num_procs=config.get("num_procs", 1),
        )

        for i in range(len(target_k)):
            iterator.add_random_variable(
                f"k{i}", "uniform", lower_bound=-2, upper_bound=3
            )

        # Run the iterator
        iterator.run()

    @staticmethod
    def generate_report(project, case_name=None):

        report = visualizer.Report()

        output_dir = os.path.join(
            project["rom_optimization_folder"], case_name
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
