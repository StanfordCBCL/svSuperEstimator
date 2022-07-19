from __future__ import annotations
import json
import os
from .. import (
    model as mdl,
    solver as slv,
    forward_models,
    iterators,
)


def run(project, config, case_name=None):

    # Setup project model and solver
    model = mdl.MultiFidelityModel(project)
    solver = slv.ZeroDSolver(cpp=True)

    optim_folder = project["rom_optimization_folder"]
    if case_name is None:
        output_folder = optim_folder
    else:
        output_folder = os.path.join(optim_folder, case_name)

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save parameters to file
    with open(os.path.join(output_folder, "parameters.json"), "w") as ff:
        json.dump(config, ff, indent=4)

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

    for i in range(4):
        iterator.add_random_variable(
            f"k{i}", "uniform", lower_bound=-2, upper_bound=3
        )

    # Run the iterator
    iterator.run()
