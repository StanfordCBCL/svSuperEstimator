from __future__ import annotations
import os

from scipy.fftpack import fft
from svsuperestimator import (
    io,
    model as mdl,
    solver as slv,
    forward_models,
    iterators,
)
import time
from rich import print

this_file_dir = os.path.dirname(__file__)


def run(
    svproject="/Users/stanford/svSuperEstimator/tmpfiles/0069_0001",
    num_procs=4,
):
    start = time.time()

    # Setup project, model, solver and webpage
    project = io.SimVascularProject(svproject)
    # webpage = io.WebPage("svSuperEstimator")
    model = mdl.MultiFidelityModel(project)
    solver = slv.ZeroDSolver(cpp=True)

    if not os.path.exists(project["rom_optimization_folder"]):
        os.makedirs(project["rom_optimization_folder"])

    forward_model = forward_models.WindkesselDistalToProximalResistance0D(
        model, solver
    )

    target_k = {
        f"k{i}": bc.resistance_distal / bc.resistance_proximal
        for i, bc in enumerate(forward_model.outlet_bcs.values())
    }

    # Running one simulation to determine targets
    y_obs = forward_model.evaluate(**target_k)

    iterator = iterators.SmcIterator(
        forward_model=forward_model,
        y_obs=y_obs,
        output_dir=project["rom_optimization_folder"],
        num_procs=num_procs,
    )
    for i in range(4):
        iterator.add_random_variable(
            f"k{i}", "uniform", lower_bound=-2, upper_bound=3
        )

    iterator.run()
    print(f"Completed in {time.time()-start:.2f}s")


# @click.command()
# @click.option(
#     "--svproject",
#     help="Path to SimVascular project folder.",
#     required=True,
#     type=str,
# )
# @click.option(
#     "--num_samples",
#     help="Number of samples from target distribution.",
#     type=int,
#     default=10,
# )
# def main(svproject, num_samples):
#     """Run svSuperEstimator."""
#     run(svproject, num_samples)


if __name__ == "__main__":
    run()
