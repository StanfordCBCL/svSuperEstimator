from __future__ import annotations
import os
from svsuperestimator import (
    io,
    model as mdl,
    solver as slv,
    forward_models,
    iterators,
)
import time
from rich import print
from svsuperestimator.problems import windkessel_smc_chopin

this_file_dir = os.path.dirname(__file__)


def run(
    svproject="/Users/stanford/svSuperEstimator/tmpfiles/0069_0001",
    num_procs=4,
):

    # Setup project, model, solver and webpage
    project = io.SimVascularProject(svproject)
    windkessel_smc_chopin.run(project, config={"num_procs": 4})


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
