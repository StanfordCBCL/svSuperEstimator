import yaml
import os

from .reader import SimVascularProject
from . import problems

from rich import print
import click


def run_file(path):
    """Run svSuperEstimator from a configuration file."""

    with open(path) as ff:
        config = yaml.safe_load(ff)

    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))

    project = SimVascularProject(config["project_folder"])

    problem_class = problems.get_problem_by_name(config["problem_type"])
    problem = problem_class(project, os.path.basename(path).split(".")[0])
    problem.run(config)

    os.chdir(cwd)


def run_folder(path):
    """Run all configuration files in a folder."""

    config_files = [f for f in os.listdir(path) if f.endswith(".yaml")]

    for filename in config_files:
        print(f"Running [bold magenta]{filename}[/bold magenta]")

        run_file(os.path.join(path, filename))


@click.command()
@click.argument("path")
def estimate(path):

    if os.path.isdir(path):
        run_folder(path)
    elif os.path.isfile(path):
        run_file(path)
    else:
        raise FileNotFoundError("The specified path does not exist.")
