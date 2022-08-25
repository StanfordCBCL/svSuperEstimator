import yaml
import os

from .reader import SimVascularProject
from . import tasks

from rich.console import Console
import click
from time import time
import platform
import sys

MAIN_CONSOLE = Console()


def run_file(path):
    """Run svSuperEstimator from a configuration file."""

    MAIN_CONSOLE.log(
        f"Loading configuration [bold magenta]{path}[/bold magenta]"
    )

    with open(path) as ff:
        config = yaml.safe_load(ff)

    # Loading the SimVascular project
    project_folder = config["project"]
    MAIN_CONSOLE.log(
        f"Loading project [bold magenta]{project_folder}[/bold magenta]"
    )
    project = SimVascularProject(project_folder)

    global_setting = config.get("global", {})

    for task_name, task_config in config["tasks"].items():
        task_class = tasks.get_task_by_name(task_name)
        task = task_class(project, {**global_setting, **task_config})
        task.run()


def run_folder(path):
    """Run all configuration files in a folder."""

    config_files = [f for f in os.listdir(path) if f.endswith(".yaml")]

    for filename in config_files:
        run_file(os.path.join(path, filename))


@click.command()
@click.argument("path")
def estimate(path):
    start = time()
    MAIN_CONSOLE.rule(
        "[#ff9100]svSuperEstimator",
        characters="=",
        style="#ff9100",
    )
    system = platform.uname()
    MAIN_CONSOLE.print(
        f"platform [bold cyan]{system.system.lower()}[/bold cyan] | python {sys.version.split()[0]} at [bold cyan]{sys.executable}[/bold cyan]"
    )

    if os.path.isdir(path):
        run_folder(path)
    elif os.path.isfile(path):
        run_file(path)
    else:
        raise FileNotFoundError("The specified path does not exist.")

    MAIN_CONSOLE.rule(f"Completed in {time()-start:.1f} seconds")
