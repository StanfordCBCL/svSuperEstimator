import os
import platform
import sys
from time import time

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich import box

from .tasks.taskutils import run_subprocess
from .reader import SimVascularProject

MAIN_CONSOLE = Console()

slurm_base = """#!/bin/bash

#SBATCH --job-name=estimator
#SBATCH --partition={partition}
#SBATCH --output={logfile}
#SBATCH --error={logfile}
#SBATCH --time={walltime}
#SBATCH --qos={qos}
#SBATCH --nodes={nodes}
#SBATCH --mem={mem}
#SBATCH --ntasks-per-node={ntasks_per_node}

module purge
module load system
module load binutils/2.38
module load qt
module load openmpi
module load mesa
module load cmake/3.23.1
module load gcc/12.1.0
module load x11
module load sqlite/3.37.2

# Command
echo "$(date): Job $SLURM_JOBID starting on $SLURM_NODELIST"
{python_path} {estimator_path} {config_file}
echo "$(date): Job $SLURM_JOBID finished on $SLURM_NODELIST"
"""

slurm_default = {
    "partition": "normal",
    "walltime": "48:00:00",
    "qos": "normal",
    "nodes": 2,
    "mem": "32GB",
    "ntasks-per-node": 24,
    "python-path": None,
}


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

        if "slurm" in config:

            parent_folder = project["parameter_estimation_folder"]
            name = task_config.get("name", None)
            if name is None:
                name = task_name
            task_output_folder = os.path.join(parent_folder, name)

            slurm_config = slurm_default.copy()
            slurm_config.update(config["slurm"])

            for key, value in slurm_config.items():
                if value is None:
                    raise ValueError(
                        f"Required option {key} not specified for slurm configuration"
                    )
            MAIN_CONSOLE.log(
                f"Scheduling task [bold cyan]{task_name}[/bold cyan] as [#ff9100]slurm[/#ff9100] job with the following configuration:"
            )
            table = Table(box=box.HORIZONTALS, expand=True, show_header=False)
            table.add_column("Configuration", style="bold cyan")
            table.add_column("Value", justify="right")
            for key, value in slurm_config.items():
                table.add_row(key, str(value))
            MAIN_CONSOLE.log(table)

            logfile = os.path.join(task_output_folder, "slurm.log")
            this_file_dir = os.path.abspath(os.path.dirname(__file__))
            estimator_path = os.path.join(this_file_dir, "main.py")

            for key in list(slurm_config.keys()):
                if "-" in key:
                    slurm_config[key.replace("-", "_")] = slurm_config[key]
                    del slurm_config[key]

            new_config_file = os.path.join(task_output_folder, "config.yaml")
            new_config = config.copy()

            del new_config["slurm"]
            new_config["tasks"] = task_config

            with open(new_config_file, "w") as ff:
                yaml.safe_dump(new_config, ff, indent=4)

            slurm_config_text = slurm_base.format(
                logfile=logfile,
                estimator_path=estimator_path,
                config_file=new_config_file,
                **slurm_config,
            )

            slurm_config_file = os.path.join(
                task_output_folder, "slurm_script.sh"
            )

            with open(slurm_config_file, "w") as ff:
                ff.write(slurm_config_text)

            run_subprocess(
                ["sbatch", slurm_config_file], logger=MAIN_CONSOLE.log
            )

        else:
            from . import tasks

            task_class = tasks.get_task_by_name(task_name)
            task = task_class(project, {**global_setting, **task_config})
            task.run()


def run_from_config(config):
    # Loading the SimVascular project
    project_folder = config["project"]
    MAIN_CONSOLE.log(
        f"Loading project [bold magenta]{project_folder}[/bold magenta]"
    )
    project = SimVascularProject(project_folder)

    global_setting = config.get("global", {})

    for task_name, task_config in config["tasks"].items():
        from . import tasks

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
