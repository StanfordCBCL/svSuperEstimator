import yaml
import os

from .reader import SimVascularProject
from . import problems

from rich import print
import click


def run_file(path, post_proc_only=False):
    """Run svSuperEstimator from a configuration file."""

    with open(path) as ff:
        config = yaml.safe_load(ff)

    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))

    project = SimVascularProject(config["project_folder"])

    problem_class = problems.get_problem_by_name(config["problem_type"])
    problem = problem_class(project, os.path.basename(path).split(".")[0])
    if post_proc_only:
        report = problem.generate_report(project_overview=True)
        report_folder = os.path.join(problem.output_folder, "report")
        report.to_html(report_folder)
        report.to_files(report_folder)
    else:
        problem.run(config)

    os.chdir(cwd)


def run_folder(path, post_proc_only=False):
    """Run all configuration files in a folder."""

    config_files = [f for f in os.listdir(path) if f.endswith(".yaml")]

    for filename in config_files:
        print(f"Running [bold magenta]{filename}[/bold magenta]")

        run_file(os.path.join(path, filename), post_proc_only=post_proc_only)


@click.command()
@click.argument("path")
@click.option(
    "-pp",
    "--post_proc_only",
    help="Only to post processing to existing results.",
    is_flag=True,
)
def estimate(path, post_proc_only):

    if os.path.isdir(path):
        run_folder(path, post_proc_only=post_proc_only)
    elif os.path.isfile(path):
        run_file(path, post_proc_only=post_proc_only)
    else:
        raise FileNotFoundError("The specified path does not exist.")
