import os
from abc import ABC, abstractmethod
from time import time

from rich import box
from rich.console import Console
from rich.table import Table

from svsuperestimator import visualizer

from ..reader import SimVascularProject


class Task(ABC):
    """Base class for svSuper Estimator task.

    A task is a set of actions to be performed on a SimVascular project. Each
    task defines 3 routines. A core_run routine in which the main calculation
    of the task is performed, a post_run routine where the data is
    postprocessed and a generate_report routine, where the postprocessed
    results are visualized.

    Attributes:
        TASKNAME: Name of the task.
        DEFAULTS: DEFAULT settings for the task.
    """

    TASKNAME = None
    DEFAULTS = {}

    def __init__(self, project: SimVascularProject, config: dict):
        """Construct the task.

        Args:
            project: SimVascular project to perform the task on.
            config: Configuration for the task.
        """
        self.project = project
        self.console = None
        self.database = {}
        self.config = self.DEFAULTS.copy()
        for key in config.keys():
            if key not in self.DEFAULTS:
                raise AttributeError(f"Unknown configuration option {key}")
        self.config.update(config)
        self.output_folder = os.path.join(
            self.project["rom_optimization_folder"], self.TASKNAME
        )

    @abstractmethod
    def core_run(self):
        """Core routine of the task."""
        raise NotImplementedError

    @abstractmethod
    def post_run(self):
        """Postprocessing routine of the task."""
        raise NotImplementedError

    @abstractmethod
    def generate_report(self) -> visualizer.Report:
        """Visualization routine of the task."""
        raise NotImplementedError

    def log(self, *args, **kwargs):
        """Log to the task console."""
        self.console.log(*args, **kwargs)

    def run(self):
        """Run the task."""

        start = time()

        # Setup task console to export stdout
        self.console = Console(record=True)

        # Log task configuration
        self.log(
            f"Task [bold cyan]{self.TASKNAME}[/bold cyan] [bold #ff9100]"
            "started[/bold #ff9100] with the following configuration:"
        )
        self._log_config()

        # Make task output directory
        os.makedirs(self.output_folder, exist_ok=True)

        # Run the task and postprocessing of the data
        self.core_run()
        self.post_run()

        # Generate task report and export data
        self.log("Generate task report")
        report = self.generate_report()
        html_report_target = os.path.join(self.output_folder, "report.html")
        self.log(f"Export report webpage to {html_report_target}")
        report.to_html(
            html_report_target,
            title=self.project.name + " - svSuperEstimator",
        )
        self.log(f"Export report files to {self.output_folder}")
        report.to_files(self.output_folder)

        # Save console output
        html_log_target = os.path.join(self.output_folder, "log.html")
        self.log(f"Save task output to {html_log_target}", style="default")
        self.console.save_html(html_log_target, clear=False)
        svg_log_target = os.path.join(self.output_folder, "log.svg")
        self.log(f"Save task output to {svg_log_target}", style="default")
        self.console.save_svg(svg_log_target, clear=False)

        self.log(
            f"Task {self.TASKNAME} [bold green]completed[/bold green] in "
            f"{time()-start:.1f} seconds"
        )

    def _log_config(self):
        """Log the task configuration"""
        table = Table(box=box.HORIZONTALS, expand=True, show_header=False)
        table.add_column("Configuration", style="bold cyan")
        table.add_column("Value", justify="right")
        for key, value in self.config.items():
            table.add_row(key, str(value))
        self.log(table)
