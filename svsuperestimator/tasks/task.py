"""This module holds the Task base class."""
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from time import perf_counter as time
from typing import Any, Dict, List, Optional
from shutil import copytree

import orjson
from rich import box
from rich.console import Console
from rich.table import Table

from svsuperestimator import visualizer

from .. import VERSION
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

    TASKNAME: Optional[str] = None
    DEFAULTS: Dict[str, Any] = {
        "report_html": True,
        "report_files": False,
        "overwrite": False,
        "name": None,
        "debug": False,
        "core_run": True,
        "post_proc": True,
        "temporary_output_folder": None,
    }

    MUST_EXIST_AT_INIT: List[str] = []

    def __init__(
        self,
        project: SimVascularProject,
        config: dict,
        prefix: str = "",
        parent_folder: Optional[str] = None,
        log_config: bool = True,
    ):
        """Construct the task.

        Args:
            project: SimVascular project to perform the task on.
            config: Configuration for the task.
            prefix: Prefix for the naming of the task folder.
            parent_folder: Parent folder for the task folder. If None,
                parameter estimation folder of project is used.
            log_config: Toggle logging of config at instantiation.
        """
        self.project = project
        self.console = Console(
            record=True, log_time_format="[%m/%d/%y %H:%M:%S]"
        )
        self.database: Dict[str, Any] = {}
        self.config = deepcopy(self.DEFAULTS)
        self.config.update(config)
        if parent_folder is None:
            parent_folder = self.project["parameter_estimation_folder"]

        if self.config["name"] is None:
            self.config["name"] = prefix + self.TASKNAME  # type: ignore

        self.final_output_folder = os.path.join(
            parent_folder, self.config["name"]  # type: ignore
        )
        if self.config["temporary_output_folder"] is None:
            self.output_folder = self.final_output_folder
        else:
            self.output_folder = self.config["temporary_output_folder"]

        if log_config:
            self.log(
                f"Created task [bold cyan]{type(self).__name__}[/bold cyan] "
                "with the following configuration:"
            )
            self._log_config()
        for key, value in self.config.items():
            if key not in self.DEFAULTS:
                self.log(f"Unused configuration option {key}")
            # if value is None:
            #     raise RuntimeError(
            #         f"Required option {key} for task "
            #         f"{type(self).__name__} not specified."
            #     )
            if key in self.MUST_EXIST_AT_INIT and not os.path.exists(
                self.config[key]
            ):
                raise FileNotFoundError(
                    f"File {self.config[key]} does not exist."
                )
        if self.output_folder != self.final_output_folder:
            copytree(self.output_folder, self.final_output_folder)

    @abstractmethod
    def core_run(self) -> None:
        """Core routine of the task."""
        raise NotImplementedError

    def post_run(self) -> None:
        """Postprocessing routine of the task."""
        pass

    def generate_report(self) -> visualizer.Report:
        """Visualization routine of the task."""
        return visualizer.Report()

    def log(self, *args: Any, **kwargs: Any) -> None:
        """Log to the task console."""
        self.console.log(*args, **kwargs)

    def run(self) -> None:
        """Run the task."""

        if not self.config["overwrite"] and self.is_completed():
            self.log(
                f"Skipping task [bold cyan]{type(self).__name__}[/bold cyan]"
            )
            return

        start_task = time()

        self.log(f"Running on version [bold magenta]{VERSION}[/bold magenta]")
        self.log(f"Starting task [bold cyan]{type(self).__name__}[/bold cyan]")
        self.database["version"] = VERSION

        # Make task output directory
        os.makedirs(self.output_folder, exist_ok=True)

        if self.config["core_run"]:
            # Run the task and postprocessing of the data
            start_core_run = time()
            self.core_run()
            end_core_run = time()
            self.database["core_runtime"] = end_core_run - start_core_run
            self.save_database()
        if self.config["post_proc"]:
            self.log("Postprocessing results")
            self.load_database()
            start_post_run = time()
            self.post_run()
            end_post_run = time()
            self.database["post_runtime"] = start_post_run - end_post_run
            self.save_database()

        # Generate task report and export data
        self.log("Generate task report")
        self.load_database()
        if self.config["report_html"] or self.config["report_files"]:
            report = self.generate_report()

        # Export report files
        if self.config["report_files"]:
            report.to_files(self.output_folder)
            self.log(f"Saved report files {self.output_folder}")

        # Save console output
        html_log_target = os.path.join(self.output_folder, "log.html")
        self.console.save_html(html_log_target, clear=False)
        self.log(f"Saved html task log {html_log_target}", style="default")
        svg_log_target = os.path.join(self.output_folder, "log.svg")
        self.console.save_svg(svg_log_target, clear=False, title="Output")
        self.log(f"Saved svg task log {svg_log_target}", style="default")

        # Export html report
        if self.config["report_html"] and report is not None:
            html_report_target = os.path.join(
                self.output_folder, "report.html"
            )
            report.to_html(
                html_report_target,
                title=self.project.name + " - svSuperEstimator",
            )
            self.log(f"Saved html report {html_report_target}")

        end_task = time()
        self.database["total_runtime"] = end_task - start_task
        self.save_database()
        self.log(
            f"Task [bold cyan]{type(self).__name__}[/bold cyan] "
            "[bold green]completed[/bold green] in "
            f"{end_task-start_task:.1f} seconds"
        )
        Path(os.path.join(self.output_folder, ".completed")).touch()

    def is_completed(self) -> bool:
        """Check if task is already completed."""
        return os.path.exists(os.path.join(self.output_folder, ".completed"))

    def _log_config(self) -> None:
        """Log the task configuration."""
        table = Table(box=box.HORIZONTALS, expand=True, show_header=False)
        table.add_column("Configuration", style="bold cyan")
        table.add_column("Value", justify="right")
        for key, value in self.config.items():
            table.add_row(key, str(value))
        self.log(table)

    def save_database(self) -> None:
        """Set problem parameters.

        Args:
            parameters: Parameters of the problem.
        """

        with open(
            os.path.join(self.output_folder, "taskdata.json"), "wb"
        ) as ff:
            ff.write(
                orjson.dumps(
                    self.database,
                    option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY,
                )
            )
        self.database = {}

    def load_database(self) -> None:
        """Return problem parameters.

        Returns:
            parameters: Parameters of the problem.
        """
        try:
            with open(
                os.path.join(self.output_folder, "taskdata.json"), "rb"
            ) as ff:
                self.database = orjson.loads(ff.read())
        except FileNotFoundError:
            self.database = {}
