from abc import abstractmethod, ABC
import os

from ..reader import SimVascularProject

from rich.console import Console
from rich.table import Table
from rich import box
import platform
import sys


class Task(ABC):

    TASKNAME = None

    def __init__(self, project: SimVascularProject, console=None):
        self.project = project
        self.options = None
        self.console = Console() if console is None else console
        self.database = {}
        self.config = {}

    @abstractmethod
    def run(self, config):
        self.config.update(config)
        self.console.rule(
            "[#ff9100]svSuperEstimator",
            characters="=",
            style="#ff9100",
        )
        system = platform.uname()
        self.print(
            f"platform [bold cyan]{system.system.lower()}[/bold cyan] | python {sys.version.split()[0]} at [bold cyan]{sys.executable}[/bold cyan]"
        )
        self.print(
            f"Runing task [bold cyan]{self.TASKNAME}[/bold cyan] on project [bold cyan]{self.project.name}"
        )
        table = Table(box=box.ROUNDED, expand=True)
        # table.add_column("Configuration")
        table.add_column("Configuration", style="bold cyan")
        table.add_column("Value", justify="right")
        for key, value in self.config.items():
            table.add_row(key, str(value))
        self.print(table)
        os.makedirs(self.output_folder, exist_ok=True)

    @abstractmethod
    def postprocess(self):
        raise NotImplementedError

    @abstractmethod
    def generate_report(self):
        raise NotImplementedError

    @property
    def output_folder(self):
        return os.path.join(self.project["rom_optimization_folder"])

    def log(self, *args, **kwargs):
        self.console.log(*args, **kwargs)

    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
