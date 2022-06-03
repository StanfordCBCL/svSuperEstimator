"""This module holds the ThreeDSolver class."""
from subprocess import call
from tempfile import TemporaryDirectory

from ..model import ThreeDModel


class ThreeDSolver:
    """3D solver.

    This class contains attributes and methods to start new 3D simulations
    using the svSolver.
    """

    def __init__(self, svsolver_exec: str, num_procs: int = 1) -> None:
        """Create a new ThreeDSolver instance.

        Args:
            svsolver_exec: Path to the svSolver executable.
            num_procs: Number of processors to use.
        """
        self._svsolver_exec = svsolver_exec
        self._num_procs = num_procs

    def run_simulation(self, model: ThreeDModel) -> None:
        """Run a new 3D solver session using the provided model.

        Args:
            model: The model to simulate.
        """

        # Create a temporary directory to perform the simulation in
        with TemporaryDirectory() as tmpdirname:

            # Create configuration files in temporary directory
            model.make_configuration(tmpdirname)

            # Call svSolver in configuration files
            call(
                args=[
                    "mpirun",
                    "-np",
                    str(self._num_procs),
                    self._svsolver_exec,
                    "solver.inp",
                ],
                cwd=tmpdirname,
            )
