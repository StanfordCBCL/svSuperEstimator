"""This module holds the ThreeDModel and auxiliary classes."""
import os
from shutil import copyfile

from ..reader import SimVascularProject


class ThreeDModel:
    """3D model.

    This class contains attributes and methods to save and modify a 3D model
    for the svSolver. It is aimed at facilitating the variation of
    input parameters.
    """

    def __init__(self, project: SimVascularProject):
        """Create a new ThreeDModel instance.

        Args:
            project: Project object to extract the model parameters from.
        """

        self._project = project
        self._input = project["3d_simulation_input"]
        self._geombc = project["3d_simulation_geombc"]
        self._restart = project["3d_simulation_restart"]
        self._numstart = project["3d_simulation_numstart"]
        self._bct = project["3d_simulation_bct"]

    def make_configuration(self, target: str) -> None:
        """Make the configuration at a specified target.

        Creates all configuration files that are needed for a svSolver session.

        Args:
            target: Target folder for the configuration files.
        """
        if not os.path.isdir(target):
            raise ValueError(
                "Configuration target for ThreeDModel must be folder."
            )
        copyfile(self._input, os.path.join(target, "solver.inp"))
        copyfile(self._geombc, os.path.join(target, "geombc.dat.1"))
        copyfile(self._restart, os.path.join(target, "restart.0.1"))
        copyfile(self._numstart, os.path.join(target, "numstart.dat"))
        copyfile(self._bct, os.path.join(target, "bct.dat"))
