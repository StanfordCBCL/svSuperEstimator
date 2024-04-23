"""This module holds the SimVascularProject class."""

import os
from typing import Any, Dict

import yaml

from ._centerline_handler import CenterlineHandler
from ._mesh_handler import MeshHandler
from ._svsolver_inflow_handler import SvSolverInflowHandler
from ._svsolver_input_handler import SvSolverInputHandler
from ._svsolver_rcr_handler import SvSolverRcrHandler
from ._svzerodsolver_input_handler import SvZeroDSolverInputHandler
from ._vtk_handler import VtkHandler

_HANDLERS = {
    "VtkHandler": VtkHandler,
    "CenterlineHandler": CenterlineHandler,
    "MeshHandler": MeshHandler,
    "SvSolverInputHandler": SvSolverInputHandler,
    "SvSolverRcrHandler": SvSolverRcrHandler,
    "SvSolverInflowHandler": SvSolverInflowHandler,
    "SvZeroDSolverInputHandler": SvZeroDSolverInputHandler,
}


class SimVascularProject:
    """Class handling the SimVascular project folder."""

    def __init__(self, folder: str, registry_override: dict = None) -> None:
        """Create a new SimVascular project instance from a folder.

        Args:
            folder: SimVascular project folder.
            registry_override: Overwrite file names from the default in
                sv_file_registry.yaml.
        """
        self._folder = os.path.abspath(folder)
        self._regex: Dict[str, str] = {
            "$CASE_NAME$": os.path.basename(self._folder)
        }
        self.name = os.path.basename(folder)

        # Read file registry
        with open(
            os.path.join(os.path.dirname(__file__), "sv_file_registry.yaml")
        ) as ff:
            self._file_registry = yaml.full_load(ff)

        if registry_override is not None:
            for key, value in registry_override.items():
                self._file_registry[key]["path"] = value

    def __getitem__(self, key: str) -> Any:
        """Get data specified by a key.

        This method defines what happens if an instance of SimVascularProject
        is indexed. For this method to return a chunk of data from a
        SimVascular project folder, the data item has to be configured in
        `sv_file_registry.yaml` with an index ID, type and path.

        Args:
            key: The index of the data element.
        """
        data: Any
        if key not in self._file_registry:
            raise KeyError(f"Unknown key: {key}")
        elif self._file_registry[key]["type"] == "data":
            target = os.path.join(
                self._folder, self._file_registry[key]["path"]
            )
            if "$" in target:
                for regex, repl in self._regex.items():
                    target = target.replace(regex, repl)
            data = _HANDLERS[
                self._file_registry[key]["handler"]
            ].from_file(  # type: ignore
                target
            )
        elif self._file_registry[key]["type"] == "path":
            target = self._file_registry[key]["path"]
            if "$" in target:
                for regex, repl in self._regex.items():
                    target = target.replace(regex, repl)
            data = os.path.join(self._folder, target)
        else:
            raise IndexError(
                f"Specified element {key} doesn't support reading."
            )
        return data

    def __setitem__(self, key: str, data: Any) -> None:
        """Set data specified by a key.

        This method defines what happens if a data item of an instance of
        SimVascularProject is set. For this method to set a chunk of data in
        a SimVascular project folder, the data item has to be configured in
        `sv_file_registry.yaml` with an index ID, type and path.

        Args:
            key: The index of the data element.
            data: The data to set.
        """
        if key not in self._file_registry:
            raise KeyError(f"Unknown key: {key}")
        elif self._file_registry[key]["type"] == "data":
            target = os.path.join(
                self._folder, self._file_registry[key]["path"]
            )
            if "$" in target:
                for regex, repl in self._regex.items():
                    target = target.replace(regex, repl)
            data.to_file(target)
        else:
            raise IndexError(
                f"Specified element {key} doesn't support setting."
            )
