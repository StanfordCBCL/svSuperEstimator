"""This module holds the SimVascularProject class."""
import json
import os
from typing import Any

import yaml


class SimVascularProject:
    """Class handling the SimVascular project folder."""

    with open(
        os.path.join(os.path.dirname(__file__), "sv_file_registry.yaml")
    ) as ff:
        _FILE_REGISTRY = yaml.full_load(ff)

    def __init__(self, folder: str) -> None:
        """Create a new SimVascular project instance from a folder.

        Args:
            folder: SimVascular project folder.
        """
        # Check if folder exists
        if not os.path.exists(os.path.join(folder, ".svproj")):
            raise ValueError(
                f"Specified directory '{folder}' is not a valid SimVascular "
                "project folder."
            )
        self._folder = os.path.abspath(folder)
        self._regex: dict[str, str] = {
            "$CASE_NAME$": os.path.basename(self._folder)
        }

    def __getitem__(self, key: str) -> Any:
        """Get data specified by a key.

        This method defines what happens if an instance of SimVascularProject
        is indexed. For this method to return a chunk of data from a
        SimVascular project folder, the data item has to be configured in
        `sv_file_registry.yaml` with an index ID, type and path.

        Args:
            key: The index of the data element.
        """
        if key not in self._FILE_REGISTRY:
            raise KeyError(f"Unknown key: {key}")
        elif self._FILE_REGISTRY[key]["type"] == "json":
            target = os.path.join(
                self._folder, self._FILE_REGISTRY[key]["path"]
            )
            if "$" in target:
                for regex, repl in self._regex.items():
                    target = target.replace(regex, repl)
            with open(target) as ff:
                data = json.load(ff)
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
        if key not in self._FILE_REGISTRY:
            raise KeyError(f"Unknown key: {key}")
        elif self._FILE_REGISTRY[key]["type"] == "json":
            target = os.path.join(self._folder, key)
            if "$" in target:
                for regex, repl in self._regex.items():
                    target = target.replace(regex, repl)
            with open(target, "w") as ff:
                json.dump(data, ff, indent=4)
            return data
