"""This module holds the PlainHandler class."""
from __future__ import annotations

import os
from typing import Type, TypeVar

from ._data_handler import DataHandler

T = TypeVar("T", bound="PlainHandler")

class PlainHandler(DataHandler):
    """Class for plain text based data."""

    def __init__(self, data: str):
        """Create a new PlainHandler instance.

        Args:
            data: Plain data.
        """
        self.data = data

    @classmethod
    def from_file(cls: Type[T], filename: str) -> T:
        """Create a new PlainHandler instance from file.

        Args:
            filename: Path to the file to read data from.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        with open(filename) as ff:
            data = ff.read()
        return cls(data)

    def to_file(self, filename: str) -> None:
        """Write data to file.

        Args:
            filename: Path to the file to save data at.
        """
        with open(filename, "w") as ff:
            ff.write(self.data)
