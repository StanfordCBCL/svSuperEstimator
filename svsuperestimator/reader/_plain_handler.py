"""This module holds the PlainHandler class."""
from __future__ import annotations

import os

from ._data_handler import DataHandler


class PlainHandler(DataHandler):
    """Class for plain text based data."""

    def __init__(self, data: str):
        """Create a new PlainHandler instance.

        Args:
            data: Plain data.
        """
        self.data = data

    @classmethod
    def from_file(cls, filename: str) -> PlainHandler:
        """Create a new PlainHandler instance from file.

        Args:
            filename: Path to the file to read data from.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        with open(filename) as ff:
            data = ff.read()
        return cls(data)
