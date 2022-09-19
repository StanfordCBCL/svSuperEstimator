"""This module holds the DataHandler class."""
from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any


class DataHandler(ABC):
    """Abstract base class for data handlers."""

    def __init__(self, data: Any):
        """Create a new DataHandler instance.

        Args:
            data: The data.
        """
        self.data = data

    @abstractclassmethod
    def from_file(cls, filename: str) -> DataHandler:
        """Create a new DataHandler instance from file.

        Args:
            filename: Path to the file to read data from.
        """
        raise NotImplementedError

    @abstractmethod
    def to_file(cls, filename: str):
        """Write the data to a file.

        Args:
            filename: Path to the file to read data from.
        """
        raise NotImplementedError
