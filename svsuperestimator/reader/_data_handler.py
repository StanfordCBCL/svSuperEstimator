"""This module holds the DataHandler class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T", bound="DataHandler")


class DataHandler(ABC):
    """Abstract base class for data handlers."""

    def __init__(self, data: Any):
        """Create a new DataHandler instance.

        Args:
            data: The data.
        """
        self.data = data

    @abstractmethod
    def to_file(cls, filename: str) -> None:
        """Write the data to a file.

        Args:
            filename: Path to the file to read data from.
        """
        raise NotImplementedError
