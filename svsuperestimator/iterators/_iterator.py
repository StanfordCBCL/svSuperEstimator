"""This module holds the Iterator class."""
from abc import ABC, abstractmethod


class Iterator:
    """Base class for all iterators."""

    def __init__(self) -> None:
        """Create a new iterator."""
        pass

    @abstractmethod
    def run(self):
        """Run the iterator."""
        raise NotImplementedError
