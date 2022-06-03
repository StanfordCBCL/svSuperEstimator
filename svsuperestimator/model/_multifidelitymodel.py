"""This module holds the MultiFidelityModel class."""
from ..io import SimVascularProject
from . import ThreeDModel, ZeroDModel


class MultiFidelityModel:
    """Multi-fidelity model.

    This class contains attributes and methods to save and modify
    multi-fidelity models.
    """

    def __init__(self, project: SimVascularProject):
        """Create a new MultiFidelityModel instance.

        Args:
            project: Project object to extract the model parameters from.
        """
        self.zerodmodel = ZeroDModel(project)
        self.threedmodel = ThreeDModel(project)
