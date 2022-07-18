"""This module holds the ForwardModel class."""


class ForwardModel:
    """Base class for forward models."""

    def __init__(self, model, solver) -> None:
        """Create a new ForwardModel instance.

        Args:
            model: The model to evaluate.
            solver: The solver to use.
        """
        self.model = model
        self.solver = solver

    def evaluate(self, **kwargs):
        """Evaluate the forward model.

        Makes one forward pass.

        Args:
            kwargs: Parameters of the forward model as keyword argument.
        """
        pass
