"""This module holds the QueensIterator class."""
from tempfile import TemporaryDirectory
import os
from pqueens.main import main as run_queens
import numpy as np
from ._iterator import Iterator


class QueensIterator(Iterator):
    """Base class for iterators based on QUEENS."""

    def __init__(
        self,
        forward_model,
        y_obs: np.ndarray,
        output_dir=None,
        num_procs=1,
        **kwargs,
    ):
        """Create a new QueensIterator instance.

        Args:
            forward_model: Forward model.
            y_obs: Matrix with row-wise observation vectors.
            output_dir: Output directory.
            num_procs: Number of parallel processes.
            kwargs: Valid options:
                database_address: Address to the database.
        """

        self._config = {
            "global_settings": {
                "output_dir": output_dir,
                "experiment_name": type(self).__name__,
            },
            "database": {
                "type": "mongodb",
                "address": kwargs.get("database_address", "localhost:27017"),
            },
            "forward_model": {
                "type": "simulation_model",
                "interface": "interface",
                "parameters": "parameters",
            },
            "interface": {
                "type": "direct_python_interface",
                "external_python_module_function": forward_model.evaluate,
                "num_workers": num_procs,
            },
            "parameters": {"random_variables": {}},
        }
        self._y_obs = y_obs

    def add_random_variable(
        self,
        name: str,
        dist_type: str,
        **kwargs: dict,
    ):
        """Add a new random variable to the iterator configuration.

        Args:
            label: Name of the variable.
            dist_type: Distribution type of the variable (`normal`, `uniform`,
                `lognormal`, `beta`)
            options: Parameters of the distribution. For `uniform` distribution
                specify `lower_bound` and `upper_bound`. For `normal
                distribution specify `mean` and `covariance`. For `beta`
                distribution specify `lower_bound`, `upper_bound`, `a`, and
                `b`. For `lognormal` specify `normal_mean` and
                `normal_covariance`.
        """
        var_config = {
            "distribution": dist_type,
            "type": "FLOAT",
            "size": 1,
            "dimension": 1,
        }

        if dist_type == "uniform":
            var_config.update(
                {
                    "lower_bound": kwargs["lower_bound"],
                    "upper_bound": kwargs["upper_bound"],
                }
            )
        elif dist_type == "normal":
            var_config.update(
                {
                    "mean": kwargs["mean"],
                    "covariance": kwargs["covariance"],
                }
            )
        elif dist_type == "beta":
            var_config.update(
                {
                    "lower_bound": kwargs["lower_bound"],
                    "upper_bound": kwargs["upper_bound"],
                    "a": kwargs["a"],
                    "b": kwargs["b"],
                }
            )
        elif dist_type == "lognormal":
            var_config.update(
                {
                    "normal_mean": kwargs["normal_mean"],
                    "normal_covariance": kwargs["normal_covariance"],
                }
            )
        else:
            raise ValueError(f"Unknown distribution type {dist_type}")

        self._config["parameters"]["random_variables"][name] = var_config

    def run(self):
        """Run the iterator."""

        with TemporaryDirectory() as tmpdir:

            # Make file for y_obs
            target_file = os.path.join(tmpdir, "targets.csv")
            with open(target_file, "w") as ff:
                ff.write("y_obs\n" + "\n".join([str(t) for t in self._y_obs]))
            self._config["model"]["experimental_csv_data_base_dir"] = tmpdir

            # Set output directory to temporary directory if not set
            if self._config["global_settings"]["output_dir"] is None:
                self._config["global_settings"]["output_dir"] = tmpdir

            # Run queens
            run_queens(options=self._config)
