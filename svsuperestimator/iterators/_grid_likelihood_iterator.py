"""This module holds the SmcIterator class."""
import numpy as np

from ._queens_iterator import QueensIterator


class GridLikelihoodIterator(QueensIterator):
    """Grid-likelihood iterator."""

    def __init__(
        self,
        forward_model,
        y_obs: np.ndarray,
        num_grid_points=100,
        output_dir=None,
        num_procs: int = 1,
        **kwargs
    ):
        """Create a new GridLikelihoodIterator instance.

        Args:
            forward_model: Forward model.
            y_obs: Matrix with row-wise observation vectors.
            output_dir: Output directory.
            num_procs: Number of parallel processes.
            kwargs: Optional parameters
                * `database_address`: Address to the database for QUEENS.
                * `num_particles`: Number of particles for SMC.
                * `resampling_threshold`: Resampling threshold for SMC.
                * `num_rejuvenation_steps`: Number of rejuvenation steps SMC.

        """
        super().__init__(forward_model, y_obs, output_dir, num_procs, **kwargs)

        self._config["method"] = {
            "method_name": "grid",
            "method_options": {
                "model": "model",
                "grid_design": {},
                "result_description": {
                    "write_results": True,
                    "plotting_options": {
                        "plot_booleans": [False],
                        "plotting_dir": "None",
                        "plot_names": ["None"],
                        "save_bool": [False],
                    },
                },
            },
        }
        self._config["model"] = {
            "type": "likelihood_model",
            "subtype": "gaussian",
            "forward_model": "forward_model",
            "output_label": "y_obs",
            "coordinate_labels": [],
            "noise_type": kwargs["noise_type"],
            "noise_value": kwargs["noise_value"],
            "experimental_file_name_identifier": "*.csv",
            "experimental_csv_data_base_dir": None,
            "parameters": "parameters",
            "noise_var_iterative_averaging": {
                "averaging_type": "moving_average",
                "num_iter_for_avg": 10,
            },
        }
        self._num_grid_points = num_grid_points

    def add_random_variable(self, name: str, dist_type: str, **kwargs: dict):
        super().add_random_variable(name, dist_type, **kwargs)

        self._config["method"]["method_options"]["grid_design"][name] = {
            "num_grid_points": self._num_grid_points,
            "axis_type": "lin",
        }
