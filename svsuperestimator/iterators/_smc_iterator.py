"""This module holds the SmcIterator class."""
import numpy as np

from ._queens_iterator import QueensIterator


class SmcIterator(QueensIterator):
    """Sequentia-Monte-Carlo iterator for static models."""

    def __init__(
        self,
        forward_model,
        y_obs: np.ndarray,
        output_dir=None,
        num_procs=1,
        **kwargs
    ):
        """Create a new SmcIterator instance.

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
            "method_name": "smc_chopin",
            "method_options": {
                "seed": 42,
                "num_particles": kwargs["num_particles"],
                "resampling_threshold": kwargs["resampling_threshold"],
                "resampling_method": "systematic",
                "feynman_kac_model": "adaptive_tempering",
                "waste_free": True,
                "num_rejuvenation_steps": kwargs["num_rejuvenation_steps"],
                "model": "model",
                "max_feval": 9e99,
                "result_description": {
                    "write_results": True,
                    "plot_results": False,
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
        }
