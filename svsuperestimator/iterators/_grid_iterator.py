import numpy as np

from ._queens_iterator import QueensIterator


class GridIterator(QueensIterator):
    def __init__(
        self, forward_model, y_obs: np.ndarray, output_dir=None, num_procs=1
    ):
        super().__init__(forward_model, y_obs, output_dir, num_procs)
