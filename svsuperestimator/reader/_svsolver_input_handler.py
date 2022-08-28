import re


class SvSolverInputHandler:
    def __init__(self, filename):
        with open(filename) as ff:
            self.data = ff.read()

    @property
    def time_step_size_3d(self):
        return float(self._find_configuration("Time Step Size"))

    @property
    def rcr_surface_ids(self):
        surface_list = self._find_configuration("List of RCR Surfaces")
        return [int(num) for num in surface_list.split()]

    def _find_configuration(self, specifier):
        return re.search(
            specifier + ":" + r".*$", self.data, re.MULTILINE
        ).group()[len(specifier) + 2 :]
