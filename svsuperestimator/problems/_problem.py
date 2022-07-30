from abc import abstractmethod, ABC
import os

from ..reader import SimVascularProject


class Problem(ABC):

    PROBLEM_NAME = None

    def __init__(self, project: SimVascularProject, case_name: str = None):
        self.project = project
        self.case_name = case_name
        self.options = None

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self):
        raise NotImplementedError

    @abstractmethod
    def generate_report(self):
        raise NotImplementedError

    @property
    def output_folder(self):
        return os.path.join(
            self.project["rom_optimization_folder"], self.case_name
        )
