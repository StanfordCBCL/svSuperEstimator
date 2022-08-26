from ._vtk_handler import VtkHandler

import numpy as np


class CenterlineHandler(VtkHandler):
    def __init__(self, data):
        super().__init__(data)
        self.array_names = self.point_data_array_names
        self.pressure_names = [
            n for n in self.array_names if n.startswith("pressure")
        ]
        self.flow_names = [
            n for n in self.array_names if n.startswith("velocity")
        ]
        self.time_steps = np.array(
            [float(k.split("_")[1]) for k in self.pressure_names]
        )

    def get_branch_data(self, branch_id):
        output = {}
        branch_data = self.threshold("BranchId", branch_id, branch_id)
        output["flow"] = np.array(
            [branch_data.get_point_data_array(n) for n in self.flow_names]
        )
        output["pressure"] = np.array(
            [branch_data.get_point_data_array(n) for n in self.pressure_names]
        )
        output["path"] = branch_data.get_point_data_array("Path")
        output["points"] = branch_data.get_points()
        return output
