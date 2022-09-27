"""This module holds the CenterlineHandler class."""
import numpy as np
import vtk

from ._vtk_handler import VtkHandler


class CenterlineHandler(VtkHandler):
    """Handler for SimVascular centerline data."""

    def __init__(self, data: vtk.vtkPolyData):
        """Create a new CenterlineHandler instance.

        Args:
            data: Centerline vtk data.
        """
        super().__init__(data)
        array_names = self.point_data_array_names
        self.pressure_names = [
            n for n in array_names if n.startswith("pressure")
        ]
        self.flow_names = [n for n in array_names if n.startswith("velocity")]
        try:
            self.time_steps = np.array(
                [float(k.split("_")[1]) for k in self.pressure_names]
            )
        except IndexError:
            self.time_steps = np.array([])

    def get_branch_data(self, branch_id: int) -> dict:
        """Get branch data by branch id.

        Args:
            branch_id: The ID of the branch.

        Returns:
            branch_data: The data of the branch saved in the centerline file.
        """
        output = {}
        branch_data = self.threshold("BranchId", branch_id, branch_id)
        if self.flow_names:
            output["flow"] = np.array(
                [branch_data.get_point_data_array(n) for n in self.flow_names]
            )
        if self.pressure_names:
            output["pressure"] = np.array(
                [
                    branch_data.get_point_data_array(n)
                    for n in self.pressure_names
                ]
            )
        try:
            output["path"] = branch_data.get_point_data_array("Path")
        except AttributeError:
            pass
        output["points"] = branch_data.points
        return output
