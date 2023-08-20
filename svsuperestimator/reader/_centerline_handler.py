"""This module holds the CenterlineHandler class."""
import os
from typing import Type, TypeVar

import numpy as np
import vtk

from ._vtk_handler import VtkHandler

T = TypeVar("T", bound="CenterlineHandler")


class CenterlineHandler(VtkHandler):
    """Handler for SimVascular centerline data."""

    def __init__(self, data: vtk.vtkPolyData, padding=False):
        """Create a new CenterlineHandler instance.

        Args:
            data: Centerline vtk data.
            padding: Use values of neighboring point on boundaries.
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

        self.pressure_arrays = np.array(
            [self.get_point_data_array(n) for n in self.pressure_names]
        )
        self.flow_arrays = np.array(
            [self.get_point_data_array(n) for n in self.flow_names]
        )

        self.point_locator = vtk.vtkPointLocator()
        self.point_locator.SetDataSet(self.data)
        self.point_locator.BuildLocator()

        if padding:
            num_points = data.GetNumberOfPoints()
            boundary_point_mapping = {}
            for i in range(num_points):
                idlist = vtk.vtkIdList()
                data.GetPointCells(i, idlist)
                if idlist.GetNumberOfIds() == 1:
                    idlist2 = vtk.vtkIdList()
                    data.GetCellPoints(idlist.GetId(0), idlist2)
                    alternative_point_id = [
                        idlist2.GetId(j)
                        for j in range(idlist2.GetNumberOfIds())
                        if idlist2.GetId(j) != i
                    ][0]
                    boundary_point_mapping[i] = alternative_point_id

            for source, target in boundary_point_mapping.items():
                self.pressure_arrays[:, source] = self.pressure_arrays[
                    :, target
                ]
                self.flow_arrays[:, source] = self.flow_arrays[:, target]

    @classmethod
    def from_file(cls: Type[T], filename: str, padding=False) -> T:
        """Create a new CenterlineHandler instance from file.

        Args:
            filename: Path to the file to read data from.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        if filename.endswith(".vtp"):
            reader = vtk.vtkXMLPolyDataReader()
        elif filename.endswith(".vtu"):
            reader = vtk.vtkXMLUnstructuredGridReader()
        else:
            raise NotImplementedError("Filetype not supported.")
        reader.SetFileName(filename)
        reader.Update()
        return cls(reader.GetOutput(), padding)

    def get_values_at_node(self, coord):
        """Get pressure and flow values at node specified by coordinates.

        Args:
            coord: Coordinates of the node.

        Returns:
            pressure: Time-dependent pressure values at node.
            flow: TIme-dependent flow values at node.
        """
        closest_point_id = self.point_locator.FindClosestPoint(coord)
        pressure = self.pressure_arrays[:, closest_point_id]
        flow = self.flow_arrays[:, closest_point_id]
        return pressure, flow

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
