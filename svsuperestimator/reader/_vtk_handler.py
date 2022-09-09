"""This module holds the VtkHandler class."""
from __future__ import annotations

import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from ._data_handler import DataHandler


class VtkHandler(DataHandler):
    """Handler for vtk based data."""

    def __init__(self, data: vtk.vtkPolyData):
        """Create a new VtkHandler instance.

        Args:
            data: VTK data.
        """
        self.data = data

    def __getitem__(self, key: str) -> np.ndarray:
        """Helper method to allow direct indexing."""
        if self.data.GetPointData().GetArray(key) is None:
            return self.get_cell_data_array(key)
        else:
            return self.get_point_data_array(key)

    @classmethod
    def from_file(cls, filename: str) -> VtkHandler:
        """Create a new VtkHandler instance from file.

        Args:
            filename: Path to the file to read data from.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        if filename.endswith(".vtp"):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(filename)
            reader.Update()
        else:
            raise NotImplementedError("Filetype not supported.")
        return cls(reader.GetOutput())

    def to_file(self, filename: str):
        """Write the data to a file.

        Args:
            filename: Path to the file to read data from.
        """
        if filename.endswith(".vtp"):
            writer = vtk.vtkXMLPolyDataWriter()
        elif filename.endswith(".vtu"):
            writer = vtk.vtkXMLUnstructuredGridWriter()
        else:
            raise NotImplementedError("Filetype not supported.")
        writer.SetFileName(filename)
        writer.SetInputData(self.data)
        writer.Update()
        writer.Write()

    @property
    def points(self) -> np.ndarray:
        """Point coordinates."""
        return vtk_to_numpy(self.data.GetPoints().GetData())

    @property
    def polys(self) -> np.ndarray:
        """Poly indices."""
        return vtk_to_numpy(self.data.GetPolys().GetData()).reshape(-1, 4)

    @property
    def bounds(self) -> np.ndarray:
        """Bounds of the data (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return np.array(self.data.GetPoints().GetBounds())

    @property
    def point_data_array_num(self) -> int:
        """Number of point data arrays."""
        return self.data.GetPointData().GetNumberOfArrays()

    @property
    def point_data_array_names(self) -> list[str]:
        """Names of point data arrays."""
        return [
            self.data.GetPointData().GetArrayName(i)
            for i in range(self.point_data_array_num)
        ]

    def threshold(
        self, label: str, lower: float = None, upper: float = None
    ) -> "VtkHandler":
        """Apply a threshold to the data and return thresholded data.

        Args:
            label: Label of the array to use for thresholding.
            lower: Lower bound of the threshold.
            upper: Upper bound of the threshold.
        """
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(self.data)
        if lower is not None:
            thresh.SetLowerThreshold(lower)
        if upper is not None:
            thresh.SetUpperThreshold(upper)
        if self.data.GetCellData().HasArray(label):
            thresh.SetInputArrayToProcess(
                0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", label
            )
        else:
            thresh.SetInputArrayToProcess(
                0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", label
            )
        thresh.Update()
        return VtkHandler(thresh.GetOutput())

    def get_point_data_array(self, label: str) -> np.ndarray:
        """Return a point data array.

        Returns:
            data_array: Point data array.
        """
        return vtk_to_numpy(self.data.GetPointData().GetArray(label))

    def get_cell_data_array(self, label: str) -> np.ndarray:
        """Return a cell data array.

        Returns:
            data_array: Cell data array.
        """
        return vtk_to_numpy(self.data.GetCellData().GetArray(label))

    def decimate(self, factor: float) -> "VtkHandler":
        """Reduce the number of polygons in the mesh.

        Args:
            factor: Factor between 0.0 and 1.0 that specifies the percentage
                of polygons to reduce by.

        Returns:
            handler: VTK handler for the decimated data.
        """
        decpro = vtk.vtkDecimatePro()
        decpro.SetInputData(self.data)
        decpro.SetTargetReduction(factor)
        decpro.Update()
        return VtkHandler(decpro.GetOutput())
