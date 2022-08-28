from vtk.util.numpy_support import vtk_to_numpy
import vtk

import numpy as np


class VtkHandler:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_file(cls, filename):
        if filename.endswith(".vtp"):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(filename)
            reader.Update()
        else:
            raise NotImplementedError("Filetype not supported.")
        return cls(reader.GetOutput())

    def threshold(
        self, label: str, lower: float = None, upper: float = None, cell=False
    ):
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(self.data)
        if lower is not None:
            thresh.SetLowerThreshold(lower)
        if upper is not None:
            thresh.SetUpperThreshold(upper)
        if cell:
            thresh.SetInputArrayToProcess(
                0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", label
            )
        else:
            thresh.SetInputArrayToProcess(
                0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", label
            )
        thresh.Update()
        return VtkHandler(thresh.GetOutput())

    def get_points(self):
        return vtk_to_numpy(self.data.GetPoints().GetData())

    def get_polys(self):
        return vtk_to_numpy(self.data.GetPolys().GetData()).reshape(-1, 4)

    def get_bounds(self):
        # (xmin,xmax, ymin,ymax, zmin,zmax)
        return np.array(self.data.GetPoints().GetBounds())

    def get_point_data_array(self, label):
        return vtk_to_numpy(self.data.GetPointData().GetArray(label))

    def get_cell_data_array(self, label):
        return vtk_to_numpy(self.data.GetCellData().GetArray(label))

    def decimate(self, factor):
        decpro = vtk.vtkDecimatePro()
        decpro.SetInputData(self.data)
        decpro.SetTargetReduction(factor)
        decpro.Update()
        return VtkHandler(decpro.GetOutput())

    @property
    def point_data_array_num(self):
        return self.data.GetPointData().GetNumberOfArrays()

    @property
    def point_data_array_names(self):
        return [
            self.data.GetPointData().GetArrayName(i)
            for i in range(self.point_data_array_num)
        ]
