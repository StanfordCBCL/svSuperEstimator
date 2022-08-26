from vtk.util.numpy_support import vtk_to_numpy
import vtk

import numpy as np


class VtkHandler:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_file(cls, filename):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        return cls(reader.GetOutput())

    def threshold(self, label: str, lower: float, upper: float):
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(self.data)
        thresh.SetLowerThreshold(lower)
        thresh.SetUpperThreshold(upper)
        thresh.SetInputArrayToProcess(
            0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", label
        )
        thresh.Update()
        return VtkHandler(thresh.GetOutput())

    def get_points(self):
        return vtk_to_numpy(self.data.GetPoints().GetData())

    def get_point_data_array(self, label):
        return vtk_to_numpy(self.data.GetPointData().GetArray(label))

    @property
    def point_data_array_num(self):
        return self.data.GetPointData().GetNumberOfArrays()

    @property
    def point_data_array_names(self):
        return [
            self.data.GetPointData().GetArrayName(i)
            for i in range(self.point_data_array_num)
        ]
