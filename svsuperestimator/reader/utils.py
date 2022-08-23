from vtk.util.numpy_support import vtk_to_numpy
import vtk


def vtk_read_polydata(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()
    return output


def vtk_point_data_threshold(
    input_data, label: str, lower: float, upper: float
):
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(input_data)
    thresh.SetLowerThreshold(lower)
    thresh.SetUpperThreshold(upper)
    thresh.SetInputArrayToProcess(
        0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", label
    )
    thresh.Update()
    output = thresh.GetOutput()
    return output
