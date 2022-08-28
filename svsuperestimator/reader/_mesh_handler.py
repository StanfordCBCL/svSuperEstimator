from ._vtk_handler import VtkHandler
import vtk
import numpy as np


class MeshHandler(VtkHandler):
    def get_boundary_centers(self):
        boundary_data = self.threshold("ModelFaceID", lower=2, cell=True)
        max_bc_id = np.max(boundary_data.get_cell_data_array("ModelFaceID"))
        middle_points = {}
        for bc_id in range(2, max_bc_id + 1):
            bc_bounds = boundary_data.threshold(
                "ModelFaceID", lower=bc_id, upper=bc_id, cell=True
            ).get_bounds()
            middle_points[bc_id] = (
                bc_bounds[[0, 2, 4]] + bc_bounds[[1, 3, 5]]
            ) / 2.0
        return middle_points
