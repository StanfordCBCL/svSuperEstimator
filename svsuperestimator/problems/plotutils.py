from __future__ import annotations
from svsuperestimator.model import ZeroDModel
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from svsuperestimator import visualizer


def create_3d_geometry_plot_with_bcs(project):

    model = ZeroDModel(project)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(project["rom_centerline"])
    reader.Update()
    polydata = reader.GetOutput()

    points = vtk_to_numpy(polydata.GetPoints().GetData())
    cells = vtk_to_numpy(polydata.GetLines().GetData()).reshape(-1, 3)

    branch_ids = vtk_to_numpy(polydata.GetPointData().GetArray("BranchId"))

    border_point_indices = np.where(
        np.unique(cells[:, [1, 2]].flatten(), return_counts=True)[1] == 1
    )
    border_points = points[border_point_indices]
    border_branch_ids = branch_ids[border_point_indices]

    branch_id_to_bc = {}
    for vessel in model._config["vessels"]:
        if "boundary_conditions" in vessel:
            branch_id = int(vessel["vessel_name"].split("_")[0][6:])
            for loc in vessel["boundary_conditions"]:
                branch_id_to_bc[branch_id] = vessel["boundary_conditions"][loc]

    hover_text = [
        branch_id_to_bc[branch_id] for branch_id in border_branch_ids
    ]

    plot = visualizer.Plot3D(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=10),
    )

    plot.add_mesh_trace_from_vtk(
        filename=project["3d_mesh"],
        color="darkred",
        name="3D Geometry",
        opacity=0.5,
    )
    plot.add_point_trace(
        x=border_points[:, 0],
        y=border_points[:, 1],
        z=border_points[:, 2],
        color="white",
        size=4,
        name="Boundary conditions",
        text=hover_text,
        showlegend=True,
    )

    return plot


def create_3d_geometry_plot_with_vessels(project, mapped_result):

    model = ZeroDModel(project)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(project["rom_centerline"])
    reader.Update()
    polydata = reader.GetOutput()

    points = vtk_to_numpy(polydata.GetPoints().GetData())
    cells = vtk_to_numpy(polydata.GetLines().GetData()).reshape(-1, 3)

    branch_ids = vtk_to_numpy(polydata.GetPointData().GetArray("BranchId"))

    border_point_indices = np.where(
        np.unique(cells[:, [1, 2]].flatten(), return_counts=True)[1] == 1
    )
    border_points = points[border_point_indices]
    border_branch_ids = branch_ids[border_point_indices]

    branch_id_to_bc = {}
    for vessel in model._config["vessels"]:
        if "boundary_conditions" in vessel:
            branch_id = int(vessel["vessel_name"].split("_")[0][6:])
            for loc in vessel["boundary_conditions"]:
                branch_id_to_bc[branch_id] = vessel["boundary_conditions"][loc]

    hover_text = [
        branch_id_to_bc[branch_id] for branch_id in border_branch_ids
    ]

    for branch_id, branch in mapped_result["branchdata"].items():

        for seg_id, segment in branch.items():
            middle = (segment["x0"] + segment["x1"]) / 2.0
            border_points = np.append(border_points, [middle], axis=0)
            hover_text.append(f"branch{branch_id}_seg{seg_id}")

    plot = visualizer.Plot3D(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=10),
    )

    plot.add_mesh_trace_from_vtk(
        filename=project["3d_mesh"],
        color="darkred",
        name="3D Geometry",
        opacity=0.5,
    )
    plot.add_point_trace(
        x=border_points[:, 0],
        y=border_points[:, 1],
        z=border_points[:, 2],
        color="white",
        size=2,
        name="Boundary conditions",
        text=hover_text,
        showlegend=True,
    )

    return plot
