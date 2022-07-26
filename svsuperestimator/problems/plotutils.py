from arviz import kde
from plotly import graph_objects as go

from svsuperestimator.model import ZeroDModel
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from svsuperestimator import visualizer

from . import statutils


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


def create_kde_plot(
    x,
    weights,
    plotrange,
    ground_truth,
    param_name,
):

    lin_x, kde, bandwidth = statutils.kernel_density_estimation_1d(
        x=x, weights=weights, bounds=plotrange
    )

    counts, bin_edges = np.histogram(
        x,
        bins=int((plotrange[1] - plotrange[0]) / bandwidth),
        weights=weights,
        density=True,
        range=plotrange,
    )

    # Create kernel density estimation plot for k0
    plot_posterior_2d = visualizer.Plot2D(
        title="Weighted histogram and kernel density estimation of "
        + param_name,
        xaxis_title=param_name,
        yaxis_title="Kernel density",
        xaxis_range=plotrange,
    )
    plot_posterior_2d.add_bar_trace(
        x=bin_edges,
        y=counts,
        name="Weighted histogram",
    )
    plot_posterior_2d.add_line_trace(
        x=lin_x, y=kde, name="Kernel density estimate"
    )
    plot_posterior_2d.add_vline_trace(x=ground_truth, text="Ground Truth")
    plot_posterior_2d.add_footnote(
        text=f"Kernel: Gaussian | Optimized Bandwith: {bandwidth:.3f} | Method: 30-fold cross-validation"
    )

    return plot_posterior_2d
