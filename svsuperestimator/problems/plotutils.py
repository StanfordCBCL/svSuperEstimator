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
    x, weights, ground_truth, param_name, num_points=1000, bw_method="scott"
):

    lin_x, kde, bandwidth = statutils.kernel_density_estimation_1d(
        x=x, weights=weights, num_points=num_points, bw_method=bw_method
    )

    counts, bin_edges = np.histogram(
        x,
        bins=50,
        weights=weights,
        density=True,
    )

    # Create kernel density estimation plot for k0
    plot_posterior_2d = visualizer.Plot2D(
        title="Weighted histogram and kernel density estimation of "
        + param_name,
        xaxis_title=param_name,
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
        text=f"Kernel: Gaussian | Optimized Bandwith: {bandwidth:.3f} | Method: {bw_method}"
    )

    return plot_posterior_2d


def create_2d_heatmap_with_marginals(
    x, y, weights, ground_truth, xparam_name, yparam_name
):

    plotrange = [[np.amin(x), np.amax(x)], [np.amin(y), np.amax(y)]]
    heatmap_plot = visualizer.Plot2D(
        title="Weighted particle density heatmap",
        xaxis_title=xparam_name,
        yaxis_title=yparam_name,
        width=750,
        height=750,
        xaxis_range=plotrange[0],
        yaxis_range=plotrange[1],
    )
    counts, xedges, yedges = np.histogram2d(
        x,
        y,
        weights=weights,
        density=True,
        bins=50,
        range=plotrange,
    )

    heatmap_plot.add_heatmap_trace(
        x=xedges, y=yedges, z=counts.T, name="Weighted particle density"
    )
    heatmap_plot.add_point_trace(x=x, y=y, color=weights, name="Particles")

    counts_x, bin_edges_x = np.histogram(
        x, bins=50, weights=weights, density=True, range=plotrange[0]
    )
    counts_y, bin_edges_y = np.histogram(
        y, bins=50, weights=weights, density=True, range=plotrange[1]
    )

    heatmap_plot.add_xy_bar_trace(
        x=bin_edges_x,
        y=bin_edges_y,
        z_x=counts_x,
        z_y=counts_y,
        name_x="Weighted histogram of " + xparam_name,
        name_y="Weighted histogram of " + yparam_name,
    )

    heatmap_plot.add_annotated_point_trace(
        x=ground_truth[0], y=ground_truth[1], text="Ground Truth"
    )

    return heatmap_plot
