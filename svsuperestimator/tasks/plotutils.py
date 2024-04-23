"""This module holds task helper function related to plotting."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .. import reader, visualizer


def create_3d_geometry_plot_with_vessels(
    project: reader.SimVascularProject,
    branch_data: Any,
    show_branches: bool = True,
) -> visualizer.Plot3D:
    """Create a 3D plot with mesh and 0D element traces.

    Args:
        project: The project.
        branch_data: branch_data with 0D element coordinates.
        show_branches: Toggle whether to show 0D element branches.
    """

    zerod_color = "cyan"

    label_points = []
    label_texts = []
    line_points = []

    for ele_name, ele_config in branch_data.items():
        if ele_name.startswith("branch") and show_branches:
            label_points.append((ele_config["x0"] + ele_config["x1"]) * 0.5)
            label_texts.append(ele_name)
            line_points += [ele_config["x0"], ele_config["x1"], [None] * 3]
        elif not ele_name.startswith("branch"):
            label_points.append(ele_config)
            label_texts.append(ele_name)

    label_points_np = np.array(label_points)
    line_points_np = np.array(line_points)

    plot = visualizer.Plot3D(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        width=750,
        height=750,
    )

    plot.add_mesh_trace_from_vtk(
        mesh_handler=project["mesh"],
        color="darkred",  # "#FF2014",
        name="3D Geometry",
        opacity=0.5,
        decimate=0.9,  # Reduce number of polygons by 90%
    )
    plot.add_point_trace(
        x=label_points_np[:, 0],
        y=label_points_np[:, 1],
        z=label_points_np[:, 2],
        color="white",
        name="",
        text=label_texts,
        textposition="middle left",
        marker_size=0.1,
        textfont_size=12,
        textfont_color=zerod_color,
    )
    if show_branches:
        plot.add_line_trace(
            x=line_points_np[:, 0],
            y=line_points_np[:, 1],
            z=line_points_np[:, 2],
            color=zerod_color,
            name="Elements",
            width=5,
            marker_size=5,
        )

    return plot


def joint_plot(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lims: list,
    output_path: str,
    color_map: str = "BuGn",
):
    """Create a joint grid plot for particles.

    Args:
        x: X-coordinates of the particles.
        y: Y-coordinates of the particles.
        weights: Weights of the particles.
        lims: X and Y limits.
        output_path: Target path for the figure.
        color_map: Color map.
    """
    plot = sns.JointGrid()

    # Create kernel density plot for joint plot
    sns.kdeplot(
        x=x,
        y=y,
        weights=weights,
        color="r",
        fill=True,
        cmap="BuGn",
        ax=plot.ax_joint,
        thresh=0.01,
    )

    # Get color for marginal plots
    cmap = plt.get_cmap(color_map)
    color = np.array(cmap(1000))[:-1]

    # Create marginal plots
    plot.ax_marg_x.hist(x=x, weights=weights, color=color, alpha=0.5)
    plot.ax_marg_y.hist(
        x=y, weights=weights, orientation="horizontal", color=color, alpha=0.5
    )

    # Set limits
    plot.ax_joint.set_xlim(lims[0])
    plot.ax_joint.set_ylim(lims[1])

    plt.savefig(output_path)
    plt.close()
