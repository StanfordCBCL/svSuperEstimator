"""This module holds various plotting classes."""
from __future__ import annotations
from typing import Union
import os
import plotly.graph_objects as go
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

from ._plot_base import PlotBase


class Plot3D(PlotBase):
    """3D plot.

    Contains functions to construct a 3D plot from different traces.
    """

    def add_point_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        name: str,
        color: Union[str, np.ndarray] = None,
        size: int = 3,
        opacity: float = 1.0,
        colorscale: str = "viridis",
        text: str = None,
        showlegend: bool = False,
    ):
        """Add a point scatter trace.

        Args:
            x: X-coordinates of the points.
            y: Y-coordinates of the points.
            z: Z-coordinates of the points.
            name: Name of the trace.
            color: Color of the points as a color specifier or an array to be
                used for color coding.
            size: Size of the points.
            opacity: Opacity of the points.
            colorscale: Colorscale to be used for color cording.
            text: Optional text to be displayed next to the points.
            showlegend: Toggle display of trace in legend.
        """
        mode = "markers"
        if text is not None:
            mode = "markers+text"
        self._fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(size=size, color=color, colorscale=colorscale),
                opacity=opacity,
                mode=mode,
                showlegend=showlegend,
                name=name,
                text=text,
            ),
        )

    def add_surface_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        name: str,
        colorscale: str = "viridis",
        opacity: float = 1.0,
        showlegend: bool = False,
    ):
        """Add a surface trace.

        Args:
            x: X-coordinates of the points.
            y: Y-coordinates of the points.
            z: Z-coordinates of the points.
            name: Name of the trace.
            colorscale: Colorscale to be used for color cording.
            opacity: Opacity of the points.
            showlegend: Toggle display of trace in legend.
        """
        self._fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                showlegend=showlegend,
                showscale=False,
                opacity=opacity,
                name=name,
                colorscale=colorscale,
            )
        )

    def add_mesh_trace_from_vtk(
        self,
        filename: str,
        name: str,
        color: str = None,
        opacity: float = 1.0,
        showlegend: bool = False,
    ):
        """Add a mesh trace from a vtk file.

        Args:
            filename: Path to the vtk file.
            name: Name of the trace.
            color: Color of the trace.
            opacity: Opacity of the trace.
            showlegend: Toggle display of trace in legend.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Error plotting {filename}. The file does not exist."
            )

        # Setup vtk reader
        if filename.endswith(".vtp"):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(filename)
            reader.Update()
            polydata = reader.GetOutput()
        else:
            raise NotImplementedError("Filetype not supported.")

        # Extract mesh
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        cells = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)

        self._fig.add_trace(
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=cells[:, 1],
                j=cells[:, 2],
                k=cells[:, 3],
                showlegend=showlegend,
                opacity=opacity,
                name=name,
                color=color,
                hoverinfo="skip",
            )
        )

    def add_annotated_point_trace(
        self,
        x: float,
        y: float,
        z: float,
        text=str,
        color: str = "orange",
        size: int = 5,
        opacity: float = 1.0,
        showlegend: bool = False,
    ):
        """Add an annotated point trace.

        Args:
            x: X-coordinate of the point.
            y: Y-coordinate of the point.
            z: Z-coordinate of the point.
            text: Text to display next to the point.
            color: Color of the point.
            size: Size of the point.
            opacity: Opacity of the point.
            showlegend: Toggle display of trace in legend.
        """
        self._fig.add_trace(
            go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[0, z],
                marker=dict(
                    size=size, color=color, symbol=["cross", "circle"]
                ),
                opacity=opacity,
                mode="lines+markers+text",
                text=[None, text],
                name=text,
                line=dict(width=size, color=color),
                showlegend=showlegend,
            ),
        )