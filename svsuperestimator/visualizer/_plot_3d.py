"""This module holds various plotting classes."""
from __future__ import annotations
import os
import plotly.graph_objects as go
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

from ._plot_base import PlotBase


class Plot3D(PlotBase):
    def add_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        name: str = None,
        color: str = None,
        size=3,
        opacity=1.0,
        text=None,
        showlegend=False,
    ):
        mode = "markers"
        if text is not None:
            mode = "markers+text"
        self._fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(size=size, color=color),
                opacity=opacity,
                mode=mode,
                showlegend=showlegend,
                name=name,
                text=text,
            ),
        )

    def add_surface(
        self,
        x,
        y,
        z,
        name: str = None,
        colorscale: str = "viridis",
        opacity=1.0,
        showlegend=False,
    ):

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

    def add_mesh_from_vtk(
        self, filename: str, name=None, color: str = None, opacity=1.0
    ):

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Error plotting {filename}. The file does not exist."
            )

        # self._layout.update(
        #     {
        #         "hovermode": False,
        #         # "scene_camera": dict(eye=dict(x=2, y=2, z=0.1)),
        #     }
        # )

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
                showlegend=True,
                opacity=opacity,
                name=name,
                color=color,
                hoverinfo="skip",
            )
        )

    def add_flag(
        self,
        x: float,
        y: float,
        z: float,
        text=str,
        color="orange",
        size=5,
        opacity=1.0,
        showlegend=False,
    ):
        """Add a flag to the 3D plot.

        A flag is a dot at coordinate (x,y,z) connected to the z=0 plane with a
        vertical line. The text is displayed next to the dot.

        Args:
            x: X-coordinate of the flag.
            y: Y-coordinate of the flag.
            z: Z-coordinate of the flag.
            text: Text to display next to the flag.
            color: Color of the flag.
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
