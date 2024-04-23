"""This module holds various plotting classes."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from ..reader import MeshHandler
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
        color: Optional[Union[str, np.ndarray]] = None,
        size: int = 3,
        opacity: float = 1.0,
        colorscale: str = "viridis",
        text: Optional[Union[Sequence, np.ndarray]] = None,
        showlegend: bool = False,
        **kwargs: Any,
    ) -> None:
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
                marker=dict(
                    size=size,
                    color=color,
                    colorscale=colorscale,
                ),
                opacity=opacity,
                mode=mode,
                showlegend=showlegend,
                name=name,
                text=text,
                **kwargs,
            ),
        )

    def add_line_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        name: str,
        color: Optional[Union[str, np.ndarray]] = None,
        width: int = 3,
        opacity: float = 1.0,
        colorscale: str = "viridis",
        showlegend: bool = False,
        **kwargs: Any,
    ) -> None:
        """Add a line scatter trace.

        Args:
            x: X-coordinates of the points.
            y: Y-coordinates of the points.
            z: Z-coordinates of the points.
            name: Name of the trace.
            color: Color of the lines as a color specifier or an array to be
                used for color coding.
            width: Size of the points.
            opacity: Opacity of the points.
            colorscale: Colorscale to be used for color cording.
            showlegend: Toggle display of trace in legend.
        """
        self._fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                line=dict(
                    width=width,
                    color=color,
                    colorscale=colorscale,
                ),
                opacity=opacity,
                mode="markers+lines",
                showlegend=showlegend,
                name=name,
                **kwargs,
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
    ) -> None:
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
        mesh_handler: MeshHandler,
        name: str,
        color: Optional[str] = None,
        opacity: float = 1.0,
        showlegend: bool = False,
        decimate: Optional[float] = None,
    ) -> None:
        """Add a mesh trace from a vtk file.

        Args:
            mesh_handler: Mesh hanlder.
            name: Name of the trace.
            color: Color of the trace.
            opacity: Opacity of the trace.
            showlegend: Toggle display of trace in legend.
            decimate: Complexity reduction factor for mesh.
        """

        # Simplify mesh
        if decimate:
            mesh_handler = mesh_handler.decimate(0.9)  # type: ignore

        # Extract mesh
        points = mesh_handler.points
        cells = mesh_handler.polys

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
        text: str,
        color: str = "orange",
        size: int = 5,
        opacity: float = 1.0,
        showlegend: bool = False,
    ) -> None:
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
