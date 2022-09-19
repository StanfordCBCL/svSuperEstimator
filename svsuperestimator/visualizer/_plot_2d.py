"""This module holds the Plot2D class."""
from __future__ import annotations

from typing import Union

import numpy as np
import plotly.graph_objects as go

from ._plot_base import PlotBase


class Plot2D(PlotBase):
    """2D plot.

    Contains functions to construct a 2D plot from different traces.
    """

    def add_point_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str,
        color: Union[str, np.ndarray] = None,
        size: int = 4,
        opacity: float = 1.0,
        colorscale: str = "viridis",
        showlegend: bool = False,
        **kwargs,
    ):
        """Add a point scatter trace.

        Args:
            x: X-coordinates of the points.
            y: Y-coordinates of the points.
            name: Name of the trace.
            color: Color of the points as a color specifier or an array to be
                used for color coding.
            size: Size of the points.
            opacity: Opacity of the points.
            colorscale: Colorscale to be used for color cording.
            showlegend: Toggle display of trace in legend.
        """
        self._fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="markers",
                marker=dict(
                    color=color,
                    size=size,
                    opacity=opacity,
                    colorscale=colorscale,
                    **kwargs,
                ),
                showlegend=showlegend,
            )
        )

    def add_line_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str,
        color: str = None,
        width: int = 2,
        opacity: float = 1.0,
        showlegend: bool = False,
        dash: str = None,
    ):
        """Add a line scatter trace.

        Args:
            x: X-coordinates of the line points.
            y: Y-coordinates of the line points.
            name: Name of the trace.
            color: Color of the line.
            width: Width of the line.
            opacity: Opacity of the line.
            showlegend: Toggle display of trace in legend.
        """
        self._fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines",
                opacity=opacity,
                line=dict(width=width, color=color, dash=dash),
                showlegend=showlegend,
            )
        )

    def add_bar_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str,
        color: str = None,
        opacity: float = 1.0,
        showlegend: bool = False,
    ):
        """Add a bar trace.

        Args:
            x: X-coordinates of the bars.
            y: Y-coordinates of the bars.
            name: Name of the trace.
            color: Color of the bars.
            opacity: Opacity of the line.
            showlegend: Toggle display of trace in legend.
        """
        self._layout_common.update(dict(bargap=0.0))
        self._fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=name,
                showlegend=showlegend,
                opacity=opacity,
                marker_color=color,
                marker_line=dict(width=1, color=color),
            )
        )

    def add_vline_trace(
        self, x: float, text: str = None, width: int = 3, color: str = "orange"
    ):
        """Add vertical line marker trace.

        Args:
            x: X-coordinate of the line.
            text: Text to display next to the line.
            width: Width of the line.
            color: Color of the line.
        """
        self._fig.add_vline(
            x=x,
            line_width=width,
            line_color=color,
            annotation_text="  " + text if text else None,
        )

    def add_annotated_point_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: list[str],
        symbol: list[str],
        color: str = "orange",
        size: int = 8,
    ):
        """Add an annotated point trace.

        Args:
            x: X-coordinate of the point.
            y: Y-coordinate of the point.
            Name: Names of points to display in legend.
            color: Colors of the points.
            symbol: Symbol of the points.
            size: Size of the point.
        """
        for i, t in enumerate(name):
            self._fig.add_trace(
                go.Scatter(
                    x=[x[i]],
                    y=[y[i]],
                    mode="markers",
                    name=name[i],
                    marker=dict(color=color[i], size=size, symbol=symbol[i]),
                    showlegend=True,
                )
            )

    def add_heatmap_trace(
        self,
        z: np.ndarray,
        name: str,
        x: np.ndarray = None,
        y: np.ndarray = None,
        colorscale: str = "viridis",
        showscale=False,
    ):
        """Add a heatmap trace.

        Args:
            x: X-coordinates of the heatmap points.
            y: Y-coordinates of the heatmap points.
            colorscale: Colorscale of the heatmap.
        """
        self._fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                name=name,
                colorscale=colorscale,
                showscale=showscale,
            )
        )

    def add_xy_bar_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z_x: np.ndarray,
        z_y: np.ndarray,
        name_x: str,
        name_y: str,
        color: str = None,
    ):
        """Add bar traces to the borders of the x and y axis.

        Args:
            x: X-coordinates of the bars.
            y: Y-coordinates of the bars.
            z_x: Z-coordinates of the x-bars.
            z_y: Z-coordinates of the y-bars.
            name_x: Name of the x bar trace.
            name_y: Name of the y bar trace.
            color: Color of the bars.
        """
        self._fig.add_trace(
            go.Bar(
                x=z_y,
                y=y,
                name=name_y,
                xaxis="x2",
                marker=dict(color=color),
                marker_line=dict(width=1, color="black"),
                orientation="h",
                showlegend=False,
            )
        )
        self._fig.add_trace(
            go.Bar(
                x=x,
                y=z_x,
                name=name_x,
                yaxis="y2",
                marker=dict(color=color),
                marker_line=dict(width=1, color="black"),
                showlegend=False,
            )
        )
        self._layout_common.update(
            dict(
                xaxis=dict(
                    zeroline=False,
                    domain=[0, 0.85],
                    showgrid=False,
                ),
                yaxis=dict(
                    zeroline=False,
                    domain=[0, 0.85],
                    showgrid=False,
                ),
                xaxis2=dict(
                    visible=False,
                    zeroline=False,
                    domain=[0.85, 1],
                    showgrid=False,
                ),
                yaxis2=dict(
                    visible=False,
                    zeroline=False,
                    domain=[0.85, 1],
                    showgrid=False,
                ),
                bargap=0,
                hovermode="closest",
            )
        )

    def add_line_with_confidence_interval_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        name: str,
    ):
        """Add a line scatter trace with considence interval.

        Args:
            x: X-coordinates of the line points.
            y: Y-coordinates of the line points.
            y_lower: Y-coordinates of the lower line points.
            y_upper: Y-coordinates of the upper line points.
            name: Name of the trace.
        """
        self._fig.add_trace(
            go.Scatter(
                name=name,
                x=x,
                y=y,
                mode="lines",
            )
        )
        self._fig.add_trace(
            go.Scatter(
                name=name,
                x=x,
                y=y_upper,
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            )
        )
        self._fig.add_trace(
            go.Scatter(
                name=name,
                x=x,
                y=y_lower,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(99, 110, 250, 0.25)",
                fill="tonexty",
            )
        )
        self._layout_common.update(dict(hovermode="x"))

    def add_parallel_coordinates_plots(
        self,
        values: list[np.ndarray],
        names: list[str] = None,
        color_by: np.ndarray = None,
        plotrange: tuple[float, float] = None,
    ):
        """Add a parallel coordinates trace to the plot.

        Args:
            values: List of arrays of values for each dimension.
            names: The names of the dimensions.
            color_by: An array to color the lines by.
            plotrange: Fixed range for each dimension.
        """

        self._fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=color_by,
                ),
                dimensions=list(
                    [
                        dict(label=name, values=value, range=plotrange)
                        for value, name in zip(values, names)
                    ]
                ),
            )
        )
