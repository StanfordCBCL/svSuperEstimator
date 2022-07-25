"""This module holds the Plot2D class."""
from __future__ import annotations
from typing import Union
import plotly.graph_objects as go
import numpy as np

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
                line=dict(width=width, color=color),
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
        x: float,
        y: float,
        text: float,
        color: str = "orange",
        textcolor: str = None,
        size: int = 8,
    ):
        """Add an annotated point trace.

        Args:
            x: X-coordinate of the point.
            y: Y-coordinate of the point.
            text: Text to display next to the point.
            color: Color of the point.
            textcolor: Color of the text.
            size: Size of the point.
        """
        self._fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                name=text,
                marker=dict(color=color, size=size),
            )
        )
        self._fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=True,
            arrowhead=1,
            font=dict(color=textcolor),
        )

    def add_heatmap_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        name: str,
        colorscale: str = "viridis",
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
                orientation="h",
            )
        )
        self._fig.add_trace(
            go.Bar(
                x=x,
                y=z_x,
                name=name_x,
                yaxis="y2",
                marker=dict(color=color),
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
                xaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
                yaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
                bargap=0.0,
                hovermode="closest",
                showlegend=False,
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
