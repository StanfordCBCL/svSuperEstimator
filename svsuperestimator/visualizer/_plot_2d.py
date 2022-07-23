"""This module holds various plotting classes."""
from __future__ import annotations
import os
import plotly.graph_objects as go
import vtk
import numpy as np

from ._plot_base import PlotBase


class Plot2D(PlotBase):
    def add_points(self):
        pass

    def add_points(
        self, x, y, color=None, size=4, opacity=1.0, colorscale="viridis"
    ):
        self._fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color=color,
                    size=size,
                    opacity=opacity,
                    colorscale=colorscale,
                ),
            )
        )

    def add_line(
        self, x, y, name=None, width=2, opacity=1.0, showlegend=False
    ):
        self._fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines",
                opacity=opacity,
                line=dict(width=width),
                showlegend=False,
            )
        )

    def add_bars(self, bin_edges, z, name=None, showlegend=False, opacity=1.0):
        self._layout_common.update(dict(bargap=0.0))
        self._fig.add_trace(
            go.Bar(
                x=bin_edges,
                y=z,
                name=name,
                showlegend=showlegend,
                opacity=opacity,
            )
        )

    def add_vline(
        self, x: float, text: str = None, line_width=3, color="orange"
    ):
        self._fig.add_vline(
            x=x,
            line_width=line_width,
            line_color=color,
            annotation_text="  " + text if text else None,
        )

    def add_line_with_confidence_interval(
        self, x, y, y_lower, y_upper, name: str = None
    ):
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

    def add_flag(self, x, y, text, color="orange", size=8):
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
            x=x, y=y, text=text, showarrow=True, arrowhead=1
        )

    def add_heatmap(self, bin_edges_x, bin_edges_y, z, colorscale="viridis"):
        self._fig.add_trace(
            go.Heatmap(
                x=bin_edges_x,
                y=bin_edges_y,
                z=z,
                colorscale=colorscale,
            )
        )

    def add_xy_bars(self, bin_edges_x, bin_edges_y, z_x, z_y):
        self._fig.add_trace(
            go.Bar(
                x=z_y,
                y=bin_edges_y,
                xaxis="x2",
                marker=dict(colorscale="viridis"),
                orientation="h",
            )
        )
        self._fig.add_trace(
            go.Bar(
                x=bin_edges_x,
                y=z_x,
                yaxis="y2",
                marker=dict(colorscale="viridis"),
            )
        )
        self._layout_common.update(
            dict(
                xaxis=dict(
                    zeroline=False,
                    domain=[0, 0.85],
                    showgrid=False,
                    range=[bin_edges_x.min(), bin_edges_x.max()],
                ),
                yaxis=dict(
                    zeroline=False,
                    domain=[0, 0.85],
                    showgrid=False,
                    range=[bin_edges_y.min(), bin_edges_y.max()],
                ),
                xaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
                yaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
                bargap=0.0,
                hovermode="closest",
                showlegend=False,
            )
        )
