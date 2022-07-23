"""This module holds various plotting classes."""
from __future__ import annotations
from typing import Any

import pandas as pd
import os
import plotly.graph_objects as go
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from dash.dcc import Graph


class PlotBase:
    """Base class for plotting classes.

    Defines common methods to handle plotly plots.
    """

    def __init__(self, **kwargs: str) -> None:
        """Create a new _PlotlyPlot instance."""
        self._fig = go.Figure()
        self._layout_common: dict[str, Any] = kwargs
        self._layout_light = {
            "template": "plotly_white",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        }
        self._layout_dark = {
            "template": "plotly_dark",
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }

    def add_footnode(self, text):
        self._fig.add_annotation(
            text=text,
            align="right",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.0,
            y=-0.2,
        )

    def to_html(self, dark: bool = True) -> str:
        """Export plot as an html encoded string.

        Args:
            dark: Toggle dark mode.

        Returns:
            html_fig: Plotly figure in static html format.
        """
        self._fig.update_layout(**self._layout_common)
        if dark:
            self._fig.update_layout(**self._layout_dark)
        else:
            self._fig.update_layout(**self._layout_light)
        return self._fig.to_html(
            include_plotlyjs="cdn", include_mathjax="cdn", full_html=False
        )

    def to_image(self, path: str, dark: bool = False) -> None:
        """Export plot as image file.

        Supported formats png, jpg, jpeg, webp, svg, pdf, eps.

        Args:
            path: Target path for image.
            dark: Toggle dark mode.
        """
        self._fig.update_layout(**self._layout_common)
        if dark:
            self._fig.update_layout(**self._layout_dark)
        else:
            self._fig.update_layout(**self._layout_light)
        self._fig.write_image(path)

    def to_dash(self, dark: bool = True, display_controls=True):
        self._fig.update_layout(**self._layout_common)
        if dark:
            self._fig.update_layout(**self._layout_dark)
        else:
            self._fig.update_layout(**self._layout_light)
        config = {"displayModeBar": False} if not display_controls else {}
        return Graph(figure=self._fig, config=config, style={"width": "100%"})
