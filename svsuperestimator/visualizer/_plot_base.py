"""This module holds the PlotBase class."""
from __future__ import annotations
from typing import Any
import plotly.graph_objects as go
from dash.dcc import Graph
from dash import html
from base64 import b64encode


class PlotBase:
    """Base class for plotting classes.

    Defines common methods to handle plotly plots.
    """

    def __init__(self, static=False, **kwargs: dict[str, Any]) -> None:
        """Create a new PlotBase instance.

        Args:
            kwargs: Plotly layout options for the figure.
        """
        self._fig = go.Figure()
        self._static = static

        # Common layout options
        self._layout_common: dict[str, Any] = {
            "font_family": "Arial",
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
        self._layout_common.update(kwargs)

        # Layout options specific for the light color scheme
        self._layout_light = {
            "template": "plotly_white",
        }

        # Layout options specific for the dark color scheme
        self._layout_dark = {
            "template": "plotly_dark",
        }

    def add_footnote(self, text: str):
        """Add a footnote to the plot.

        Args:
            text: The text to display in the footnote.
        """
        self._fig.add_annotation(
            text=text,
            align="right",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.0,
            y=-0.3,
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

        if self._static:
            img_bytes = self._fig.to_image(format="svg")
            encoding = b64encode(img_bytes).decode()
            img_b64 = "data:image/svg+xml;base64," + encoding
            html_str = f'<img src="{img_b64}">'
        else:
            html_str = self._fig.to_html(
                include_plotlyjs="cdn",
                include_mathjax="cdn",
                full_html=False,
            )
        return html_str

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
        """Export the plot as a dash graph.

        Args:
            dark: Toggle dark mode.
            display_controls: Display plotly controls.
        """
        self._fig.update_layout(**self._layout_common)
        if dark:
            self._fig.update_layout(**self._layout_dark)
        else:
            self._fig.update_layout(**self._layout_light)
        config = {"displayModeBar": False} if not display_controls else {}
        return html.Div(
            Graph(
                figure=self._fig,
                config=config,
            )
        )
