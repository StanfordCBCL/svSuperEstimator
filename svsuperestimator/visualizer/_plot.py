"""This module holds various plotting classes."""
from __future__ import annotations
from typing import Any
from matplotlib.pyplot import legend

import pandas as pd
import os
import plotly.graph_objects as go
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from scipy.interpolate import griddata

from pqueens.utils.pdf_estimation import (
    estimate_bandwidth_for_kde,
    estimate_pdf,
)

from scipy import ndimage


class _PlotlyPlot:
    """Base class for plotting classes based on plotly.

    Defines common methods to handle plotly plots.
    """

    # Default layout for plots generated by plotly
    _DEFAULT_LAYOUT: dict[str, Any] = {
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "template": "plotly_dark",
    }

    def __init__(self, **kwargs: str) -> None:
        """Create a new _PlotlyPlot instance."""
        self._fig = go.Figure()
        self._layout: dict[str, Any] = self._DEFAULT_LAYOUT.copy()
        self._fig.update_layout(**self._layout)
        self.configure(**kwargs)  # type: ignore

    @property
    def fig(self):
        self._fig.update_layout(**self._layout)
        return self._fig

    def to_html(self) -> str:
        """Export plot as an html encoded string.

        Returns:
            html_fig: Plotly figure in html format.
        """
        self._fig.update_layout(**self._layout)
        return self._fig.to_html(
            include_plotlyjs="cdn", include_mathjax="cdn", full_html=False
        )

    def to_png(self, path: str) -> None:
        """Export plot as image file.

        Supported formats png, jpg, jpeg, webp, svg, pdf, eps.

        Args:
            path: Target path for image.
        """
        self._fig.update_layout(**self._layout)
        self._fig.update_layout(
            {
                "template": "plotly_white",
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
            }
        )
        self._fig.write_image(path)

    def configure(
        self,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        legend_title: str = None,
        width: int = None,
        height: int = None,
    ) -> None:
        """Set various configurations for the plot.

        Args:
            title: Figure title.
            xlabel: Label of the x-axis.
            ylabel: Label of the y-axis.
            legend_title: Title of the legend.
            width: Width of the figure.
            height: Height of the figure.
        """
        if title is not None:
            self._layout["title"] = title
        if xlabel is not None:
            self._layout["xaxis_title"] = xlabel
        if ylabel is not None:
            self._layout["yaxis_title"] = ylabel
        if legend_title is not None:
            self._layout["legend_title"] = legend_title
        if width is not None:
            self._layout["width"] = width
        if height is not None:
            self._layout["height"] = height


class LinePlot(_PlotlyPlot):
    """Line plot."""

    def __init__(
        self,
        x: str = None,
        y: str = None,
        name: str = None,
        **kwargs: str,
    ) -> None:
        """Create a new LinePlot instance.

        Args:

            name: Name of line.
        """
        super().__init__(**kwargs)
        self._fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines",
                line=dict(color="rgba(99, 110, 250, 0.5)"),
                showlegend=False,
            )
        )

    def add_trace(
        self,
        dataframe: pd.DataFrame,
        x: str = None,
        y: str = None,
        name=None,
    ):
        """Add a trace to the line plot.

        Args:
            dataframe: The dataframe to plot.
            x: Label of the dataframe to use for the x-axis.
            y: Label of the dataframe to use for the y-axis.
            name: Name of line.
        """
        self._fig.add_trace(
            go.Scatter(
                name=name,
                x=dataframe[x],
                y=dataframe[y],
                mode="lines",
                line=dict(color="rgba(99, 110, 250, 0.5)"),
                showlegend=False,
            )
        )


class LinePlotWithUpperLower(_PlotlyPlot):
    """Line plot with error bars between upper and lower."""

    def __init__(
        self,
        dataframe_mean: pd.DataFrame,
        dataframe_upper: pd.DataFrame,
        dataframe_lower: pd.DataFrame,
        x: str = None,
        y: str = None,
        **kwargs: str,
    ) -> None:
        """Create a new LinePlotWithUpperLower instance.

        Args:
            dataframe_mean: The dataframe for the center line.
            dataframe_upper: The dataframe for the upper line.
            dataframe_lower: The dataframe for the lower line.
            x: Label of the dataframe to use for the x-axis.
            y: Label of the dataframe to use for the y-axis.
        """
        super().__init__(**kwargs)

        # print(pio.templates["plotly_dark"])
        # colors= [#636efa, #EF553B, #00cc96, #ab63fa, #FFA15A,
        # #19d3f3, #FF6692, #B6E880, #FF97FF, #FECB52],

        self._fig.add_trace(
            go.Scatter(
                name="Mean",
                x=dataframe_mean[x],
                y=dataframe_mean[y],
                mode="lines",
            )
        )
        self._fig.add_trace(
            go.Scatter(
                name="Upper",
                x=dataframe_upper[x],
                y=dataframe_upper[y],
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            )
        )
        self._fig.add_trace(
            go.Scatter(
                name="Standard Deviation",
                hovertemplate="%{y}<extra>Lower</extra>",
                x=dataframe_lower[x],
                y=dataframe_lower[y],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(99, 110, 250, 0.25)",
                fill="tonexty",
            )
        )
        self._fig.update_layout(
            hovermode="x",
        )

    def add_trace(
        self,
        dataframe: pd.DataFrame,
        x: str = None,
        y: str = None,
        name=None,
    ):
        """Add a trace to the line plot.

        Args:
            dataframe: The dataframe to plot.
            x: Label of the dataframe to use for the x-axis.
            y: Label of the dataframe to use for the y-axis.
            name: Name of line.
        """
        self._fig.add_trace(
            go.Scatter(
                name=name,
                x=dataframe[x],
                y=dataframe[y],
                mode="lines",
                line=dict(color="#FFA15A", dash="dash"),
            )
        )


class ViolinPlot(_PlotlyPlot):
    """Violin plot."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        **kwargs: str,
    ) -> None:
        """Create a new ViolinPlot instance.

        Args:
            dataframe: The dataframe to plot.
            x: Label of the dataframe to use for the x-axis.
            y: Label of the dataframe to use for the y-axis.
            color: Label of the dataframe to use for the color.
        """
        super().__init__(**kwargs)
        for col in dataframe.columns:
            self._fig.add_trace(
                go.Violin(
                    y=dataframe[col],
                    name=col,
                    points="all",
                    jitter=0.05,
                    box_visible=True,
                    meanline_visible=True,
                )
            )

    def add_lines(
        self,
        x: list[str],
        y: list[float],
        name=None,
    ):
        """Add a horizontal line marker.

        Args:
            x: Position in x-direction.
            y: Position in y-direction.
            name: Name of line.
        """
        self._fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker_symbol=41,
                marker_line_color="#FFA15A",
                marker_color="#FFA15A",
                marker_line_width=2,
                marker_size=40,
                name=name,
            )
        )


class ParticlePlot3d(_PlotlyPlot):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        surface=True,
        marginals=False,
        ground_truth=None,
        **kwargs: str,
    ) -> None:
        """Create a new LinePlot instance.

        Args:
            dataframe: The dataframe to plot.
            x: Label of the dataframe to use for the x-axis.
            y: Label of the dataframe to use for the y-axis.
            name: Name of line.
        """
        super().__init__(**kwargs)

        # Make surface mesh from particles
        xi = np.linspace(x.min(), x.max(), 1000)
        yi = np.linspace(y.min(), y.max(), 1000)
        X, Y = np.meshgrid(xi, yi)

        Z = np.clip(
            griddata((x, y), z, (X, Y), method="linear"),
            a_min=0.0,
            a_max=None,
        )
        # Z = ndimage.gaussian_filter(Z, sigma=5.0)
        Z[Z == 0.0] = np.nan

        self._fig = go.Figure(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(size=3),
                opacity=1.0,
                mode="markers",
                showlegend=False,
            ),
        )

        if surface:
            self._fig.add_trace(
                go.Surface(
                    x=xi,
                    y=yi,
                    z=Z,
                    showlegend=False,
                    showscale=False,
                    opacity=0.7,
                    name="",
                    colorscale="viridis",
                    hoverinfo="skip",
                )
            )
        if ground_truth:
            self._fig.add_trace(
                go.Scatter3d(
                    x=[ground_truth[0], ground_truth[0]],
                    y=[ground_truth[1], ground_truth[1]],
                    z=[0, np.nanmax(Z) * 1.1],
                    marker=dict(
                        size=5, color="orange", symbol=["cross", "circle"]
                    ),
                    opacity=1.0,
                    mode="lines+markers+text",
                    text=[None, "Ground Truth"],
                    line=dict(width=5, color="orange"),
                    showlegend=False,
                ),
            )
        self._fig.update_layout(
            margin=dict(l=20, b=20, r=20),
            scene=dict(
                xaxis=dict(
                    showbackground=False, title=kwargs.get("xlabel", None)
                ),
                yaxis=dict(
                    showbackground=False, title=kwargs.get("ylabel", None)
                ),
                zaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                ),
                zaxis_visible=False,
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
        )


class DistPlot(_PlotlyPlot):
    def __init__(self, samples, ground_truth=None, **kwargs):
        super().__init__(**kwargs)

        bandwidth_x = estimate_bandwidth_for_kde(
            samples, np.amin(samples), np.amax(samples)
        )
        pdf_x, support_points = estimate_pdf(samples, bandwidth_x)

        self._fig.add_trace(
            go.Histogram(
                x=samples.ravel(),
                histnorm="probability density",
                showlegend=False,
                opacity=0.5,
                nbinsx=50,
                # marker=dict(colorscale="viridis"),
            )
        )
        self._fig.add_trace(
            go.Scatter(
                x=support_points.ravel(),
                y=pdf_x.ravel(),
                name="PDF",
                mode="lines",
                opacity=1,
                line=dict(width=2),
                showlegend=False,
            )
        )
        if ground_truth:
            self._fig.add_vline(
                x=ground_truth,
                line_width=3,
                # line_dash="dash",
                line_color="orange",
                annotation_text="  Ground Truth",
            )

        self._fig.add_annotation(
            text=f"Kernel: Gaussian | Optimized bandwith: {bandwidth_x:.3f}",
            align="right",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.0,
            y=-0.2,
            # bordercolor="black",
            # borderwidth=1,
        )


class HistogramContourPlot2D(_PlotlyPlot):
    def __init__(self, x: np.ndarray, y: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._fig = go.Figure(
            go.Histogram2dContour(
                x=x,
                y=y,
                xaxis="x",
                yaxis="y",
                colorscale="viridis",
                # nbinsx=50,
                # nbinsy=50,
            )
        )
        self._fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                xaxis="x",
                yaxis="y",
                mode="markers",
                marker=dict(color="whitesmoke", size=2, opacity=0.5),
            )
        )
        self._fig.add_trace(
            go.Histogram(
                y=y, xaxis="x2", marker=dict(colorscale="viridis"), nbinsy=50
            )
        )

        self._fig.add_trace(
            go.Histogram(
                x=x, yaxis="y2", marker=dict(colorscale="viridis"), nbinsx=50
            )
        )
        self._fig.update_layout(
            xaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
            yaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
            xaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
            yaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
            bargap=0,
            hovermode="closest",
            showlegend=False,
        )

    def add_dot(self, x, y):
        self._fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                name="Ground Truth",
                marker=dict(color="orange", size=8),
            )
        )
        self._fig.add_annotation(
            x=x, y=y, text="Ground Truth", showarrow=True, arrowhead=1
        )


class Vtk3dPlot(_PlotlyPlot):
    """3d plot from vtk file."""

    def __init__(
        self, filename: str, color: str = None, name=None, **kwargs: str
    ):
        """Create a new Vtk3dPlot instance.

        Args:
            filename: Name of the vtk file to plot.
            color: Color of the mesh.
        """
        super().__init__(**kwargs)

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Error plotting {filename}. The file does not exist."
            )

        camera = dict(eye=dict(x=2, y=2, z=0.1))

        self._layout.update({"hovermode": False, "scene_camera": camera})

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
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        i, j, k = cells[:, 1], cells[:, 2], cells[:, 3]

        mesh_args = {}
        if color is not None:
            mesh_args["color"] = color
        self._fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    showlegend=True,
                    opacity=0.5,
                    name=name,
                    **mesh_args,
                )
            ],
        )
        self._fig.update_scenes(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False
        )
        self._fig.update_layout(margin=dict(l=0, r=0, b=0, t=10))
