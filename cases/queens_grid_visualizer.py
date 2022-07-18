import pickle
import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import plotly.graph_objects as go


input_folder = "/Users/stanford/queens/my_case/minimal_grid/output"

from svsuperestimator import io


with open(
    os.path.join(input_folder, "grid_RCR_bloodvessel.pickle"), "rb"
) as ff:
    processed_results = pickle.load(ff)

input_data = processed_results["input_data"]
values = processed_results["raw_output_data"]["mean"]
x = input_data[:, 0]
y = input_data[:, 1]
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)

Z = griddata((x, y), np.squeeze(values), (X, Y), method="linear", rescale=True)

fig = go.Figure(
    go.Surface(
        x=xi,
        y=yi,
        z=Z,
        # showlegend=False,
        # showscale=False,
        name="",
    )
)
fig.update_layout(
    margin=dict(l=20, b=20, r=20),
    scene=dict(
        xaxis=dict(title="Rp"),
        yaxis=dict(title="Rd"),
        zaxis=dict(title="pressure_out"),
    ),
)
fig.update_layout({"xaxis_title": "$R_p$", "yaxis_title": "$R_d$"})

webpage = io.WebPage("svSuperEstimator")
plot = io._plot._PlotlyPlot()
plot._fig = fig
webpage.add_plots([plot])
webpage.build("./dashboard")
