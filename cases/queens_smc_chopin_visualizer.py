import pickle
import os
import numpy as np
import pandas as pd


input_folder = "/Users/stanford/queens/my_case/0069_0001/output"

from svsuperestimator import io


with open(
    os.path.join(input_folder, "smc_RCR_bloodvessel.pickle"), "rb"
) as ff:
    processed_results = pickle.load(ff)

mean = processed_results["mean"]
var = processed_results["var"]
raw_output_data = processed_results["raw_output_data"]

particles = raw_output_data["particles"]
weights = raw_output_data["weights"]
log_posterior = raw_output_data["log_posterior"]
mean = raw_output_data["mean"]
var = raw_output_data["var"]

x = particles[:, 0]
y = particles[:, 1]
z = np.exp(log_posterior - log_posterior.max())
z = z / np.mean(z)

webpage = io.WebPage("svSuperEstimator")
# particle_plot3d = io.ParticlePlot3d(
#     x,
#     y,
#     z,
#     title="Bivariate and marginal posterior probabilities",
#     xlabel=r"Rp",
#     ylabel=r"Rd",
# )
# webpage.add_heading("Selected Model")

ground_truth = [
    {
        "bc_name": "RCR_0",
        "bc_type": "RCR",
        "bc_values": {"C": 2.87e-05, "Pd": 0.0, "Rd": 75427.0, "Rp": 6559.0},
    },
    {
        "bc_name": "RCR_1",
        "bc_type": "RCR",
        "bc_values": {
            "C": 0.00013459000000000002,
            "Pd": 0.0,
            "Rd": 12846.0,
            "Rp": 1427.0,
        },
    },
    {
        "bc_name": "RCR_2",
        "bc_type": "RCR",
        "bc_values": {
            "C": 1.263e-05,
            "Pd": 0.0,
            "Rd": 175097.99999999997,
            "Rp": 11176.0,
        },
    },
    {
        "bc_name": "RCR_3",
        "bc_type": "RCR",
        "bc_values": {
            "C": 1.607e-05,
            "Pd": 0.0,
            "Rd": 131794.0,
            "Rp": 14644.0,
        },
    },
]


for i, label in enumerate([1, 4, 5, 6]):

    prox_plot = io.ViolinPlot(
        pd.DataFrame(
            particles[:, i * 2],
            columns=[
                "$R_{p," + str(label) + "}$",
            ],
        ),
        title="Optimized proximal resistance",
        ylabel="",
    )
    prox_gt = ground_truth[i]["bc_values"]["Rp"]
    prox_plot.add_lines(
        ["$R_{p," + str(label) + "}$"], [prox_gt], name="Ground Truth"
    )
    dist_plot = io.ViolinPlot(
        pd.DataFrame(
            particles[:, i * 2 + 1],
            columns=[
                "$R_{d," + str(label) + "}$",
            ],
        ),
        title="Optimized distal resistance",
        ylabel="",
    )
    dist_gt = ground_truth[i]["bc_values"]["Rd"]
    dist_plot.add_lines(
        ["$R_{d," + str(label) + "}$"], [dist_gt], name="Ground Truth"
    )
    webpage.add_heading(f"Boundary condition RCR{label}")
    webpage.add_plots([prox_plot, dist_plot])

webpage.build("./dashboard")
