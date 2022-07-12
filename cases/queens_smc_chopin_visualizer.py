import pickle
import os
import numpy as np
import pandas as pd


input_folder = "/Users/stanford/queens/my_case/minimal_smc/output"

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
z = np.exp(log_posterior)
z = z / np.mean(z)

webpage = io.WebPage("svSuperEstimator")
particle_plot3d = io.ParticlePlot3d(
    x,
    y,
    z,
    title="Bivariate and marginal posterior probabilities",
    xlabel=r"Rp",
    ylabel=r"Rd",
)
webpage.add_heading("Selected Model")


particle_plot2d = io.ViolinPlot(
    pd.DataFrame(particles, columns=["$R_p$", "$R_d$"]),
    title="Optimized proximal and distal resistance",
    ylabel="",
)

webpage.add_plots([particle_plot3d, particle_plot2d])

webpage.build("./dashboard")
