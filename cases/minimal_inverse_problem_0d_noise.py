"""Inverse problem in 0D optimization problem with Noise.

This case implements a 0D model optimization routine. It optimizes the
distal to proximal resistance ration (denoted k) at each outlet to match
the outlet pressure and flow. k is initialized with a values of 10.0.
"""
import time

import pandas as pd
from scipy import optimize

from svsuperestimator import io
from svsuperestimator import model as mdl
from svsuperestimator import solver as slv

import numpy as np

start = time.time()

SIMVASC_FOLDER = "tmpfiles/0069_0001"
K_START = [10.0, 10.0, 10.0, 10.0]
BOUNDS = [(1e-8, 20), (1e-8, 20), (1e-8, 20), (1e-8, 20)]

# Setup project, model, solver and webpage
project = io.SimVascularProject(SIMVASC_FOLDER)
model = mdl.MultiFidelityModel(project)
solver = slv.ZeroDSolver()
webpage = io.WebPage("svSuperEstimator")

# Plot 3d model
try:
    plot3d = io.Vtk3dPlot(
        project["3d_mesh"],
        title="0063_1001",
        width=500,
        height=500,
        color="darkred",
    )
    webpage.add_heading("Selected Model")
    webpage.add_plots([plot3d])
except FileNotFoundError:
    pass

bc_map = {}
to_drop = []
for branch_id, vessel_data in enumerate(model.zerodmodel._config["vessels"]):
    if "boundary_conditions" in vessel_data:
        for bc_type, bc_name in vessel_data["boundary_conditions"].items():
            bc_map[bc_name] = {"name": f"V{branch_id}"}
            if bc_type == "inlet":
                bc_map[bc_name]["flow"] = "flow_in"
                bc_map[bc_name]["pressure"] = "pressure_in"
            else:
                bc_map[bc_name]["flow"] = "flow_out"
                bc_map[bc_name]["pressure"] = "pressure_out"
    else:
        to_drop.append(f"V{branch_id}")

rename_map = {bc_map[key]["name"]: key for key in bc_map.keys()}


def format_result(result: pd.DataFrame):
    result = result.replace(rename_map)
    for i in to_drop:
        result = result[mean.name != i]
    result.flow_out *= 3.6  # Convert cm^3/s to l/h
    result.pressure_out *= 0.00075006156130264  # Convert g/(cm s^2) to mmHg
    return result


def get_result_plots(
    mean: pd.DataFrame, upper: pd.DataFrame, lower: pd.DataFrame
) -> tuple[io.LinePlot, io.LinePlot]:
    plot_mean = format_result(mean.copy())
    plot_upper = format_result(upper.copy())
    plot_lower = format_result(lower.copy())
    flow_plot = io.LinePlotWithUpperLower(
        plot_mean,
        plot_upper,
        plot_lower,
        x="time",
        y="flow_out",
        color="name",
        title="Flow over time",
        xlabel=r"$s$",
        ylabel=r"$\frac{l}{h}$",
        legend_title="BC Name",
    )
    pres_plot = io.LinePlotWithUpperLower(
        plot_mean,
        plot_upper,
        plot_lower,
        x="time",
        y="pressure_out",
        color="name",
        title="Pressure over time",
        xlabel=r"$s$",
        ylabel=r"$mmHg$",
        legend_title="BC Name",
    )
    return flow_plot, pres_plot


# Running one simulation to determine targets
target = solver.run_simulation(model.zerodmodel)

# Plot ground truth
# webpage.add_heading("Target")
# webpage.add_plots(
#     get_result_plots(target[target.name != bc_map["INFLOW"]["name"]])
# )

outlet_bcs = {
    name: bc
    for name, bc in model.zerodmodel.boundary_conditions.items()
    if name != "INFLOW"
}

q_target = np.array(
    [
        target[target.name == bc_map[bc]["name"]][bc_map[bc]["flow"]].mean()
        for bc in outlet_bcs
    ]
)

p_target = np.array(
    [
        target[target.name == bc_map[bc]["name"]][
            bc_map[bc]["pressure"]
        ].mean()
        for bc in outlet_bcs
    ]
)

k_opt = [
    bc.resistance_distal / bc.resistance_proximal for bc in outlet_bcs.values()
]  # Optimal distal to proximal resistance ratio for each outlet


r_i = [
    bc.resistance_distal + bc.resistance_proximal for bc in outlet_bcs.values()
]  # Internal resistance for each outlet


def set_boundary_conditions(k: list[float]):
    """Set boundary conditions based on distal to proximal resistance ratio."""
    for ki, r_ii, bc in zip(k, r_i, outlet_bcs.values()):
        bc.resistance_proximal = r_ii * 1 / (1.0 + ki)
        bc.resistance_distal = r_ii * ki / (1.0 + ki)


def objective_function(k, p_target_noisy, q_target_noisy):
    """Objective function for the optimization.

    Evaluates the sum of the offset for the input output pressure relation for
    each outlet.
    """

    set_boundary_conditions(k)
    result = solver.run_simulation(model.zerodmodel)

    offset = sum(
        [
            abs(
                (
                    p_target_noisy[i]
                    - (
                        result[result.name == bc_map[bc]["name"]][
                            bc_map[bc]["pressure"]
                        ].mean()
                    )
                )
                / p_target_noisy[i]
            )
            + abs(
                (
                    q_target_noisy[i]
                    - result[result.name == bc_map[bc]["name"]][
                        bc_map[bc]["flow"]
                    ].mean()
                )
                / q_target_noisy[i]
            )
            for i, bc in enumerate(outlet_bcs)
        ]
    )
    # print(offset)
    return offset


# Run optimization
frames = []
keys = []

np.random.seed(0)

for i in range(10):

    p_target_noisy = (
        p_target + np.random.randn(len(p_target)) * p_target / 20.0
    )
    q_target_noisy = (
        q_target + np.random.randn(len(q_target)) * q_target / 20.0
    )
    print("Pressure target: ", p_target_noisy)
    print("Flow target: ", q_target_noisy)

    optimized_k = optimize.minimize(
        fun=lambda l: objective_function(l, p_target_noisy, q_target_noisy),
        x0=K_START,
        method="Nelder-Mead",
        bounds=BOUNDS,
        tol=1e-2,
        options={"maxiter": 20},
    ).x

    set_boundary_conditions(optimized_k)
    optimized_result = solver.run_simulation(model.zerodmodel)
    optimized_result["noise_iteration"] = [i] * len(optimized_result)

    print("Optimized k: ", optimized_k)

    frames.append(optimized_result)
    keys.append(i)

mean = frames[0].copy()

for key in ["flow_in", "flow_out", "pressure_in", "pressure_out"]:
    mean[key] *= 0
    for frame in frames:
        mean[key] += frame[key]
    mean[key] /= len(frames)

std = frames[0].copy()

for key in ["flow_in", "flow_out", "pressure_in", "pressure_out"]:
    std[key] *= 0
    for frame in frames:
        std[key] += (frame[key] - mean[key]) * (frame[key] - mean[key])
    std[key] = (std[key] / len(frames)) ** 0.5

upper = frames[0].copy()
lower = frames[0].copy()

for key in ["flow_in", "flow_out", "pressure_in", "pressure_out"]:
    upper[key] = mean[key] + std[key]
    lower[key] = mean[key] - std[key]


for bc_name in bc_map:

    if bc_name != "INFLOW":

        webpage.add_heading("Optimized Result for " + bc_name)
        webpage.add_plots(
            get_result_plots(
                mean[mean.name == bc_map[bc_name]["name"]],
                upper[upper.name == bc_map[bc_name]["name"]],
                lower[lower.name == bc_map[bc_name]["name"]],
            )
        )

# Build webpage
webpage.build("./dashboard")
