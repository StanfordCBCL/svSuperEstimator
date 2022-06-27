"""Inverse problem in 0D optimization problem.

This case implements a 0D model optimization routine. It optimizes the
distal to proximal resistance ration (denoted k) at each outlet to match
the inlet-outlet pressure difference. k is initialized with a values of 10.0.
"""
import time

import pandas as pd
from scipy import optimize

from svsuperestimator import io
from svsuperestimator import model as mdl
from svsuperestimator import solver as slv

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


def get_result_plots(result: pd.DataFrame) -> tuple[io.LinePlot, io.LinePlot]:
    """Create plots for flow and pressure result.

    Args:
        result: Result dataframe.

    Returns:
        flow_plot: Line plot for the flow over time.
        pres_plot: Line plot for pressure over time.
    """
    result = result[result.name != bc_map["INFLOW"]["name"]]
    result = result.replace(rename_map)
    for i in to_drop:
        result = result[result.name != i]
    plot_result = result.copy()
    plot_result.flow_out *= 3.6  # Convert cm^3/s to l/h
    plot_result.pressure_out *= (
        0.00075006156130264  # Convert g/(cm s^2) to mmHg
    )
    flow_plot = io.LinePlot(
        plot_result,
        x="time",
        y="flow_out",
        color="name",
        title="Flow over time",
        xlabel=r"$s$",
        ylabel=r"$\frac{l}{h}$",
        legend_title="BC Name",
    )
    pres_plot = io.LinePlot(
        plot_result,
        x="time",
        y="pressure_out",
        color="name",
        title="Pressure over time",
        xlabel=r"$s$",
        ylabel=r"$mmHg$",
        legend_title="BC Name",
    )
    return flow_plot, pres_plot


# Running one simulation to determine ground truth
ground_truth = solver.run_simulation(model.zerodmodel)
p_avg_inlet_gt = ground_truth.query(f"name=='{bc_map['INFLOW']['name']}'")[
    bc_map["INFLOW"]["pressure"]
].mean()
q_avg_inlet_gt = ground_truth.query(f"name=='{bc_map['INFLOW']['name']}'")[
    bc_map["INFLOW"]["flow"]
].mean()
outlet_bcs = {
    name: bc
    for name, bc in model.zerodmodel.boundary_conditions.items()
    if name != "INFLOW"
}
r_i = [
    bc.resistance_distal + bc.resistance_proximal for bc in outlet_bcs.values()
]  # Internal resistance for each outlet
p_diff_io_gt = [
    p_avg_inlet_gt
    - ground_truth[ground_truth.name == bc_map[bc]["name"]][
        bc_map[bc]["pressure"]
    ].mean()
    for bc in outlet_bcs
]  # Pressure difference between inlet and each outlet
q_avg_outlet_gt = [
    ground_truth[ground_truth.name == bc_map[bc]["name"]][
        bc_map[bc]["flow"]
    ].mean()
    for bc in outlet_bcs
]
k_opt = [
    bc.resistance_distal / bc.resistance_proximal for bc in outlet_bcs.values()
]  # Optimal distal to proximal resistance ratio for each outlet

# Plot ground truth
webpage.add_heading("Ground Truth")
webpage.add_plots(
    get_result_plots(
        ground_truth[ground_truth.name != bc_map["INFLOW"]["name"]]
    )
)


def set_boundary_conditions(k: list[float]):
    """Set boundary conditions based on distal to proximal resistance ratio."""
    for ki, r_ii, bc in zip(k, r_i, outlet_bcs.values()):
        bc.resistance_proximal = r_ii * 1 / (1.0 + ki)
        bc.resistance_distal = r_ii * ki / (1.0 + ki)


opt_progress = {"offset": []}


def objective_function(k):
    """Objective function for the optimization.

    Evaluates the sum of the offset for the input output pressure relation for
    each outlet.
    """

    set_boundary_conditions(k)
    result = solver.run_simulation(model.zerodmodel)

    offset = 0.0
    p_avg_inlet = result.query(f"name=='{bc_map['INFLOW']['name']}'")[
        bc_map["INFLOW"]["pressure"]
    ].mean()
    offset = sum(
        [
            abs(
                p_diff_io_gt[i]
                - (
                    p_avg_inlet
                    - result[result.name == bc_map[bc]["name"]][
                        bc_map[bc]["pressure"]
                    ].mean()
                )
            )
            / p_avg_inlet_gt
            + abs(
                q_avg_outlet_gt[i]
                - result[result.name == bc_map[bc]["name"]][
                    bc_map[bc]["flow"]
                ].mean()
            )
            / q_avg_inlet_gt
            for i, bc in enumerate(outlet_bcs)
        ]
    )
    opt_progress["offset"].append(offset)
    print(offset)
    return offset


# Run optimization
optimized_k = optimize.minimize(
    objective_function,
    K_START,
    method="Nelder-Mead",
    bounds=BOUNDS,
).x

# Get result for optimized k
set_boundary_conditions(optimized_k)
optimized_result = solver.run_simulation(model.zerodmodel)

# Plot optimized results
webpage.add_heading("Optimized Result")
webpage.add_plots(
    get_result_plots(
        optimized_result[optimized_result.name != bc_map["INFLOW"]["name"]]
    )
)

# Plot optimization error
webpage.add_heading("Optimization Error")
diff = (
    optimized_result[["flow_in", "pressure_in", "flow_out", "pressure_out"]]
    - ground_truth[["flow_in", "pressure_in", "flow_out", "pressure_out"]]
)
diff["name"], diff["time"] = optimized_result.name, optimized_result.time
for bc_name in outlet_bcs:
    diff.loc[
        diff.name == bc_map[bc_name]["name"], bc_map[bc_name]["flow"]
    ] *= 100.0 / (
        ground_truth[ground_truth.name == bc_map[bc_name]["name"]][
            bc_map[bc_name]["flow"]
        ].max()
        - ground_truth[ground_truth.name == bc_map[bc_name]["name"]][
            bc_map[bc_name]["flow"]
        ].min()
    )
    diff.loc[
        diff.name == bc_map[bc_name]["name"], bc_map[bc_name]["pressure"]
    ] *= (
        100.0
        / ground_truth[ground_truth.name == bc_map[bc_name]["name"]][
            bc_map[bc_name]["pressure"]
        ].mean()
    )
flow_plot, pres_plot = get_result_plots(
    diff[diff.name != bc_map["INFLOW"]["name"]]
)
flow_plot.configure(title="Flow error over time", xlabel="$s$", ylabel=r"$\%$")
pres_plot.configure(
    title="Pressure error over time", xlabel="$s$", ylabel=r"$\%$"
)
webpage.add_plots([flow_plot, pres_plot])

# Plot optimization progress
webpage.add_heading("Optimizer Performance")
progress_plot = io.LinePlot(
    pd.DataFrame(data=opt_progress),
    y="offset",
    title="Objective function over iteration",
    xlabel="Iteration number",
    ylabel="Result",
)
webpage.add_plots([progress_plot])

# Build webpage
webpage.build("./dashboard")

print("Optimized: ", optimized_k)
print("Ground truth: ", k_opt)
print(
    f"Completed in {time.time()-start:.2f}s with {len(opt_progress['offset'])} evaluations"
)
