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
solver = slv.ZeroDSolver(model.zerodmodel)
webpage = io.WebPage("svSuperEstimator")

# Running one simulation to determine ground truth
ground_truth = solver.run_simulation()
p_avg_inlet_gt = ground_truth.query("name=='INFLOW'")["pressure"].mean()
outlet_bcs = {
    name: bc
    for name, bc in model.zerodmodel.boundary_conditions.items()
    if name != "INFLOW"
}
r_i = [
    bc.resistance_distal + bc.resistance_proximal for bc in outlet_bcs.values()
]  # Internal resistance for each outlet
p_diff_io_gt = [
    p_avg_inlet_gt - ground_truth[ground_truth.name == bc]["pressure"].mean()
    for bc in outlet_bcs
]  # Pressure difference between inlet and each outlet
k_opt = [
    bc.resistance_distal / bc.resistance_proximal for bc in outlet_bcs.values()
]  # Optimal distal to proximal resistance ratio for each outlet

# Plot ground truth
webpage.add_heading("Ground Truth")
webpage.add_plots(solver.get_result_plots(ground_truth))


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
    result = solver.run_simulation()

    offset = 0.0
    p_avg_inlet = result.query("name=='INFLOW'")["pressure"].mean()
    offset = sum(
        [
            abs(
                p_diff_io_gt[i]
                - (p_avg_inlet - result[result.name == bc]["pressure"].mean())
            )
            for i, bc in enumerate(outlet_bcs)
        ]
    )
    opt_progress["offset"].append(offset)
    return offset


# Run optimization
optimized_k = optimize.minimize(
    objective_function,
    K_START,
    method="Nelder-Mead",
    bounds=BOUNDS,
    options={"maxiter": 10},
).x

# Get result for optimized k
set_boundary_conditions(optimized_k)
optimized_result = solver.run_simulation()

# Plot optimized results
webpage.add_heading("Optimized Result")
webpage.add_plots(solver.get_result_plots(optimized_result))

# Plot optimization error
webpage.add_heading("Optimization Error")
diff = (
    optimized_result[["flow", "pressure"]] - ground_truth[["flow", "pressure"]]
)
diff["name"], diff["time"] = optimized_result.name, optimized_result.time
flow_plot, pres_plot = solver.get_result_plots(diff)
flow_plot.configure(
    title="Flow error over time", xlabel="$s$", ylabel=r"$\frac{l}{h}$"
)
pres_plot.configure(
    title="Pressure error over time", xlabel="$s$", ylabel="$mmHg$"
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

print(f"Completed in {time.time()-start:.2f}s")
