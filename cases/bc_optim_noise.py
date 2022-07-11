"""Inverse problem in 0D optimization problem with Noise.

This case implements a 0D model optimization routine. It optimizes the
distal to proximal resistance ration (denoted k) at each outlet to match
the outlet pressure and flow. k is initialized with a values of 10.0.
"""
import time

import pandas as pd
from scipy import optimize
import click

from svsuperestimator import io
from svsuperestimator import model as mdl
from svsuperestimator import solver as slv
from scipy import stats

import numpy as np
import os


def run(svproject, num_samples):
    """Run the opimization.

    Args:
        svproject: Path to SimVascular project folder.
        num_samples: Number of samples from target distribution.
    """
    start = time.time()

    # Setup project, model, solver and webpage
    project = io.SimVascularProject(svproject)
    model = mdl.MultiFidelityModel(project)
    solver = slv.ZeroDSolver()
    webpage = io.WebPage("svSuperEstimator")

    if not os.path.exists(project["rom_optimization_folder"]):
        os.makedirs(project["rom_optimization_folder"])

    # Plot 3d model
    try:
        plot3d = io.Vtk3dPlot(
            project["3d_mesh"],
            title=project.name,
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
    for branch_id, vessel_data in enumerate(
        model.zerodmodel._config["vessels"]
    ):
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
            result = result[result.name != i]
        result.flow_out *= 3.6  # Convert cm^3/s to l/h
        result.pressure_out *= (
            0.00075006156130264  # Convert g/(cm s^2) to mmHg
        )
        return result

    # Running one simulation to determine targets
    target = solver.run_simulation(model.zerodmodel)

    outlet_bcs = {
        name: bc
        for name, bc in model.zerodmodel.boundary_conditions.items()
        if name != "INFLOW"
    }

    q_target = np.array(
        [
            target[target.name == bc_map[bc]["name"]][
                bc_map[bc]["flow"]
            ].mean()
            for bc in outlet_bcs
        ]
    )
    pd.DataFrame([list(q_target)], columns=list(outlet_bcs.keys()),).to_csv(
        os.path.join(project["rom_optimization_folder"], "flow_target.csv")
    )
    q_norm_factor = 1.0 / (q_target.min() - q_target.max())

    p_target = target[target.name == bc_map["INFLOW"]["name"]][
        bc_map["INFLOW"]["pressure"]
    ].mean() - np.array(
        [
            target[target.name == bc_map[bc]["name"]][
                bc_map[bc]["pressure"]
            ].mean()
            for bc in outlet_bcs
        ]
    )
    pd.DataFrame([list(p_target)], columns=list(outlet_bcs.keys()),).to_csv(
        os.path.join(project["rom_optimization_folder"], "pressure_target.csv")
    )
    p_norm_factor = 1.0 / (np.mean(p_target))

    k_opt = [
        bc.resistance_distal / bc.resistance_proximal
        for bc in outlet_bcs.values()
    ]  # Optimal distal to proximal resistance ratio for each outlet
    pd.DataFrame([list(k_opt)], columns=list(outlet_bcs.keys()),).to_csv(
        os.path.join(project["rom_optimization_folder"], "rd_rp_ratio.csv")
    )

    r_i = [
        bc.resistance_distal + bc.resistance_proximal
        for bc in outlet_bcs.values()
    ]  # Internal resistance for each outlet

    def get_result_plots(
        mean: pd.DataFrame,
        upper: pd.DataFrame,
        lower: pd.DataFrame,
        gt: pd.DataFrame,
    ) -> tuple[io.LinePlot, io.LinePlot]:
        plot_mean = format_result(mean.copy())
        plot_upper = format_result(upper.copy())
        plot_lower = format_result(lower.copy())
        gt = format_result(gt.copy())
        flow_plot = io.LinePlotWithUpperLower(
            plot_mean,
            plot_upper,
            plot_lower,
            x="time",
            y="flow_out",
            title="Flow over time",
            xlabel=r"$s$",
            ylabel=r"$\frac{l}{h}$",
            legend_title="BC Name",
        )
        flow_plot.add_trace(gt, x="time", y="flow_out", name="Ground Truth")
        pres_plot = io.LinePlotWithUpperLower(
            plot_mean,
            plot_upper,
            plot_lower,
            x="time",
            y="pressure_out",
            title="Pressure over time",
            xlabel=r"$s$",
            ylabel=r"$mmHg$",
            legend_title="BC Name",
        )
        pres_plot.add_trace(
            gt, x="time", y="pressure_out", name="Ground Truth"
        )
        return flow_plot, pres_plot

    def set_boundary_conditions(k: list[float]):
        """Set BCs based on distal to proximal resistance ratio."""
        for ki, r_ii, bc in zip(k, r_i, outlet_bcs.values()):
            bc.resistance_proximal = r_ii * 1 / (1.0 + ki)
            bc.resistance_distal = bc.resistance_proximal * ki

    bc_names = [bc_map[bc]["name"] for bc in outlet_bcs]
    bc_pressure = [bc_map[bc]["pressure"] for bc in outlet_bcs]
    bc_flow = [bc_map[bc]["flow"] for bc in outlet_bcs]
    inflow_name = bc_map["INFLOW"]["name"]
    inflow_pressure = bc_map["INFLOW"]["pressure"]

    def objective_function(
        k, p_target_noisy, q_target_noisy, offsets, sample_id, eval_stats
    ):
        """Objective function for the optimization.

        Evaluates the sum of the offset for the input output pressure relation
        for each outlet.
        """
        start = time.time()

        set_boundary_conditions(k)

        result = solver.run_simulation(model.zerodmodel, True)

        p_inflow = result.loc[result.name == inflow_name][
            inflow_pressure
        ].iloc[0]

        p_offsets = [
            (
                p_target_noisy[i]
                - (
                    p_inflow
                    - result.loc[result.name == name][pressure_id].iloc[0]
                )
            )
            * p_norm_factor
            for i, (name, pressure_id) in enumerate(zip(bc_names, bc_pressure))
        ]
        q_offsets = [
            (
                q_target_noisy[i]
                - result.loc[result.name == name][flow_id].iloc[0]
            )
            * q_norm_factor
            for i, (name, flow_id) in enumerate(zip(bc_names, bc_flow))
        ]

        offset = np.linalg.norm(p_offsets) + np.linalg.norm(q_offsets)
        print(
            f"Model: {project.name} | Sample: {sample_id:5.0f} | "
            f"Iteration: {len(offsets):5.0f} | Offset: {offset}"
        )
        offsets.append(offset)
        eval_stats["eval_time"] += time.time() - start
        return offset

    # Run optimization
    frames = []
    optimized_ks = pd.DataFrame(columns=list(outlet_bcs.keys()))
    p_targets = pd.DataFrame(columns=list(outlet_bcs.keys()))
    q_targets = pd.DataFrame(columns=list(outlet_bcs.keys()))
    offsets = []
    eval_stats = {}
    eval_stats["eval_time"] = 0.0

    np.random.seed(0)

    k_start = [10.0] * len(outlet_bcs)
    bounds = [(0.1, 100)] * len(outlet_bcs)
    for i in range(num_samples):

        p_target_noisy = (
            p_target + np.random.randn(len(p_target)) * p_target / 10.0
        )
        q_target_noisy = (
            q_target + np.random.randn(len(q_target)) * q_target / 10.0
        )
        offset = []
        optimized_k = optimize.minimize(
            fun=lambda l: objective_function(
                l, p_target_noisy, q_target_noisy, offset, i + 1, eval_stats
            ),
            x0=k_start,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 200},
        ).x

        set_boundary_conditions(optimized_k)
        optimized_result = solver.run_simulation(model.zerodmodel)
        offsets.append(
            pd.DataFrame(
                [[j, o] for j, o in enumerate(offset)],
                columns=["iteration", "offset"],
            ),
        )

        frames.append(optimized_result)
        optimized_ks = pd.concat(
            [
                optimized_ks,
                pd.DataFrame(
                    [list(optimized_k)],
                    columns=list(outlet_bcs.keys()),
                ),
            ],
            ignore_index=True,
        )
        p_targets = pd.concat(
            [
                p_targets,
                pd.DataFrame(
                    [list(p_target_noisy)],
                    columns=list(outlet_bcs.keys()),
                ),
            ]
        )
        q_targets = pd.concat(
            [
                q_targets,
                pd.DataFrame(
                    [list(q_target_noisy)],
                    columns=list(outlet_bcs.keys()),
                ),
            ]
        )
    optimized_ks.to_csv(
        os.path.join(
            project["rom_optimization_folder"], "rd_rp_ratio_samples.csv"
        )
    )
    p_targets.to_csv(
        os.path.join(
            project["rom_optimization_folder"], "pressure_target_samples.csv"
        )
    )
    q_targets.to_csv(
        os.path.join(
            project["rom_optimization_folder"], "flow_target_samples.csv"
        )
    )

    mean = frames[0].copy()

    for key in ["flow_in", "flow_out", "pressure_in", "pressure_out"]:
        mean[key] *= 0
        for frame in frames:
            mean[key] += frame[key]
        mean[key] /= len(frames)

    mean.to_csv(os.path.join(project["rom_optimization_folder"], "mean.csv"))

    std = frames[0].copy()

    for key in ["flow_in", "flow_out", "pressure_in", "pressure_out"]:
        std[key] *= 0
        for frame in frames:
            std[key] += (frame[key] - mean[key]) * (frame[key] - mean[key])
        std[key] = (std[key] / len(frames)) ** 0.5

    std.to_csv(os.path.join(project["rom_optimization_folder"], "std.csv"))

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
                    target[target.name == bc_map[bc_name]["name"]],
                )
            )

    offset_plot = io.LinePlot(
        offsets[0],
        x="iteration",
        y="offset",
        title="Error over iteration",
    )
    for i in range(1, len(offsets)):
        offset_plot.add_trace(
            offsets[i],
            x="iteration",
            y="offset",
        )

    webpage.add_heading("Random optimization targets")

    q_target_plot = io.ViolinPlot(
        q_targets * 3.6,  # Convert cm^3/s to l/h
        title="Flow out",
        ylabel=r"$\frac{l}{h}$",
    )
    q_target_plot.add_lines(
        list(outlet_bcs.keys()), q_target * 3.6, name="Reference"
    )
    p_target_plot = io.ViolinPlot(
        p_targets * 0.00075006156130264,  # Convert g/(cm s^2) to mmHg
        title="Pressure drop to inflow",
        ylabel=r"$mmHg$",
    )
    p_target_plot.add_lines(
        list(outlet_bcs.keys()),
        p_target * 0.00075006156130264,
        name="Reference",
    )
    webpage.add_plots(
        [
            q_target_plot,
            p_target_plot,
        ]
    )

    webpage.add_heading("Optimization details")
    k_plot = io.ViolinPlot(
        optimized_ks[
            (np.abs(stats.zscore(optimized_ks)) < 3).all(axis=1)
        ],  # Remove outliers
        ylabel=r"$\frac{R_d}{R_p}$",
        title="Optimized distal to proximal resistance ratio",
    )
    k_plot.add_lines(
        list(outlet_bcs.keys()),
        k_opt,
        name="Ground Truth",
    )
    webpage.add_plots([k_plot, offset_plot])

    # Build webpage
    webpage.build(project["rom_optimization_folder"])
    sum_evals = sum([len(o) for o in offsets])
    print(
        f"Completed in {time.time()-start:.2f}s with {sum_evals} evaluations "
        f"and {sum_evals/eval_stats['eval_time']} evaluations/second."
    )


@click.command()
@click.option(
    "--svproject",
    help="Path to SimVascular project folder.",
    required=True,
    type=str,
)
@click.option(
    "--num_samples",
    help="Number of samples from target distribution.",
    type=int,
    default=10,
)
def main(svproject, num_samples):
    """Run svSuperEstimator."""
    run(svproject, num_samples)
