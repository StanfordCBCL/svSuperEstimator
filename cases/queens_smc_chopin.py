from __future__ import annotations
import os
from pqueens.main import main as run_queens
from tempfile import TemporaryDirectory
import json

from scipy.fftpack import fft
from svsuperestimator import io, model as mdl, solver as slv
import time
import click
import pandas as pd
import numpy as np
from rich import print

this_file_dir = os.path.dirname(__file__)


def run(
    svproject="/Users/stanford/svSuperEstimator/tmpfiles/0069_0001",
    num_procs=4,
):
    start = time.time()

    # Setup project, model, solver and webpage
    project = io.SimVascularProject(svproject)
    webpage = io.WebPage("svSuperEstimator")
    model = mdl.MultiFidelityModel(project)
    solver = slv.ZeroDSolver(cpp=True)

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

    k_opt = [
        bc.resistance_distal / bc.resistance_proximal
        for bc in outlet_bcs.values()
    ]  # Optimal distal to proximal resistance ratio for each outlet
    pd.DataFrame([list(k_opt)], columns=list(outlet_bcs.keys()),).to_csv(
        os.path.join(project["rom_optimization_folder"], "rd_rp_ratio.csv")
    )

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

    config = {
        "experiment_name": "queens_smc_chopin",
        "database": {"type": "mongodb", "address": "localhost:27017"},
        "method": {
            "method_name": "smc_chopin",
            "method_options": {
                "seed": 42,
                "num_particles": 100,
                "resampling_threshold": 0.5,
                "resampling_method": "systematic",
                "feynman_kac_model": "adaptive_tempering",
                "waste_free": True,
                "num_rejuvenation_steps": 2,
                "model": "model",
                "max_feval": 10000,
                "result_description": {
                    "write_results": True,
                    "plot_results": False,
                },
            },
        },
        "model": {
            "type": "likelihood_model",
            "subtype": "gaussian",
            "forward_model": "forward_model",
            "output_label": "y_obs",
            "coordinate_labels": [],
            "noise_type": "fixed_variance",
            "noise_value": 1.0,
            "experimental_file_name_identifier": "*.csv",
            "experimental_csv_data_base_dir": None,
            "parameters": "parameters",
        },
        "forward_model": {
            "type": "simulation_model",
            "interface": "interface",
            "parameters": "parameters",
        },
        "interface": {
            "type": "direct_python_interface",
            "function_name": "objective_function",
            "external_python_module_function": os.path.join(
                this_file_dir, "objective_function.py"
            ),
            "num_workers": num_procs,
        },
        "parameters": {
            "random_variables": {
                "k0": {
                    "distribution": "uniform",
                    "type": "FLOAT",
                    "size": 1,
                    "lower_bound": -2,
                    "upper_bound": 3,
                    "dimension": 1,
                },
                "k1": {
                    "distribution": "uniform",
                    "type": "FLOAT",
                    "size": 1,
                    "lower_bound": -2,
                    "upper_bound": 3,
                    "dimension": 1,
                },
                "k2": {
                    "distribution": "uniform",
                    "type": "FLOAT",
                    "size": 1,
                    "lower_bound": -2,
                    "upper_bound": 3,
                    "dimension": 1,
                },
                "k3": {
                    "distribution": "uniform",
                    "type": "FLOAT",
                    "size": 1,
                    "lower_bound": -2,
                    "upper_bound": 3,
                    "dimension": 1,
                },
            }
        },
    }

    with TemporaryDirectory() as tmpdir:
        target_file = os.path.join(tmpdir, "targets.csv")
        config["model"]["experimental_csv_data_base_dir"] = tmpdir
        with open(target_file, "w") as ff:
            ff.write(
                "y_obs\n"
                + "\n".join([str(t) for t in list(p_target) + list(q_target)])
            )
        input_file = os.path.join(tmpdir, "config.json")
        with open(input_file, "w") as ff:
            json.dump(config, ff)
        run_queens(
            [
                "--input",
                input_file,
                "--output_dir",
                project["rom_optimization_folder"],
            ]
        )

    # Build webpage
    webpage.build(project["rom_optimization_folder"])
    print(f"Completed in {time.time()-start:.2f}s")


# @click.command()
# @click.option(
#     "--svproject",
#     help="Path to SimVascular project folder.",
#     required=True,
#     type=str,
# )
# @click.option(
#     "--num_samples",
#     help="Number of samples from target distribution.",
#     type=int,
#     default=10,
# )
# def main(svproject, num_samples):
#     """Run svSuperEstimator."""
#     run(svproject, num_samples)


if __name__ == "__main__":
    run()
