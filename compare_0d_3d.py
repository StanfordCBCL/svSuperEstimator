from svsuperestimator.reader import *

from rich import print, box
from rich.console import Console
from rich.table import Table


from svsuperestimator.tasks import utils as taskutils

from plotly import express as px

import numpy as np

from svzerodsolver import runnercpp
import os
import pickle


threed_results_folder = "/Users/stanford/data/centerlines_luca"
threed_input_folder = (
    "/Users/stanford/data/centerlines_luca/boundary_conditions"
)
projects_folder = "/Users/stanford/data/projects"

threed_result_files = [
    os.path.join(threed_results_folder, n)
    for n in os.listdir(threed_results_folder)
    if n.endswith(".vtp")
]

model_mapping = {}

CONSOLE = Console(record=True)

for path in threed_result_files:
    model_name = path.split("/")[-1].split(".")[0]
    var_id = int(path.split("/")[-1].split(".")[1])

    if not model_name in model_mapping:
        model_mapping[model_name] = {}

    if not var_id in model_mapping[model_name]:
        model_mapping[model_name][var_id] = {}

    model_mapping[model_name][var_id].update(
        {
            "project_folder": os.path.join(projects_folder, model_name),
            "threed_result": path,
            "inflow_file": os.path.join(
                threed_input_folder, f"inflow_{model_name}.{var_id}.flow"
            ),
            "rcr_file": os.path.join(
                threed_input_folder, f"rcrt.dat_{model_name}.{var_id}.flow"
            ),
        }
    )

for model_name, variations in model_mapping.items():

    data_3d = {
        "pressure_systole": [],
        "pressure_diastole": [],
        "flow_systole": [],
        "flow_diastole": [],
    }
    data_0d = {
        "pressure_systole": [],
        "pressure_diastole": [],
        "flow_systole": [],
        "flow_diastole": [],
    }
    data_0d_opt = {
        "pressure_systole": [],
        "pressure_diastole": [],
        "flow_systole": [],
        "flow_diastole": [],
    }

    for var_id, var_config in variations.items():

        print(f"Running {model_name} variation {var_id}")

        project = SimVascularProject(var_config["project_folder"])
        threed_input = project["3d_simulation_input_path"]
        mesh_file = project["3d_mesh"]
        rcr_file = var_config["rcr_file"]
        inflow_file = var_config["inflow_file"]
        threed_result = var_config["threed_result"]

        handler3d = SvSolverInputHandler(threed_input)

        mapped_data = taskutils.extract_0d_element_coordinates(
            project["rom_simulation_config"],
            project["rom_centerline"],
        )

        handlermesh = MeshHandler.from_file(mesh_file)

        face_ids = handlermesh.get_boundary_centers()

        bc_ids = list(face_ids.keys())
        bc_ids_coords = list(face_ids.values())

        bc_mapping = {}
        for bc_name, bc_coord in mapped_data.items():
            if not bc_name.startswith("branch"):
                index = np.argmin(
                    np.linalg.norm(bc_ids_coords - bc_coord, axis=1)
                )
                bc_mapping[bc_name] = bc_ids[index]

        handler_rcr = SvSolverRcrHandler(rcr_file)
        threed_bc = handler_rcr.get_boundary_conditions()
        rcr_surface_ids = handler3d.rcr_surface_ids
        for i, bc in enumerate(threed_bc):
            surface_id = rcr_surface_ids[i]

            for bc_name, bc_id in bc_mapping.items():
                if bc_id == surface_id:
                    bc_mapping[bc_name] = bc
        handler_inflow = SvSolverInflowHandler(inflow_file)
        bc_mapping["INFLOW"] = handler_inflow.get_boundary_condition()

        zerod_config = project["rom_simulation_config"]

        for bc_config in zerod_config["boundary_conditions"]:
            if not bc_config["bc_name"] == "INFLOW":
                assert len(bc_mapping[bc_config["bc_name"]]["Pd"]) == 2
                assert len(bc_mapping[bc_config["bc_name"]]["t"]) == 2
                assert (
                    bc_mapping[bc_config["bc_name"]]["Pd"][0]
                    == bc_mapping[bc_config["bc_name"]]["Pd"][1]
                )
                bc_mapping[bc_config["bc_name"]]["Pd"] = bc_mapping[
                    bc_config["bc_name"]
                ]["Pd"][0]

            before = bc_config["bc_values"].copy()

            for name in bc_config["bc_values"]:
                bc_config["bc_values"][name] = bc_mapping[
                    bc_config["bc_name"]
                ][name]

            after = bc_config["bc_values"].copy()

            if bc_config["bc_name"] == "INFLOW":
                fig1 = px.line(before["Q"])
                fig2 = px.line(after["Q"])

            fig1.show()
            fig2.show()
            raise SystemExit

            if not bc_config["bc_name"] == "INFLOW":
                table = Table()
                table = Table(box=box.HORIZONTALS)
                table.add_column(bc_config["bc_name"], style="bold cyan")
                table.add_column("Value", justify="right")
                for key, value in before.items():
                    table.add_row(key, f"{value:.2e} -> {after[key]:.2e}")
                table.add_row(
                    "time constant",
                    f"{before['Rd']*before['C']:.2e} -> {after['Rd']*after['C']:.2e}",
                )
                CONSOLE.print(table)
            else:
                table = Table()
                table = Table(box=box.HORIZONTALS)
                table.add_column(bc_config["bc_name"], style="bold cyan")
                table.add_column("Value", justify="right")
                table.add_row(
                    "Q_min",
                    f"{np.min(before['Q']):.2e} -> {np.min(after['Q']):.2e}",
                )
                table.add_row(
                    "Q_max",
                    f"{np.max(before['Q']):.2e} -> {np.max(after['Q']):.2e}",
                )
                CONSOLE.print(table)

        zerod_opt_config = project["rom_simulation_config_optimized"]

        for bc_config in zerod_opt_config["boundary_conditions"]:
            if not bc_config["bc_name"] == "INFLOW" and not isinstance(
                bc_mapping[bc_config["bc_name"]]["Pd"], float
            ):
                assert len(bc_mapping[bc_config["bc_name"]]["Pd"]) == 2
                assert len(bc_mapping[bc_config["bc_name"]]["t"]) == 2
                assert (
                    bc_mapping[bc_config["bc_name"]]["Pd"][0]
                    == bc_mapping[bc_config["bc_name"]]["Pd"][1]
                )
                bc_mapping[bc_config["bc_name"]]["Pd"] = bc_mapping[
                    bc_config["bc_name"]
                ]["Pd"][0]

            for name in bc_config["bc_values"]:
                bc_config["bc_values"][name] = bc_mapping[
                    bc_config["bc_name"]
                ][name]

        # CONSOLE.save_svg("log.svg")
        # raise SystemExit

        result_0d = runnercpp.run_from_config(zerod_config)
        result_0d_opt = runnercpp.run_from_config(zerod_opt_config)

        pts_per_cycle = zerod_config["simulation_parameters"][
            "number_of_time_pts_per_cardiac_cycle"
        ]

        branch_data, times = taskutils.map_centerline_result_to_0d(
            threed_result,
            zerod_config,
            handler3d.time_step_size_3d,
        )

        for branch_id, branch in branch_data.items():

            for seg_id, segment in branch.items():
                vessel_name = "V" + str(segment["vessel_id"])

                vessel_result_0d = result_0d[result_0d.name == vessel_name]
                vessel_result_0d_opt = result_0d_opt[
                    result_0d.name == vessel_name
                ]

                fig1 = px.line(segment["pressure_out"])
                fig2 = px.line(
                    vessel_result_0d["pressure_out"][-pts_per_cycle:]
                )
                fig3 = px.line(
                    vessel_result_0d_opt["pressure_out"][-pts_per_cycle:]
                )

                fig1.show()
                fig2.show()
                # fig3.show()
                raise SystemExit

                # Collect 3d results
                data_3d["pressure_systole"].append(
                    np.amax(segment["pressure_out"])
                )
                data_3d["pressure_diastole"].append(
                    np.amin(segment["pressure_out"])
                )
                data_3d["flow_systole"].append(np.amax(segment["flow_out"]))
                data_3d["flow_diastole"].append(np.amin(segment["flow_out"]))

                # Collect 0d results
                data_0d["pressure_systole"].append(
                    np.amax(vessel_result_0d["pressure_out"][-pts_per_cycle:])
                )
                data_0d["pressure_diastole"].append(
                    np.amin(vessel_result_0d["pressure_out"][-pts_per_cycle:])
                )
                data_0d["flow_systole"].append(
                    np.amax(vessel_result_0d["flow_out"][-pts_per_cycle:])
                )
                data_0d["flow_diastole"].append(
                    np.amin(vessel_result_0d["flow_out"][-pts_per_cycle:])
                )

                # Collect 0d results
                data_0d_opt["pressure_systole"].append(
                    np.amax(
                        vessel_result_0d_opt["pressure_out"][-pts_per_cycle:]
                    )
                )
                data_0d_opt["pressure_diastole"].append(
                    np.amin(
                        vessel_result_0d_opt["pressure_out"][-pts_per_cycle:]
                    )
                )
                data_0d_opt["flow_systole"].append(
                    np.amax(vessel_result_0d_opt["flow_out"][-pts_per_cycle:])
                )
                data_0d_opt["flow_diastole"].append(
                    np.amin(vessel_result_0d_opt["flow_out"][-pts_per_cycle:])
                )

        break

    with open(f"results/{model_name}.pickle", "wb") as ff:
        pickle.dump({"3d": data_3d, "0d": data_0d, "0d_opt": data_0d_opt}, ff)

    break

CONSOLE.save_svg("log.svg")
