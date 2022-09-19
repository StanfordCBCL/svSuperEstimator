from svsuperestimator.reader import *

from rich import print, box
from rich.console import Console
from rich.table import Table


from svsuperestimator.reader import utils as readutils
from svsuperestimator.tasks import taskutils

from plotly import express as px

import numpy as np

from svzerodsolver import runnercpp
import os
import pickle
import plotly.graph_objects as go


threed_results_folder = "/Users/stanford/data/aorta_simulations_3"
threed_input_folder = "/Users/stanford/data/aorta_simulations_3/files"
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

for mname in model_mapping.keys():
    model_mapping[mname][-1] = {
        "project_folder": os.path.join(projects_folder, mname),
        "threed_result": f"/Users/stanford/data/3d_centerline/{mname}.vtp",
        "inflow_file": None,
        "rcr_file": None,
    }

# ['0090_0001', '0091_0001', '0093_0001', '0094_0001', '0095_0001']

for model_name, variations in model_mapping.items():

    data_3d = {}
    data_0d = {}
    data_0d_opt = {}

    for var_id, var_config in variations.items():

        print(f"Running {model_name} variation {var_id}")

        project = SimVascularProject(var_config["project_folder"])
        handlermesh = project["mesh"]
        if var_config["rcr_file"] is not None:
            rcr_file = var_config["rcr_file"]
            handler_rcr = SvSolverRcrHandler.from_file(rcr_file)
        else:
            handler_rcr = project["3d_simulation_rcr"]
        if var_config["inflow_file"] is not None:
            inflow_file = var_config["inflow_file"]
            handler_inflow = SvSolverInflowHandler.from_file(inflow_file)
        else:
            handler_inflow = project["3d_simulation_inflow"]
        threed_result = var_config["threed_result"]
        threed_result_handler = CenterlineHandler.from_file(threed_result)

        handler3d = project["3d_simulation_input"]

        mapped_data = readutils.get_0d_element_coordinates(project)

        face_ids = handlermesh.boundary_centers

        bc_ids = list(face_ids.keys())
        bc_ids_coords = list(face_ids.values())

        bc_mapping = {}
        for bc_name, bc_coord in mapped_data.items():
            if not bc_name.startswith("branch"):
                index = np.argmin(
                    np.linalg.norm(bc_ids_coords - bc_coord, axis=1)
                )
                bc_mapping[bc_name] = bc_ids[index]

        threed_bc = handler_rcr.get_rcr_data()
        rcr_surface_ids = handler3d.rcr_surface_ids
        for i, bc in enumerate(threed_bc):
            surface_id = rcr_surface_ids[i]

            for bc_name, bc_id in bc_mapping.items():
                if bc_id == surface_id:
                    bc_mapping[bc_name] = bc
        bc_mapping["INFLOW"] = handler_inflow.get_inflow_data()

        zerod_handler = project["0d_simulation_input"]
        zerod_config = zerod_handler.data

        for bc_config in zerod_config["boundary_conditions"]:
            # print(bc_config["bc_name"])
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

            # if bc_config["bc_name"] == "INFLOW":
            #     fig1 = px.line(before["Q"])
            #     fig2 = px.line(after["Q"])

            # fig1.show()
            # fig2.show()
            # raise SystemExit

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
                # CONSOLE.print(table)
            else:
                table = Table()
                table = Table(box=box.HORIZONTALS)
                table.add_column(bc_config["bc_name"], style="bold cyan")
                table.add_column("Value", justify="right")
                table.add_row(
                    "Q_min",
                    f"{np.min(before['Q']):.3e} -> {np.min(after['Q']):.3e}",
                )
                table.add_row(
                    "Q_max",
                    f"{np.max(before['Q']):.3e} -> {np.max(after['Q']):.3e}",
                )
                # CONSOLE.print(table)

        zerod_opt_handler = project["rom_simulation_config_optimized"]
        zerod_opt_config = zerod_opt_handler.data

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

        branch_data, times = taskutils.map_centerline_result_to_0d_2(
            project,
            threed_result_handler,
        )

        taskutils.set_initial_condition(zerod_handler, branch_data)
        taskutils.set_initial_condition(zerod_opt_handler, branch_data)
        zerod_handler.update_simparams(
            steady_initial=False, num_cycles=1, last_cycle_only=False
        )
        zerod_opt_handler.update_simparams(
            steady_initial=False, num_cycles=1, last_cycle_only=False
        )

        result_0d = runnercpp.run_from_config(zerod_handler.data)
        result_0d_opt = runnercpp.run_from_config(zerod_opt_handler.data)

        # print(result_0d)

        if len(times) < 10:
            continue

        data_3d[var_id] = {
            "pressure_systole": [],
            "pressure_diastole": [],
            "flow_systole": [],
            "flow_diastole": [],
        }
        data_0d[var_id] = {
            "pressure_systole": [],
            "pressure_diastole": [],
            "flow_systole": [],
            "flow_diastole": [],
        }
        data_0d_opt[var_id] = {
            "pressure_systole": [],
            "pressure_diastole": [],
            "flow_systole": [],
            "flow_diastole": [],
        }

        i = 0

        for branch_id, branch in branch_data.items():

            for seg_id, segment in branch.items():
                vessel_name = f"branch{branch_id}_seg{seg_id}"

                vessel_result_0d = result_0d[result_0d.name == vessel_name]
                vessel_result_0d_opt = result_0d_opt[
                    result_0d_opt.name == vessel_name
                ]

                # fig1 = px.line(segment["pressure_out"])
                # fig2 = px.line(
                #     vessel_result_0d["pressure_out"]
                # )
                # fig3 = px.line(
                #     vessel_result_0d_opt["pressure_out"]
                # )

                # fig1.show()
                # fig2.show()
                # fig3.show()
                # raise SystemExit

                # Collect 3d results
                data_3d[var_id]["pressure_systole"].append(
                    np.amax(segment["pressure_out"])
                )
                data_3d[var_id]["pressure_diastole"].append(
                    np.amin(segment["pressure_out"])
                )
                data_3d[var_id]["flow_systole"].append(
                    np.amax(segment["flow_out"])
                )
                data_3d[var_id]["flow_diastole"].append(
                    np.amin(segment["flow_out"])
                )

                # Collect 0d results
                data_0d[var_id]["pressure_systole"].append(
                    np.amax(vessel_result_0d["pressure_out"])
                )
                data_0d[var_id]["pressure_diastole"].append(
                    np.amin(vessel_result_0d["pressure_out"])
                )
                data_0d[var_id]["flow_systole"].append(
                    np.amax(vessel_result_0d["flow_out"])
                )
                data_0d[var_id]["flow_diastole"].append(
                    np.amin(vessel_result_0d["flow_out"])
                )

                # Collect 0d results
                data_0d_opt[var_id]["pressure_systole"].append(
                    np.amax(vessel_result_0d_opt["pressure_out"])
                )
                data_0d_opt[var_id]["pressure_diastole"].append(
                    np.amin(vessel_result_0d_opt["pressure_out"])
                )
                data_0d_opt[var_id]["flow_systole"].append(
                    np.amax(vessel_result_0d_opt["flow_out"])
                )
                data_0d_opt[var_id]["flow_diastole"].append(
                    np.amin(vessel_result_0d_opt["flow_out"])
                )

                if False:

                    print(
                        "3D:",
                        data_3d["pressure_systole"][-1],
                        "\n0D:",
                        data_0d["pressure_systole"][-1],
                        "\n0D:",
                        data_0d_opt["pressure_systole"][-1],
                    )

                    fig = go.Figure()
                    fig.update_layout(title=f"branch{branch_id}_seg{seg_id}")

                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=np.array(segment["pressure_out"]),
                            name="3D",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=np.array(vessel_result_0d["time"]),
                            y=np.array(vessel_result_0d["pressure_out"]),
                            name="0D",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=np.array(vessel_result_0d_opt["time"]),
                            y=np.array(vessel_result_0d_opt["pressure_out"]),
                            name="0D_opt",
                        )
                    )

                    fig.show()
                    if i == 3:
                        raise SystemExit

                i += 1

    with open(f"results/{model_name}.pickle", "wb") as ff:
        pickle.dump({"3d": data_3d, "0d": data_0d, "0d_opt": data_0d_opt}, ff)


CONSOLE.save_svg("log.svg")
