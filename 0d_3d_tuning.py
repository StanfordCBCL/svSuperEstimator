import json
import vtk
import numpy as np
from scipy import interpolate
import plotly.graph_objects as go
import os

from rich import print

from svzerodsolver import runnercpp, runner
from scipy import optimize

from vtk.util.numpy_support import vtk_to_numpy


def fit_zerod_to_threed(zerod_file, centerline_file, output_file):

    # Extract polydata from centerline file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(centerline_file)
    reader.Update()
    polydata = reader.GetOutput()
    num_arrays = polydata.GetPointData().GetNumberOfArrays()
    array_names = [
        polydata.GetPointData().GetArrayName(i) for i in range(num_arrays)
    ]
    pressure_names = [
        name for name in array_names if name.startswith("pressure")
    ]
    flow_names = [name for name in array_names if name.startswith("velocity")]

    # Read 0d data

    with open(zerod_file) as ff:
        zerod_config = json.load(ff)

    def get_branch_polydata(branch_id):

        thresh = vtk.vtkThreshold()
        thresh.SetInputData(polydata)
        thresh.SetLowerThreshold(branch_id)
        thresh.SetUpperThreshold(branch_id)
        thresh.SetInputArrayToProcess(
            0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "BranchId"
        )
        thresh.Update()
        branch_pdata = thresh.GetOutput()

        return branch_pdata

    branch_data = {}

    for vessel_config in zerod_config["vessels"]:

        name = vessel_config["vessel_name"]
        branch_id, seg_id = name.split("_")
        branch_id = int(branch_id[6:])
        seg_id = int(seg_id[3:])

        if not branch_id in branch_data:
            branch_data[branch_id] = {}

        branch_data[branch_id][seg_id] = {}
        branch_data[branch_id][seg_id]["length"] = vessel_config[
            "vessel_length"
        ]
        branch_data[branch_id][seg_id]["id"] = vessel_config["vessel_id"]
        branch_data[branch_id][seg_id]["values"] = vessel_config[
            "zero_d_element_values"
        ]

    def extract_pressure_flow(polydata, idx):
        flow = []
        pressure = []

        for flow_name in flow_names:
            flow.append(
                vtk_to_numpy(polydata.GetPointData().GetArray(flow_name))[idx]
            )

        for pressure_name in pressure_names:
            pressure.append(
                vtk_to_numpy(polydata.GetPointData().GetArray(pressure_name))[
                    idx
                ]
            )

        return np.array(flow), np.array(pressure)

    for branch_id, branch in branch_data.items():

        branch_pdata = get_branch_polydata(branch_id)

        path_length = vtk_to_numpy(
            branch_pdata.GetPointData().GetArray("Path")
        )

        seg_start = 0.0
        seg_start_index = 0

        for seg_id in range(len(branch)):
            segment = branch[seg_id]
            length = segment["length"]

            seg_end_index = (np.abs(path_length - length - seg_start)).argmin()
            seg_end = path_length[seg_end_index]

            flow_0, pressure_0 = extract_pressure_flow(
                branch_pdata, seg_start_index
            )
            flow_1, pressure_1 = extract_pressure_flow(
                branch_pdata, seg_end_index
            )

            branch_data[branch_id][seg_id]["flow_0"] = flow_0
            branch_data[branch_id][seg_id]["flow_1"] = flow_1
            branch_data[branch_id][seg_id]["pressure_0"] = pressure_0
            branch_data[branch_id][seg_id]["pressure_1"] = pressure_1

            seg_start = seg_end
            seg_start_index = seg_end_index

    times = (
        np.array([float(k.split("_")[1]) for k in pressure_names]) * 0.000339
    )  # TODO: Assumption about time step size not always correct
    cycle_period = (
        zerod_config["boundary_conditions"][0]["bc_values"]["t"][-1]
        - zerod_config["boundary_conditions"][0]["bc_values"]["t"][0]
    )
    start_last_cycle = (np.abs(times - (times[-1] - cycle_period))).argmin()

    def interpolate_values_to_times(values, sim_times, target_times):

        ip = interpolate.interp1d(sim_times, values)
        return ip(target_times)

    def run_single_bv_simulation(
        inflow, outpres, t, R, C, L, stenosis_coefficient
    ):

        config = {
            "boundary_conditions": [
                {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {"Q": inflow.tolist(), "t": t.tolist()},
                },
                {
                    "bc_name": "OUTPRES",
                    "bc_type": "PRESSURE",
                    "bc_values": {"P": outpres.tolist(), "t": t.tolist()},
                },
            ],
            "junctions": [],
            "simulation_parameters": {
                "number_of_cardiac_cycles": 5,
                "number_of_time_pts_per_cardiac_cycle": zerod_config[
                    "simulation_parameters"
                ]["number_of_time_pts_per_cardiac_cycle"],
                "steady_initial": True,
            },
            "vessels": [
                {
                    "boundary_conditions": {
                        "inlet": "INFLOW",
                        "outlet": "OUTPRES",
                    },
                    "vessel_id": 0,
                    "vessel_length": 10.0,
                    "vessel_name": "branch0_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "C": C,
                        "L": L,
                        "R_poiseuille": R,
                        "stenosis_coefficient": stenosis_coefficient,
                    },
                }
            ],
        }

        result = runnercpp.run_from_config(config)

        sim_times = np.array(result["time"])
        sim_inpres = np.array(result["pressure_in"])
        sim_outflow = np.array(result["flow_out"])

        inpres = interpolate_values_to_times(sim_inpres, sim_times, times)
        outflow = interpolate_values_to_times(sim_outflow, sim_times, times)

        return inpres, outflow

    # Start optimizing the branches
    for branch_id, branch in branch_data.items():

        for seg_id in range(len(branch)):
            segment = branch[seg_id]

            inpres = segment["pressure_0"]
            outpres = segment["pressure_1"]
            inflow = segment["flow_0"]
            outflow = segment["flow_1"]

            R = segment["values"]["R_poiseuille"]
            C = segment["values"]["C"]
            L = segment["values"]["L"]
            stenosis_coefficient = segment["values"]["stenosis_coefficient"]

            theta_log_start = np.log([R, C, L, stenosis_coefficient])

            pres_norm_factor = np.mean(inpres)
            flow_norm_factor = np.amax(outflow) - np.amin(outflow)

            def objective_function(theta_log):
                theta = np.exp(theta_log)
                inpres_sim, outflow_sim = run_single_bv_simulation(
                    inflow, outpres, times, *theta
                )

                mse = np.linalg.norm(
                    (inpres_sim[start_last_cycle:] - inpres[start_last_cycle:])
                    / pres_norm_factor
                ) + np.linalg.norm(
                    (
                        outflow_sim[start_last_cycle:]
                        - outflow[start_last_cycle:]
                    )
                    / flow_norm_factor
                )

                return mse

            print("Optimizing branch", branch_id, "segment", seg_id)
            theta_log_optimized = optimize.minimize(
                fun=objective_function,
                x0=theta_log_start,
                method="Nelder-Mead",
                options={"maxiter": 200},
            ).x

            # inpres_sim, outflow_sim = run_single_bv_simulation(
            #     inflow, outpres, times, *np.exp(theta_log_optimized)
            # )
            # inpres_start, outflow_start = run_single_bv_simulation(
            #     inflow, outpres, times, *np.exp(theta_log_start)
            # )
            # print("Before: ", np.exp(theta_log_start))
            # print("Optimized: ", np.exp(theta_log_optimized))

            # fig = go.Figure()
            # # fig.add_trace(go.Scatter(x=times, y=inpres, name="Inpres 3d"))
            # # fig.add_trace(
            # #     go.Scatter(x=times, y=inpres_sim, name="Inpres 0d opt")
            # # )
            # # fig.add_trace(
            # #     go.Scatter(x=times, y=inpres_start, name="Inpres 0d")
            # # )
            # fig.add_trace(go.Scatter(x=times, y=outflow, name="Outflow 3d"))
            # fig.add_trace(
            #     go.Scatter(x=times, y=outflow_sim, name="Outflow 0d opt")
            # )
            # fig.add_trace(
            #     go.Scatter(x=times, y=outflow_start, name="Outflow 0d")
            # )
            # fig.show()

            # raise SystemExit

            theta_optimized = np.exp(theta_log_optimized)

            branch_data[branch_id][seg_id]["values_optimized"] = {
                "C": theta_optimized[1],
                "L": theta_optimized[2],
                "R_poiseuille": theta_optimized[0],
                "stenosis_coefficient": theta_optimized[3],
            }

    for i, vessel_config in enumerate(zerod_config["vessels"]):

        name = vessel_config["vessel_name"]
        branch_id, seg_id = name.split("_")
        branch_id = int(branch_id[6:])
        seg_id = int(seg_id[3:])

        zerod_config["vessels"][i]["zero_d_element_values"] = branch_data[
            branch_id
        ][seg_id]["values_optimized"]

    with open(output_file, "w") as ff:
        json.dump(zerod_config, ff, indent=4)


# centerline_folder = "/Users/stanford/data/3d_centerline"
# zerod_folder = "/Users/stanford/data/0d"

# output_folder = "/Users/stanford/data/0d_optimized"

# i = 0

# for filename in os.listdir(centerline_folder):
#     label = filename[:-4]
#     print(i, label)

#     centerline_file = os.path.join(centerline_folder, filename)
#     zerod_file = os.path.join(zerod_folder, label + "_0d.in")
#     output_file = os.path.join(output_folder, label + "_0d.in")

#     if not os.path.exists(output_file) and not label in [
#         "0070_0001",
#         "0069_0001",
#         "0176_0000",
#     ]:
#         fit_zerod_to_threed(zerod_file, centerline_file, output_file)

#     i += 1


zerod_file = "/Users/stanford/data/0d/0069_0001_0d.in"
zerod_file_opt = "/Users/stanford/data/0d/0069_0001_0d_optimized.in"
centerline_file = "/Users/stanford/data/3d_centerline/0069_0001.vtp"

fit_zerod_to_threed(zerod_file, centerline_file, zerod_file_opt)
