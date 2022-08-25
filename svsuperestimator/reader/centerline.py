from vtk.util.numpy_support import vtk_to_numpy
import vtk

import numpy as np

from . import utils


def map_centerline_result_to_0d(centerline, zerod_config, dt3d):

    # Read centerline vtk file
    polydata = utils.vtk_read_polydata(centerline)

    # Extract names for pressure and flow time step solutions
    num_arrays = polydata.GetPointData().GetNumberOfArrays()
    array_names = [
        polydata.GetPointData().GetArrayName(i) for i in range(num_arrays)
    ]
    pressure_names = [n for n in array_names if n.startswith("pressure")]
    flow_names = [n for n in array_names if n.startswith("velocity")]

    # calculate cycle period
    cycle_period = (
        zerod_config["boundary_conditions"][0]["bc_values"]["t"][-1]
        - zerod_config["boundary_conditions"][0]["bc_values"]["t"][0]
    )

    # Extract time steps
    times = np.array([float(k.split("_")[1]) for k in pressure_names]) * dt3d

    # Calculate start of last cycle
    start_last_cycle = (
        np.abs(times - (times[-1] - cycle_period))
    ).argmin() - 1

    # Extract branch information of 0D config
    branchdata = {}
    for vessel_config in zerod_config["vessels"]:

        # Extract branch and segment id from name
        name = vessel_config["vessel_name"]
        branch_id, seg_id = name.split("_")
        branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

        if not branch_id in branchdata:
            branchdata[branch_id] = {}

        branchdata[branch_id][seg_id] = {}
        branchdata[branch_id][seg_id]["length"] = vessel_config[
            "vessel_length"
        ]
        branchdata[branch_id][seg_id]["vessel_id"] = vessel_config["vessel_id"]

    def extract_pressure_flow(polydata, idx):
        flow = np.array(
            [
                vtk_to_numpy(polydata.GetPointData().GetArray(n))[idx]
                for n in flow_names
            ]
        )[start_last_cycle:-1]
        pressure = np.array(
            [
                vtk_to_numpy(polydata.GetPointData().GetArray(n))[idx]
                for n in pressure_names
            ]
        )[start_last_cycle:-1]
        return flow, pressure

    for branch_id, branch in branchdata.items():

        branch_pdata = utils.vtk_point_data_threshold(
            polydata, "BranchId", branch_id, branch_id
        )

        path_length = vtk_to_numpy(
            branch_pdata.GetPointData().GetArray("Path")
        )
        points = vtk_to_numpy(branch_pdata.GetPoints().GetData())

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

            branchdata[branch_id][seg_id]["flow_0"] = flow_0
            branchdata[branch_id][seg_id]["flow_1"] = flow_1
            branchdata[branch_id][seg_id]["pressure_0"] = pressure_0
            branchdata[branch_id][seg_id]["pressure_1"] = pressure_1
            branchdata[branch_id][seg_id]["x0"] = points[seg_start_index]
            branchdata[branch_id][seg_id]["x1"] = points[seg_end_index]

            seg_start = seg_end
            seg_start_index = seg_end_index

    times = times[start_last_cycle:-1] - [np.amin(times[start_last_cycle])]

    return branchdata, times
