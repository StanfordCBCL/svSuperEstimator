import json
import vtk
import numpy as np
from scipy import interpolate
import plotly.graph_objects as go
import os

from rich import print

from svzerodsolver import runnercpp, runner
from scipy import optimize
from svsuperestimator import visualizer

from vtk.util.numpy_support import vtk_to_numpy

zerod_file = "/Users/stanford/data/0d/0069_0001_0d.in"
zerod_file_opt = "/Users/stanford/data/0d/0069_0001_0d_optimized.in"
centerline_file = "/Users/stanford/data/3d_centerline/0069_0001.vtp"

with open(zerod_file) as ff:
    zerod_config = json.load(ff)

with open(zerod_file_opt) as ff:
    zerod_opt_config = json.load(ff)

zerod_result = result = runnercpp.run_from_config(zerod_config)
zerod_opt_result = result = runnercpp.run_from_config(zerod_opt_config)

# Extract polydata from centerline file
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(centerline_file)
reader.Update()
polydata = reader.GetOutput()
num_arrays = polydata.GetPointData().GetNumberOfArrays()
array_names = [
    polydata.GetPointData().GetArrayName(i) for i in range(num_arrays)
]
pressure_names = [name for name in array_names if name.startswith("pressure")]
flow_names = [name for name in array_names if name.startswith("velocity")]


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
    branch_data[branch_id][seg_id]["length"] = vessel_config["vessel_length"]
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
            vtk_to_numpy(polydata.GetPointData().GetArray(pressure_name))[idx]
        )

    return np.array(flow), np.array(pressure)


for branch_id, branch in branch_data.items():

    branch_pdata = get_branch_polydata(branch_id)

    path_length = vtk_to_numpy(branch_pdata.GetPointData().GetArray("Path"))

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
        flow_1, pressure_1 = extract_pressure_flow(branch_pdata, seg_end_index)

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

end_3d = times[-1]

zerod_result = zerod_result[zerod_result["time"] < end_3d]
zerod_result = zerod_result[zerod_result["time"] > end_3d - cycle_period]
zerod_opt_result = zerod_opt_result[zerod_opt_result["time"] < end_3d]
zerod_opt_result = zerod_opt_result[
    zerod_opt_result["time"] > end_3d - cycle_period
]

result_idx_3d = times > end_3d - cycle_period

report = visualizer.Report()
for branch_id, branch in branch_data.items():

    for seg_id in range(len(branch)):
        segment = branch[seg_id]
        report.add(f"Branch {branch_id} segment {seg_id}")
        vessel_id = segment["id"]

        inpres_plot = visualizer.Plot2D(title="Inlet pressure")
        inpres_plot.add_line_trace(
            x=times[result_idx_3d],
            y=segment["pressure_0"][result_idx_3d],
            name="3D",
            showlegend=True,
        )
        inpres_plot.add_line_trace(
            x=zerod_result[zerod_result.name == f"V{vessel_id}"]["time"],
            y=zerod_result[zerod_result.name == f"V{vessel_id}"][
                "pressure_in"
            ],
            name="0D",
            showlegend=True,
        )
        inpres_plot.add_line_trace(
            x=zerod_opt_result[zerod_opt_result.name == f"V{vessel_id}"][
                "time"
            ],
            y=zerod_opt_result[zerod_opt_result.name == f"V{vessel_id}"][
                "pressure_in"
            ],
            name="0D optimized",
            showlegend=True,
        )
        inflow_plot = visualizer.Plot2D(title="Inlet flow")
        inflow_plot.add_line_trace(
            x=times[result_idx_3d],
            y=segment["flow_0"][result_idx_3d],
            name="3D",
            showlegend=True,
        )
        inflow_plot.add_line_trace(
            x=zerod_result[zerod_result.name == f"V{vessel_id}"]["time"],
            y=zerod_result[zerod_result.name == f"V{vessel_id}"]["flow_in"],
            name="0D",
            showlegend=True,
        )
        inflow_plot.add_line_trace(
            x=zerod_opt_result[zerod_opt_result.name == f"V{vessel_id}"][
                "time"
            ],
            y=zerod_opt_result[zerod_opt_result.name == f"V{vessel_id}"][
                "flow_in"
            ],
            name="0D optimized",
            showlegend=True,
        )
        report.add([inpres_plot, inflow_plot])

report.to_html("/Users/stanford/svSuperEstimator")
