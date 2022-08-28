import numpy as np
from scipy.interpolate import CubicSpline

from ..reader import CenterlineHandler

import numpy as np


def cgs_pressure_to_mmgh(cgs_pressure):
    """Convert pressure from g/(cm s^2) to mmHg.

    Args:
        cgs_pressure: Pressure in CGS format.

    Returns:
        Pressure in mmHg.
    """
    return np.array(cgs_pressure * 0.00075006156130264)


def cgs_flow_to_lh(cgs_flow):
    """Convert flow from cm^3/s to l/h.

    Args:
        cgs_flow: Flow in CGS format.

    Returns:
        Flow in l/h.
    """
    return np.array(cgs_flow * 3.6)


def refine_with_cubic_spline(y, num):
    y = y.copy()
    y[-1] = y[0]
    x_old = np.linspace(0.0, 100.0, len(y))
    x_new = np.linspace(0.0, 100.0, num)
    y_new = CubicSpline(x_old, y, bc_type="periodic")(x_new)
    return y_new


def map_centerline_result_to_0d(centerline, zerod_config, dt3d):

    cl_handler = CenterlineHandler.from_file(centerline)

    # calculate cycle period
    cycle_period = (
        zerod_config["boundary_conditions"][0]["bc_values"]["t"][-1]
        - zerod_config["boundary_conditions"][0]["bc_values"]["t"][0]
    )

    # Extract time steps
    times = cl_handler.time_steps * dt3d

    # Calculate start of last cycle
    start_last_cycle = (
        np.abs(times - (times[-1] - cycle_period))
    ).argmin() - 1

    def filter_last_cycle(data, seg_end_index):
        if start_last_cycle == -1:
            return data[:, seg_end_index]
        return data[start_last_cycle:-1, seg_end_index]

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

    for branch_id, branch in branchdata.items():

        branch_data = cl_handler.get_branch_data(branch_id)

        seg_start = 0.0
        seg_start_index = 0

        for seg_id in range(len(branch)):
            segment = branch[seg_id]
            length = segment["length"]

            seg_end_index = (
                np.abs(branch_data["path"] - length - seg_start)
            ).argmin()
            seg_end = branch_data["path"][seg_end_index]

            segment.update(
                {
                    "flow_in": filter_last_cycle(
                        branch_data["flow"], seg_end_index
                    ),
                    "flow_out": filter_last_cycle(
                        branch_data["flow"], seg_end_index
                    ),
                    "pressure_in": filter_last_cycle(
                        branch_data["pressure"], seg_end_index
                    ),
                    "pressure_out": filter_last_cycle(
                        branch_data["pressure"], seg_end_index
                    ),
                    "x0": branch_data["points"][seg_start_index],
                    "x1": branch_data["points"][seg_end_index],
                }
            )

            seg_start = seg_end
            seg_start_index = seg_end_index

    times = times[start_last_cycle:-1] - np.amin(times[start_last_cycle])

    return branchdata, times


def extract_0d_element_coordinates(zerod_config, centerline):

    cl_handler = CenterlineHandler.from_file(centerline)

    elements = {}

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
        branchdata[branch_id][seg_id][
            "boundary_conditions"
        ] = vessel_config.get("boundary_conditions", {})

    for branch_id, branch in branchdata.items():

        branch_data = cl_handler.get_branch_data(branch_id)

        seg_start = 0.0
        seg_start_index = 0

        for seg_id in range(len(branch)):
            segment = branch[seg_id]
            length = segment["length"]

            seg_end_index = (
                np.abs(branch_data["path"] - length - seg_start)
            ).argmin()
            seg_end = branch_data["path"][seg_end_index]

            elements[f"branch{branch_id}_seg{seg_id}"] = {
                "x0": branch_data["points"][seg_start_index],
                "x1": branch_data["points"][seg_end_index],
            }

            if "inlet" in segment["boundary_conditions"]:
                elements[
                    segment["boundary_conditions"]["inlet"]
                ] = branch_data["points"][seg_start_index]
            if "outlet" in segment["boundary_conditions"]:
                elements[
                    segment["boundary_conditions"]["outlet"]
                ] = branch_data["points"][seg_end_index]

            seg_start = seg_end
            seg_start_index = seg_end_index

    return elements
