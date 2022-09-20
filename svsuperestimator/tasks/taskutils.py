import subprocess
from time import sleep
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline


def cgs_pressure_to_mmgh(cgs_pressure):
    """Convert pressure from g/(cm s^2) to mmHg.

    Args:
        cgs_pressure: Pressure in CGS format.

    Returns:
        Pressure in mmHg.
    """
    return np.array(np.array(cgs_pressure) * 0.00075006156130264)


def cgs_flow_to_lh(cgs_flow):
    """Convert flow from cm^3/s to l/h.

    Args:
        cgs_flow: Flow in CGS format.

    Returns:
        Flow in l/h.
    """
    return np.array(np.array(cgs_flow) * 3.6)


def cgs_flow_to_lmin(cgs_flow):
    """Convert flow from cm^3/s to l/min.

    Args:
        cgs_flow: Flow in CGS format.

    Returns:
        Flow in l/h.
    """
    return np.array(np.array(cgs_flow) * 60.0 / 1000.0)


def refine_with_cubic_spline(y: np.ndarray, num: np.ndarray):
    """Refine a curve using cubic spline interpolation.

    Args:
        y: The data to refine.
        num: New number of points of the refined data.
    """
    y = y.copy()
    y[-1] = y[0]
    x_old = np.linspace(0.0, 100.0, len(y))
    x_new = np.linspace(0.0, 100.0, num)
    y_new = CubicSpline(x_old, y, bc_type="periodic")(x_new)
    return y_new


def run_subprocess(
    args: list,
    logger: Callable,
    refresh_rate=1.0,
    logprefix: str = "",
    cwd=None,
) -> None:
    """Run a subprocess.

    Args:
        args: Arguments for the subprocess.
        logger: Logger to use for logging.
        refresh_rate: Rate to update logging (in seconds).
        logprefix: Prefix to put in front of lines when logging.
        cwd: Working directory of the subprocess.
    """
    process = subprocess.Popen(
        " ".join(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        cwd=cwd,
    )

    def check_io():
        while True:
            output = process.stdout.readline().decode().strip()
            if output:
                logger(logprefix + output)
            else:
                break

    while process.poll() is None:
        check_io()
        sleep(refresh_rate)

    if process.returncode != 0:
        raise RuntimeError("Subprocess failed")


def map_centerline_result_to_0d(zerod_handler, centerline_handler, dt3d):
    """Map centerine result onto 0d elements."""

    cl_handler = centerline_handler

    # calculate cycle period
    cycle_period = (
        zerod_handler.boundary_conditions["INFLOW"]["bc_values"]["t"][-1]
        - zerod_handler.boundary_conditions["INFLOW"]["bc_values"]["t"][0]
    )

    # Extract time steps
    times = centerline_handler.time_steps * dt3d

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
    for vessel_config in zerod_handler.vessels.values():

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

        for seg_id, segment in branch.items():
            segment = branch[seg_id]
            length = segment["length"]

            seg_end_index = (
                np.abs(branch_data["path"] - length - seg_start)
            ).argmin()
            seg_end = branch_data["path"][seg_end_index]

            segment.update(
                {
                    "flow_in": filter_last_cycle(
                        branch_data["flow"], seg_start_index
                    ),
                    "flow_out": filter_last_cycle(
                        branch_data["flow"], seg_end_index
                    ),
                    "pressure_in": filter_last_cycle(
                        branch_data["pressure"], seg_start_index
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

    if start_last_cycle == -1:
        times -= times[0]
    else:
        times = times[start_last_cycle:-1] - np.amin(times[start_last_cycle])

    return branchdata, times


def map_centerline_result_to_0d_2(
    zerod_handler, cl_handler, threed_handler, results_handler
):
    """Map centerine result onto 0d elements."""

    # calculate cycle period
    cycle_period = (
        zerod_handler.boundary_conditions["INFLOW"]["bc_values"]["t"][-1]
        - zerod_handler.boundary_conditions["INFLOW"]["bc_values"]["t"][0]
    )

    # Extract time steps
    times = results_handler.time_steps * threed_handler.time_step_size

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
    for vessel_config in zerod_handler.vessels.values():

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

    results_branch_starts = {}
    for branch_id, branch in branchdata.items():
        results_branch_starts[branch_id] = results_handler.get_branch_data(
            branch_id
        )["points"][0]

    keys = list(results_branch_starts.keys())
    starts = np.array(list(results_branch_starts.values()))

    for branch_id, branch in branchdata.items():

        cl_data = cl_handler.get_branch_data(branch_id)
        start = cl_data["points"][0]

        new_id = keys[np.argmin(np.linalg.norm(starts - start, axis=1))]

        branch_data = results_handler.get_branch_data(new_id)

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
                        branch_data["flow"], seg_start_index
                    ),
                    "flow_out": filter_last_cycle(
                        branch_data["flow"], seg_end_index
                    ),
                    "pressure_in": filter_last_cycle(
                        branch_data["pressure"], seg_start_index
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

    if start_last_cycle == -1:
        times -= times[0]
    else:
        times = times[start_last_cycle:-1] - np.amin(times[start_last_cycle])

    return branchdata, times


def set_initial_condition(zerod_handler, mapped_data):

    nodes = zerod_handler.nodes

    bcs = zerod_handler.boundary_conditions
    vessels = zerod_handler.vessels

    initial_condition = {}
    for ele1, ele2 in nodes:
        try:
            branch_id, seg_id = ele1.split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])
            pressure = mapped_data[branch_id][seg_id]["pressure_out"][0]
            flow = mapped_data[branch_id][seg_id]["flow_out"][0]
            initial_condition[f"pressure:{ele1}:{ele2}"] = pressure
            initial_condition[f"flow:{ele1}:{ele2}"] = flow
        except:
            pass

        try:
            branch_id, seg_id = ele2.split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])
            pressure = mapped_data[branch_id][seg_id]["pressure_in"][0]
            flow = mapped_data[branch_id][seg_id]["flow_in"][0]
            initial_condition[f"pressure:{ele1}:{ele2}"] = pressure
            initial_condition[f"flow:{ele1}:{ele2}"] = flow
        except:
            pass

        if ele2.startswith("RCR"):
            initial_condition[f"pressure_c:{ele2}"] = (
                initial_condition[f"pressure:{ele1}:{ele2}"]
                - bcs[ele2]["bc_values"]["Rp"]
                * initial_condition[f"flow:{ele1}:{ele2}"]
            )

        if ele2.startswith("branch"):
            vessel_params = vessels[ele2]["zero_d_element_values"]
            initial_condition[f"pressure_c:{ele2}"] = (
                initial_condition[f"pressure:{ele1}:{ele2}"]
                - (
                    vessel_params["R_poiseuille"]
                    + vessel_params["stenosis_coefficient"]
                    * abs(initial_condition[f"flow:{ele1}:{ele2}"])
                )
                * initial_condition[f"flow:{ele1}:{ele2}"]
            )

    for junction_name, junction in zerod_handler.junctions.items():
        if junction["junction_type"] == "resistive_junction":

            for node in nodes:
                if node[1] == junction_name:
                    branch_id, seg_id = node[0].split("_")
                    branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])
                    initial_condition[
                        f"pressure_c:{node[1]}"
                    ] = initial_condition[f"pressure:{node[0]}:{node[1]}"]

    zerod_handler.data["initial_condition"] = initial_condition
