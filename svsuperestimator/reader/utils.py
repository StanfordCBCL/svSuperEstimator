"""This module contains input/output utils."""
import numpy as np

from ._svproject import SimVascularProject


def get_0d_element_coordinates(project: SimVascularProject):
    """Extract 0D elements with coordinates from a project.

    Args:
        project: SimVascular project.
    """
    zerod_handler = project["0d_simulation_input"]
    cl_handler = project["centerline"]

    elements = {}

    # Extract branch information of 0D config
    branchdata = {}
    for vessel_config in zerod_handler.vessels:

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
