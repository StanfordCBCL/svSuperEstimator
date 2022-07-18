from svsuperestimator import model as mdl, solver as slv, io
import numpy as np

svproject = "/Users/stanford/svSuperEstimator/tmpfiles/0069_0001"
project = io.SimVascularProject(svproject)
model = mdl.MultiFidelityModel(project)
solver = slv.ZeroDSolver()
from rich import print

outlet_bcs = {
    name: bc
    for name, bc in model.zerodmodel.boundary_conditions.items()
    if name != "INFLOW"
}

r_i = [
    bc.resistance_distal + bc.resistance_proximal for bc in outlet_bcs.values()
]  # Internal resistance for each outlet

bc_map = {}
to_drop = []
for branch_id, vessel_data in enumerate(model.zerodmodel._config["vessels"]):
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

bc_names = [bc_map[bc]["name"] for bc in outlet_bcs]
bc_pressure = [bc_map[bc]["pressure"] for bc in outlet_bcs]
bc_flow = [bc_map[bc]["flow"] for bc in outlet_bcs]
inflow_name = bc_map["INFLOW"]["name"]
inflow_pressure = bc_map["INFLOW"]["pressure"]


def objective_function(**kwargs):
    """Objective function for the optimization.

    Evaluates the sum of the offset for the input output pressure relation
    for each outlet.
    """
    # Set the resistance based on k
    for i, bc in enumerate(outlet_bcs.values()):
        ki = np.exp(kwargs[f"k{i}"])
        bc.resistance_proximal = r_i[i] * 1 / (1.0 + ki)
        bc.resistance_distal = bc.resistance_proximal * ki

    try:
        result = solver.run_simulation(model.zerodmodel, True)
    except RuntimeError:
        return np.array([np.nan] * len(outlet_bcs) * 2)

    p_inflow = result.loc[result.name == inflow_name][inflow_pressure].iloc[0]

    p_eval = [
        p_inflow - result.loc[result.name == name][pressure_id].iloc[0]
        for name, pressure_id in zip(bc_names, bc_pressure)
    ]
    q_eval = [
        result.loc[result.name == name][flow_id].iloc[0]
        for name, flow_id in zip(bc_names, bc_flow)
    ]
    return np.array(p_eval + q_eval)
