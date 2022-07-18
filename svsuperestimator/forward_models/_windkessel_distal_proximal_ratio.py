"""This module holds the WindkesselDistalToProximalResistance0D class."""
from ._forward_model import ForwardModel
from svsuperestimator import model as mdl, solver as slv

import numpy as np


class WindkesselDistalToProximalResistance0D(ForwardModel):
    """Windkessel distal to proximal resistance 0D forward model.

    This forward model performs evaluations of a 0D model based on a
    given distal to proximal resistance ratio."""

    def __init__(self, model: mdl.ZeroDModel, solver: slv.ZeroDSolver) -> None:
        super().__init__(model, solver)

        self.outlet_bcs = {
            name: bc
            for name, bc in model.boundary_conditions.items()
            if name != "INFLOW"
        }

        self.r_i = [
            bc.resistance_distal + bc.resistance_proximal
            for bc in self.outlet_bcs.values()
        ]  # Internal resistance for each outlet

        self.bc_map = {}
        for branch_id, vessel_data in enumerate(model._config["vessels"]):
            if "boundary_conditions" in vessel_data:
                for bc_type, bc_name in vessel_data[
                    "boundary_conditions"
                ].items():
                    self.bc_map[bc_name] = {"name": f"V{branch_id}"}
                    if bc_type == "inlet":
                        self.bc_map[bc_name]["flow"] = "flow_in"
                        self.bc_map[bc_name]["pressure"] = "pressure_in"
                    else:
                        self.bc_map[bc_name]["flow"] = "flow_out"
                        self.bc_map[bc_name]["pressure"] = "pressure_out"

        self.bc_names = [self.bc_map[bc]["name"] for bc in self.outlet_bcs]
        self.bc_pressure = [
            self.bc_map[bc]["pressure"] for bc in self.outlet_bcs
        ]
        self.bc_flow = [self.bc_map[bc]["flow"] for bc in self.outlet_bcs]
        self.inflow_name = self.bc_map["INFLOW"]["name"]
        self.inflow_pressure = self.bc_map["INFLOW"]["pressure"]

    def evaluate(self, **kwargs):
        """Objective function for the optimization.

        Evaluates the sum of the offset for the input output pressure relation
        for each outlet.
        """
        # Set the resistance based on k
        for i, bc in enumerate(self.outlet_bcs.values()):
            ki = np.exp(kwargs[f"k{i}"])
            bc.resistance_proximal = self.r_i[i] * 1 / (1.0 + ki)
            bc.resistance_distal = bc.resistance_proximal * ki

        try:
            result = self.solver.run_simulation(self.model, True)
        except RuntimeError:
            return np.array([np.nan] * len(self.outlet_bcs) * 2)

        p_inflow = result.loc[result.name == self.inflow_name][
            self.inflow_pressure
        ].iloc[0]

        p_eval = [
            p_inflow - result.loc[result.name == name][pressure_id].iloc[0]
            for name, pressure_id in zip(self.bc_names, self.bc_pressure)
        ]
        q_eval = [
            result.loc[result.name == name][flow_id].iloc[0]
            for name, flow_id in zip(self.bc_names, self.bc_flow)
        ]
        return np.array(p_eval + q_eval)
