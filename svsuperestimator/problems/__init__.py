"""svSuperEstimator's problem subpackage.

Contains the different optimization problems.
"""
from ._windkessel_smc_chopin import WindkesselSMCChopin
from ._bivariant_windkessel_smc_chopin import BivariantWindkesselSMCChopin
import os
import json


__all__ = ["WindkesselSMCChopin"]

_problem_mapping = {
    WindkesselSMCChopin.PROBLEM_NAME: WindkesselSMCChopin,
    BivariantWindkesselSMCChopin.PROBLEM_NAME: BivariantWindkesselSMCChopin,
}

VALID_PROBLEMS = list(_problem_mapping.keys())


def get_problem_by_name(name):
    return _problem_mapping[name]


def get_problem_by_run_name(project, run_name):
    with open(
        os.path.join(
            project["rom_optimization_folder"], run_name, "parameters.json"
        )
    ) as ff:
        case_id = json.load(ff)["problem_type"]
    return _problem_mapping[case_id]
