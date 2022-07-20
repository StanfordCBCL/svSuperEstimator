"""svSuperEstimator's problem subpackage.

Contains the different optimization problems.
"""
from ._windkessel_smc_chopin import WindkesselSMCChopin
import os


__all__ = ["WindkesselSMCChopin"]

_problem_mapping = {WindkesselSMCChopin.PROBLEM_NAME: WindkesselSMCChopin}

VALID_PROBLEMS = list(_problem_mapping.keys())


def get_problem_by_name(name):
    return _problem_mapping[name]


def get_problem_by_run_name(project, run_name):
    with open(
        os.path.join(project["rom_optimization_folder"], run_name, "case.txt")
    ) as ff:
        case_id = ff.read()
    return _problem_mapping[case_id]
