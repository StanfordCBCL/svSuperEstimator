"""svSuperEstimator's problem subpackage.

Contains the different optimization problems.
"""
from .blood_vessel_tuning import BloodVesselTuning

__all__ = ["BloodVesselTuning"]

_task_mapping = {
    BloodVesselTuning.TASKNAME: BloodVesselTuning,
}

VALID_TASKS = list(_task_mapping.keys())


def get_task_by_name(name):
    return _task_mapping[name]
