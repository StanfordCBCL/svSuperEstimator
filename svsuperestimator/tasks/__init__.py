"""svSuperEstimator's problem subpackage.

Contains the different optimization problems.
"""
from .blood_vessel_tuning import BloodVesselTuning
from .windkessel_tuning import WindkesselTuning

__all__ = ["BloodVesselTuning"]

_task_mapping = {
    BloodVesselTuning.TASKNAME: BloodVesselTuning,
    WindkesselTuning.TASKNAME: WindkesselTuning,
}

VALID_TASKS = list(_task_mapping.keys())


def get_task_by_name(name):
    return _task_mapping[name]
