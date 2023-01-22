"""svSuperEstimator's task subpackage.

Contains all the tasks for performing multi-fidelity parameter estimation.
"""
from typing import Type

from .task import Task


def get_task_by_name(name: str) -> Type[Task]:
    """Get the task class by it's name.

    Args:
        name: Name of the task.

    Returns:
        cls: Class of the task.
    """
    from .map_three_d_result_on_centerline import MapThreeDResultOnCenterline
    from .map_zero_d_result_to_three_d import MapZeroDResultToThreeD
    from .model_calibration import ModelCalibration
    from .multi_fidelity_tuning import MultiFidelityTuning
    from .three_d_simulation import ThreeDSimulation
    from .windkessel_tuning import WindkesselTuning

    task_mapping = {
        ModelCalibration.TASKNAME: ModelCalibration,
        WindkesselTuning.TASKNAME: WindkesselTuning,
        MapZeroDResultToThreeD.TASKNAME: MapZeroDResultToThreeD,
        ThreeDSimulation.TASKNAME: ThreeDSimulation,
        MapThreeDResultOnCenterline.TASKNAME: MapThreeDResultOnCenterline,
        MultiFidelityTuning.TASKNAME: MultiFidelityTuning,
    }
    return task_mapping[name]
