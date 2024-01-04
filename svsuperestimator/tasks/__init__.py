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
    from .map_zero_d_result_to_three_d import MapZeroDResultToThreeD
    from .model_calibration import ModelCalibration
    from .model_calibration_least_squares import ModelCalibrationLeastSquares
    from .multi_fidelity_tuning import MultiFidelityTuning
    from .three_d_simulation import AdaptiveThreeDSimulation
    from .windkessel_tuning import WindkesselTuning

    task_mapping = {
        ModelCalibration.TASKNAME: ModelCalibration,
        WindkesselTuning.TASKNAME: WindkesselTuning,
        MapZeroDResultToThreeD.TASKNAME: MapZeroDResultToThreeD,
        AdaptiveThreeDSimulation.TASKNAME: AdaptiveThreeDSimulation,
        MultiFidelityTuning.TASKNAME: MultiFidelityTuning,
        ModelCalibrationLeastSquares.TASKNAME: ModelCalibrationLeastSquares
    }
    return task_mapping[name]
