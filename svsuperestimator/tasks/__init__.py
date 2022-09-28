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
    from .blood_vessel_tuning import BloodVesselTuning
    from .map_three_d_result_on_centerline import MapThreeDResultOnCenterline
    from .map_zero_d_result_to_three_d import MapZeroDResultToThreeD
    from .multi_fidelity_tuning import MultiFidelityTuning
    from .three_d_simulation import ThreeDSimulation
    from .windkessel_tuning import WindkesselTuning

    task_mapping = {
        BloodVesselTuning.__name__: BloodVesselTuning,
        WindkesselTuning.__name__: WindkesselTuning,
        MapZeroDResultToThreeD.__name__: MapZeroDResultToThreeD,
        ThreeDSimulation.__name__: ThreeDSimulation,
        MapThreeDResultOnCenterline.__name__: MapThreeDResultOnCenterline,
        MultiFidelityTuning.__name__: MultiFidelityTuning,
    }
    return task_mapping[name]
