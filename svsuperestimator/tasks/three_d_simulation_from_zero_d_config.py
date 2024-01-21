"""This module holds the ThreeDSimulationFromZeroDConfig task."""
from __future__ import annotations

import os
from typing import Any

from .map_zero_d_result_to_three_d import MapZeroDResultToThreeD
from .model_calibration_least_squares import ModelCalibrationLeastSquares
from .task import Task
from .three_d_simulation import AdaptiveThreeDSimulation
from .windkessel_tuning import WindkesselTuning


class ThreeDSimulationFromZeroDConfig(Task):

    TASKNAME = "three_d_simulation_from_zero_d_config"
    DEFAULTS: dict[str, Any] = {
        "zerod_config_file": None,
        "num_procs": 1,
        "three_d_max_asymptotic_error": 0.01,
        "three_d_max_cardiac_cycles": 10,
        "three_d_time_step_size": "auto",
        "svpre_executable": None,
        "svsolver_executable": None,
        "svpost_executable": None,
        "svslicer_executable": None,
        WindkesselTuning.TASKNAME: {},
        MapZeroDResultToThreeD.TASKNAME: {},
        AdaptiveThreeDSimulation.TASKNAME: {},
        ModelCalibrationLeastSquares.TASKNAME: {},
        **Task.DEFAULTS,
    }

    def core_run(self) -> None:
        """Core routine of the task."""

        zerod_config_file = self.config["zerod_config_file"]

        task_sequence: list[Task] = []

        global_config = {
            "report_html": self.config["report_html"],
            "report_files": self.config["report_files"],
            "overwrite": self.config["overwrite"],
            "debug": self.config["debug"],
            "core_run": self.config["core_run"],
            "post_proc": self.config["post_proc"],
        }

        map_zero_three_task = MapZeroDResultToThreeD(
            project=self.project,
            config={
                "zerod_config_file": zerod_config_file,
                **global_config,
                **self.config[MapZeroDResultToThreeD.TASKNAME],
            },
            parent_folder=self.output_folder,
        )
        task_sequence.append(map_zero_three_task)
        three_d_sim_task = AdaptiveThreeDSimulation(
            project=self.project,
            config={
                "num_procs": self.config["num_procs"],
                "rcrt_dat_path": os.path.join(
                    map_zero_three_task.output_folder, "rcrt.dat"
                ),
                "initial_vtu_path": os.path.join(
                    map_zero_three_task.output_folder, "initial.vtu"
                ),
                "zerod_config_file": zerod_config_file,
                "max_asymptotic_error": self.config[
                    "three_d_max_asymptotic_error"
                ],
                "max_cardiac_cycles": self.config[
                    "three_d_max_cardiac_cycles"
                ],
                "time_step_size": self.config["three_d_time_step_size"],
                "svpre_executable": self.config["svpre_executable"],
                "svsolver_executable": self.config["svsolver_executable"],
                "svpost_executable": self.config["svpost_executable"],
                "svslicer_executable": self.config["svslicer_executable"],
                **global_config,
                **self.config[AdaptiveThreeDSimulation.TASKNAME],
            },
            parent_folder=self.output_folder,
        )
        task_sequence.append(three_d_sim_task)

        for task in task_sequence:
            task.run()
