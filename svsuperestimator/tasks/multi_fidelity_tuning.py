"""This module holds the MultiFidelityTuning task."""
from __future__ import annotations

import os
from typing import Any

from .map_three_d_result_on_centerline import MapThreeDResultOnCenterline
from .map_zero_d_result_to_three_d import MapZeroDResultToThreeD
from .model_calibration import ModelCalibration
from .task import Task
from .three_d_simulation import ThreeDSimulation
from .windkessel_tuning import WindkesselTuning


class MultiFidelityTuning(Task):
    """Map 3D result on centerline task."""

    TASKNAME = "multi_fidelity_tuning"
    DEFAULTS: dict[str, Any] = {
        "num_procs": 1,
        "num_iter": 1,
        "theta_obs": None,
        "y_obs": None,
        "smc_num_particles": 100,
        "smc_num_rejuvenation_steps": 2,
        "smc_resampling_threshold": 0.5,
        "smc_noise_factor": 0.05,
        "smc_waste_free": True,
        "smc_kernel_density_estimation": False,
        "num_cardiac_cycles_3d": 2,
        "svpre_executable": None,
        "svsolver_executable": None,
        "svpost_executable": None,
        "slicer_executable": None,
        "WindkesselTuning": {},
        "MapZeroDResultToThreeD": {},
        "ThreeDSimulation": {},
        "MapThreeDResultOnCenterline": {},
        ModelCalibration.TASKNAME: {},
        **Task.DEFAULTS,
    }

    def core_run(self) -> None:
        """Core routine of the task."""

        zerod_config_file = self.project["0d_simulation_input_path"]

        task_sequence: list[Task] = []

        global_config = {
            "report_html": self.config["report_html"],
            "report_files": self.config["report_files"],
            "overwrite": self.config["overwrite"],
            "debug": self.config["debug"],
            "core_run": self.config["core_run"],
            "post_proc": self.config["post_proc"],
        }

        for i in range(self.config["num_iter"]):
            windkessel_task = WindkesselTuning(
                project=self.project,
                config={
                    "zerod_config_file": zerod_config_file,
                    "num_procs": self.config["num_procs"],
                    "theta_obs": self.config["theta_obs"],
                    "y_obs": self.config["y_obs"],
                    "num_particles": self.config["smc_num_particles"],
                    "num_rejuvenation_steps": self.config[
                        "smc_num_rejuvenation_steps"
                    ],
                    "resampling_threshold": self.config[
                        "smc_resampling_threshold"
                    ],
                    "noise_factor": self.config["smc_noise_factor"],
                    "waste_free": self.config["smc_waste_free"],
                    "kernel_density_estimation": self.config[
                        "smc_kernel_density_estimation"
                    ],
                    **global_config,
                    **self.config["WindkesselTuning"],
                },
                prefix=f"{i*5}_",
                parent_folder=self.output_folder,
            )
            task_sequence.append(windkessel_task)
            map_zero_three_task = MapZeroDResultToThreeD(
                project=self.project,
                config={
                    "zerod_config_file": os.path.join(
                        windkessel_task.output_folder, "solver_0d_map.in"
                    ),
                    **global_config,
                    **self.config["MapZeroDResultToThreeD"],
                },
                prefix=f"{i*5+1}_",
                parent_folder=self.output_folder,
            )
            task_sequence.append(map_zero_three_task)
            three_d_sim_task = ThreeDSimulation(
                project=self.project,
                config={
                    "num_procs": self.config["num_procs"],
                    "rcrt_dat_path": os.path.join(
                        map_zero_three_task.output_folder, "rcrt.dat"
                    ),
                    "initial_vtu_path": os.path.join(
                        map_zero_three_task.output_folder, "initial.vtu"
                    ),
                    "svpre_executable": self.config["svpre_executable"],
                    "svsolver_executable": self.config["svsolver_executable"],
                    "svpost_executable": self.config["svpost_executable"],
                    "num_cardiac_cycles": self.config["num_cardiac_cycles_3d"],
                    **global_config,
                    **self.config["ThreeDSimulation"],
                },
                prefix=f"{i*5+2}_",
                parent_folder=self.output_folder,
            )
            task_sequence.append(three_d_sim_task)
            map_three_zero_task = MapThreeDResultOnCenterline(
                project=self.project,
                config={
                    "num_procs": self.config["num_procs"],
                    "slicer_executable": self.config["slicer_executable"],
                    "threed_result_file": os.path.join(
                        three_d_sim_task.output_folder, "result.vtu"
                    ),
                    **global_config,
                    **self.config["MapThreeDResultOnCenterline"],
                },
                prefix=f"{i*5+3}_",
                parent_folder=self.output_folder,
            )
            task_sequence.append(map_three_zero_task)
            bv_tuning_task = ModelCalibration(
                project=self.project,
                config={
                    "zerod_config_file": os.path.join(
                        windkessel_task.output_folder, "solver_0d_map.in"
                    ),
                    "threed_solution_file": os.path.join(
                        map_three_zero_task.output_folder,
                        "result_mapped_on_centerline.vtp",
                    ),
                    "num_procs": self.config["num_procs"],
                    **global_config,
                    **self.config[ModelCalibration.TASKNAME],
                },
                prefix=f"{i*5+4}_",
                parent_folder=self.output_folder,
            )
            task_sequence.append(bv_tuning_task)
            zerod_config_file = os.path.join(
                bv_tuning_task.output_folder, "solver_0d.in"
            )

        for task in task_sequence:
            task.run()
