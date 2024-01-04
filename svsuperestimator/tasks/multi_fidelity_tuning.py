"""This module holds the MultiFidelityTuning task."""
from __future__ import annotations

import os
from typing import Any

from .map_zero_d_result_to_three_d import MapZeroDResultToThreeD
from .model_calibration_least_squares import ModelCalibrationLeastSquares
from .task import Task
from .three_d_simulation import AdaptiveThreeDSimulation
from .windkessel_tuning import WindkesselTuning


class MultiFidelityTuning(Task):
    """Map 3D result on centerline task."""

    TASKNAME = "multi_fidelity_tuning"
    DEFAULTS: dict[str, Any] = {
        "num_procs": 1,
        "theta_obs": None,
        "y_obs": None,
        "smc_num_particles": 100,
        "smc_num_rejuvenation_steps": 2,
        "smc_resampling_threshold": 0.5,
        "smc_noise_factor": 0.05,
        "smc_waste_free": True,
        "smc_kernel_density_estimation": False,
        "three_d_theta_source": "map",
        "three_d_max_asymptotic_error": 0.001,
        "three_d_max_cardiac_cycles": 10,
        "three_d_time_step_size": "auto",
        "lsq_calibrate_stenosis_coefficient": True,
        "lsq_initial_damping_factor": 1.0,
        "lsq_maximum_iterations": 100,
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

        if self.config["three_d_theta_source"] == "map":
            zero_d_input_file_name = "solver_0d_map.in"
        elif self.config["three_d_theta_source"] == "mean":
            zero_d_input_file_name = "solver_0d_mean.in"
        else:
            raise ValueError(
                "Invalid configuration for 'three_d_theta_source': "
                f"{self.config['three_d_theta_source']}"
            )

        for i in range(2):
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
                    **self.config[WindkesselTuning.TASKNAME],
                },
                prefix=f"{i*5}_",
                parent_folder=self.output_folder,
            )
            task_sequence.append(windkessel_task)
            if i >= 1:
                break
            map_zero_three_task = MapZeroDResultToThreeD(
                project=self.project,
                config={
                    "zerod_config_file": os.path.join(
                        windkessel_task.output_folder, zero_d_input_file_name
                    ),
                    **global_config,
                    **self.config[MapZeroDResultToThreeD.TASKNAME],
                },
                prefix=f"{i*5+1}_",
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
                    "zerod_config_file": os.path.join(
                        windkessel_task.output_folder, zero_d_input_file_name
                    ),
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
                prefix=f"{i*5+2}_",
                parent_folder=self.output_folder,
            )
            task_sequence.append(three_d_sim_task)
            bv_tuning_task = ModelCalibrationLeastSquares(
                project=self.project,
                config={
                    "zerod_config_file": os.path.join(
                        windkessel_task.output_folder, zero_d_input_file_name
                    ),
                    "threed_solution_file": os.path.join(
                        three_d_sim_task.output_folder,
                        "result.vtp",
                    ),
                    "num_procs": self.config["num_procs"],
                    "calibrate_stenosis_coefficient": self.config["lsq_calibrate_stenosis_coefficient"],
                    "initial_damping_factor": self.config["lsq_initial_damping_factor"],
                    "maximum_iterations": self.config["lsq_maximum_iterations"],
                    **global_config,
                    **self.config[ModelCalibrationLeastSquares.TASKNAME],
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
