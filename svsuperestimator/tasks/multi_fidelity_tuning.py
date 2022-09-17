from __future__ import annotations

import os

from .task import Task
from .windkessel_tuning import WindkesselTuning
from .map_zero_d_result_to_three_d import MapZeroDResultToThreeD
from .three_d_simulation import ThreeDSimulation
from .map_three_d_result_on_centerline import MapThreeDResultOnCenterline
from .blood_vessel_tuning import BloodVesselTuning


class MultiFidelityTuning(Task):
    """Map 3D result on centerline task."""

    TASKNAME = "multi_fidelity_tuning"
    DEFAULTS = {
        "num_procs": 1,
        "num_iter": 1,
        "smc_num_particles": 100,
        "smc_num_rejuvenation_steps": 2,
        "smc_resampling_threshold": 0.5,
        "smc_noise_factor": 0.05,
        "num_cardiac_cycles_3d": 2,
        "svpre_executable": None,
        "svsolver_executable": None,
        "svpost_executable": None,
        "slicer_executable": None,
        **Task.DEFAULTS,
    }

    def core_run(self):
        """Core routine of the task."""

        zerod_config_file = self.project["0d_simulation_input"]

        task_sequence = []

        for i in range(self.config["num_iter"]):
            suffix = f"_{i}"
            windkessel_task = WindkesselTuning(
                project=self.project,
                config={
                    "zerod_config_file": zerod_config_file,
                    "num_procs": self.config["num_procs"],
                    "num_particles": self.config["smc_num_particles"],
                    "num_rejuvenation_steps": self.config[
                        "smc_num_rejuvenation_steps"
                    ],
                    "resampling_threshold": self.config[
                        "smc_resampling_threshold"
                    ],
                    "noise_factor": self.config["smc_noise_factor"],
                },
                suffix=suffix,
            )
            task_sequence.append(windkessel_task)
            map_zero_three_task = MapZeroDResultToThreeD(
                project=self.project,
                config={
                    "zerod_config_file": os.path.join(
                        windkessel_task.output_folder, "solver_0d_map.in"
                    )
                },
                suffix=suffix,
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
                    "num_cardiac_cycles": self.config["num_cardiac_cycles_3d"]
                },
                suffix=suffix,
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
                },
                suffix=suffix,
            )
            task_sequence.append(map_three_zero_task)
            bv_tuning_task = BloodVesselTuning(
                project=self.project,
                config={
                    "zerod_config_file": zerod_config_file,
                    "threed_solution_file": os.path.join(
                        map_three_zero_task.output_folder,
                        "result_mapped_on_centerline.vtp",
                    ),
                    "num_procs": self.config["num_procs"],
                },
                suffix=suffix,
            )
            task_sequence.append(bv_tuning_task)
            zerod_config_file = os.path.join(
                bv_tuning_task.output_folder, "solver_0d.in"
            )

        for task in task_sequence:
            task.run()

    def post_run(self):
        """Postprocessing routine of the task."""
        pass

    def generate_report(self):
        """Generate the task report."""
        pass
