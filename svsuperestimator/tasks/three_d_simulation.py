"""This module holds the ThreeDSimulation task."""
from __future__ import annotations

import os
from shutil import copy2, copytree, ignore_patterns, rmtree

import numpy as np

from .. import reader, visualizer
from .task import Task
from .taskutils import run_subprocess


class ThreeDSimulation(Task):
    """3D Simulation task."""

    TASKNAME = "three_d_simulation"
    DEFAULTS = {
        "num_procs": 1,
        "rcrt_dat_path": None,
        "initial_vtu_path": None,
        "svpre_executable": None,
        "svsolver_executable": None,
        "svpost_executable": None,
        "svslicer_executable": None,
        **Task.DEFAULTS,
    }

    MUST_EXIST_AT_INIT = [
        "svpre_executable",
        "svsolver_executable",
        "svpost_executable",
        "svslicer_executable"
    ]

    def core_run(self) -> None:
        """Core routine of the task."""

        if os.path.exists(self._solver_output_folder):
            raise RuntimeError(f"Solver output folder already exists: {self._solver_output_folder}")

        self._setup_input_files()

        self._run_preprocessor()

        simulated_steps = 0
        i_cardiac_cycle = 0

        while True:
            i_cardiac_cycle += 1

            current_last_step = self._get_current_last_step()
            self.log(f"Current last time step: {current_last_step}")

            start_step = simulated_steps
            end_step = simulated_steps + self._steps_per_cycle

            self.log(
                f"Simulating cardiac cycle {i_cardiac_cycle} "
                f"(time step {start_step} to {end_step})"
            )

            self._run_solver()

            three_d_result_file = os.path.join(
                self.output_folder, f"result_cycle_{i_cardiac_cycle}.vtu"
            )
            self._run_postprocessor(start_step, end_step, three_d_result_file)

            centerline_result_file = os.path.join(
                self.output_folder, f"result_cycle_{i_cardiac_cycle}.vtp"
            )
            self._run_slicer(three_d_result_file, centerline_result_file)

            simulated_steps = end_step

    def post_run(self) -> None:
        """Postprocessing routine of the task."""

        pass

    def generate_report(self) -> visualizer.Report:
        """Generate the task report."""

        return visualizer.Report()

    def _setup_input_files(self):
        sim_folder_path = self.project["3d_simulation_folder_path"]

        self.log("Collect mesh-complete")
        source = os.path.join(sim_folder_path, "mesh-complete")
        target = os.path.join(self.output_folder, "mesh-complete")
        copytree(
            source,
            target,
            ignore=ignore_patterns("initial.vtu"),
            dirs_exist_ok=True,
        )

        self.log("Collect inflow.flow")
        source = os.path.join(sim_folder_path, "inflow.flow")
        target = os.path.join(self.output_folder, "inflow.flow")
        copy2(source, target)

        self.log("Collect solver.inp")
        input_handler = self.project["3d_simulation_input"]
        inflow_data = reader.SvSolverInflowHandler.from_file(
            os.path.join(self.output_folder, "inflow.flow")
        ).get_inflow_data()
        self._cardiac_cycle_period = inflow_data["t"][-1] - inflow_data["t"][0]
        self.log(f"Found cardiac cycle period {self._cardiac_cycle_period} s ")
        time_step_size = input_handler.time_step_size
        self.log(f"Found 3D simulation time step size {time_step_size} s ")
        self._steps_per_cycle = int(
            np.rint(self._cardiac_cycle_period / time_step_size)
        )
        self.log(
            "Set number of 3D simulation time steps "
            f"to {self._steps_per_cycle} s"
        )
        input_handler.num_time_steps = self._steps_per_cycle
        input_handler.to_file(os.path.join(self.output_folder, "solver.inp"))

        self.log(f"Collect {self.project.name}.svpre")
        source = os.path.join(sim_folder_path, f"{self.project.name}.svpre")
        target = os.path.join(self.output_folder, f"{self.project.name}.svpre")
        copy2(source, target)

        self.log("Collect rcrt.dat")
        target = os.path.join(self.output_folder, "rcrt.dat")
        copy2(self.config["rcrt_dat_path"], target)

        self.log("Collect initial.vtu")
        target = os.path.join(
            self.output_folder, "mesh-complete", "initial.vtu"
        )
        copy2(self.config["initial_vtu_path"], target)

    def _run_preprocessor(self):
        self.log("Running preprocessor")
        run_subprocess(
            [
                self.config["svpre_executable"],
                f"{self.project.name}.svpre",
            ],
            logger=self.log,
            logprefix=r"\[svpre]: ",
            cwd=self.output_folder,
        )

    def _run_solver(self):
        self.log(f"Running solver for {self._steps_per_cycle} time steps")
        run_subprocess(
            [
                "UCX_POSIX_USE_PROC_LINK=n srun",
                self.config["svsolver_executable"],
                "solver.inp",
            ],
            logger=self.log,
            logprefix=r"\[svsolver]: ",
            cwd=self.output_folder,
        )

    def _run_postprocessor(
        self,
        start_step: int,
        stop_step: int,
        output_file: str,
        interval: int = 5,
    ):
        self.log(
            f"Postprocessing steps {start_step} to {stop_step} "
            f"(interval {interval})"
        )
        run_subprocess(
            [
                self.config["svpost_executable"],
                f"-start {start_step}",
                f"-stop {stop_step}",
                f"-incr {interval}",
                "-sol -vtkcombo",
                f"-vtu {output_file}",
            ],
            logger=self.log,
            logprefix=r"\[svpost]: ",
            cwd=self._solver_output_folder,
        )
        self.log(f"Saved postprocessed output file: {output_file}")
        self.log("Completed postprocessing")

    def _run_slicer(
        self, three_d_result_file: str, centerline_result_file: str
    ):
        centerline_file = self.project["centerline_path"]
        self.log(f"Slicing 3D output file {three_d_result_file}")
        run_subprocess(
            [
                f"OMP_NUM_THREADS={self.config['num_procs']}",
                self.config["svslicer_executable"],
                three_d_result_file,
                centerline_file,
                centerline_result_file,
            ],
            logger=self.log,
            logprefix=r"\[svslicer] ",
            cwd=self.output_folder,
        )
        self.log(f"Saved centerline output file: {centerline_result_file}")
        self.log("Completed slicing")

    @property
    def _solver_output_folder(self):
        return os.path.join(
            self.output_folder, f"{self.config['num_procs']}-procs_case"
        )

    def _get_current_last_step(self):
        if not os.path.exists(self._solver_output_folder):
            return 0
        restart_files = [
            f
            for f in os.listdir(self._solver_output_folder)
            if f.startswith("restart")
        ]
        if len(restart_files) == 0:
            return 0
        return np.max([f.split(".")[1] for f in restart_files])
