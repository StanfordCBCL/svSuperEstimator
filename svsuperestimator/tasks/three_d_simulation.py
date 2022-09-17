from __future__ import annotations

import os
from shutil import copytree, copy2, ignore_patterns, rmtree
import numpy as np

from .taskutils import run_subprocess
from .. import reader

from .task import Task


class ThreeDSimulation(Task):
    """3D Simulation task."""

    TASKNAME = "three_d_simulation"
    DEFAULTS = {
        "num_procs": 1,
        "num_cardiac_cycles": 2,
        "rcrt_dat_path": None,
        "initial_vtu_path": None,
        "svpre_executable": None,
        "svsolver_executable": None,
        "svpost_executable": None,
        **Task.DEFAULTS,
    }

    def core_run(self):
        """Core routine of the task."""

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

        self.log("Collect solver.inp")
        input_handler = self.project["3d_simulation_input"]
        inflow_data = reader.SvSolverInflowHandler.from_file(
            os.path.join(self.output_folder, "inflow.flow")
        ).get_inflow_data()
        cardiac_cycle_period = inflow_data["t"][-1] - inflow_data["t"][0]
        time_step_size = input_handler.time_step_size
        steps_per_cycle = int(np.rint(cardiac_cycle_period / time_step_size))
        # TODO: Check if time step calculation is correct
        num_time_steps = steps_per_cycle * self.config["num_cardiac_cycles"]
        input_handler.num_time_steps = num_time_steps
        input_handler.to_file(os.path.join(self.output_folder, "solver.inp"))

        self.log("Collect inflow.flow")
        source = os.path.join(sim_folder_path, "inflow.flow")
        target = os.path.join(self.output_folder, "inflow.flow")
        copy2(source, target)

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

        self.log("Calling svpre")
        run_subprocess(
            [
                self.config["svpre_executable"],
                f"{self.project.name}.svpre",
            ],
            logger=self.log,
            logprefix="\[svpre]: ",
            cwd=self.output_folder,
        )

        self.log("Calling svsolver")
        run_subprocess(
            [
                "sh",
                self.config["svsolver_executable"],
                "solver.inp",
            ],
            logger=self.log,
            logprefix="\[svsolver]: ",
            cwd=self.output_folder,
        )

        self.log("Calling svpost")
        run_subprocess(
            [
                self.config["svpost_executable"],
                f"-start {num_time_steps-steps_per_cycle}",
                f"-stop {num_time_steps}",
                f"-incr {5}",
                "-sol -vtkcombo",
                "-vtu ../result.vtu",
            ],
            logger=self.log,
            logprefix="\[svpost]: ",
            cwd=os.path.join(
                self.output_folder, f"{self.config['num_procs']}-procs"
            ),
        )

        # self.log("Cleaning up")
        # rmtree(os.path.join(self.output_folder, f"{self.config['num_procs']}-procs"))

    def post_run(self):
        """Postprocessing routine of the task."""

        pass

    def generate_report(self):
        """Generate the task report."""

        pass
