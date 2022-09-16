from __future__ import annotations

import os
from shutil import copytree, copy2, ignore_patterns

from .taskutils import run_subprocess

from .task import Task


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
        **Task.DEFAULTS,
    }

    def core_run(self):
        """Core routine of the task."""

        sim_folder_path = self.project["3d_simulation_folder_path"]

        self.log("Collect mesh-complete")
        source = os.path.join(sim_folder_path, "mesh-complete")
        target = os.path.join(self.output_folder, "mesh-complete")
        copytree(source, target, ignore=ignore_patterns("initial.vtu"), dirs_exist_ok=True)

        self.log("Collect solver.inp")
        source = os.path.join(sim_folder_path, "solver.inp")
        target = os.path.join(self.output_folder, "solver.inp")
        copy2(source, target)

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
                f"OMP_NUM_THREADS={self.config['num_procs']}",
                self.config["svpre_executable"],
                os.path.join(self.output_folder, f"{self.project.name}.svpre"),
            ],
            logger=self.log,
            logprefix="\[svpre]: ",
        )

        self.log("Calling svsolver")
        run_subprocess(
            [
                f"OMP_NUM_THREADS={self.config['num_procs']}",
                self.config["svsolver_executable"],
                os.path.join(self.output_folder, "solver.inp"),
            ],
            logger=self.log,
            logprefix="\[svsolver]: ",
        )

        self.log("Calling svpost")
        run_subprocess(
            [
                f"OMP_NUM_THREADS={self.config['num_procs']}",
                self.config["svpost_executable"],
                os.path.join(self.output_folder, "solver.inp"),
            ],
            logger=self.log,
            logprefix="\[svpost]: ",
        )

    def post_run(self):
        """Postprocessing routine of the task."""

        pass

    def generate_report(self):
        """Generate the task report."""

        pass
