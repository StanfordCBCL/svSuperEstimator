"""This module holds the MapThreeDResultOnCenterline task."""
from __future__ import annotations

import os

from .task import Task
from .taskutils import run_subprocess


class MapThreeDResultOnCenterline(Task):
    """Map 3D result on centerline task."""

    TASKNAME = "map_three_d_result_on_centerline"
    DEFAULTS = {
        "num_procs": 1,
        "slicer_executable": None,
        "threed_result_file": None,
        **Task.DEFAULTS,
    }

    MUST_EXIST_AT_INIT = ["slicer_executable"]

    def core_run(self) -> None:
        """Core routine of the task."""

        centerline_file = self.project["centerline_path"]

        target = os.path.join(
            self.output_folder, "result_mapped_on_centerline.vtp"
        )

        self.log("Starting slicer")
        run_subprocess(
            [
                f"OMP_NUM_THREADS={self.config['num_procs']}",
                self.config["slicer_executable"],
                self.config["threed_result_file"],
                centerline_file,
                target,
            ],
            logger=self.log,
            logprefix=r"\[slicer] ",
            cwd=self.output_folder,
        )
