"""This module holds the ThreeDSimulation task."""
from __future__ import annotations

import os
from shutil import copy2, copytree, ignore_patterns
import multiprocessing

import numpy as np
from svzerodsolver import runnercpp

from .. import reader, visualizer
from ..reader import CenterlineHandler, SvZeroDSolverInputHandler
from .task import Task
from .taskutils import (
    map_centerline_result_to_0d_2,
    refine_with_cubic_spline,
    run_subprocess,
)


class AdaptiveThreeDSimulation(Task):
    """3D Simulation task."""

    TASKNAME = "adaptive_three_d_simulation"
    DEFAULTS = {
        "num_procs": 1,
        "rcrt_dat_path": None,
        "initial_vtu_path": None,
        "zerod_config_file": None,
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
        "svslicer_executable",
    ]

    def core_run(self) -> None:
        """Core routine of the task."""

        if os.path.exists(self._solver_output_folder):
            raise RuntimeError(
                "Solver output folder already exists: "
                f"{self._solver_output_folder}"
            )

        # Setup all input files
        self._setup_input_files()

        # Preprocess the input files
        self._run_preprocessor()

        simulated_steps = 0
        i_cardiac_cycle = 0
        self.database["asymptotic_errors"] = []
        self.database["max_asymptotic_errors"] = []

        while True:
            i_cardiac_cycle += 1

            start_step = simulated_steps
            end_step = simulated_steps + self._steps_per_cycle

            self.log(
                f"Simulating cardiac cycle {i_cardiac_cycle} "
                f"(time step {start_step} to {end_step})"
            )

            # Run the simulation
            self._run_solver()
            simulated_steps = end_step

            # Run the postprocessing
            three_d_result_file = os.path.join(
                self.output_folder, f"result_cycle_{i_cardiac_cycle}.vtu"
            )
            self._run_postprocessor(start_step, end_step, three_d_result_file)

            # Map the postprocessed results to the centerline
            centerline_result_file = os.path.join(
                self.output_folder, f"result_cycle_{i_cardiac_cycle}.vtp"
            )
            self._run_slicer(three_d_result_file, centerline_result_file)

            # Check if simulation is periodic
            (
                max_symptotic_error,
                asymptotic_errors_dict,
            ) = self._calculate_max_asymptotic_error(
                centerline_result_file, self.config["zerod_config_file"]
            )
            self.database["asymptotic_errors"].append(asymptotic_errors_dict)
            self.database["max_asymptotic_errors"].append(max_symptotic_error)
            if max_symptotic_error <= 0.01:
                break

        self.log(
            f"Periodic state reached after {i_cardiac_cycle} cardiac cycles."
        )
        centerline_result_file_final = os.path.join(
            self.output_folder, "result.vtp"
        )
        copy2(centerline_result_file, centerline_result_file_final)

    def post_run(self) -> None:
        """Postprocessing routine of the task."""

        pass

    def generate_report(self) -> visualizer.Report:
        """Generate the task report."""

        return visualizer.Report()

    def _setup_input_files(self) -> None:
        """Generate all input files for the 3D fluid simulation."""

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
        self.input_handler = self.project["3d_simulation_input"]
        inflow_data = reader.SvSolverInflowHandler.from_file(
            os.path.join(self.output_folder, "inflow.flow")
        ).get_inflow_data()
        self._cardiac_cycle_period = inflow_data["t"][-1] - inflow_data["t"][0]
        self.database["cardiac_cycle_period"] = self._cardiac_cycle_period
        self.log(f"Found cardiac cycle period {self._cardiac_cycle_period} s ")
        time_step_size = self.input_handler.time_step_size
        self.database["time_step_size"] = time_step_size
        self.log(f"Found 3D simulation time step size {time_step_size} s ")
        self._steps_per_cycle = int(
            np.rint(self._cardiac_cycle_period / time_step_size)
        )
        self.database["steps_per_cycle"] = self._steps_per_cycle
        self.log(
            "Set number of 3D simulation time steps "
            f"to {self._steps_per_cycle} s"
        )
        self.input_handler.num_time_steps = self._steps_per_cycle
        self.input_handler.to_file(
            os.path.join(self.output_folder, "solver.inp")
        )

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

    def _run_preprocessor(self) -> None:
        """Run svPre."""
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

    def _run_solver(self) -> None:
        """Run the 3D fluid dynamics simulation using svSolver."""
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
    ) -> None:
        """Postprocess the raw output files to a single vtu file."""
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
    ) -> None:
        """Run svSlicer to map the volumetric 3D results on the centerline."""
        centerline_file = self.project["centerline_path"]
        self.log(f"Slicing 3D output file {three_d_result_file}")
        run_subprocess(
            [
                f"OMP_NUM_THREADS={min(self.config['num_procs'], multiprocessing.cpu_count())}",
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
    def _solver_output_folder(self) -> str:
        """The folder where the solver saves the raw output files in."""
        return os.path.join(
            self.output_folder, f"{self.config['num_procs']}-procs_case"
        )

    def _calculate_max_asymptotic_error(
        self, current_centerline_result: str, zero_d_input_file: str
    ) -> tuple[float, dict]:
        """Determine the maximum asymptotic error for each Windkessel outlet.

        Calculates the asymptotic error of each Windkessel outlet according to
        Pfaller et al. "On the Periodicity of Cardiovascular Fluid Dynamics
        Simulations". This is a measure for determining whether a 3D
        simulation has reached a periodic state.
        """
        zerod_input_handler = SvZeroDSolverInputHandler.from_file(
            zero_d_input_file
        )
        centerline = self.project["centerline"]
        cl_handler_current = CenterlineHandler.from_file(
            current_centerline_result
        )

        # Map centerline 3D result to the 0D elements (helps to extract
        # pressure and flow values at the outlets)
        mapped_data_current, _ = map_centerline_result_to_0d_2(
            zerod_input_handler,
            centerline,
            self.input_handler,
            cl_handler_current,
        )

        zerod_boundary_conditions = zerod_input_handler.boundary_conditions
        zerod_pts_per_cycle = zerod_input_handler.num_pts_per_cycle

        zero_d_times = np.linspace(
            zerod_boundary_conditions["INFLOW"]["bc_values"]["t"][0],
            zerod_boundary_conditions["INFLOW"]["bc_values"]["t"][-1],
            zerod_pts_per_cycle,
        )

        asymptotic_errors = []
        asymptotic_errors_dict = {}

        for (
            bc_name,
            bc_details,
        ) in zerod_input_handler.vessel_to_bc_map.items():
            if bc_name == "INFLOW":
                continue

            branch_id, seg_id = bc_details["name"].split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

            # Extract pressure and flow at the Windkessel boundary condition
            three_d_pressure = mapped_data_current[branch_id][seg_id][
                bc_details["pressure"]
            ]
            three_d_flow = mapped_data_current[branch_id][seg_id][
                bc_details["flow"]
            ]

            # Refine the pressure and flow curve with cubic splines
            three_d_pressure_refined = refine_with_cubic_spline(
                three_d_pressure, zerod_pts_per_cycle
            )
            three_d_flow_refined = refine_with_cubic_spline(
                three_d_flow, zerod_pts_per_cycle
            )

            # Simulate a single Windkessel with the same parameters. Prescribe
            # the 3D flow at the inlet to the Windkessel and simulate the
            # corresponding pressure
            zerod_pressure = self._simulate_windkessel_bc(
                zerod_boundary_conditions[bc_name]["bc_values"],
                zero_d_times,
                three_d_flow_refined,
                zerod_input_handler.num_cycles,
                zerod_pts_per_cycle,
            )

            # Compare the simulated pressure with the actual 3D pressure to
            # compute the asymptotic error
            asymptotic_error = (
                np.mean(three_d_pressure_refined) - np.mean(zerod_pressure)
            ) / np.mean(zerod_pressure)

            self.log(
                f"Asymptotic error for {bc_name}: {asymptotic_error*100:.2f} %"
            )
            asymptotic_errors.append(asymptotic_error)
            asymptotic_errors_dict[bc_name] = asymptotic_error

        # Return the maximum asymptotic error
        max_asymptotic_error = np.max(asymptotic_errors)

        self.log(f"Maximum asymptotic error: {max_asymptotic_error*100:.2f} %")

        return max_asymptotic_error, asymptotic_errors_dict

    @classmethod
    def _simulate_windkessel_bc(
        cls,
        bc_values: dict,
        bc_times: np.ndarray,
        bc_inflow: np.ndarray,
        num_cycles: int,
        num_pts_per_cycle: int,
    ) -> np.ndarray:
        """Simulate a single Windkessel boundary condition.

        Runs a simulation with a single Windkessel model. Prescribed the
        flow at the inlet of the Windkessel model and simulates the
        corresponding pressure at the inlet.
        """

        config = {
            "boundary_conditions": [
                {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {
                        "Q": bc_inflow.tolist(),
                        "t": bc_times.tolist(),
                    },
                },
                {
                    "bc_name": "RCR_0",
                    "bc_type": "RCR",
                    "bc_values": bc_values,
                },
            ],
            "junctions": [],
            "simulation_parameters": {
                "number_of_cardiac_cycles": num_cycles,
                "number_of_time_pts_per_cardiac_cycle": num_pts_per_cycle,
                "steady_initial": False,
                "output_last_cycle_only": True,
            },
            "vessels": [
                {
                    "boundary_conditions": {
                        "inlet": "INFLOW",
                        "outlet": "RCR_0",
                    },
                    "vessel_id": 0,
                    "vessel_name": "branch0_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "C": 0.0,
                        "L": 0.0,
                        "R_poiseuille": 0.0,
                        "stenosis_coefficient": 0.0,
                    },
                }
            ],
        }
        result = runnercpp.run_from_config(config)

        return np.array(result["pressure_out"])
