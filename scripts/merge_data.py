import os
import shutil

project_path = "/Users/stanford/data/projects"
centerline_path = "/Users/stanford/data/centerlines_raw"
solver_in_path = "/Users/stanford/data/0d"


for project_name in [
    f for f in os.listdir(project_path) if not f.startswith(".")
]:
    project_folder = os.path.join(project_path, project_name)

    zerod_folder = os.path.join(project_folder, "ROMSimulations", project_name)

    if not os.path.exists(zerod_folder):
        os.makedirs(zerod_folder)

    solver_target = os.path.join(
        project_folder, "ROMSimulations", project_name, "solver_0d.in"
    )
    solver_source = os.path.join(solver_in_path, project_name + "_0d.in")
    if not os.path.exists(solver_target):
        if not os.path.exists(solver_source):
            raise FileNotFoundError(
                f"Solver file for {project_name} does not exist."
            )
        print(f"Copying {solver_source} to {solver_target}")
        shutil.copyfile(solver_source, solver_target)

    centerline_target = os.path.join(
        project_folder, "ROMSimulations", project_name, project_name + ".vtp"
    )
    centerline_source = os.path.join(centerline_path, project_name + ".vtp")
    if not os.path.exists(centerline_target):
        if not os.path.exists(centerline_source):
            raise FileNotFoundError(
                f"Centerline file for {project_name} does not exist."
            )
        print(f"Copying {centerline_source} to {centerline_target}")
        shutil.copyfile(centerline_source, centerline_target)
