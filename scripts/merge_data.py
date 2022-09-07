import os
import shutil
import click


@click.command()
@click.argument("project_path")
@click.argument("centerline_path")
@click.argument("solver_in_path")
def main(project_path, centerline_path, solver_in_path):

    for project_name in [
        f for f in os.listdir(project_path) if not f.startswith(".")
    ]:
        project_folder = os.path.join(project_path, project_name)

        zerod_folder = os.path.join(
            project_folder, "ROMSimulations", project_name
        )

        if not os.path.exists(zerod_folder):
            os.makedirs(zerod_folder)

        solver_target = os.path.join(
            project_folder, "ROMSimulations", project_name, "solver_0d.in"
        )
        solver_source = os.path.join(solver_in_path, project_name + "_0d.in")
        if not os.path.exists(solver_target):
            if not os.path.exists(solver_source):
                print(f"Solver file for {project_name} does not exist.")
            print(f"Copying {solver_source} to {solver_target}")
            shutil.copyfile(solver_source, solver_target)

        centerline_target = os.path.join(
            project_folder,
            "ROMSimulations",
            project_name,
            project_name + ".vtp",
        )
        centerline_source = os.path.join(
            centerline_path, project_name + ".vtp"
        )
        if not os.path.exists(centerline_target):
            if not os.path.exists(centerline_source):
                print(f"Centerline file for {project_name} does not exist.")
            print(f"Copying {centerline_source} to {centerline_target}")
            shutil.copyfile(centerline_source, centerline_target)


if __name__ == "__main__":
    main()
