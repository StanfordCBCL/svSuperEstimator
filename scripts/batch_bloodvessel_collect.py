import os
import shutil
import click


@click.command()
@click.argument("model_folder")
@click.argument("target_folder")
def main(model_folder, target_folder):

    for project_name in [
        f for f in os.listdir(model_folder) if not f.startswith(".")
    ]:
        project_folder = os.path.join(model_folder, project_name)

        zerod_folder = os.path.join(project_folder, "ParameterEstimation")

        report_source = os.path.join(
            zerod_folder, "BloodVesselTuning", "report.html"
        )
        # print(report_source)
        report_target = os.path.join(target_folder, project_name + ".html")

        if os.path.exists(report_source):
            shutil.copyfile(report_source, report_target)

        solver_source = os.path.join(
            zerod_folder, "BloodVesselTuning", "solver_0d.in"
        )
        solver_target = os.path.join(target_folder, project_name + "_0d.in")

        if os.path.exists(solver_source):
            shutil.copyfile(solver_source, solver_target)


if __name__ == "__main__":
    main()
