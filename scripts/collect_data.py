import os
import shutil

project_path = "/Users/stanford/data/projects"

target_path = "/Users/stanford/svSuperEstimator/collected_optimized_input"


for project_name in [
    f for f in os.listdir(project_path) if not f.startswith(".")
]:
    project_folder = os.path.join(project_path, project_name)

    zerod_folder = os.path.join(project_folder, "ParameterEstimation")

    report_source = os.path.join(
        zerod_folder, "BloodVesselTuning", "report.html"
    )

    report_target = os.path.join(target_path, project_name + ".html")

    if os.path.exists(report_source):
        shutil.copyfile(report_source, report_target)
