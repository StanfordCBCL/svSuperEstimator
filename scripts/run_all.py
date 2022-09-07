import os
from svsuperestimator.main import run_from_config

model_folder = "/Users/stanford/data/projects"

if __name__ == "__main__":

    for model in [
        n for n in os.listdir(model_folder) if not n.startswith(".")
    ]:

        config = {
            "project": f"/Users/stanford/data/projects/{model}",
            "global": {"num_procs": 4},
            "tasks": {
                "BloodVesselTuning": {
                    "threed_solution_file": f"/Users/stanford/data/3d_centerline/{model}.vtp",
                    "maxfev": 2000,
                }
            },
        }
        try:
            run_from_config(config)
            print(model, "successful")
        except Exception as excp:
            print(model, "failed: ", excp)
