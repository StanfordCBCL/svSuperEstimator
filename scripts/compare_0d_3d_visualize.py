import pickle
from svsuperestimator import visualizer
import os

from svsuperestimator.tasks import taskutils, plotutils
from svsuperestimator.reader import utils as readutils, SimVascularProject

model_name = "0091_0001"

for result_file in [f for f in os.listdir("results") if f.endswith(".pickle")]:

    report = visualizer.Report()

    model_name = result_file[:-7]
    print(model_name)

    with open(f"results/{model_name}.pickle", "rb") as ff:
        data = pickle.load(ff)

    project_folder = "/Users/stanford/data/projects"
    project = SimVascularProject(os.path.join(project_folder, model_name))

    branch_data = readutils.get_0d_element_coordinates(project)
    model_plot = plotutils.create_3d_geometry_plot_with_vessels(
        project, branch_data
    )
    report.add([model_plot])

    # print(data["0d"]["pressure_systole"])

    plot_opts = {
        "width": 750,
        "height": 700,
    }

    flow_opts = {
        "xaxis_title": r"Systolic flow 3D [l/min]",
        "yaxis_title": r"Systolic flow 0D [l/min]",
    }

    pres_opts = {
        "xaxis_title": r"Systolic pressure 3D [mmHg]",
        "yaxis_title": r"Systolic pressure 0D [mmHg]",
    }

    subtitle = "Systolic pressure at different nodes of the model for multiple variations of boundary conditions"

    report.add("Systolic pressure")
    pres_plots = []
    for data_0d, title in zip(
        [data["0d"], data["0d_opt"]],
        ["Pure geometric model", "Model tuned to one 3D result"],
    ):

        pres_plot = visualizer.Plot2D(
            title=title + f"<br><sup>{subtitle}</sup>",
            **plot_opts,
            **pres_opts,
        )
        max_pres = -9e99
        min_pres = 9e99
        for var_id in sorted(data["3d"].keys()):

            max_pres = max(
                max(data["3d"][var_id]["pressure_systole"]),
                max(data_0d[var_id]["pressure_systole"]),
                max_pres,
            )
            min_pres = min(
                min(data["3d"][var_id]["pressure_systole"]),
                min(data_0d[var_id]["pressure_systole"]),
                min_pres,
            )
            if var_id == -1 and title == "Model tuned to one 3D result":
                opts = dict(
                    name="Training model",
                    symbol="star-open",
                    color="white",
                    size=8,
                )
            elif var_id == -1:
                continue
            else:
                opts = dict(name=f"Variation {var_id}")
            pres_plot.add_point_trace(
                x=taskutils.cgs_pressure_to_mmgh(
                    data["3d"][var_id]["pressure_systole"]
                ),
                y=taskutils.cgs_pressure_to_mmgh(
                    data_0d[var_id]["pressure_systole"]
                ),
                showlegend=True,
                **opts,
            )
        range = taskutils.cgs_pressure_to_mmgh(
            [min_pres - 0.5e4, max_pres + 0.5e4]
        )
        pres_plot._fig.update_xaxes(range=range, tickmode="linear", dtick=5)
        pres_plot._fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            range=range,
            tickmode="linear",
            dtick=5,
        )
        pres_plot.add_line_trace(
            x=range,
            y=range,
            name="1:1 relation",
            showlegend=True,
            color="white",
            dash="dot",
            opacity=0.5,
        )
        pres_plots.append(pres_plot)

    report.add(pres_plots)

    subtitle = "Systolic flow at different nodes of the model for multiple variations of boundary conditions"

    report.add("Systolic flow")
    flow_plots = []
    for data_0d, title in zip(
        [data["0d"], data["0d_opt"]],
        ["Pure geometric model", "Model tuned to one 3D result"],
    ):

        flow_plot = visualizer.Plot2D(
            title=title + f"<br><sup>{subtitle}</sup>",
            **plot_opts,
            **flow_opts,
        )
        max_flow = -9e99
        min_flow = 9e99
        for var_id in sorted(data["3d"].keys()):

            max_flow = max(
                max(data["3d"][var_id]["flow_systole"]),
                max(data_0d[var_id]["flow_systole"]),
                max_flow,
            )
            min_flow = min(
                min(data["3d"][var_id]["flow_systole"]),
                min(data_0d[var_id]["flow_systole"]),
                min_flow,
            )
            if var_id == -1 and title == "Model tuned to one 3D result":
                opts = dict(
                    name="Training model",
                    symbol="star-open",
                    color="white",
                    size=8,
                )
            elif var_id == -1:
                continue
            else:
                opts = dict(name=f"Variation {var_id}")
            flow_plot.add_point_trace(
                x=taskutils.cgs_flow_to_lmin(
                    data["3d"][var_id]["flow_systole"]
                ),
                y=taskutils.cgs_flow_to_lmin(data_0d[var_id]["flow_systole"]),
                showlegend=True,
                **opts,
            )
        range = taskutils.cgs_flow_to_lmin([min_flow - 20, max_flow + 20])
        flow_plot._fig.update_xaxes(range=range, tickmode="linear", dtick=1.0)
        flow_plot._fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            range=range,
            tickmode="linear",
            dtick=1.0,
        )
        flow_plot.add_line_trace(
            x=range,
            y=range,
            name="1:1 relation",
            showlegend=True,
            color="white",
            dash="dot",
            opacity=0.5,
        )
        flow_plots.append(flow_plot)

    report.add(flow_plots)

    report.to_html("results/" + model_name + ".html", title=model_name)
