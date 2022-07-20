import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from .. import model as mdl, visualizer, reader
from ..problems import windkessel_smc_chopin
import os
import pandas as pd
import numpy as np

from . import helpers

# Create app
app = dash.Dash(
    "svSuperEstimator",
    title="svSuperEstimator",
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Open+Sans:wght@500&display=swap",
    ],
)


MODEL_FOLDER = "/Users/stanford/svSuperEstimator/models"


app.layout = html.Div(
    id="parent",
    children=[
        helpers.create_top_bar("svSuperEstimator"),
        html.H1("Select Model"),
        html.Div(
            dcc.Dropdown(
                sorted(
                    [
                        f
                        for f in os.listdir(MODEL_FOLDER)
                        if not f.startswith(".")
                    ]
                ),
                id="model-name",
                placeholder="Select model",
            ),
            className="item",
        ),
        html.Div(id="project-info"),
        html.Div(id="result-selection"),
        html.Div(id="configuration"),
        html.Div(id="input-parameter"),
        dcc.Loading(
            id="loading",
            children=[html.Div(id="results")],
            style={"margin-top": 20, "margin-bottom": 20},
        ),
        dcc.Loading(
            id="loading-2",
            children=[
                html.Div(id="new-case-results"),
            ],
            style={"margin-top": 20, "margin-bottom": 20},
        ),
    ],
)


VALID_CASES = [
    "Nelder-Mead Optimization",
    "Grid Approximation",
    "Sequential-Monte-Carlo",
]


@app.callback(
    Output("project-info", "children"),
    Output("result-selection", "children"),
    Input("model-name", "value"),
)
def show_project_information(model_name):
    if model_name is None:
        return None, None
    project = reader.SimVascularProject(os.path.join(MODEL_FOLDER, model_name))
    try:
        plot3d = visualizer.Vtk3dPlot(
            project["3d_mesh"],
            color="darkred",
        )
        graph = dcc.Graph(figure=plot3d.fig)
    except FileNotFoundError:
        graph = "No 3D geometry found"
    try:
        model = mdl.ZeroDModel(project)
        bc_info = model.get_boundary_condition_info()
    except ValueError:
        return html.Div("Model not supported", className="item"), None
    cases = sorted(
        [
            name
            for name in os.listdir(project["rom_optimization_folder"])
            if name.startswith("case_")
        ]
    )
    cases.append("Create new case")
    return (
        [
            html.H1("Project Overview"),
            helpers.create_columns([graph, helpers.create_table(bc_info)]),
        ],
        [
            html.H1("Parameter Estimation"),
            html.Div(
                dcc.Dropdown(
                    cases,
                    id="selected-case",  # className="dropdown"
                    value=cases[0]
                    # placeholder="Select optimization case",
                ),
                className="item",
            ),
        ],
    )


@app.callback(
    Output("configuration", "children"),
    Input("selected-case", "value"),
    Input("model-name", "value"),
)
def configure_optimization(selected_case, model_name):
    if model_name is None or selected_case != "Create new case":
        return None
    return [
        html.H1("Configuration"),
        html.Div(
            dcc.Dropdown(
                VALID_CASES,
                id="case-selection",  # className="dropdown"
                placeholder="Select problem type",
            ),
            className="item",
        ),
    ]


@app.callback(
    Output("results", "children"),
    Input("selected-case", "value"),
    Input("model-name", "value"),
)
def displace_results(selected_case, model_name):
    if selected_case != "Create new case" and model_name is not None:
        project = reader.SimVascularProject(
            os.path.join(MODEL_FOLDER, model_name)
        )
        return windkessel_smc_chopin.generate_report(
            project, selected_case
        ).to_dash()
    return None


@app.callback(
    Output("input-parameter", "children"),
    Input("selected-case", "value"),
    Input("case-selection", "value"),
    Input("model-name", "value"),
)
def configure_optimization(selected_case, case_selection, model_name):
    if (
        selected_case != "Create new case"
        or model_name == None
        or case_selection == None
    ):
        return None

    options = [
        dcc.Input(
            id="num_procs",
            type="number",
            placeholder="Number of workers",
            className="input",
            debounce=True,
        ),
        dcc.Input(
            id="num_particles",
            type="number",
            placeholder="Number of particles",
            className="input",
            debounce=True,
        ),
        dcc.Input(
            id="num_rejuvenation_steps",
            type="number",
            placeholder="Number of rejuvenation steps",
            className="input",
            debounce=True,
        ),
        dcc.Input(
            id="resampling_threshold",
            type="number",
            placeholder="Resampling threshold",
            className="input",
            debounce=True,
        ),
    ]

    return html.Div(
        [
            helpers.create_columns(options),
            html.Div(
                html.Button(
                    "Start estimation",
                    id="start-simulation",
                    n_clicks=0,
                    className="button",
                ),
                className="item",
            ),
        ]
    )


@app.callback(
    Output("new-case-results", "children"),
    Input("num_procs", "value"),
    Input("num_particles", "value"),
    Input("num_rejuvenation_steps", "value"),
    Input("resampling_threshold", "value"),
    Input("start-simulation", "n_clicks"),
    Input("model-name", "value"),
    Input("selected-case", "value"),
)
def start_thread(
    num_procs,
    num_particles,
    num_rejuvenation_steps,
    resampling_threshold,
    start_simulation,
    model_name,
    selected_case,
):
    if (
        not None
        in [
            num_procs,
            num_particles,
            num_rejuvenation_steps,
            resampling_threshold,
            model_name,
        ]
        and start_simulation
        and selected_case == "Create new case"
    ):
        project = reader.SimVascularProject(
            os.path.join(MODEL_FOLDER, model_name)
        )

        config = {
            "num_procs": num_procs,
            "num_particles": num_particles,
            "num_rejuvenation_steps": num_rejuvenation_steps,
            "resampling_threshold": resampling_threshold,
        }

        case_name = f"case_smc_chopin_np{config['num_particles']}_rt{10*config['resampling_threshold']:.0f}_rs{config['num_rejuvenation_steps']}"
        windkessel_smc_chopin.run(project, config, case_name)

        return windkessel_smc_chopin.generate_report(
            project, case_name
        ).to_dash()

    return None


def run():
    app.run_server(debug=False)
