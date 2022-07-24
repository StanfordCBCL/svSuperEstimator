import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from .. import model as mdl, reader, problems
from ..problems import plotutils
import os
import click
import pandas as pd

from . import helpers


@click.command()
@click.argument("model_folder")
def run(model_folder):

    # Create app
    app = dash.Dash(
        "svSuperEstimator",
        title="svSuperEstimator",
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=Open+Sans:wght@500&display=swap",
        ],
    )

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
                            for f in os.listdir(model_folder)
                            if not f.startswith(".")
                        ]
                    ),
                    id="selected-model-name",
                    placeholder="Select model",
                ),
                className="item",
            ),
            html.Div(id="project-info-section"),
            html.Div(id="case-selection-section"),
            html.Div(id="new-case-configuration-section"),
            html.Div(id="new-case-parameter-section"),
            dcc.Loading(
                id="loading-old-case-results-section",
                children=[html.Div(id="old-case-results-section")],
                style={"margin-top": 20, "margin-bottom": 20},
            ),
            dcc.Loading(
                id="loading-new-case-results-section",
                children=[
                    html.Div(id="new-case-results-section"),
                ],
                style={"margin-top": 20, "margin-bottom": 20},
            ),
        ],
    )

    @app.callback(
        Output("project-info-section", "children"),
        Output("case-selection-section", "children"),
        Input("selected-model-name", "value"),
    )
    def show_project_information(model_name):
        if model_name is None:
            return None, None
        project = reader.SimVascularProject(
            os.path.join(model_folder, model_name)
        )
        try:
            graph = plotutils.create_3d_geometry_plot_with_bcs(
                project
            ).to_dash(display_controls=False)
        except FileNotFoundError:
            graph = "No 3D geometry found"
        try:
            model = mdl.ZeroDModel(project)
            bc_info = model.get_boundary_condition_info()
        except ValueError:
            return html.Div("Model not supported", className="item"), None

        cases = []
        if os.path.exists(project["rom_optimization_folder"]):
            cases = sorted(
                [
                    name
                    for name in os.listdir(project["rom_optimization_folder"])
                    if os.path.isdir(
                        os.path.join(project["rom_optimization_folder"], name)
                    )
                ]
            )
        cases.append("Create new case")
        return (
            [
                html.H1("Project Overview"),
                helpers.create_columns([helpers.create_table(bc_info), graph]),
            ],
            [
                html.H1("Parameter Estimation"),
                html.Div(
                    dcc.Dropdown(
                        cases, id="selected-case-result", value=cases[0]
                    ),
                    className="item",
                ),
            ],
        )

    @app.callback(
        Output("new-case-configuration-section", "children"),
        Input("selected-case-result", "value"),
        Input("selected-model-name", "value"),
    )
    def configure_optimization(selected_case, model_name):
        if model_name is None or selected_case != "Create new case":
            return None
        return [
            html.H1("Create new case"),
            html.Div(
                dcc.Dropdown(
                    problems.VALID_PROBLEMS,
                    id="new-case-type-selection",  # className="dropdown"
                    placeholder="Select problem type",
                ),
                className="item",
            ),
        ]

    @app.callback(
        Output("old-case-results-section", "children"),
        Input("selected-case-result", "value"),
        Input("selected-model-name", "value"),
    )
    def displace_results(selected_case, model_name):
        if selected_case != "Create new case" and model_name is not None:
            project = reader.SimVascularProject(
                os.path.join(model_folder, model_name)
            )
            problem_class = problems.get_problem_by_run_name(
                project, selected_case
            )
            problem = problem_class(project, selected_case)

            return problem.generate_report().to_dash()
        return None

    @app.callback(
        Output("new-case-parameter-section", "children"),
        Input("selected-case-result", "value"),
        Input("new-case-type-selection", "value"),
        Input("selected-model-name", "value"),
    )
    def configure_optimization(selected_case, case_selection, model_name):
        if (
            selected_case != "Create new case"
            or model_name == None
            or case_selection == None
        ):
            return None

        problem_class = problems.get_problem_by_name(case_selection)
        project = reader.SimVascularProject(
            os.path.join(model_folder, model_name)
        )
        problem = problem_class(project, case_selection)

        config_df = pd.DataFrame(
            [[key, value] for key, value in problem.options.items()],
            columns=["Name", "Value"],
        )

        config_table = helpers.create_editable_table(
            config_df, table_id="new-case-type-parameters"
        )

        return html.Div(
            [
                helpers.create_box(config_table),
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
        Output("new-case-results-section", "children"),
        Input("new-case-type-parameters", "data"),
        Input("start-simulation", "n_clicks"),
        Input("selected-model-name", "value"),
        Input("selected-case-result", "value"),
        Input("new-case-type-selection", "value"),
    )
    def start_thread(
        raw_config,
        start_simulation,
        model_name,
        selected_case,
        case_type,
    ):
        if (
            model_name is not None
            and start_simulation
            and selected_case == "Create new case"
        ):
            project = reader.SimVascularProject(
                os.path.join(model_folder, model_name)
            )

            config = {item["Name"]: item["Value"] for item in raw_config}

            problem_class = problems.get_problem_by_name(case_type)
            problem = problem_class(project, None)
            problem.run(config)

            return problem.generate_report().to_dash()

        return None

    app.run_server(debug=False)
