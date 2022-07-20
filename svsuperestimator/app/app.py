import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from .. import model as mdl, visualizer, reader, problems
import os
import click

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

        cases = []
        if os.path.exists(project["rom_optimization_folder"]):
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

            return problem_class.generate_report(
                project, selected_case
            ).to_dash()
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
        Output("new-case-results-section", "children"),
        Input("num_procs", "value"),
        Input("num_particles", "value"),
        Input("num_rejuvenation_steps", "value"),
        Input("resampling_threshold", "value"),
        Input("start-simulation", "n_clicks"),
        Input("selected-model-name", "value"),
        Input("selected-case-result", "value"),
        Input("new-case-type-selection", "value"),
    )
    def start_thread(
        num_procs,
        num_particles,
        num_rejuvenation_steps,
        resampling_threshold,
        start_simulation,
        model_name,
        selected_case,
        case_type,
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
                os.path.join(model_folder, model_name)
            )

            config = {
                "num_procs": num_procs,
                "num_particles": num_particles,
                "num_rejuvenation_steps": num_rejuvenation_steps,
                "resampling_threshold": resampling_threshold,
            }

            case_name = f"case_smc_chopin_np{config['num_particles']}_rt{10*config['resampling_threshold']:.0f}_rs{config['num_rejuvenation_steps']}"

            problem_class = problems.get_problem_by_name(case_type)
            problem_class.run(project, config, case_name)

            return problem_class.generate_report(project, case_name).to_dash()

        return None

    app.run_server(debug=False)
