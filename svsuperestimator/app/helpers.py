from dash import html, dash_table


def create_top_bar(title):
    """Create top bar in dash.

    Args:
        title: The title displayed in the bar.

    Returns:
        The dash html block.
    """
    return html.Div(
        children=[
            html.H1(
                id="H1",
                children=title,
            ),
        ],
        className="topbar",
    )


def create_box(element):
    """Wrap the element in the box with a highlighted color.

    Args:
        element: The element to wrap.

    Returns:
        The dash html block.
    """
    return html.Div(element, className="item")


def create_columns(elements):
    """Wrap the elements in columns.

    Args:
        element: The element to wrap.

    Returns:
        The dash html block.
    """
    return html.Div(
        [html.Div(ele, className="item") for ele in elements],
        className="container",
    )


def create_table(dataframe):
    """Create a table from a pandas dataframe.

    Args:
        dataframe: The dataframe to convert.

    Returns:
        The dash html block.
    """
    return dash_table.DataTable(
        dataframe.to_dict("records"),
        [{"name": i, "id": i} for i in dataframe.columns],
        style_header={
            "font-weight": "bold",
            "backgroundColor": "transparent",
        },
        style_data={
            "backgroundColor": "transparent",
        },
        style_cell={
            "textAlign": "left",
            "font-size": 11,
            "backgroundColor": "transparent",
        },
        editable=False,
    )


def create_editable_table(dataframe, table_id):
    """Create a table from a pandas dataframe.

    Args:
        dataframe: The dataframe to convert.

    Returns:
        The dash html block.
    """
    return dash_table.DataTable(
        dataframe.to_dict("records"),
        [{"name": i, "id": i} for i in dataframe.columns],
        style_header={
            "font-weight": "bold",
            "backgroundColor": "transparent",
            "color": "ffffffe8",
        },
        style_data={
            "backgroundColor": "transparent",
            "color": "ffffffe8",
        },
        style_cell={
            "textAlign": "left",
            "font-size": 11,
            "color": "ffffffe8",
        },
        css=[
            {
                "selector": "td.cell--selected, td.focused",
                "rule": "background-color: #FF4136;",
            },
            {
                "selector": "td.cell--selected *, td.focused *",
                "rule": "color: white !important;text-align: left;font-style: italic;",
            },
        ],
        id=table_id,
        editable=True,
    )
