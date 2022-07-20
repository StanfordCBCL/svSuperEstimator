from dash import html, dcc, dash_table


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
            "font-family": "sans-serif",
        },
    )
