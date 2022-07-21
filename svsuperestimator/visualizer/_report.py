from __future__ import annotations
from enum import Enum
from typing import Sequence
import os
from datetime import datetime
from typing import Any

from dash import html, dcc
import pandas as pd

from ._plot import TablePlot

from ..app.helpers import create_columns, create_box, create_table


class _ContentType(Enum):
    HEADING = 1
    PLOT = 2
    PLOTS = 3
    TABLE = 4


class Report:
    def __init__(self) -> None:

        self._content = []

    def add_title(self, title):
        self._content.append((_ContentType.HEADING, title))

    def add_plots(self, plots):
        if isinstance(plots, Sequence):
            self._content.append((_ContentType.PLOTS, plots))
        else:
            self._content.append((_ContentType.PLOT, plots))

    def add_table(self, dataframe):
        self._content.append((_ContentType.TABLE, dataframe))

    def to_html(self, folder):
        """Convert the report to a static html website in the folder.

        The main page can be accessed by opening the `index.html` file in the
        folder.

        Args:
            path: Target folder for the webpage.
        """

        formatted_content = []

        for item_type, item in self._content:
            if item_type == _ContentType.HEADING:
                formatted_content.append(_HtmlHeading(item))
            elif item_type in [_ContentType.PLOTS, _ContentType.PLOT]:
                formatted_content.append(_HtmlFlexbox(item))
            elif item_type == _ContentType.TABLE:
                formatted_content.append(_HtmlFlexbox(item))
            else:
                raise RuntimeError("Unknown content type.")

        sceleton = """<!DOCTYPE html>
<html lang="en">
<meta charset="UTF-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@500&display=swap" rel="stylesheet"> 
<style>{style}</style>
<script src=""></script>
<body>
<div class="topbar"><h1>{title} Dashboard</h1></div>
{body}
</body>
</html>
"""  # noqa

        stylesheet = """body {
    margin: 0;
    font-family: 'Open Sans', sans-serif;
    background-color: #121212;
    color: #ffffffe8;
  }

  table {
    width: 100%;
    font-family: monospace, monospace;
    font-size: 10pt;
    font-weight: lighter;
    text-align: left;
  }

  th,
  td {
    text-align: left;
    padding: 5px;
  }

  .topbar {
    overflow: hidden;
    background-color: #212121;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.07), 0 2px 4px rgba(0, 0, 0, 0.07), 0 4px 8px rgba(0, 0, 0, 0.07), 0 8px 16px rgba(0, 0, 0, 0.07), 0 16px 32px rgba(0, 0, 0, 0.07), 0 32px 64px rgba(0, 0, 0, 0.07);
  }

  .element {
    display: inline-block;
    width: 33%;
  }

  .timestamp {
    text-align: right;
    color: #ffffffb5;
    padding-right: 20px;
    font-size: 10pt;
  }

  .topbar h1 {
    margin-top: 10px;
    margin-bottom: 10px;
    font-weight: lighter;
  }

  .container {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-start;
    padding: 10px;
    align-items: flex-start;
  }

  .item {
    flex: 0 calc(50% - 40px);
    background-color: #212121;
    border-radius: 10px;
    padding: 10px;
    margin-left: 10px;
    margin-right: 10px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.07), 0 2px 4px rgba(0, 0, 0, 0.07), 0 4px 8px rgba(0, 0, 0, 0.07), 0 8px 16px rgba(0, 0, 0, 0.07), 0 16px 32px rgba(0, 0, 0, 0.07), 0 32px 64px rgba(0, 0, 0, 0.07);
  }

  h1 {
    font-size: 16pt;
    margin-left: 20px;
    margin-top: 10px;
    margin-bottom: 10px;
  }

  h2 {
    font-size: 14pt;
  }
"""

        # Create the folder if it doesn't exist
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # Build and write html page
        with open(os.path.join(folder, "index.html"), "w") as ff:
            ff.write(
                sceleton.format(
                    title="svSuperEstimator",
                    style=stylesheet,
                    timestamp=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    body="\n".join([c.to_html() for c in formatted_content]),
                )
            )

    def to_dash(self):
        formatted_content = []
        for item_type, item in self._content:
            if item_type == _ContentType.HEADING:
                formatted_content.append(html.H1(item))
            elif item_type in [_ContentType.PLOT]:
                formatted_content.append(
                    create_box(dcc.Graph(figure=item.fig))
                )
            elif item_type in [_ContentType.PLOTS]:
                formatted_content.append(
                    create_columns(
                        [dcc.Graph(figure=iitem.fig) for iitem in item]
                    )
                )
            elif item_type in [_ContentType.TABLE]:
                formatted_content.append(create_box(create_table(item)))
            else:
                raise RuntimeError("Unknown content type.")

        return formatted_content

    def to_files(self, folder):

        current_heading = ""
        item_in_section_counter = 0
        for item_type, item in self._content:
            if item_type == _ContentType.HEADING:
                current_heading = item
                item_in_section_counter = 0
            elif item_type == _ContentType.PLOT:
                item.to_png(
                    os.path.join(
                        folder, current_heading + f"_{item_in_section_counter}"
                    )
                )
                item_in_section_counter += 1
            elif item_type == _ContentType.PLOTS:
                for iitem in item:
                    try:
                        iitem.to_png(
                            os.path.join(
                                folder,
                                current_heading
                                + f"_{item_in_section_counter}.png",
                            )
                        )
                    except AttributeError:
                        iitem.to_csv(
                            os.path.join(
                                folder,
                                current_heading
                                + f"_{item_in_section_counter}.csv",
                            )
                        )
                    item_in_section_counter += 1
            elif item_type in [_ContentType.TABLE]:
                item.to_csv(
                    os.path.join(
                        folder,
                        current_heading + f"_{item_in_section_counter}.csv",
                    )
                )
                item_in_section_counter += 1
            else:
                raise RuntimeError("Unknown content type.")


class _HtmlHeading:
    """Auxiliary class for generating html heading."""

    def __init__(self, text: str, level: int = 1) -> None:
        """Create a new instance of _HtmlHeading.

        Args:
            text: The text of the heading.
            level: The level of the heading from 1-6.
        """
        self._text = text
        self._level = level

    def to_html(self) -> str:
        """Get html string respresentation of content."""
        return f"<h{self._level}>{self._text}</h{self._level}>"


class _HtmlFlexbox:
    """Auxiliary class for generating html flexboxes."""

    def __init__(self, items: list[Any]) -> None:
        """Create a new instance of _HtmlFlexbox.

        Args:
            items: Items for the flexbox.
        """
        if not isinstance(items, list) and not isinstance(items, tuple):
            self._items = [items]
        else:
            self._items = items

    def to_html(self) -> str:
        """Get html string respresentation of content."""
        html = "<div class='container'>\n"
        for item in self._items:
            if isinstance(item, pd.DataFrame):
                html += (
                    f"<div class='item'>{item.to_html(index=False)}</div>\n"
                )
            else:
                html += f"<div class='item'>{item.to_html()}</div>\n"
        html += "</div>"
        return html
