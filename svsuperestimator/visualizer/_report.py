"""This module holds the Report class."""
from __future__ import annotations
import os
from datetime import datetime
from typing import Any

from dash import html
import pandas as pd

from ..app.helpers import create_columns


class Report:
    """Class for formatting results."""

    def __init__(self) -> None:
        """Create a new report."""
        self._content = []

    def add(self, content: Any):
        """Add new content to the report."""
        self._content.append(content)

    def to_html(self, folder: str) -> None:
        """Convert the report to a static html website in the folder.

        The main page can be accessed by opening the `index.html` file in the
        folder.

        Args:
            path: Target folder for the webpage.
        """

        formatted_content = []

        for item in self._content:
            if isinstance(item, str):
                formatted_content.append(_HtmlHeading(item))
            else:
                formatted_content.append(_HtmlFlexbox(item))

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

    def to_dash(self) -> list:
        """Convert the report to dash content."""
        formatted_content = []
        for item in self._content:
            if isinstance(item, str):
                formatted_content.append(html.H1(item))
            else:
                formatted_content.append(
                    create_columns([iitem.to_dash() for iitem in item])
                )

        return formatted_content

    def to_files(self, folder: str) -> None:
        """Convert to report to seperate files.

        Image will be saved as png files and tables as csv file.

        Args:
            folder: Folder to save the files in.
        """
        current_heading = ""
        item_in_section_counter = 0
        for item in self._content:
            if isinstance(item, str):
                current_heading = item
                item_in_section_counter = 0
            else:
                for iitem in item:
                    if isinstance(iitem, pd.DataFrame):
                        iitem.to_csv(
                            os.path.join(
                                folder,
                                current_heading
                                + f"_{item_in_section_counter}.csv",
                            )
                        )
                    else:
                        iitem.to_image(
                            os.path.join(
                                folder,
                                current_heading
                                + f"_{item_in_section_counter}.png",
                            )
                        )
                    item_in_section_counter += 1


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
