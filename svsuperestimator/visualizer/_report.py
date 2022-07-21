from __future__ import annotations
from enum import Enum
from typing import Sequence
import os
from datetime import datetime
from typing import Any

from dash import html, dcc

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
                pass  # TODO
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
<div class="topbar"><div class="element">
</div><div class="element"><h1>{title} Dashboard</h1></div>
<div class="element"><div class="timestamp">{timestamp}</div></div>
</div>{body}</body>
</html>
"""  # noqa

        # Create the folder if it doesn't exist
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # Read css stylesheet
        stylesheet_path = os.path.join(
            os.path.dirname(__file__), "../app/assets/stylesheet.css"
        )
        with open(stylesheet_path) as ff:
            style = ff.read()

        # Build and write html page
        with open(os.path.join(folder, "index.html"), "w") as ff:
            ff.write(
                sceleton.format(
                    title="svSuperEstimator",
                    style=style,
                    timestamp=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    body="\n".join([c.get_html() for c in formatted_content]),
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

    def to_pngs(self, folder):

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
                    iitem.to_png(
                        os.path.join(
                            folder,
                            current_heading
                            + f"_{item_in_section_counter}.png",
                        )
                    )
                    item_in_section_counter += 1
            elif item_type in [_ContentType.TABLE]:
                # TODO: Save png and not csv
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

    def get_html(self) -> str:
        """Get html string respresentation of content."""
        return f"<h{self._level}>{self._text}</h{self._level}>"


class _HtmlFlexbox:
    """Auxiliary class for generating html flexboxes."""

    def __init__(self, items: list[Any]) -> None:
        """Create a new instance of _HtmlFlexbox.

        Args:
            items: Items for the flexbox.
        """
        self._items = list(items)

    def get_html(self) -> str:
        """Get html string respresentation of content."""
        html = "<div class='container'>\n"
        for item in self._items:
            html += f"<div class='item'>{item.to_html()}</div>\n"
        html += "</div>"
        return html
