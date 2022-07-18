"""This module holds the WebPage class."""
from __future__ import annotations
import os
from datetime import datetime
from typing import Any, Iterable, Union


class WebPage:
    """Web page class.

    This class faciliates creation of a simple wepage for presenting
    plots in a comprehensive manner.
    """

    _SCELETON = """<!DOCTYPE html>
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

    def __init__(self, title: str) -> None:
        """Create a new WebPage instance."""
        self._title = title
        self._content: list[Any] = []

    def add_heading(self, text: str, level: int = 1) -> None:
        """Add a heading to the webpage.

        Args:
            text: The text of the heading.
            level: The level of the heading from 1-6.
        """
        self._content.append(_HtmlHeading(text, level))

    def add_plots(self, plots: Any) -> None:
        """Add plots to the webpage.

        Args:
            plots: List of plots to add to the webpage.
        """
        # Append to existing flexbox if one is at the end
        if self._content and isinstance(self._content[-1], _HtmlFlexbox):
            self._content[-1].append(plots)
        else:
            self._content.append(_HtmlFlexbox(plots))

    def build(self, path: str) -> None:
        """Build the webpage at the specified location.

        This method creates a folder for the webpage. The main page can be
        accessed by opening the `index.html` file in the folder.

        Args:
            path: Target folder for the webpage.
        """

        # Create the folder if it doesn't exist
        if not os.path.isdir(path):
            os.mkdir(path)

        # Read css stylesheet
        stylesheet_path = os.path.join(
            os.path.dirname(__file__), "stylesheet.css"
        )
        with open(stylesheet_path) as ff:
            style = ff.read()

        # Build and write html page
        with open(os.path.join(path, "index.html"), "w") as ff:
            ff.write(
                self._SCELETON.format(
                    title="svSuperEstimator",
                    style=style,
                    timestamp=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    body="\n".join([c.get_html() for c in self._content]),
                )
            )


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

    def append(self, items: Union[list[Any], Any]) -> None:
        """Append items to the flexbox.

        Args:
            items: Item(s) to append.
        """
        if isinstance(items, Iterable):
            self._items.extend(list(items))
        else:
            self._items.append(list(items))

    def get_html(self) -> str:
        """Get html string respresentation of content."""
        html = "<div class='container'>\n"
        for item in self._items:
            html += f"<div class='item'>{item.to_html()}</div>\n"
        html += "</div>"
        return html
