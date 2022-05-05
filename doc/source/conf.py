"""Configuration file for the Sphinx documentation builder."""
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath(".."))

project = "svSuperEstimator"
copyright = "Stanford University, The Regents of the University of California, and others."

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "m2r2",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
]

source_suffix = [".rst", ".md"]
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "press"
html_theme_path = [
    "_themes",
]
html_sidebars = {"**": ["util/searchbox.html", "util/sidetoc.html"]}
html_logo = "img/logo.png"
html_favicon = "img/favicon.jpg"
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
set_type_checking_flag = False
mermaid_output_format = "svg"
mermaid_cmd = "./node_modules/.bin/mmdc"

html_theme_options = {
    "external_links": [
        ("View on GitHub", "https://github.com/SimVascular/svSuperEstimator"),
        ("About SimVascular", "https://simvascular.github.io"),
    ]
}
