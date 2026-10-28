# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# make the mcio package importable for autodoc without installing it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# add dynamic class property table generator
sys.path.append(str(Path(__file__).resolve().parent / "_ext"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mcio'
copyright = '2026, UKAEA'
author = 'R. Worrall'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google/NumPy style docstrings
    "sphinx.ext.mathjax",   # enable MathJax rendering
    "myst_parser",          # Markdown support
    "sphinx_copybutton",    # copy button on code blocks
    "decomparams",          # decomposition parameters table directive
]

# MyST markdown extensions, including dollar math
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
    "attrs_block",
    "attrs_inline",
    "linkify",
    "strikethrough",
    "dollarmath",
    "amsmath"
]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
