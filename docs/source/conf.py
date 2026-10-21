# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mcio'
copyright = '2025, R. Worrall'
author = 'R. Worrall'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_static_path = ["_static"]

def setup(app):
    app.add_css_file("custom.css")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # for Google/NumPy style docstrings
    "myst_parser",          # if you want Markdown support
    "sphinx.ext.mathjax"    # enable MathJax rendering
]

# Tell MyST to parse dollar math
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

autosummary_generate = True  # make Sphinx create stub pages automatically

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

html_sidebars = {
    "**": ["search-field.html", "globaltoc.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"],
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # so autodoc can find your package

# add dynamic class property table generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "_ext")))
extensions.append("decomparams")