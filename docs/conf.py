# -- Project information -----------------------------------------------------
project = 'deshima-sensitivity'
copyright = '2020, Akio Taniguchi'
author = 'Akio Taniguchi'
    
# -- APIDOC location ---------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.viewcode",
        "sphinx.ext.napoleon",
        "sphinx.ext.autosummary",
]
    
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
    
# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
        "github_url": "https://github.com/deshima-dev/deshima-sensitivity/",
}
    
html_static_path = ['_static']
