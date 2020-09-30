# -- Project information -----------------------------------------------------
project = 'tiempo_deshima'
copyright = '2018-2020 DESHIMA software team'
author = 'Akira Endo'
release = '0.1.0'
    
# -- APIDOC location ---------------------------------------------------------
import os
#import sys
#sys.path.insert(0, os.path.abspath('..'))

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
        "github_url": "https://github.com/Stefanie-B/DESHIMAmodel/",
}
    
html_static_path = ['_static']
