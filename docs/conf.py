"""Sphinx configuration for pyheom."""

import sys
import os

sys.path.insert(0, os.path.abspath('..'))

project   = 'pyheom'
author    = 'Tatsushi Ikeda'
copyright = '2020-2026, Tatsushi Ikeda'

with open('../VERSION.txt') as f:
    version = release = f.read().strip()

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'myst_parser',
]

source_suffix = {'.md': 'markdown'}
master_doc    = 'index'

# MyST settings
myst_enable_extensions = ['colon_fence', 'dollarmath']

# Napoleon settings (for Google/NumPy docstrings)
napoleon_numpy_docstring  = True
napoleon_google_docstring = False

# autodoc settings
autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

html_theme = 'alabaster'
html_theme_options = {
    'description': 'Hierarchical equations of motion (HEOM) for open quantum systems',
    'github_user': 'tatsushi-ikeda',
    'github_repo': 'pyheom',
    'fixed_sidebar': True,
}
