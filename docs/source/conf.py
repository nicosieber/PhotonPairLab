import os
import sys

# Make project importable for autodoc:
# If package root is at <repo_root>/photonpairlab/, then:
sys.path.insert(0, os.path.abspath("../../src"))  # repo root

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PhotonPairLab'
copyright = '2026, Nico Sieber'
author = 'Nico Sieber'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = []

language = 'y'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
# Docstring handling
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
