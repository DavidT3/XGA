# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import shutil


# -- Extra Setup for XGA -----------------------------------------------------

# This is where XGA stores the configuration file when it generates it initially
config_path = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config', 'xga'))

# Generates the directory structure
# If this is running on a local machine, then this directory may already exist with populated
#  config and census files, we don't want them to be deleted if that is the case
if not os.path.exists(config_path):
    os.makedirs(config_path)

    # This is the absolute path of where the config file should live
    config_file_path = os.path.join(config_path, 'xga.cfg')
    census_file_path = os.path.join(config_path, 'census.csv')
    # Finds the current working directory
    current_path = os.getcwd()
    # As this file lives in the docs/source folder, we go two levels up for the absolute path the XGA repo folder
    xga_path = '/'.join(current_path.split('/')[:-2])

    # The test data includes a mostly complete config file, and census
    test_cfg_path = os.path.join(xga_path, 'test_data', 'xga.cfg')
    test_census_path = os.path.join(xga_path, 'test_data', 'census.csv')
    test_data_path = os.path.join(xga_path, 'test_data')

    # Copy the config and census over to the place they should be on whatever system is building this
    shutil.copy(test_cfg_path, config_file_path)
    shutil.copy(test_census_path, census_file_path)

    with open(config_file_path, 'r') as cfg:
        lines = cfg.readlines()

    lines = [line.replace('root_xmm_dir = ./', 'root_xmm_dir = {}'.format(test_data_path)) for line in lines]

    with open(config_file_path, 'w') as cfg:
        cfg.write(''.join(lines))

# I don't like to import in the code itself, but it has to be here so we can be sure a valid config file exists
# from xga import utils
# utils.XGA_MODE = "DOCS"

# -- Project information -----------------------------------------------------

project = 'XMM: Generate and Analyse (XGA)'
copyright = '2020, David J Turner'
author = 'David J Turner'

# The full version, including alpha/beta/rc tags
# release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.graphviz',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'sphinx_rtd_theme']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autodoc_mock_imports = ["fitsio", 'regions', 'corner', 'emcee', 'abel']

