#!/usr/bin/env python

#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 10/03/2021, 16:22. Copyright (c) David J Turner

from os import path

from setuptools import setup, find_packages

import versioneer

# Uses the README as the long description
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='xga',
      packages=find_packages(),
      version=versioneer.get_version(),
      description='Python package to easily generate and analyse XMM data products',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='David Turner',
      author_email='david.turner@sussex.ac.uk',
      url='http://github.com/DavidT3/XGA',
      setup_requires=[],
      install_requires=["astropy>=4.0", "numpy>=1.18", "tqdm>=4.45", "regions==0.4", "pandas>=1.0.3",
                        "fitsio>=1.1.2", "matplotlib>=3.1.3", "scipy>=1.4.1", "pyabel>=0.8.3", "corner>=2.1.0",
                        "emcee>=3.0.2", "tabulate>=0.8.9", "getdist>=1.1.3"],
      include_package_data=True,
      python_requires='>=3')

