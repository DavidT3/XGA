#!/usr/bin/env python

#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/08/2025, 14:12. Copyright (c) The Contributors

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
      cmdclass=versioneer.get_cmdclass(),
      description='Python package to easily generate and analyse X-ray astronomy data products, ideal for '
                  'investigating large samples.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='David J Turner',
      author_email='djturner@umbc.edu',
      url='http://github.com/DavidT3/XGA',
      setup_requires=[],
      install_requires=["astropy>=6.1.2", "numpy>=1.26.4", "tqdm>=4.66.4", "regions>=0.9", "pandas>=2.2.2",
                        "fitsio>=1.2.1", "matplotlib>=3.9.0", "scipy>=1.14.0", "pyabel>=0.9", "corner>=2.2.2",
                        "emcee>=3.1.6", "tabulate>=0.9.0", "getdist>=1.4.7", "exceptiongroup>=1.0.4"],
      include_package_data=True,
      python_requires='>=3.10')


