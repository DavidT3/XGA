#!/usr/bin/env python

#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 01/07/2020, 19:32. Copyright (c) David J Turner

from setuptools import setup, find_packages

import versioneer

setup(name='xga',
      packages=find_packages(),
      version=versioneer.get_version(),
      description='Python package to easily generate and analyse XMM data products',
      author='David Turner',
      author_email='david.turner@sussex.ac.uk',
      url='http://github.com/DavidT3/XGA',
      setup_requires=[],
      install_requires=["astropy>=4.0.1", "numpy>=1.18.1", "tqdm>=4.45.0", "regions==0.4", "pandas>=1.0.3",
                        "fitsio~=1.1.2", "matplotlib~=3.1.3", "scipy~=1.4.1"],
      include_package_data=True,
      python_requires='>=3')

