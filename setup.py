#!/usr/bin/env python
from setuptools import setup, find_packages
import versioneer

# TODO Add something here that checks for the presence of SAS environment variables

# TODO add attempted import of xspec, as it can't go in the install requires section

setup(name='xga',
      packages=find_packages(),
      version=versioneer.get_version(),
      description='Python package to easily generate and analyse XMM data products',
      author='David Turner',
      author_email='david.turner@sussex.ac.uk',
      url='http://github.com/DavidT3/XGA',
      setup_requires=[],
      install_requires=["astropy", "numpy", "tqdm", "regions~=0.4", "pandas>=1.0.3"],
      include_package_data=True,
      python_requires='>=3')

