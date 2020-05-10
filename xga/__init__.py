#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 30/04/2020, 09:44. Copyright (c) David J Turner

from ._version import get_versions

from xga.utils import xga_conf, CENSUS, OUTPUT, COMPUTE_MODE, NUM_CORES

__version__ = get_versions()['version']
del get_versions
