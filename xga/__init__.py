#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 25/08/2020, 11:49. Copyright (c) David J Turner
from ._version import get_versions

__version__ = get_versions()['version']

from xga.utils import xga_conf, CENSUS, OUTPUT, COMPUTE_MODE, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT, \
    MODEL_PARS, MODEL_UNITS, ABUND_TABLES, XSPEC_FIT_METHOD, COUNTRATE_CONV_SCRIPT

del get_versions
