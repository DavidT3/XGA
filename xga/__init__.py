#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/04/2023, 15:15. Copyright (c) The Contributors
from ._version import get_versions

__version__ = get_versions()['version']

from .utils import xga_conf, CENSUS, OUTPUT, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT, MODEL_PARS, \
    MODEL_UNITS, ABUND_TABLES, XSPEC_FIT_METHOD, COUNTRATE_CONV_SCRIPT, NHC, BLACKLIST, HY_MASS, MEAN_MOL_WEIGHT, \
    SAS_VERSION, XSPEC_VERSION, SAS_AVAIL, DEFAULT_COSMO

del get_versions
