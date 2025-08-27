#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/08/2025, 15:06. Copyright (c) The Contributors
from . import _version
__version__ = _version.get_versions()['version']

from .utils import xga_conf, CENSUS, OUTPUT, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT, CROSS_ARF_XSPEC_SCRIPT, \
    MODEL_PARS, MODEL_UNITS, ABUND_TABLES, XSPEC_FIT_METHOD, COUNTRATE_CONV_SCRIPT, NHC, BLACKLIST, HY_MASS, \
    MEAN_MOL_WEIGHT, SAS_VERSION, XSPEC_VERSION, SAS_AVAIL, DEFAULT_COSMO, MISSION_COL_DB
