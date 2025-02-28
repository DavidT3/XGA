#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 28/02/2025, 11:50. Copyright (c) The Contributors
from . import _version
__version__ = _version.get_versions()['version']

from .utils import xga_conf, CENSUS, OUTPUT, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT, MODEL_PARS, MODEL_UNITS, \
    ABUND_TABLES, XSPEC_FIT_METHOD, COUNTRATE_CONV_SCRIPT, NHC, BLACKLIST, HY_MASS, MEAN_MOL_WEIGHT, SAS_VERSION, \
    XSPEC_VERSION, SAS_AVAIL, DEFAULT_COSMO, TELESCOPES, USABLE, DEFAULT_TELE_SEARCH_DIST, COMBINED_INSTS, eSASS_AVAIL, \
    SRC_REGION_COLOURS, check_telescope_choices, PRETTY_TELESCOPE_NAMES, CIAO_AVAIL, CIAO_VERSION, CALDB_AVAIL, \
    CALDB_VERSION
