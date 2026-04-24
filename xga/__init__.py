#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 09/12/2025, 15:20. Copyright (c) The Contributors
from . import _version
__version__ = _version.get_versions()['version']

from .utils import (xga_conf, CENSUS, OUTPUT, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT, MODEL_PARS,
                    MODEL_UNITS, ABUND_TABLES, XSPEC_FIT_METHOD, COUNTRATE_CONV_SCRIPT, NHC, BLACKLIST, HY_MASS, MEAN_MOL_WEIGHT,
                    SAS_VERSION, XSPEC_VERSION, SAS_AVAIL, DEFAULT_COSMO, TELESCOPES, USABLE, DEFAULT_TELE_SEARCH_DIST, COMBINED_INSTS,
                    ESASS_AVAIL, SRC_REGION_COLOURS, check_telescope_choices, PRETTY_TELESCOPE_NAMES, CIAO_AVAIL, CIAO_VERSION, CALDB_AVAIL,
                    CALDB_VERSION, ESASS_VERSION, RAD_MATCH_PRECISION)

import sys
from . import generate
# Here we set up 'shims' to ensure that pre-multi-mission imports of SAS wrapper functions
#  still function as intended.
sys.modules['xga.sas'] = generate.sas
sys.modules['xga.sas.phot'] = generate.sas.phot
sys.modules['xga.sas.misc'] = generate.sas.misc
sys.modules['xga.sas.spec'] = generate.sas.spec
sys.modules['xga.sas.lightcurve'] = generate.sas.lightcurve
sys.modules['xga.sas.run'] = generate.sas.run

# We also need to make sure that the modules are accessible as attributes of the xga module
#  itself, as some parts of DAXA (and other older code) may use them that way.
sas = generate.sas
