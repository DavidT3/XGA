#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/13/26, 5:06 PM. Copyright (c) The Contributors.

from .fit import single_temp_apec, power_law, single_temp_apec_profile, blackbody, multi_temp_dem_apec, \
    single_temp_mekal, double_temp_apec
from .run import execute_cmd, xspec_call
