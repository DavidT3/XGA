#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/04/2023, 12:16. Copyright (c) The Contributors

from .fit import single_temp_apec, power_law, single_temp_apec_profile, blackbody, multi_temp_dem_apec, \
    single_temp_mekal
from .run import execute_cmd, xspec_call
