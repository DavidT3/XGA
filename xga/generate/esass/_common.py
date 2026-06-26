#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 12/12/2025, 14:46. Copyright (c) The Contributors

from astropy.units import Quantity

# The default lower and upper energy limits for an image to use as a extent map
EROSITA_EXTMAP_LO_EN = Quantity(0.2, 'keV')
EROSITA_EXTMAP_HI_EN = Quantity(10.0, 'keV')

# The default values of eSASS time step for survey and pointed observation types
T_STEP_SURVEY = 0.5
T_STEP_POINT = 100