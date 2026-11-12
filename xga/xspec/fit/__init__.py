#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors
"""
Subpackage providing functions for fitting XSPEC spectral models to global and spatially resolved sets of X-ray
spectra.
"""

from .general import *
from .models import *

# Perhaps this one will be removed
from .profile import *
