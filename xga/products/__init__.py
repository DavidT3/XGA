#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/08/2025, 21:42. Copyright (c) The Contributors

from .base import BaseProduct, BaseAggregateProduct, BaseProfile1D, BaseAggregateProfile1D
from .phot import Image, ExpMap, RateMap, PSF, PSFGrid
from .misc import EventList
from .relation import ScalingRelation
from .spec import Spectrum, AnnularSpectra

# Defining a dictionary to map from string product names to their associated classes
PROD_MAP = {"image": Image, "expmap": ExpMap, "events": EventList, "spectrum": Spectrum, "psf": PSF,
            "psfgrid": PSFGrid}












