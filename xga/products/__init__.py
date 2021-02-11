#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 11/02/2021, 12:30. Copyright (c) David J Turner

from .base import BaseProduct, BaseAggregateProduct, BaseProfile1D, BaseAggregateProfile1D
from .misc import EventList
from .phot import Image, ExpMap, RateMap, PSF, PSFGrid
from .relation import ScalingRelation
from .spec import Spectrum, AnnularSpectra

# Defining a dictionary to map from string product names to their associated classes
PROD_MAP = {"image": Image, "expmap": ExpMap, "events": EventList, "spectrum": Spectrum, "psf": PSF,
            "psfgrid": PSFGrid}












