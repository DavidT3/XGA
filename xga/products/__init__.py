#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:06 AM. Copyright (c) The Contributors.

from .base import BaseProduct, BaseAggregateProduct, BaseProfile1D, BaseAggregateProfile1D
from .lightcurve import LightCurve, AggregateLightCurve
from .misc import EventList
from .phot import Image, ExpMap, RateMap, PSF, PSFGrid
from .relation import ScalingRelation
from .spec import Spectrum, AnnularSpectra

# Defining a dictionary to map from string product names to their associated classes
PROD_MAP = {"image": Image, "expmap": ExpMap, "events": EventList, "spectrum": Spectrum, "psf": PSF,
            "psfgrid": PSFGrid, 'badpix': BaseProduct, 'attitude': BaseProduct}












