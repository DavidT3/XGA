#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 10/6/26, 9:36 AM. Copyright (c) The Contributors.

from .base import BaseAggregateProduct, BaseAggregateProfile1D, BaseProduct, BaseProfile1D
from .events import EventList
from .housekeeping import BadPixels, MissionAttitude, MissionMask
from .lightcurve import AggregateLightCurve, LightCurve
from .phot import PSF, ExpMap, Image, PSFGrid, RateMap
from .relation import ScalingRelation
from .spec import AnnularSpectra, Spectrum

# Defining a dictionary to map from string product names to their associated classes
PROD_MAP = {
    "image": Image,
    "expmap": ExpMap,
    "events": EventList,
    "spectrum": Spectrum,
    "psf": PSF,
    "psfgrid": PSFGrid,
    "lightcurve": LightCurve,
    "badpix": BaseProduct,
    "attitude": BaseProduct,
}
