#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 16/01/2024, 14:14. Copyright (c) The Contributors
"""The __init__ for the xga.generate.sas submodule, which wraps XMM-Newton SAS software tools."""

from ._common import region_setup
from .lightcurve import evselect_lightcurve
from .phot import eexpmap, emosaic, evselect_image, psfgen
from .spec import cross_arf, evselect_spectrum, spectrum_set

__all__ = [
    "cross_arf",
    "eexpmap",
    "emosaic",
    "evselect_image",
    "evselect_lightcurve",
    "evselect_spectrum",
    "psfgen",
    "region_setup",
    "spectrum_set",
]
