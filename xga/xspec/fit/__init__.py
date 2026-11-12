#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors
"""
Subpackage providing functions for fitting XSPEC spectral models to global and spatially resolved sets of X-ray
spectra.
"""

from .general import (
    FIT_FUNC_MODEL_NAMES,
    blackbody,
    double_temp_apec,
    general_model_fit,
    multi_temp_dem_apec,
    power_law,
    single_temp_apec,
    single_temp_mekal,
)
from .models import custom_xspec_fit, describe_xspec_model, validate_and_parse_model

from .profile import PROF_FIT_FUNC_MODEL_NAMES, single_temp_apec_profile

__all__ = [
    "FIT_FUNC_MODEL_NAMES",
    "PROF_FIT_FUNC_MODEL_NAMES",
    "blackbody",
    "custom_xspec_fit",
    "describe_xspec_model",
    "double_temp_apec",
    "general_model_fit",
    "multi_temp_dem_apec",
    "power_law",
    "single_temp_apec",
    "single_temp_apec_profile",
    "single_temp_mekal",
    "validate_and_parse_model",
]
