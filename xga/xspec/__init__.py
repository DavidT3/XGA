#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/4/26, 9:01 AM. Copyright (c) The Contributors.
"""Subpackage for running XSPEC spectral fits and simulations on XGA sources and samples."""

from .fit import (
    blackbody,
    custom_xspec_fit,
    describe_xspec_model,
    double_temp_apec,
    general_model_fit,
    multi_temp_dem_apec,
    power_law,
    single_temp_apec,
    single_temp_apec_profile,
    single_temp_mekal,
)
from .run import execute_cmd, xspec_call

__all__ = [
    "blackbody",
    "custom_xspec_fit",
    "describe_xspec_model",
    "double_temp_apec",
    "execute_cmd",
    "general_model_fit",
    "multi_temp_dem_apec",
    "power_law",
    "single_temp_apec",
    "single_temp_apec_profile",
    "single_temp_mekal",
    "xspec_call",
]
