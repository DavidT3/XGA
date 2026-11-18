#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/04/2023, 14:51. Copyright (c) The Contributors
"""Subpackage providing end-to-end XGA analysis pipelines."""

from .clusters import luminosity_temperature_pipeline

__all__ = ["luminosity_temperature_pipeline"]
