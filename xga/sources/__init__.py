#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/25/26, 3:48 PM. Copyright (c) The Contributors.
"""The __init__ for XGA's 'sources' submodule, where source classes from the various files are imported."""

from .base import BaseSource, NullSource
from .extended import GalaxyCluster
from .general import ExtendedSource, PointSource
from .point import Star

__all__ = ["BaseSource", "NullSource", "GalaxyCluster", "ExtendedSource", "PointSource", "Star"]
