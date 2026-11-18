#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 5:23 PM. Copyright (c) The Contributors.

"""Tests for the GalaxyCluster source class."""

import unittest

from astropy.units import Quantity

from .. import get_test_source


class TestGalaxyCluster(unittest.TestCase):
    """Tests for GalaxyCluster properties and methods."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise the shared multi-mission test GalaxyCluster source."""
        cls.src = get_test_source("all")

    def test_r500(self) -> None:
        """Tests that the r500 property returns the expected overdensity radius."""
        assert self.src.r500 == Quantity(500, "kpc")
