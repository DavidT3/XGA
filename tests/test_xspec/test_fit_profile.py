#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/19/26, 4:26 PM. Copyright (c) The Contributors.

"""Tests for XSPEC annular spectral profile fitting functions in xga.xspec.fit.profile."""

import unittest

from astropy.units import Quantity

from xga.sources import GalaxyCluster
from xga.xspec.fit import single_temp_apec_profile

from .. import get_test_source


class TestFitProfileFuncs(unittest.TestCase):
    """Tests for XSPEC spectral profile fitting functions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise the shared multi-mission test GalaxyCluster source."""
        cls.src = get_test_source("all")

    def test_single_temp_apec_profile_stacked_spectra_false(self) -> None:
        """Tests fitting a single-temperature APEC profile to non-stacked annular spectra."""
        res = single_temp_apec_profile(self.src, Quantity([0, 150, 1000], "kpc"))

        assert isinstance(res, GalaxyCluster)

    def test_single_temp_apec_profile_stacked_spectra_false_carf(self) -> None:
        """Tests fitting a single-temperature APEC profile to non-stacked annular spectra with cross-ARFs enabled."""
        res = single_temp_apec_profile(self.src, Quantity([0, 150, 1000], "kpc"), use_cross_arf=True)

        assert isinstance(res, GalaxyCluster)
