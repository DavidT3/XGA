#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/5/26, 11:50 PM. Copyright (c) The Contributors.

"""Tests for XGA hydrostatic mass profile generation functions in xga.sourcetools.mass."""

import unittest

from astropy.units import Quantity

from xga.products.profile import HydrostaticMass
from xga.sourcetools.mass import inv_abel_dens_onion_temp

from .. import get_test_source


class TestMassFuncs(unittest.TestCase):
    """Tests for hydrostatic mass profile measurement functions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise the shared multi-mission test GalaxyCluster source."""
        cls.src = get_test_source("all")

    def test_inv_abel_dens_onion_temp(self) -> None:
        """Tests that inv_abel_dens_onion_temp generates HydrostaticMass profiles for XMM and eRASS data."""
        res = inv_abel_dens_onion_temp(
            self.src, Quantity(600, "kpc"), "beta", "king", "vikhlinin_temp", Quantity(600, "kpc"), stacked_spectra=True
        )
        assert isinstance(res, dict)
        assert set(res.keys()) == {"erass", "xmm"}
        assert isinstance(res["erass"][0], HydrostaticMass)
