#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 6:08 PM. Copyright (c) The Contributors.

"""Tests for internal setup helper functions in xga.sourcetools._common."""

import unittest

from astropy.units import Quantity

from xga import NUM_CORES
from xga.products.profile import GasDensity3D, GasTemperature3D
from xga.sources import GalaxyCluster
from xga.sourcetools._common import _get_all_telescopes, _setup_global, _setup_inv_abel_dens_onion_temp

from .. import get_test_source


class TestSetupFuncs(unittest.TestCase):
    """Tests for common sourcetools setup and telescope validation helper functions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise the shared multi-mission test GalaxyCluster source."""
        cls.src = get_test_source("all")

    def test_get_all_telescopes_list_input(self) -> None:
        """Tests that _get_all_telescopes extracts all associated telescopes when passed a list of sources."""
        res = _get_all_telescopes([self.src])
        assert set(res) == {"erass", "xmm"}

    def test_get_all_telescopes_source_input(self) -> None:
        """Tests that _get_all_telescopes extracts all associated telescopes when passed a single source."""
        res = _get_all_telescopes(self.src)
        assert set(res) == {"erass", "xmm"}

    def test_setup_global(self) -> None:
        """Tests that _setup_global prepares the source list, outer radii, and per-telescope return dictionaries."""
        res = _setup_global(
            self.src,
            Quantity(600, "kpc"),
            Quantity(600, "kpc"),
            "angr",
            True,
            5,
            None,
            None,
            NUM_CORES,
            4,
            True,
            ["xmm", "erass"],
        )

        assert isinstance(res[0][0], GalaxyCluster)
        assert isinstance(res[1][0], Quantity)
        assert isinstance(res[2], dict)
        assert set(res[2].keys()) == {"erass", "xmm"}
        assert len(res[2]["erass"]) == 1

    def test_setup_inv_abel_dens_onion_temp(self) -> None:
        """Tests that _setup_inv_abel_dens_onion_temp generates the required 3D gas density and temperature profiles."""
        res = _setup_inv_abel_dens_onion_temp(
            self.src, Quantity(600, "kpc"), "beta", "king", "vikhlinin_temp", Quantity(600, "kpc"), stacked_spectra=True
        )

        assert isinstance(res[0], list)
        assert len(res[0]) == 1
        assert isinstance(res[1], dict)
        assert isinstance(res[1][repr(self.src)]["erass"], GasDensity3D)
        assert isinstance(res[1][repr(self.src)]["xmm"], GasDensity3D)
        assert isinstance(res[2], dict)
        assert isinstance(res[2][repr(self.src)]["erass"], GasTemperature3D)
        assert isinstance(res[2][repr(self.src)]["xmm"], GasTemperature3D)
        assert isinstance(res[3], dict)
        assert isinstance(res[3][repr(self.src)], str)
        assert res[3][repr(self.src)] == "king"
        assert isinstance(res[4], dict)
        assert isinstance(res[4][repr(self.src)], str)
        assert res[4][repr(self.src)] == "vikhlinin_temp"
