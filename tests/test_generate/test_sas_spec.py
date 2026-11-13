#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 7:14 PM. Copyright (c) The Contributors.
"""Tests for XGA's SAS spectral product generation wrapper functions."""

import unittest

from astropy.units import Quantity

from xga.exceptions import NoProductAvailableError
from xga.generate.sas.lightcurve import evselect_lightcurve
from xga.generate.sas.spec import evselect_spectrum, spectrum_set
from xga.products import AnnularSpectra, LightCurve, Spectrum

from .. import get_test_source
from ..utils import require_sas


class TestSasSpecFuncs(unittest.TestCase):
    """Test cases for the XMM SAS spectral product generation functions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the XMM test source required for the SAS spectral generation tests."""
        cls.src = get_test_source("xmm")

    @require_sas
    def test_evselect_spectrum(self) -> None:
        """Testing that evselect_spectrum generates and stores XMM spectra for a source."""
        evselect_spectrum(self.src, "r500")

        try:
            spec = self.src.get_spectra("r500", telescope="xmm")
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        if isinstance(spec, list):
            for sp in spec:
                assert isinstance(sp, Spectrum)
                assert sp.telescope == "xmm"
        else:
            assert isinstance(spec, Spectrum)
            assert spec.telescope == "xmm"

    @require_sas
    def test_spectrum_set(self) -> None:
        """Testing spectrum_set for annular spectrum generation."""
        radii = Quantity([0, 100, 200, 500], "kpc")
        spectrum_set(self.src, radii)

        try:
            ann_spec = self.src.get_annular_spectra(radii, telescope="xmm")
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        assert isinstance(ann_spec, AnnularSpectra)
        assert ann_spec.telescope == "xmm"
        assert len(ann_spec) == 3

    @require_sas
    def test_evselect_lightcurve(self) -> None:
        """Testing evselect_lightcurve generation."""
        evselect_lightcurve(self.src, "r500")

        try:
            lc = self.src.get_lightcurves("r500", telescope="xmm")
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        if isinstance(lc, list):
            for cur_lc in lc:
                assert isinstance(cur_lc, LightCurve)
                assert cur_lc.telescope == "xmm"
        else:
            assert isinstance(lc, LightCurve)
            assert lc.telescope == "xmm"
