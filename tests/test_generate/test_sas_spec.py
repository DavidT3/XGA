#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 7:14 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.exceptions import NoProductAvailableError
from xga.generate.sas.lightcurve import evselect_lightcurve
from xga.generate.sas.spec import evselect_spectrum, spectrum_set
from xga.products import Spectrum, LightCurve, AnnularSpectra
from .. import get_test_source
from ..utils import require_sas


class TestSasSpecFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('xmm')

    @require_sas
    def test_evselect_spectrum(self):
        evselect_spectrum(self.src, 'r500')

        try:
            spec = self.src.get_spectra('r500', telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        if isinstance(spec, list):
            for sp in spec:
                assert isinstance(sp, Spectrum)
                assert sp.telescope == 'xmm'
        else:
            assert isinstance(spec, Spectrum)
            assert spec.telescope == 'xmm'

    @require_sas
    def test_spectrum_set(self):
        """
        Testing spectrum_set for annular spectrum generation.
        """
        radii = Quantity([0, 100, 200, 500], 'kpc')
        spectrum_set(self.src, radii)

        try:
            ann_spec = self.src.get_annular_spectra(radii, telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        assert isinstance(ann_spec, AnnularSpectra)
        assert ann_spec.telescope == 'xmm'
        assert len(ann_spec) == 3

    @require_sas
    def test_evselect_lightcurve(self):
        """
        Testing evselect_lightcurve generation.
        """
        evselect_lightcurve(self.src, 'r500')

        try:
            lc = self.src.get_lightcurves('r500', telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        if isinstance(lc, list):
            for l in lc:
                assert isinstance(l, LightCurve)
                assert l.telescope == 'xmm'
        else:
            assert isinstance(lc, LightCurve)
            assert lc.telescope == 'xmm'



