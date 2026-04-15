import unittest
import sys 
import os

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.spec import evselect_spectrum, spectrum_set
from xga.generate.sas.lightcurve import evselect_lightcurve
from xga.products import Spectrum, LightCurve, AnnularSpectra

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import SRC_ALL_TELS

class TestSasSpecFuncs(unittest.TestCase):
    def test_evselect_spectrum(self):
        evselect_spectrum(SRC_ALL_TELS, 'r500')

        spec = SRC_ALL_TELS.get_spectra('r500', telescope='xmm')

        if isinstance(spec, list):
            for sp in spec:
                assert isinstance(sp, Spectrum)
                assert sp.telescope == 'xmm'
        else:
            assert isinstance(spec, Spectrum)
            assert spec.telescope == 'xmm'

    def test_spectrum_set(self):
        """
        Testing spectrum_set for annular spectrum generation.
        """
        radii = Quantity([0, 100, 200, 500], 'kpc')
        spectrum_set(SRC_ALL_TELS, radii)

        ann_spec = SRC_ALL_TELS.get_annular_spectra(radii, telescope='xmm')
        assert isinstance(ann_spec, AnnularSpectra)
        assert ann_spec.telescope == 'xmm'
        assert len(ann_spec) == 3

    def test_evselect_lightcurve(self):
        """
        Testing evselect_lightcurve generation.
        """
        evselect_lightcurve(SRC_ALL_TELS, 'r500')

        lc = SRC_ALL_TELS.get_lightcurves('r500', telescope='xmm')

        if isinstance(lc, list):
            for l in lc:
                assert isinstance(l, LightCurve)
                assert l.telescope == 'xmm'
        else:
            assert isinstance(lc, LightCurve)
            assert lc.telescope == 'xmm'
