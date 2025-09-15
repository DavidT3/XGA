import unittest
import sys 
import os

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.spec import evselect_spectrum
from xga.products import Spectrum

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
