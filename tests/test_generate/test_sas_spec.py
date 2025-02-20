import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.spec import evselect_spectrum
from xga.products import Spectrum

from .. import SRC_ALL_TELS

class TestSasSpecFuncs(unittest.TestCase):
    def test_evselect_spectrum(self):
        evselect_spectrum(SRC_ALL_TELS, 'r500')

        spec = SRC_ALL_TELS.get_spectra('r500', telescope='xmm')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'xmm'
