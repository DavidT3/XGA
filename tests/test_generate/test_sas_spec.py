import unittest

import os
import shutil
import numpy as np

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.spec import evselect_spectrum
from xga.products import Spectrum

from .. import SRC_INFO

class TestTempFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], 
                                              r500=Quantity(500, 'kpc'), name=SRC_INFO['name'], 
                                              use_peak=False, search_distance={'erosita': 
                                              Quantity(3.6, 'deg')})

    def test_evselect_spectrum(self):
        evselect_spectrum(self.test_src, 'r500')

        spec = self.test_src.get_spectra('r500', telescope='xmm')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'xmm'
