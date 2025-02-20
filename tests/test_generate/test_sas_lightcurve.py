import unittest

import os
import shutil
import numpy as np

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.lightcurve import evselect_lightcurve
from xga.products import Lightcurve

from .. import SRC_INFO

class TestLCFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
    
    def test_evselect_lc(self):
        evselect_lightcurve(self.test_src, 'r500')

        lc = self.test_src.get_lightcurves('r500', telescope='xmm')

        assert lc.telescope == 'xmm'
        assert isinstance(lc, Lightcurve)
