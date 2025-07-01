import unittest
import os
import sys

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.lightcurve import evselect_lightcurve
from xga.products.lightcurve import LightCurve


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import SRC_ALL_TELS

class TestSasLcFuncs(unittest.TestCase):    
    def test_evselect_lc(self):
        evselect_lightcurve(SRC_ALL_TELS, 'r500')

        lc = SRC_ALL_TELS.get_lightcurves('r500', telescope='xmm')

        assert lc.telescope == 'xmm'
        assert isinstance(lc, LightCurve)
