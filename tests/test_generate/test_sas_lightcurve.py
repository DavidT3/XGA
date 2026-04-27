#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:20 AM. Copyright (c) The Contributors.

import os
import sys
import unittest

from xga.generate.sas.lightcurve import evselect_lightcurve
from xga.products.lightcurve import LightCurve
from ..utils import require_sas

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import get_test_source

class TestSasLcFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('xmm')

    @require_sas
    def test_evselect_lc(self):
        evselect_lightcurve(self.src, 'r500')

        lc = self.src.get_lightcurves('r500', telescope='xmm')
        if isinstance(lc, list):
            lc = lc[0]

        assert lc.telescope == 'xmm'
        assert isinstance(lc, LightCurve)



