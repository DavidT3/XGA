#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:38 AM. Copyright (c) The Contributors.

import os
import sys
import unittest

from astropy.units import Quantity

from xga.sources import GalaxyCluster
from xga.xspec.fit import single_temp_apec_profile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import get_test_source

class TestFitProfileFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    def test_single_temp_apec_profile_stacked_spectra_false(self):
        res = single_temp_apec_profile(self.src, Quantity([0, 150, 1000], 'kpc'))

        assert type(res) == GalaxyCluster


