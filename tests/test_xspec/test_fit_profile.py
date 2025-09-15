import unittest
import sys
import os

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.xspec.fit import single_temp_apec_profile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import SRC_ALL_TELS

class TestFitProfileFuncs(unittest.TestCase):
    def test_single_temp_apec_profile_stacked_spectra_false(self):
        res = single_temp_apec_profile(SRC_ALL_TELS, Quantity([0, 150, 1000], 'kpc'))

        assert type(res) == GalaxyCluster