import unittest
import sys
import os

from astropy.units import Quantity

import xga

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import SRC_ALL_TELS


class TestGalaxyCluster(unittest.TestCase):
    def test_r500(self):
        assert SRC_ALL_TELS.r500 == Quantity(500, 'kpc')