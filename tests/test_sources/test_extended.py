#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:35 AM. Copyright (c) The Contributors.

import os
import sys
import unittest

from astropy.units import Quantity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import get_test_source


class TestGalaxyCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    def test_r500(self):
        assert self.src.r500 == Quantity(500, 'kpc')


