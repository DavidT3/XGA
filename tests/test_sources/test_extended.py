#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 5:23 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from .. import get_test_source


class TestGalaxyCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    def test_r500(self):
        assert self.src.r500 == Quantity(500, 'kpc')


