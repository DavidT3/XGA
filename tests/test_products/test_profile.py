#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/20/26, 12:46 PM. Copyright (c) The Contributors.

import os
import unittest

from astropy.units import Quantity

from xga.products.profile import ProjectedGasTemperature1D
from xga.sourcetools.misc import rad_to_ang
from .. import MISC_OUTPUT_TESTS


class TestProfileView(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        radii = Quantity([100, 200, 300, 400], 'kpc')
        deg_radii = rad_to_ang(radii, z=0.16)

        cls.tp_one = ProjectedGasTemperature1D(radii,
                                               Quantity([1, 3, 4, 3.4], 'keV'),
                                               Quantity([0, 0], 'deg'),
                                               "magic-galaxy-cluster",
                                               "an-obsid",
                                               "an-inst",
                                               deg_radii=deg_radii)

        cls.tp_two = ProjectedGasTemperature1D(radii,
                                               Quantity([10, 8, 7.1, 3.4], 'keV'),
                                               Quantity([1, 1], 'deg'),
                                               "another-magic-galaxy-cluster",
                                               "an-obsid",
                                               "an-inst",
                                               deg_radii=deg_radii)

    def test_default_view(self):
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        self.tp_one.save_view(os.path.join(test_out_path, "tp_one_default.png"))

    def test_default_aggregate_view(self):
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        agg_prof = self.tp_one + self.tp_two

        agg_prof.view(save_path=os.path.join(test_out_path, "tp_one_two_agg_default.png"))
