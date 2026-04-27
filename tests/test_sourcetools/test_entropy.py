#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:36 AM. Copyright (c) The Contributors.

import os
import sys
import unittest

from astropy.units import Quantity

from xga.products.profile import SpecificEntropy
# Now when xga is imported it will make a new census with the test_data
from xga.sourcetools.entropy import entropy_inv_abel_dens_onion_temp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import get_test_source

class TestEntropyFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    def test_entropy_inv_abel_dens_onion_temp(self):
        res = entropy_inv_abel_dens_onion_temp(self.src, Quantity(600, 'kpc'), 'beta', 'king', 
                                               'vikhlinin_temp', Quantity(600, 'kpc'), 
                                               stacked_spectra=True)
        assert type(res) == dict
        assert set(res.keys()) == set(['erosita', 'xmm'])
        assert type(res['erosita'][0]) == SpecificEntropy



