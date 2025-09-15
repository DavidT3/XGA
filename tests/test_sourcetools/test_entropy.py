#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 03/07/2025, 10:55. Copyright (c) The Contributors

import os
import sys
import unittest

from astropy.units import Quantity

from xga.products.profile import SpecificEntropy
# Now when xga is imported it will make a new census with the test_data
from xga.sourcetools.entropy import entropy_inv_abel_dens_onion_temp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import SRC_ALL_TELS

class TestEntropyFuncs(unittest.TestCase):
    def test_entropy_inv_abel_dens_onion_temp(self):
        res = entropy_inv_abel_dens_onion_temp(SRC_ALL_TELS, Quantity(600, 'kpc'), 'beta', 'king', 
                                               'vikhlinin_temp', Quantity(600, 'kpc'), 
                                               stacked_spectra=True)
        assert type(res) == dict
        assert set(res.keys()) == set(['erosita', 'xmm'])
        assert type(res['erosita'][0]) == SpecificEntropy
