import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.xspec.fit import single_temp_apec_profile

from .. import SRC_ALL_TELS

class TestSetupFuncs(unittest.TestCase):
    def test_single_temp_apec_profile_stacked_spectra_false(self):
        res = single_temp_apec_profile(SRC_ALL_TELS, Quantity([0, 150, 1000], 'kpc'))

        assert type(res) == GalaxyCluster