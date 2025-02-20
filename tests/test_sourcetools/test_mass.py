import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.sourcetools.mass import inv_abel_dens_onion_temp
from xga.products.profile import HydrostaticMass

from .. import SRC_INFO

class TestSetupFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     telescope='erosita',
                                     search_distance={'erosita': Quantity(3.6, 'deg')})

    def test_inv_abel_dens_onion_temp(self):
        res = inv_abel_dens_onion_temp(self.test_src, Quantity(600, 'kpc'), 'beta', 'king', 
                                       'vikhlinin_temp', Quantity(600, 'kpc'), stacked_spectra=True)

        assert type(res) == dict
        assert set(res.keys()) == set(['erosita', 'xmm'])
        assert type(res['erosita']) == HydrostaticMass