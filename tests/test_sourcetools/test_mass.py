import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.sourcetools.mass import inv_abel_dens_onion_temp
from xga.products.profile import HydrostaticMass

from .. import SRC_ALL_TELS

class TestSetupFuncs(unittest.TestCase):
    def test_inv_abel_dens_onion_temp(self):
        res = inv_abel_dens_onion_temp(SRC_ALL_TELS, Quantity(600, 'kpc'), 'beta', 'king', 
                                       'vikhlinin_temp', Quantity(600, 'kpc'), stacked_spectra=True)

        assert type(res) == dict
        assert set(res.keys()) == set(['erosita', 'xmm'])
        assert type(res['erosita']) == HydrostaticMass