import unittest

from astropy.units import Quantity

from .. import set_up_test_config, restore_og_cfg

# I know it is horrible to write code in the middle of importing modules, but this needs to 
# happen before xga is imported, as we are moving config files
set_up_test_config()

# Now when xga is imported it will make a new census with the test_data
import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.sourcetools.entropy import entropy_inv_abel_dens_onion_temp
from xga.products.profile import SpecificEntropy

from .. import SRC_ALL_TELS

class TestEntropyFuncs(unittest.TestCase):
    def test_entropy_inv_abel_dens_onion_temp(self):
        res = entropy_inv_abel_dens_onion_temp(SRC_ALL_TELS, Quantity(600, 'kpc'), 'beta', 'king', 
                                               'vikhlinin_temp', Quantity(600, 'kpc'), 
                                               stacked_spectra=True)

        assert type(res) == dict
        assert set(res.keys()) == set(['erosita', 'xmm'])
        assert type(res['erosita']) == SpecificEntropy
