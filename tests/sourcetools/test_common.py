import unittest

import os
import shutil
import numpy as np

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
from xga.products.profile import GasDensity3D, GasTemperature3D
from xga.sourcetools._common import _get_all_telescopes, _setup_global, \
                                    _setup_inv_abel_dens_onion_temp
from xga import NUM_CORES

class TestSetupFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(226.0318, -2.8046, 0.2093, r500=Quantity(500, 'kpc'),
                    name="1eRASS_J150407.6-024816", use_peak=False,
                    search_distance={'erosita': Quantity(3.6, 'deg')})

    @classmethod
    def tearDownClass(cls):
        """
        This is run once after all the tests.
        """
        # This function restores the user's original config file and deletes the test one made
        restore_og_cfg()
        # Then we will delete all the products that xga has made so there aren't loads of big files
        # in the package
#        shutil.rmtree('tests/test_data/xga_output')

    def test_get_all_telescopes_list_input(self):
        res = _get_all_telescopes([self.test_src])
        assert set(res) == set(['erosita', 'xmm'])

    def test_get_all_telescopes_source_input(self):
        res = _get_all_telescopes(self.test_src)
        assert set(res) == set(['erosita', 'xmm'])

    def test_setup_global(self):
        res = _setup_global(self.test_src, Quantity(600, 'kpc'), Quantity(600, 'kpc'), 'angr', True,
                            5, None, None, NUM_CORES, 4, True)
        
        assert res[0][0] == GalaxyCluster
        assert type(res[1][0]) == Quantity
        assert type(res[2]) == dict
        assert set(res[2].keys()) == set(['erosita', 'xmm'])
        assert len(res[2]['erosita']) == 1 

    def test_setup_inv_abel_dens_onion_temp(self):
        res = _setup_inv_abel_dens_onion_temp(self.test_src, Quantity(600, 'kpc'), 
                                              'beta', 'king', 'vikhlinin_temp', 
                                              Quantity(600, 'kpc'), 
                                              stacked_spectra=True)
        
        assert type(res[0]) == list
        assert len(res[0]) == 1
        assert type(res[1]) == dict
        assert type(res[1][repr(self.test_src)]['erosita']) == GasDensity3D
        assert type(res[1][repr(self.test_src)]['xmm']) == GasDensity3D
        assert type(res[2]) == dict
        assert type(res[2][repr(self.test_src)]['erosita']) == GasTemperature3D
        assert type(res[2][repr(self.test_src)]['xmm']) == GasTemperature3D
        assert type(res[3]) == dict
        assert type(res[3][repr(self.test_src)]) == str
        assert res[3][repr(self.test_src)] == 'king'
        assert type(res[4]) == dict
        assert type(res[4][repr(self.test_src)]) == str
        assert res[3][repr(self.test_src)] == 'vikhlinin_temp'