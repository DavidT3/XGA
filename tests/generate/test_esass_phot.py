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

class TestPhotFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(226.0318, -2.8046, 0.2093, r500=Quantity(500, 'kpc'),
                                     name="1eRASS_J150407.6-024816", use_peak=False,
                                     telescope='erosita',
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
        cls.test_src_all_tels = GalaxyCluster(226.0318, -2.8046, 0.2093, r500=Quantity(500, 'kpc'),
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

    def test_evtool_image(self):
        evtool_image(self.test_src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        im = self.test_src.get_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert im.telescope == 'erosita'
        assert im.energy_bounds[0] == Quantity(0.4, 'keV')
        assert im.energy_bounds[1] == Quantity(3, 'keV')

    def test_evtool_image_combined_obs(self):
        evtool_image(self.test_src, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        im = self.test_src.get_combined_images(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert im.telescope == 'erosita'
        assert im.energy_bounds[0] == Quantity(0.4, 'keV')
        assert im.energy_bounds[1] == Quantity(3, 'keV')

    def test_expmap(self):
        expmap(self.test_src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        exp = self.test_src.get_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert exp.telescope == 'erosita'
        assert exp.energy_bounds[0] == Quantity(0.4, 'keV')
        assert exp.energy_bounds[1] == Quantity(3, 'keV')

    def test_expmap_combined_obs(self):
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        exp = self.test_src.get_combined_expmaps(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert exp.telescope == 'erosita'
        assert exp.energy_bounds[0] == Quantity(0.5, 'keV')
        assert exp.energy_bounds[1] == Quantity(3, 'keV')

if __name__ == "__main__":
     unittest.main()
