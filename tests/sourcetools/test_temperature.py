import unittest

import unittest
import os
import shutil

from astropy.units import Quantity

from .. import set_up_test_config, restore_og_cfg

# I know it is horrible to write code in the middle of importing modules, but this needs to 
# happen before xga is imported, as we are moving config files
set_up_test_config()

# Now when xga is imported it will make a new census with the test_data
import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap
from xga.sourcetools.temperature import _ann_bins_setup

class TestAnnBinsSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(149.59209, -11.05972, 0.16, r500=Quantity(1200, 'kpc'), 
                                      r200=Quantity(1700, 'kpc'), name="A907", use_peak=False,
                                      telescope='erosita',
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

    def test_ann_bins_setup_working_with_erosita(self):
        """
        Testing _ann_bins_setup doesn't error unexpectedly with erosita data.
        """

        # need combined ratemaps already generated to have _ann_bins_setup run
        evtool_image(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)

        print(self.test_src.get_combined_ratemaps(Quantity(0.5, 'keV'), Quantity(2, 'keV'), None, None, None, None, None, telescope='erosita'))
        _ann_bins_setup(self.test_src,  Quantity(500, 'kpc'), Quantity(20, 'kpc'), 
                        Quantity(0.5, 'keV'), Quantity(2, 'keV'))





if __name__ == "__main__":
     unittest.main()