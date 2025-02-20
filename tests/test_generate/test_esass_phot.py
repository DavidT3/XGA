import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap

from .. import SRC_INFO

class TestPhotFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     telescope='erosita',
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
        cls.test_src_all_tels = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], 
                                              r500=Quantity(500, 'kpc'), name=SRC_INFO['name'], 
                                              use_peak=False, search_distance={'erosita': 
                                              Quantity(3.6, 'deg')})

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
