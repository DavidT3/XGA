import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap

from .. import SRC_ALL_TELS

class TestPhotFuncs(unittest.TestCase):
    def test_evtool_image(self):
        evtool_image(SRC_ALL_TELS, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        im = SRC_ALL_TELS.get_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert im.telescope == 'erosita'
        assert im.energy_bounds[0] == Quantity(0.4, 'keV')
        assert im.energy_bounds[1] == Quantity(3, 'keV')

    def test_evtool_image_combined_obs(self):
        evtool_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        im = SRC_ALL_TELS.get_combined_images(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert im.telescope == 'erosita'
        assert im.energy_bounds[0] == Quantity(0.4, 'keV')
        assert im.energy_bounds[1] == Quantity(3, 'keV')

    def test_expmap(self):
        expmap(SRC_ALL_TELS, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        exp = SRC_ALL_TELS.get_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert exp.telescope == 'erosita'
        assert exp.energy_bounds[0] == Quantity(0.4, 'keV')
        assert exp.energy_bounds[1] == Quantity(3, 'keV')

    def test_expmap_combined_obs(self):
        expmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        exp = SRC_ALL_TELS.get_combined_expmaps(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert exp.telescope == 'erosita'
        assert exp.energy_bounds[0] == Quantity(0.5, 'keV')
        assert exp.energy_bounds[1] == Quantity(3, 'keV')

if __name__ == "__main__":
     unittest.main()
