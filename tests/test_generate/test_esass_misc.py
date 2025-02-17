import numpy as np
import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.misc import evtool_combine_evts

from ..import SRC_INFO

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

    def test_evtool_combine_evts(self):
        evtool_combine_evts(self.test_src)

        evtlist = self.test_src.get_products("combined_events", just_obj=False, telescope='erosita')[0]

        assert evtlist.telescope == 'erosita'
        assert set(evtlist.obs_ids) == set(self.test_src.obs_ids['erosita'])

if __name__ == "__main__":
     unittest.main()