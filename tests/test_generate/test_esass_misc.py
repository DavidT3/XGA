import unittest

from astropy.units import Quantity
import sys
import os

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.misc import evtool_combine_evts


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..import SRC_ALL_TELS

class TestEsassMiscFuncs(unittest.TestCase):
    def test_evtool_combine_evts(self):
        evtool_combine_evts(SRC_ALL_TELS)

        evtlist = SRC_ALL_TELS.get_products("combined_events", just_obj=False, telescope='erosita')[0][0]
        print(evtlist)
        assert evtlist.telescope == 'erosita'
        assert set(evtlist.obs_ids) == set(SRC_ALL_TELS.obs_ids['erosita'])

if __name__ == "__main__":
     unittest.main()