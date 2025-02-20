import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.misc import evtool_combine_evts

from ..import SRC_ALL_TELS

class TestPhotFuncs(unittest.TestCase):
    def test_evtool_combine_evts(self):
        evtool_combine_evts(SRC_ALL_TELS)

        evtlist = SRC_ALL_TELS.get_products("combined_events", just_obj=False, telescope='erosita')[0]

        assert evtlist.telescope == 'erosita'
        assert set(evtlist.obs_ids) == set(SRC_ALL_TELS.obs_ids['erosita'])

if __name__ == "__main__":
     unittest.main()