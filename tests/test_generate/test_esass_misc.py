#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/5/26, 11:51 PM. Copyright (c) The Contributors.

import unittest

from xga.generate.esass.misc import evtool_combine_evts
from .. import get_test_source
from ..utils import require_esass


class TestEsassMiscFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('erass')

    @require_esass
    def test_evtool_combine_evts(self):
        evtool_combine_evts(self.src)

        evtlist = self.src.get_products("combined_events", just_obj=False, telescope='erass')[0][-1]
        assert evtlist.telescope == 'erass'
        assert set(evtlist.obs_ids) == set(self.src.obs_ids['erass'])

if __name__ == "__main__":
     unittest.main()
