#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/24/26, 1:36 PM. Copyright (c) The Contributors.

import os
import sys
import unittest

from xga.generate.esass.misc import evtool_combine_evts
from ..utils import require_esass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import get_test_source

class TestEsassMiscFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('erosita')

    @require_esass
    def test_evtool_combine_evts(self):
        evtool_combine_evts(self.src)

        evtlist = self.src.get_products("combined_events", just_obj=False, telescope='erosita')[0][-1]
        assert evtlist.telescope == 'erosita'
        assert set(evtlist.obs_ids) == set(self.src.obs_ids['erosita'])

if __name__ == "__main__":
     unittest.main()
