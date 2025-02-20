import unittest

import os
import shutil
import numpy as np

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.lightcurve import srctool_lightcurve
from xga.products import Lightcurve

from .. import SRC_INFO

class TestLCFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
    
    def test_srctool_lightcurve_combine_insts_f_combine_obs_f(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=False and combine_obs=False
        """
        srctool_lightcurve(self.test_src, 'r500', combine_tm=False, combine_obs=False)

        lc = self.test_src.get_lightcurves('r500', telescope='erosita', inst='TM1')

        assert isinstance(lc, Lightcurve)
        assert lc.telescope == 'erosita'
        assert lc.obs_id != 'combined'
        assert lc.instrument == 'TM1'

    def test_srctool_lightcurve_combine_insts_t_combine_obs_f(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=True and combine_obs=False
        """
        srctool_lightcurve(self.test_src, 'r500', combine_tm=True, combine_obs=False)

        lc = self.test_src.get_lightcurves('r500', telescope='erosita', inst='combined')

        assert isinstance(lc, Lightcurve)
        assert lc.telescope == 'erosita'
        assert lc.obs_id != 'combined'
        assert lc.instrument == 'combined'

    def test_srctool_lightcurve_combine_insts_f_combine_obs_t(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=False and combine_obs=True
        """
        srctool_lightcurve(self.test_src, 'r500', combine_tm=False, combine_obs=True)

        lc = self.test_src.get_combined_lightcurves('r500', telescope='erosita', inst='TM1')

        assert isinstance(lc, Lightcurve)
        assert lc.telescope == 'erosita'
        assert lc.obs_id == 'combined'
        assert lc.instrument == 'TM1'

    def test_srctool_lightcurve_combine_insts_t_combine_obs_t(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=True and combine_obs=True
        """
        srctool_lightcurve(self.test_src, 'r500', combine_tm=True, combine_obs=True)

        lc = self.test_src.get_combined_lightcurves('r500', telescope='erosita')

        assert isinstance(lc, Lightcurve)
        assert lc.telescope == 'erosita'
        assert lc.obs_id == 'combined'
        assert lc.instrument == 'combined'