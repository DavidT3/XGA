import unittest

from astropy.units import Quantity
import os 
import sys

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.lightcurve import srctool_lightcurve
from xga.products.lightcurve import LightCurve


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import SRC_ALL_TELS

class TestEsassLcFuncs(unittest.TestCase):    
    def test_srctool_lightcurve_combine_insts_f_combine_obs_f(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=False and combine_obs=False
        """
        srctool_lightcurve(SRC_ALL_TELS, 'r500', combine_tm=False, combine_obs=False)

        lc = SRC_ALL_TELS.get_lightcurves('r500', telescope='erosita', inst='tm1')

        assert isinstance(lc, LightCurve)
        assert lc.telescope == 'erosita'
        assert lc.obs_id != 'combined'
        assert lc.instrument == 'tm1'

    def test_srctool_lightcurve_combine_insts_t_combine_obs_f(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=True and combine_obs=False
        """
        srctool_lightcurve(SRC_ALL_TELS, 'r500', combine_tm=True, combine_obs=False)

        lc = SRC_ALL_TELS.get_lightcurves('r500', telescope='erosita', inst='combined')

        if isinstance(lc, list):
            for l in lc:
                assert isinstance(l, LightCurve)
                assert l.telescope == 'erosita'
                assert l.obs_id != 'combined'
                assert l.instrument == 'combined'
        else:
            assert isinstance(lc, LightCurve)
            assert lc.telescope == 'erosita'
            assert lc.obs_id != 'combined'
            assert lc.instrument == 'combined'


    def test_srctool_lightcurve_combine_insts_f_combine_obs_t(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=False and combine_obs=True
        """
        srctool_lightcurve(SRC_ALL_TELS, 'r500', combine_tm=False, combine_obs=True)

        lc = SRC_ALL_TELS.get_combined_lightcurves('r500', telescope='erosita', inst='tm1')

        assert isinstance(lc, LightCurve)
        assert lc.telescope == 'erosita'
        assert lc.obs_id == 'combined'
        assert lc.instrument == 'tm1'

    def test_srctool_lightcurve_combine_insts_t_combine_obs_t(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=True and combine_obs=True
        """
        srctool_lightcurve(SRC_ALL_TELS, 'r500', combine_tm=True, combine_obs=True)

        lc = SRC_ALL_TELS.get_combined_lightcurves('r500', telescope='erosita')

        if isinstance(lc, list):
            for l in lc:
                assert isinstance(l, LightCurve)
                assert l.telescope == 'erosita'
                assert l.obs_id == 'combined'
                assert l.instrument == 'combined'
        else:
            assert isinstance(lc, LightCurve)
            assert lc.telescope == 'erosita'
            assert lc.obs_id == 'combined'
            assert lc.instrument == 'combined'