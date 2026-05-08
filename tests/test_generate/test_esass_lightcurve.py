#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/8/26, 5:28 PM. Copyright (c) The Contributors.

import unittest

from xga.exceptions import NoProductAvailableError
from xga.generate.esass.lightcurve import srctool_lightcurve
from xga.products.lightcurve import LightCurve
from .. import get_test_source
from ..utils import require_esass


class TestEsassLcFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('erass')

    @require_esass
    def test_srctool_lightcurve_combine_insts_f_combine_obs_f(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=False and combine_obs=False
        """
        # Run the light curve generation - one light curve per eROSITA TM per Obs associated with this source
        srctool_lightcurve(self.src, 'r500', combine_tm=False, combine_obs=False)

        try:
            lc = self.src.get_lightcurves('r500', telescope='erass', inst='tm1')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        # There should be more than one light curve returned in this test, so the
        #  lc variable should be a list of LightCurve instances
        assert type(lc) == list

        for cur_lc in lc:
            assert isinstance(cur_lc, LightCurve)
            assert cur_lc.telescope == 'erass'
            assert cur_lc.obs_id != 'combined'
            assert cur_lc.instrument == 'tm1'

    @require_esass
    def test_srctool_lightcurve_combine_insts_t_combine_obs_f(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=True and combine_obs=False
        """
        # Run the light curve generation - one light curve per eROSITA Obs, combining all TMs
        srctool_lightcurve(self.src, 'r500', combine_tm=True, combine_obs=False)

        try:
            lc = self.src.get_lightcurves('r500', telescope='erass', inst='combined')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        # There should be more than one light curve returned in this test, so the
        #  lc variable should be a list of LightCurve instances
        assert type(lc) == list

        for cur_lc in lc:
            assert isinstance(cur_lc, LightCurve)
            assert cur_lc.telescope == 'erass'
            assert cur_lc.obs_id != 'combined'
            assert cur_lc.instrument == 'combined'

    @require_esass
    def test_srctool_lightcurve_combine_insts_f_combine_obs_t(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=False and combine_obs=True
        """
        # Run the light curve generation - one light curve per eROSITA TM, combining all ObsIDs
        #  associated with this source
        srctool_lightcurve(self.src, 'r500', combine_tm=False, combine_obs=True)

        try:
            lc = self.src.get_combined_lightcurves('r500', telescope='erass', inst='tm1')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving TM1 combined-ObsID light curve.")

        assert isinstance(lc, LightCurve)
        assert lc.telescope == 'erass'
        assert lc.obs_id == 'combined'
        assert lc.instrument == 'tm1'

        try:
            all_tm_lc = self.src.get_combined_lightcurves('r500', telescope='erass', inst=None)
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving combined-ObsID "
                      "light curves for every separate TM.")

        assert type(all_tm_lc) == list

        for cur_lc in all_tm_lc:
            assert isinstance(cur_lc, LightCurve)
            assert cur_lc.telescope == 'erass'
            assert cur_lc.obs_id == 'combined'
            assert cur_lc.instrument[:2] == 'tm'

    @require_esass
    def test_srctool_lightcurve_combine_insts_t_combine_obs_t(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=True and combine_obs=True
        """

        srctool_lightcurve(self.src, 'r500', combine_tm=True, combine_obs=True)

        lc = self.src.get_combined_lightcurves('r500', telescope='erass', inst='combined')

        if isinstance(lc, list):
            for l in lc:
                assert isinstance(l, LightCurve)
                assert l.telescope == 'erass'
                assert l.obs_id == 'combined'
                assert l.instrument == 'combined'
        else:
            assert isinstance(lc, LightCurve)
            assert lc.telescope == 'erass'
            assert lc.obs_id == 'combined'
            assert lc.instrument == 'combined'
