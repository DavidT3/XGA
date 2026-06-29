#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/8/26, 5:49 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.exceptions import NoProductAvailableError
from xga.generate.esass.lightcurve import srctool_lightcurve
from xga.products.lightcurve import LightCurve
from .. import get_test_source
from ..utils import require_esass


class TestEsassLcFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('erass')

        # Specify the lower and upper energy bounds for the light curves generated and
        #  retrieved in these tests.
        cls.lc_lo_en = Quantity(0.5, 'keV')
        cls.lc_hi_en = Quantity(2.0, 'keV')

    @require_esass
    def test_srctool_lightcurve_combine_insts_f_combine_obs_f(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=False and combine_obs=False
        """
        # Run the light curve generation - one light curve per eROSITA TM per Obs associated with this source
        srctool_lightcurve(self.src,
                           'r500',
                           lo_en=self.lc_lo_en,
                           hi_en=self.lc_hi_en,
                           combine_tm=False, combine_obs=False)

        try:
            lc = self.src.get_lightcurves('r500',
                                          lo_en=self.lc_lo_en,
                                          hi_en=self.lc_hi_en,
                                          telescope='erass',
                                          inst='tm1')
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
            # Might as well test the energy bounds as well
            assert cur_lc.energy_bounds[0] == self.lc_lo_en
            assert cur_lc.energy_bounds[1] == self.lc_hi_en

    @require_esass
    def test_srctool_lightcurve_combine_insts_t_combine_obs_f(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=True and combine_obs=False
        """
        # Run the light curve generation - one light curve per eROSITA Obs, combining all TMs
        srctool_lightcurve(self.src,
                           'r500',
                           lo_en=self.lc_lo_en,
                           hi_en=self.lc_hi_en,
                           combine_tm=True,
                           combine_obs=False)

        try:
            lc = self.src.get_lightcurves('r500',
                                          lo_en=self.lc_lo_en,
                                          hi_en=self.lc_hi_en,
                                          telescope='erass',
                                          inst='combined')
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
            # Might as well test the energy bounds as well
            assert cur_lc.energy_bounds[0] == self.lc_lo_en
            assert cur_lc.energy_bounds[1] == self.lc_hi_en

    @require_esass
    def test_srctool_lightcurve_combine_insts_f_combine_obs_t(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=False and combine_obs=True
        """
        # Run the light curve generation - one light curve per eROSITA TM, combining all ObsIDs
        #  associated with this source
        srctool_lightcurve(self.src,
                           'r500',
                           lo_en=self.lc_lo_en,
                           hi_en=self.lc_hi_en,
                           combine_tm=False,
                           combine_obs=True)

        try:
            lc = self.src.get_combined_lightcurves('r500',
                                                   lo_en=self.lc_lo_en,
                                                   hi_en=self.lc_hi_en,
                                                   telescope='erass',
                                                   inst='tm1')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving TM1 combined-ObsID light curve.")

        assert isinstance(lc, LightCurve)
        assert lc.telescope == 'erass'
        assert lc.obs_id == 'combined'
        assert lc.instrument == 'tm1'
        assert lc.energy_bounds[0] == self.lc_lo_en
        assert lc.energy_bounds[1] == self.lc_hi_en

        try:
            all_tm_lc = self.src.get_combined_lightcurves('r500',
                                                          lo_en=self.lc_lo_en,
                                                          hi_en=self.lc_hi_en,
                                                          telescope='erass',
                                                          inst=None)
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving combined-ObsID "
                      "light curves for every separate TM.")

        assert type(all_tm_lc) == list

        for cur_lc in all_tm_lc:
            assert isinstance(cur_lc, LightCurve)
            assert cur_lc.telescope == 'erass'
            assert cur_lc.obs_id == 'combined'
            assert cur_lc.instrument[:2] == 'tm'
            # Might as well test the energy bounds as well
            assert cur_lc.energy_bounds[0] == self.lc_lo_en
            assert cur_lc.energy_bounds[1] == self.lc_hi_en

    @require_esass
    def test_srctool_lightcurve_combine_insts_t_combine_obs_t(self):
        """
        Testing srctool_lightcurve with the arguments combine_insts=True and combine_obs=True
        """

        # Run the light curve generation - this should only result in a single light curve as
        #  we're combining all ObsIDs and all their TMs into a single light curve.
        srctool_lightcurve(self.src,
                           'r500',
                           lo_en=self.lc_lo_en,
                           hi_en=self.lc_hi_en,
                           combine_tm=True,
                           combine_obs=True)

        try:
            # Specifying the lo_en and hi_en is particularly important for this test, as
            #  we expect a single combined ObsID - combined TM light curve to be returned,
            #  and if there have been other equivalent LC generations run but with different
            #  energy bounds then we'll get a list returned (though given we control these
            #  tests, and they should be run in a clean environment I'm not THAT worried).
            lc = self.src.get_combined_lightcurves('r500',
                                                   lo_en=self.lc_lo_en,
                                                   hi_en=self.lc_hi_en,
                                                   telescope='erass',
                                                   inst='combined')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        # This MUST be a single LightCurve return, if we're getting multiple instances in a
        #  list then something has gone wrong
        assert type(lc) ==  LightCurve
        assert lc.telescope == 'erass'
        assert lc.obs_id == 'combined'
        assert lc.instrument == 'combined'
        assert lc.energy_bounds[0] == self.lc_lo_en
        assert lc.energy_bounds[1] == self.lc_hi_en
