#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 04/07/2025, 12:05. Copyright (c) The Contributors

import os
import sys
import unittest

import numpy as np
from astropy.units import Quantity

from xga.generate.esass.phot import evtool_image, expmap
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.products.profile import GasDensity3D
from xga.sourcetools.temperature import _ann_bins_setup, _snr_bins, _cnt_bins, \
    min_snr_proj_temp_prof, min_cnt_proj_temp_prof, \
    onion_deproj_temp_prof

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import SRC_ERO, SRC_ALL_TELS

class TestTempFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = SRC_ERO
        cls.all_tels = SRC_ALL_TELS

    def test_ann_bins_setup_working_with_erosita(self):
        """
        Testing _ann_bins_setup doesn't error unexpectedly with erosita data.
        """
        # need combined ratemaps already generated to have _ann_bins_setup run
        evtool_image(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)

        retrn = _ann_bins_setup(self.test_src,  Quantity(500, 'kpc'), Quantity(20, 'kpc'), 
                                Quantity(0.5, 'keV'), Quantity(2, 'keV'), 'erosita')
        
        # checking the rtmap is the correct telescope
        assert retrn[0].telescope == 'erosita'
        # checking cur_rads is the correct type
        assert type(retrn[1]) == np.ndarray
        # checking max_ann is the correct type
        assert type(retrn[2]) == int
        # checking ann_masks is the correct type
        assert type(retrn[3]) == np.ndarray
        # checking back_mask is the correct type
        assert type(retrn[4]) == np.ndarray
        # checking pix_centre is the correct type and has correct units
        assert retrn[5].unit == 'pix'
        # checking corr_mask is the correct type
        assert type(retrn[6]) == np.ndarray
        # checking pix_to_deg is the correct type and has correct units
        assert retrn[7].unit == 'deg/pix'


    def test_snr_bins_working_with_erosita(self):
        """
        Testing _snr_bins doesn't error unexpectedly with erosita data.
        """
        # need combined ratemaps already generated to have _snr_bins_setup run
        evtool_image(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)

        retrn = _snr_bins(self.test_src, Quantity(500, 'kpc'), 3, Quantity(20, 'kpc'), 
                          Quantity(0.5, 'keV'), Quantity(2, 'keV'), 'erosita')
        
        # Checking that final_rads is the right type and in the right units
        assert retrn[0].unit == 'arcsec'
        # checking snrs is the correct type
        assert type(retrn[1]) == np.ndarray
        # checking max_ann is the correct type
        assert type(retrn[2]) == int

    def test_cnt_bins_working_with_erosita(self):
        """
        Testing _cnt_bins doesn't error unexpectedly with erosita data.
        """
        # need combined ratemaps already generated to have _snr_bins_setup run
        evtool_image(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)

        retrn = _cnt_bins(self.test_src, Quantity(500, 'kpc'), 10, Quantity(20, 'kpc'), 
                          Quantity(0.5, 'keV'), Quantity(2, 'keV'), 'erosita')
    
        # Checking that final_rads is the right type and in the right units
        assert retrn[0].unit == 'arcsec'
        # checking cnts is the correct type
        assert type(retrn[1]) == Quantity
        # checking max_ann is the correct type
        assert type(retrn[2]) == int

    def test_min_snr_proj_temp_prof_w_two_tscopes(self):
        
        # need combined ratemaps already generated for erosita
        evtool_image(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        # and for xmm
        evselect_image(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        eexpmap(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        emosaic(self.all_tels, 'image')
        emosaic(self.all_tels, 'expmap')

        all_rads = min_snr_proj_temp_prof(self.all_tels, Quantity(500, 'kpc'),
                                          stacked_spectra=True)
        
        assert all_rads['erosita'][0].unit == 'arcsec'
        assert all_rads['xmm'][0].unit == 'arcsec'


    def test_min_cnt_proj_temp_prof_w_two_tscopes(self):
        
        # need combined ratemaps already generated for erosita
        evtool_image(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        # and for xmm
        evselect_image(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        eexpmap(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        emosaic(self.all_tels, 'image')
        emosaic(self.all_tels, 'expmap')

        all_rads = min_cnt_proj_temp_prof(self.all_tels, Quantity(500, 'kpc'), 
                                          stacked_spectra=True)
        
        assert all_rads['erosita'].unit == 'arcsec'
        assert all_rads['xmm'].unit == 'arcsec'

    def test_onion_deproj_temp_prof(self):

        # need combined ratemaps already generated for erosita
        evtool_image(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        # and for xmm
        evselect_image(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        eexpmap(self.all_tels, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        emosaic(self.all_tels, 'image')
        emosaic(self.all_tels, 'expmap')

        res = onion_deproj_temp_prof(self.all_tels, Quantity(600, 'kpc'), 
                                     stacked_spectra=True)
        
        assert type(res['xmm'][0]) == GasDensity3D
        assert type(res['erosita'][0]) == GasDensity3D


if __name__ == "__main__":
     unittest.main()