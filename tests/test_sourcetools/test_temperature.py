import unittest
import sys 
import os
import numpy as np

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.sourcetools.temperature import _ann_bins_setup, _snr_bins, _cnt_bins, \
                                        min_snr_proj_temp_prof, min_cnt_proj_temp_prof, \
                                        onion_deproj_temp_prof
from xga.products.profile import GasDensity3D


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import SRC_ALL_TELS

class TestTempFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = SRC_ALL_TELS.disassociate_obs('xmm')

    def test_ann_bins_setup_working_with_erosita(self):
        """
        Testing _ann_bins_setup doesn't error unexpectedly with erosita data.
        """
        # need combined ratemaps already generated to have _ann_bins_setup run
        evtool_image(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)

        retrn = _ann_bins_setup(self.test_src,  Quantity(500, 'kpc'), Quantity(20, 'kpc'), 
                        Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        
        # checking the rtmap is the correct telescope
        assert retrn[0]['erosita'].telescope == 'erosita'
        # checking cur_rads is the correct type
        assert type(retrn[1]['erosita']) == np.ndarray
        # checking max_ann is the correct type
        assert type(retrn[2]['erosita']) == int
        # checking ann_masks is the correct type
        assert type(retrn[3]['erosita']) == np.ndarray
        # checking back_mask is the correct type
        assert type(retrn[4]['erosita']) == np.ndarray
        # checking pix_centre is the correct type and has correct units
        assert retrn[5]['erosita'].unit == 'pix'
        # checking corr_mask is the correct type
        assert type(retrn[6]['erosita']) == np.ndarray
        # checking pix_to_deg is the correct type and has correct units
        assert retrn[7]['erosita'].unit == 'deg/pix'


    def test_snr_bins_working_with_erosita(self):
        """
        Testing _snr_bins doesn't error unexpectedly with erosita data.
        """
        # need combined ratemaps already generated to have _snr_bins_setup run
        evtool_image(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)

        retrn = _snr_bins(self.test_src, Quantity(500, 'kpc'), 3, Quantity(20, 'kpc'), 
                          Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        
        # Checking that final_rads is the right type and in the right units
        assert retrn[0]['erosita'].unit == 'arcsec'
        # checking snrs is the correct type
        assert type(retrn[1]['erosita']) == np.ndarray
        # checking max_ann is the correct type
        assert type(retrn[2]['erosita']) == int

    def test_cnt_bins_working_with_erosita(self):
        """
        Testing _cnt_bins doesn't error unexpectedly with erosita data.
        """
        # need combined ratemaps already generated to have _snr_bins_setup run
        evtool_image(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)

        retrn = _cnt_bins(self.test_src, Quantity(500, 'kpc'), 10, Quantity(20, 'kpc'), 
                  Quantity(0.5, 'keV'), Quantity(2, 'keV'))
    
        # Checking that final_rads is the right type and in the right units
        assert retrn[0]['erosita'].unit == 'arcsec'
        # checking cnts is the correct type
        assert type(retrn[1]['erosita']) == Quantity
        # checking max_ann is the correct type
        assert type(retrn[2]['erosita']) == int

    def test_min_snr_proj_temp_prof_w_two_tscopes(self):
        
        # need combined ratemaps already generated for erosita
        evtool_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(self.test_src, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        # and for xmm
        evselect_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        eexpmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        emosaic(SRC_ALL_TELS, 'image')
        emosaic(SRC_ALL_TELS, 'expmap')

        all_rads = min_snr_proj_temp_prof(SRC_ALL_TELS, Quantity(500, 'kpc'),
                                          stacked_spectra=True)
        
        assert all_rads['erosita'][0].unit == 'arcsec'
        assert all_rads['xmm'][0].unit == 'arcsec'


    def test_min_cnt_proj_temp_prof_w_two_tscopes(self):
        
        # need combined ratemaps already generated for erosita
        evtool_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        # and for xmm
        evselect_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        eexpmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        emosaic(SRC_ALL_TELS, 'image')
        emosaic(SRC_ALL_TELS, 'expmap')

        all_rads = min_cnt_proj_temp_prof(SRC_ALL_TELS, Quantity(500, 'kpc'), 
                                          stacked_spectra=True)
        
        assert all_rads['erosita'].unit == 'arcsec'
        assert all_rads['xmm'].unit == 'arcsec'

    def test_onion_deproj_temp_prof(self):

        # need combined ratemaps already generated for erosita
        evtool_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        expmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'), combine_obs=True)
        # and for xmm
        evselect_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        eexpmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2, 'keV'))
        emosaic(SRC_ALL_TELS, 'image')
        emosaic(SRC_ALL_TELS, 'expmap')

        res = onion_deproj_temp_prof(SRC_ALL_TELS, Quantity(600, 'kpc'), 
                                     stacked_spectra=True)
        
        assert type(res['xmm'][0]) == GasDensity3D
        assert type(res['erosita'][0]) == GasDensity3D


if __name__ == "__main__":
     unittest.main()