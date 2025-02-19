import unittest

import os
import shutil
import numpy as np

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.exceptions import TelescopeNotAssociatedError
from xga.products import Image, ExpMap

from .. import SRC_INFO

class TestPhotFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], 
                                              r500=Quantity(500, 'kpc'), name=SRC_INFO['name'], 
                                              use_peak=False, search_distance={'erosita': 
                                              Quantity(3.6, 'deg')})

        cls.test_src_ero = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], 
                                        r500=Quantity(500, 'kpc'), name=SRC_INFO['name'], 
                                        use_peak=False,
                                        telescope='erosita',
                                        search_distance={'erosita': Quantity(3.6, 'deg')})
    
    def test_evselect_image_no_tel_error(self):
        """
        Testing that TelescopeNotAssociatedError is raised when telescope isn't associated.
        """
        self.assertRaises(TelescopeNotAssociatedError, evselect_image(self.test_src_ero))

    def test_evselect_image(self):
        evselect_image(self.test_src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        im = self.test_src.get_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='xmm')

        assert im.telescope == 'xmm'
        assert im.energy_bounds[0] == Quantity(0.4, 'keV')
        assert im.energy_bounds[1] == Quantity(3, 'keV')
        assert isinstance(im, Image)
    
    def test_eexpmap(self):
        eexpmap(self.test_src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        exp = self.test_src.get_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert exp.telescope == 'erosita'
        assert exp.energy_bounds[0] == Quantity(0.4, 'keV')
        assert exp.energy_bounds[1] == Quantity(3, 'keV')
        assert isinstance(exp, ExpMap)
    
    def test_emosaic_incorrect_input(self):
        self.assertRaises(ValueError, emosaic(self.test_src, 'wrong'))
    
    def test_emosaic_image(self):
        emosaic(self.test_src, 'image', lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'))
        
        im = self.test_src.get_combined_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                               telescope='xmm')
        assert im.telescope == 'xmm'
        assert im.energy_bounds[0] == Quantity(0.4, 'keV')
        assert im.energy_bounds[1] == Quantity(3, 'keV')
        assert isinstance(im, Image)


    def test_emosaic_expmap(self):
        emosaic(self.test_src, 'expmap', lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'))
        
        exp = self.test_src.get_combined_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                               telescope='xmm')
        assert exp.telescope == 'xmm'
        assert exp.energy_bounds[0] == Quantity(0.4, 'keV')
        assert exp.energy_bounds[1] == Quantity(3, 'keV')
        assert isinstance(exp, ExpMap)



    

    