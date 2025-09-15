import unittest
import sys
import os

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.exceptions import TelescopeNotAssociatedError
from xga.products import Image, ExpMap


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import SRC_ALL_TELS, SRC_ERO

class TestSasPhotFuncs(unittest.TestCase):    
    def test_evselect_image_no_tel_error(self):
        """
        Testing that TelescopeNotAssociatedError is raised when telescope isn't associated.
        """
        with self.assertRaises(TelescopeNotAssociatedError):
            evselect_image(SRC_ERO)

    def test_evselect_image(self):
        evselect_image(SRC_ALL_TELS, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        im = SRC_ALL_TELS.get_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='xmm')
        if isinstance(im, list):
            for i in im:
                assert i.telescope == 'xmm'
                assert i.energy_bounds[0] == Quantity(0.4, 'keV')
                assert i.energy_bounds[1] == Quantity(3, 'keV')
                assert isinstance(i, Image)
        else:
            assert im.telescope == 'xmm'
            assert im.energy_bounds[0] == Quantity(0.4, 'keV')
            assert im.energy_bounds[1] == Quantity(3, 'keV')
            assert isinstance(im, Image)

    
    def test_eexpmap(self):
        eexpmap(SRC_ALL_TELS, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        exp = SRC_ALL_TELS.get_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                       telescope='xmm')
        if isinstance(exp, list):
            for e in exp:
                assert e.telescope == 'xmm'
                assert e.energy_bounds[0] == Quantity(0.4, 'keV')
                assert e.energy_bounds[1] == Quantity(3, 'keV')
                assert isinstance(e, ExpMap)
        else:
            assert exp.telescope == 'xmm'
            assert exp.energy_bounds[0] == Quantity(0.4, 'keV')
            assert exp.energy_bounds[1] == Quantity(3, 'keV')
            assert isinstance(exp, ExpMap)

    
    def test_emosaic_incorrect_input(self):
        with self.assertRaises(ValueError):
            emosaic(SRC_ALL_TELS, 'wrong')
    
    def test_emosaic_image(self):
        emosaic(SRC_ALL_TELS, 'image', lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'))
        
        im = SRC_ALL_TELS.get_combined_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                               telescope='xmm')
        if isinstance(im,list):
            for i in im:
                assert i.telescope == 'xmm'
                assert i.energy_bounds[0] == Quantity(0.4, 'keV')
                assert i.energy_bounds[1] == Quantity(3, 'keV')
                assert isinstance(i, Image)
        else:
            assert im.telescope == 'xmm'
            assert im.energy_bounds[0] == Quantity(0.4, 'keV')
            assert im.energy_bounds[1] == Quantity(3, 'keV')
            assert isinstance(im, Image)


    def test_emosaic_expmap(self):
        emosaic(SRC_ALL_TELS, 'expmap', lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'))
        
        exp = SRC_ALL_TELS.get_combined_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                               telescope='xmm')
        if isinstance(exp, list):
            for e in exp:
                assert e.telescope == 'xmm'
                assert e.energy_bounds[0] == Quantity(0.4, 'keV')
                assert e.energy_bounds[1] == Quantity(3, 'keV')
                assert isinstance(e, ExpMap)
        else:
            assert exp.telescope == 'xmm'
            assert exp.energy_bounds[0] == Quantity(0.4, 'keV')
            assert exp.energy_bounds[1] == Quantity(3, 'keV')
            assert isinstance(exp, ExpMap)



    

    