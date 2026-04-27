#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:23 AM. Copyright (c) The Contributors.

import os
import sys
import unittest

from astropy.units import Quantity

from xga.exceptions import TelescopeNotAssociatedError
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.products import Image, ExpMap
from ..utils import require_sas

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import get_test_source

class TestSasPhotFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('xmm')

    @require_sas
    def test_evselect_image_no_tel_error(self):
        """
        Testing that TelescopeNotAssociatedError is raised when telescope isn't associated.
        """
        with self.assertRaises(TelescopeNotAssociatedError):
            evselect_image(self.src)

    @require_sas
    def test_evselect_image(self):
        evselect_image(self.src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        im = self.src.get_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
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

    @require_sas
    def test_eexpmap(self):
        eexpmap(self.src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        exp = self.src.get_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
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

    @require_sas
    def test_emosaic_incorrect_input(self):
        with self.assertRaises(ValueError):
            emosaic(self.src, 'wrong')
    
    @require_sas
    def test_emosaic_image(self):
        emosaic(self.src, 'image', lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'))
        
        im = self.src.get_combined_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
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


    @require_sas
    def test_emosaic_expmap(self):
        emosaic(self.src, 'expmap', lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'))
        
        exp = self.src.get_combined_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
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



    

    


