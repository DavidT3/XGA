#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 7:14 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.exceptions import TelescopeNotAssociatedError, NoProductAvailableError
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.products import Image, ExpMap
from .. import get_test_source
from ..utils import require_sas


class TestSasPhotFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Grab a source with XMM in
        cls.src = get_test_source('xmm')

        # Additionally, grab one that we know DOESN'T have any XMM
        cls.no_xmm_src = get_test_source('erass')

    @require_sas
    def test_evselect_image_no_tel_error(self):
        """
        Testing that TelescopeNotAssociatedError is raised when a source with no XMM data is passed
        to the XMM-specific evselect_image function.
        """
        with self.assertRaises(TelescopeNotAssociatedError):
            evselect_image(self.no_xmm_src)

    @require_sas
    def test_evselect_image(self):
        evselect_image(self.src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        try:
            im = self.src.get_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'),
                                          telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

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

        try:
            exp = self.src.get_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'),
                                           telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

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

        try:
            im = self.src.get_combined_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'),
                                                   telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

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

        try:
            exp = self.src.get_combined_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'),
                                                   telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

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



    

    


