#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 7:14 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.exceptions import NoProductAvailableError
from xga.generate.esass.phot import evtool_image, combine_phot_prod
from xga.generate.sas.phot import evselect_image, emosaic
from .. import get_test_source


class TestProductRetrievalCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('xmm')

    def test_xmm_image_retrieval_specificity(self):
        """
        Test that XMM retrieval distinguishes between specific instrument and 'combined'.
        """
        if 'xmm' not in self.src.telescopes:
            self.skipTest("XMM data not associated")

        lo, hi = Quantity(0.5, 'keV'), Quantity(2.0, 'keV')
        evselect_image(self.src, lo, hi)
        emosaic(self.src, 'image', lo, hi)

        # 1. Request specific instrument
        # Should return only PN/MOS1/MOS2 images, NOT the combined one
        try:
            imgs = self.src.get_images(lo_en=lo, hi_en=hi, telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving individual XMM images.")

        if not isinstance(imgs, list):
            imgs = [imgs]

        for im in imgs:
            self.assertNotEqual(im.instrument, 'combined', "Retrieved combined image when individual ones expected")
            self.assertNotEqual(im.obs_id, 'combined')

        # 2. Request combined instrument
        try:
            comb_img = self.src.get_combined_images(lo_en=lo, hi_en=hi, telescope='xmm')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving combined XMM image.")

        self.assertEqual(comb_img.instrument, 'combined')
        self.assertEqual(comb_img.obs_id, 'combined')

    def test_erass_image_retrieval_specificity(self):
        """
        Test that eRASS retrieval distinguishes between combination modes.
        """
        if 'erass' not in self.src.telescopes:
            self.skipTest("eRASS data not associated")

        lo, hi = Quantity(0.5, 'keV'), Quantity(2.0, 'keV')
        # Generate individual-obs images
        evtool_image(self.src, lo, hi, combine_obs=False)
        # Generate multi-obs combined images
        combine_phot_prod(self.src, 'image', lo, hi)

        # 1. Request individual images
        try:
            imgs = self.src.get_images(lo_en=lo, hi_en=hi, telescope='erass')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving individual eRASS images.")

        if not isinstance(imgs, list):
            imgs = [imgs]

        for im in imgs:
            self.assertNotEqual(im.obs_id, 'combined', "Retrieved combined image when individual ones expected")

        # 2. Request combined image
        try:
            comb_img = self.src.get_combined_images(lo_en=lo, hi_en=hi, telescope='erass')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving combined eRASS image.")

        # Should return the combined one
        if isinstance(comb_img, list):
            comb_img = comb_img[0]
        self.assertEqual(comb_img.obs_id, 'combined')

if __name__ == "__main__":
    unittest.main()



