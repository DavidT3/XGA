import unittest
import os
import sys
from astropy.units import Quantity

from xga.generate.sas.phot import evselect_image, emosaic
from xga.generate.esass.phot import evtool_image, combine_phot_prod
from xga.products.phot import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests import SRC_ALL_TELS, SRC_ERO

class TestProductRetrievalCorrectness(unittest.TestCase):
    """
    Ensures that product retrieval methods return the specific products requested,
    avoiding accidental retrieval of combined products when individual ones are
    requested, and vice versa.
    """

    def test_xmm_image_retrieval_specificity(self):
        """
        Test that XMM retrieval distinguishes between specific instrument and 'combined'.
        """
        if 'xmm' not in SRC_ALL_TELS.telescopes:
            self.skipTest("XMM data not associated")

        lo, hi = Quantity(0.5, 'keV'), Quantity(2.0, 'keV')
        evselect_image(SRC_ALL_TELS, lo, hi)
        emosaic(SRC_ALL_TELS, 'image', lo, hi)

        # 1. Request specific instrument
        # Should return only PN/MOS1/MOS2 images, NOT the combined one
        imgs = SRC_ALL_TELS.get_images(lo_en=lo, hi_en=hi, telescope='xmm')
        if not isinstance(imgs, list):
            imgs = [imgs]

        for im in imgs:
            self.assertNotEqual(im.instrument, 'combined', "Retrieved combined image when individual ones expected")
            self.assertNotEqual(im.obs_id, 'combined')

        # 2. Request combined instrument
        comb_img = SRC_ALL_TELS.get_combined_images(lo_en=lo, hi_en=hi, telescope='xmm')
        self.assertEqual(comb_img.instrument, 'combined')
        self.assertEqual(comb_img.obs_id, 'combined')

    def test_erosita_image_retrieval_specificity(self):
        """
        Test that eROSITA retrieval distinguishes between combination modes.
        """
        if 'erosita' not in SRC_ALL_TELS.telescopes:
            self.skipTest("eROSITA data not associated")

        lo, hi = Quantity(0.5, 'keV'), Quantity(2.0, 'keV')
        # Generate individual-obs images
        evtool_image(SRC_ALL_TELS, lo, hi, combine_obs=False)
        # Generate multi-obs combined images
        combine_phot_prod(SRC_ALL_TELS, 'image', lo, hi)

        # 1. Request individual images
        imgs = SRC_ALL_TELS.get_images(lo_en=lo, hi_en=hi, telescope='erosita')
        if not isinstance(imgs, list):
            imgs = [imgs]

        for im in imgs:
            self.assertNotEqual(im.obs_id, 'combined', "Retrieved combined image when individual ones expected")

        # 2. Request combined image
        comb_img = SRC_ALL_TELS.get_combined_images(lo_en=lo, hi_en=hi, telescope='erosita')
        # Should return the combined one
        if isinstance(comb_img, list):
            comb_img = comb_img[0]
        self.assertEqual(comb_img.obs_id, 'combined')

if __name__ == "__main__":
    unittest.main()
