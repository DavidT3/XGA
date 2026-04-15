import unittest
import os
import sys

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.sas.phot import evselect_image, eexpmap
from xga.generate.sas.spec import evselect_spectrum
from xga.generate.ciao.phot import chandra_image_expmap
from xga.generate.ciao.spec import specextract_spectrum
from xga.products.phot import Image, ExpMap
from xga.products.spec import Spectrum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests import SRC_INFO, SRC_ALL_TELS


class TestProductLoading(unittest.TestCase):
    """
    Testing that multi-mission products (XMM and Chandra) are correctly loaded
    by the _existing_xga_products mechanism in BaseSource.
    """

    def test_xmm_image_loading(self):
        """
        Test that existing XMM images are correctly identified and loaded.
        """
        if 'xmm' not in SRC_ALL_TELS.telescopes:
            self.skipTest("XMM data not associated with test source")

        # Ensure image is generated
        evselect_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))

        # Re-declare source to trigger product loading
        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                            name=SRC_INFO['name'], use_peak=False,
                            search_distance={'xmm': Quantity(30, 'arcmin')}, load_profiles=False)

        # Retrieve and verify
        imgs = src.get_images(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(2.0, 'keV'), telescope='xmm')
        if not isinstance(imgs, list):
            imgs = [imgs]

        self.assertTrue(len(imgs) > 0, "No XMM images loaded")
        self.assertTrue(all(isinstance(im, Image) for im in imgs))
        self.assertTrue(all(im.telescope == 'xmm' for im in imgs))

    def test_xmm_expmap_loading(self):
        """
        Test that existing XMM exposure maps are correctly identified and loaded.
        """
        if 'xmm' not in SRC_ALL_TELS.telescopes:
            self.skipTest("XMM data not associated with test source")

        # Ensure expmap is generated
        eexpmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))

        # Re-declare source
        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                            name=SRC_INFO['name'], use_peak=False,
                            search_distance={'xmm': Quantity(30, 'arcmin')}, load_profiles=False)

        # Retrieve and verify
        exps = src.get_expmaps(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(2.0, 'keV'), telescope='xmm')
        if not isinstance(exps, list):
            exps = [exps]

        self.assertTrue(len(exps) > 0, "No XMM exposure maps loaded")
        self.assertTrue(all(isinstance(ex, ExpMap) for ex in exps))

    def test_xmm_spectrum_loading(self):
        """
        Test that existing XMM spectra are correctly identified and loaded.
        """
        if 'xmm' not in SRC_ALL_TELS.telescopes:
            self.skipTest("XMM data not associated with test source")

        # Ensure spectrum is generated
        evselect_spectrum(SRC_ALL_TELS, 'r500')

        # Re-declare source
        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                            name=SRC_INFO['name'], use_peak=False,
                            search_distance={'xmm': Quantity(30, 'arcmin')}, load_profiles=False)

        # Retrieve and verify
        specs = src.get_spectra('r500', telescope='xmm')
        if not isinstance(specs, list):
            specs = [specs]

        self.assertTrue(len(specs) > 0, "No XMM spectra loaded")
        self.assertTrue(all(isinstance(sp, Spectrum) for sp in specs))

    def test_chandra_image_loading(self):
        """
        Test that existing Chandra images are correctly identified and loaded.
        """
        if 'chandra' not in SRC_ALL_TELS.telescopes:
            self.skipTest("Chandra data not associated with test source")

        # Ensure image is generated
        chandra_image_expmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))

        # Re-declare source
        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                            name=SRC_INFO['name'], use_peak=False,
                            search_distance={'chandra': Quantity(10, 'arcmin')}, load_profiles=False)

        # Retrieve and verify
        imgs = src.get_images(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(2.0, 'keV'), telescope='chandra')
        if not isinstance(imgs, list):
            imgs = [imgs]

        self.assertTrue(len(imgs) > 0, "No Chandra images loaded")
        self.assertTrue(all(isinstance(im, Image) for im in imgs))
        self.assertTrue(all(im.telescope == 'chandra' for im in imgs))

    def test_chandra_spectrum_loading(self):
        """
        Test that existing Chandra spectra are correctly identified and loaded.
        """
        if 'chandra' not in SRC_ALL_TELS.telescopes:
            self.skipTest("Chandra data not associated with test source")

        # Ensure spectrum is generated
        specextract_spectrum(SRC_ALL_TELS, 'r500')

        # Re-declare source
        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                            name=SRC_INFO['name'], use_peak=False,
                            search_distance={'chandra': Quantity(10, 'arcmin')}, load_profiles=False)

        # Retrieve and verify
        specs = src.get_spectra('r500', telescope='chandra')
        if not isinstance(specs, list):
            specs = [specs]

        self.assertTrue(len(specs) > 0, "No Chandra spectra loaded")
        self.assertTrue(all(isinstance(sp, Spectrum) for sp in specs))


if __name__ == "__main__":
    unittest.main()
