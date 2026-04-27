#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 5:23 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from tests import SRC_INFO, get_test_source
from xga.generate.ciao.phot import chandra_image_expmap
from xga.generate.ciao.spec import specextract_spectrum
from xga.generate.sas.phot import evselect_image, eexpmap
from xga.generate.sas.spec import evselect_spectrum
from xga.products.phot import Image, ExpMap
from xga.products.spec import Spectrum
from xga.sources import GalaxyCluster


class TestProductLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('xmm')

    def test_xmm_image_loading(self):
        """
        Test that existing XMM images are correctly identified and loaded.
        """
        if 'xmm' not in self.src.telescopes:
            self.skipTest("XMM data not associated with test source")

        # Ensure image is generated
        evselect_image(self.src, Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))

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
        if 'xmm' not in self.src.telescopes:
            self.skipTest("XMM data not associated with test source")

        # Ensure expmap is generated
        eexpmap(self.src, Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))

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
        if 'xmm' not in self.src.telescopes:
            self.skipTest("XMM data not associated with test source")

        # Ensure spectrum is generated
        evselect_spectrum(self.src, 'r500')

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
        if 'chandra' not in self.src.telescopes:
            self.skipTest("Chandra data not associated with test source")

        # Ensure image is generated
        chandra_image_expmap(self.src, Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))

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
        if 'chandra' not in self.src.telescopes:
            self.skipTest("Chandra data not associated with test source")

        # Ensure spectrum is generated
        specextract_spectrum(self.src, 'r500')

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



