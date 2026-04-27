#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 5:23 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.generate.esass import evtool_image
from xga.generate.esass import srctool_spectrum
from xga.products.phot import Image
from xga.products.spec import Spectrum
from xga.sources import GalaxyCluster
from .. import SRC_INFO, get_test_source, EXPECTED_ERO_OBS, EXPECTED_XMM_OBS


class TestBaseSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    def test_obs_ids_assigned(self):
        obs = self.src.obs_ids

        xmm_obs = set(obs['xmm'])
        ero_obs = set(obs['erosita'])

        assert EXPECTED_ERO_OBS == ero_obs
        assert EXPECTED_XMM_OBS == xmm_obs

    def test_existing_prods_loaded_in_img(self):
        """
        Testing _existing_xga_products() in BaseSource for images.
        """
        src = self.src
        evtool_image(src, Quantity(0.3, 'keV'), Quantity(3, 'keV'))

        del(src)

        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})

        img = src.get_images(lo_en=Quantity(0.3, 'keV'), hi_en=Quantity(3, 'keV'), telescope='erosita')

        assert all([isinstance(im, Image) for im in img])

    def test_existing_prods_loaded_in_img_combined(self):
        """
        Testing _existing_xga_products() in BaseSource for combined images.
        """
        src = self.src
        evtool_image(src, Quantity(0.3, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        del(src)

        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})

        img = src.get_combined_images(lo_en=Quantity(0.3, 'keV'), hi_en=Quantity(3, 'keV'), telescope='erosita')

        assert isinstance(img, Image)
        assert img.obs_id == 'combined'

    def test_existing_prods_loaded_in_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra.
        """
        src = self.src
        srctool_spectrum(src, 'r500', combine_obs=False)

        del(src)

        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})

        spec = src.get_spectra('r500', telescope='erosita')

        assert all(isinstance(sp, Spectrum) for sp in spec)

    def test_existing_prods_loaded_in_comb_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra made from combined obs.
        """
        src = self.src
        srctool_spectrum(src, 'r500', combine_obs=True)

        del(src)

        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})

        spec = src.get_combined_spectra('r500', inst='combined', telescope='erosita')

        assert isinstance(spec, Spectrum)

    def test_existing_prods_loaded_in_idv_inst_comb_obs_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra made from combined obs.
        """
        src = self.src
        srctool_spectrum(src, 'r500', combine_obs=True, combine_tm=False)

        del(src)

        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})

        spec = src.get_combined_spectra('r500', telescope='erosita', inst='tm1')

        assert isinstance(spec, Spectrum)
        assert spec.instrument == 'tm1'

# TODO combined_spectrum should get things with instruments and obsids

if __name__ == "__main__":
     unittest.main()


