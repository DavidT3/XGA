import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass import evtool_image
from xga.products.phot import Image
from xga.products.spec import Spectrum
from xga.generate.esass import srctool_spectrum

from .. import SRC_INFO, SRC_ALL_TELS


class TestBaseSource(unittest.TestCase):
    def test_obs_ids_assigned(self):
        obs = SRC_ALL_TELS.obs_ids

        expected_xmm_obs = set(['0404910601', '0201901401', '0201903501'])
        expected_ero_obs = set(['147099', '148102', '151102', '150099'])
        xmm_obs = set(obs['xmm'])
        ero_obs = set(obs['erosita'])

        assert expected_ero_obs == ero_obs
        assert expected_xmm_obs == xmm_obs

    def test_existing_prods_loaded_in_img(self):
        """
        Testing _existing_xga_products() in BaseSource for images.
        """
        src = SRC_ALL_TELS
        evtool_image(src, Quantity(0.3, 'keV'), Quantity(3, 'keV'))

        del(src)

        src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
        
        img = src.get_images(lo_en=Quantity(0.3, 'keV'), hi_en=Quantity(3, 'keV'))

        assert all([isinstance(im, Image) for im in img])

    def test_existing_prods_loaded_in_img_combined(self):
        """
        Testing _existing_xga_products() in BaseSource for combined images.
        """
        src = SRC_ALL_TELS
        evtool_image(src, Quantity(0.3, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        del(src)

        src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
        
        img = src.get_combined_images(lo_en=Quantity(0.3, 'keV'), hi_en=Quantity(3, 'keV'))

        assert isinstance(img, Image)
        assert img.obs_id == 'combined'

    def test_existing_prods_loaded_in_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra.
        """
        src = SRC_ALL_TELS
        srctool_spectrum(src, 'r500', combine_obs=False)
        
        del(src)

        src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
        
        spec = src.get_spectra('r500', telescope='erosita')

        assert all(isinstance(sp, Spectrum) for sp in spec)

    def test_existing_prods_loaded_in_comb_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra made from combined obs.
        """
        src = SRC_ALL_TELS
        srctool_spectrum(src, 'r500', combine_obs=True)
        
        del(src)

        src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
        
        spec = src.get_combined_spectra('r500', inst='combined', telescope='erosita')

        assert isinstance(spec, Spectrum)

    def test_existing_prods_loaded_in_idv_inst_comb_obs_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra made from combined obs.
        """
        src = SRC_ALL_TELS
        srctool_spectrum(src, 'r500', combine_obs=True, combine_tm=False)
        
        del(src)

        src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
        
        spec = src.get_combined_spectra('r500', telescope='erosita', inst='tm1')

        assert isinstance(spec, Spectrum)
        assert spec.instrument == 'tm1'

# TODO combined_spectrum should get things with instruments and obsids

if __name__ == "__main__":
     unittest.main()