import unittest

import os
import shutil
import numpy as np

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.spec import srctool_spectrum
from xga.products import Spectrum

from .. import SRC_INFO

class TestTempFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     telescope='erosita',
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
        cls.test_src_all_tels = GalaxyCluster(SRC_INFO['RA'], SRC_INFO['dec'], SRC_INFO['z'], 
                                              r500=Quantity(500, 'kpc'), name=SRC_INFO['name'], 
                                              use_peak=False, search_distance={'erosita': 
                                              Quantity(3.6, 'deg')})

    def test_srctool_spectrum_combine_insts_f_combine_obs_f(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=False and combine_obs=False
        """
        srctool_spectrum(self.test_src, 'r500', combine_tm=False, combine_obs=False)

        spec = self.test_src.get_spectra('r500', telescope='erosita', inst='TM1')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id != 'combined'
        assert spec.instrument == 'TM1'

    def test_srctool_spectrum_combine_insts_t_combine_obs_f(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=True and combine_obs=False
        """
        srctool_spectrum(self.test_src, 'r500', combine_tm=True, combine_obs=False)

        spec = self.test_src.get_spectra('r500', telescope='erosita', inst='combined')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id != 'combined'
        assert spec.instrument == 'combined'

    def test_srctool_spectrum_combine_insts_f_combine_obs_t(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=False and combine_obs=True
        """
        srctool_spectrum(self.test_src, 'r500', combine_tm=False, combine_obs=True)

        spec = self.test_src.get_combined_spectra('r500', telescope='erosita', inst='TM1')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id == 'combined'
        assert spec.instrument == 'TM1'

    def test_srctool_spectrum_combine_insts_t_combine_obs_t(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=True and combine_obs=True
        """
        srctool_spectrum(self.test_src, 'r500', combine_tm=True, combine_obs=True)

        spec = self.test_src.get_combined_spectra('r500', telescope='erosita')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id == 'combined'
        assert spec.instrument == 'combined'
    



# TODO test that annular spectra with combined obs but individual instruments are stored correctly