import unittest
import sys
import os

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.spec import srctool_spectrum
from xga.products import Spectrum


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import SRC_ALL_TELS

class TestEsassSpecFuncs(unittest.TestCase):
    def test_srctool_spectrum_combine_insts_f_combine_obs_f(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=False and combine_obs=False
        """
        srctool_spectrum(SRC_ALL_TELS, 'r500', combine_tm=False, combine_obs=False)

        spec = SRC_ALL_TELS.get_spectra('r500', telescope='erosita', inst='TM1')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id != 'combined'
        assert spec.instrument == 'TM1'

    def test_srctool_spectrum_combine_insts_t_combine_obs_f(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=True and combine_obs=False
        """
        srctool_spectrum(SRC_ALL_TELS, 'r500', combine_tm=True, combine_obs=False)

        spec = SRC_ALL_TELS.get_spectra('r500', telescope='erosita', inst='combined')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id != 'combined'
        assert spec.instrument == 'combined'

    def test_srctool_spectrum_combine_insts_f_combine_obs_t(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=False and combine_obs=True
        """
        srctool_spectrum(SRC_ALL_TELS, 'r500', combine_tm=False, combine_obs=True)

        spec = SRC_ALL_TELS.get_combined_spectra('r500', telescope='erosita', inst='TM1')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id == 'combined'
        assert spec.instrument == 'TM1'

    def test_srctool_spectrum_combine_insts_t_combine_obs_t(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=True and combine_obs=True
        """
        srctool_spectrum(SRC_ALL_TELS, 'r500', combine_tm=True, combine_obs=True)

        spec = SRC_ALL_TELS.get_combined_spectra('r500', telescope='erosita')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id == 'combined'
        assert spec.instrument == 'combined'
    



# TODO test that annular spectra with combined obs but individual instruments are stored correctly