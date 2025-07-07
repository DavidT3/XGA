#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 03/07/2025, 10:55. Copyright (c) The Contributors

import os
import sys
import unittest

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

        spec = SRC_ALL_TELS.get_spectra('r500', telescope='erosita', inst='tm1', obs_id='227093')
        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id != 'combined'
        assert spec.instrument == 'tm1'

    def test_srctool_spectrum_combine_insts_t_combine_obs_f(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=True and combine_obs=False
        """
        srctool_spectrum(SRC_ALL_TELS, 'r500', combine_tm=True, combine_obs=False)

        spec = SRC_ALL_TELS.get_spectra('r500', telescope='erosita', inst='combined', 
                                        obs_id='227093')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id != 'combined'
        assert spec.instrument == 'combined'

    def test_srctool_spectrum_combine_insts_f_combine_obs_t(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=False and combine_obs=True
        """
        srctool_spectrum(SRC_ALL_TELS, 'r500', combine_tm=False, combine_obs=True)

        spec = SRC_ALL_TELS.get_combined_spectra('r500', telescope='erosita', inst='tm1')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id == 'combined'
        assert spec.instrument == 'tm1'

    def test_srctool_spectrum_combine_insts_t_combine_obs_t(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=True and combine_obs=True
        """
        srctool_spectrum(SRC_ALL_TELS, 'r500', combine_tm=True, combine_obs=True)

        spec = SRC_ALL_TELS.get_combined_spectra('r500', telescope='erosita')

        aa = SRC_ALL_TELS.get_products('combined_spectrum', telescope='erosita')
        print(aa)
        for spec in aa:
            print(spec.instrument)
            print(spec.obs_id)
            print(spec.path)

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id == 'combined'
        assert spec.instrument == 'combined'
    



# TODO test that annular spectra with combined obs but individual instruments are stored correctly