#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:25 AM. Copyright (c) The Contributors.

import os
import sys
import unittest

from xga.generate.esass.spec import srctool_spectrum
from xga.products import Spectrum
from ..utils import require_esass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import get_test_source

class TestEsassSpecFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('erosita')

    @require_esass
    def test_srctool_spectrum_combine_insts_f_combine_obs_f(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=False and combine_obs=False
        """
        srctool_spectrum(self.src, 'r500', combine_tm=False, combine_obs=False)

        spec = self.src.get_spectra('r500', telescope='erosita', inst='tm1', obs_id='227093')
        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id != 'combined'
        assert spec.instrument == 'tm1'

    @require_esass
    def test_srctool_spectrum_combine_insts_t_combine_obs_f(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=True and combine_obs=False
        """
        srctool_spectrum(self.src, 'r500', combine_tm=True, combine_obs=False)

        spec = self.src.get_spectra('r500', telescope='erosita', inst='combined', 
                                        obs_id='227093')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id != 'combined'
        assert spec.instrument == 'combined'

    @require_esass
    def test_srctool_spectrum_combine_insts_f_combine_obs_t(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=False and combine_obs=True
        """
        srctool_spectrum(self.src, 'r500', combine_tm=False, combine_obs=True)

        spec = self.src.get_combined_spectra('r500', telescope='erosita', inst='tm1')

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id == 'combined'
        assert spec.instrument == 'tm1'

    @require_esass
    def test_srctool_spectrum_combine_insts_t_combine_obs_t(self):
        """
        Testing srctool_spectrum with the arguments combine_insts=True and combine_obs=True
        """
        srctool_spectrum(self.src, 'r500', combine_tm=True, combine_obs=True)

        spec = self.src.get_combined_spectra('r500', telescope='erosita', inst='combined')

        if isinstance(spec, list):
            spec = spec[0]

        assert isinstance(spec, Spectrum)
        assert spec.telescope == 'erosita'
        assert spec.obs_id == 'combined'
        assert spec.instrument == 'combined'
    



# TODO test that annular spectra with combined obs but individual instruments are stored correctly


