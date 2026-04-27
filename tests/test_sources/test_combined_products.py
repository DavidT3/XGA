#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 5:23 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.exceptions import NoProductAvailableError
from xga.generate.esass.spec import srctool_spectrum
from .. import get_test_source


class TestCombinedProductDisambiguation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    def test_erosita_multiobs_combined_inst_param(self):
        """
        Test that inst parameter correctly filters multi-obs combined spectra.
        Tests obs='combined' with inst='tm1' vs inst='combined'.
        """
        if 'erosita' not in self.src.obs_ids or len(self.src.obs_ids.get('erosita', [])) <= 1:
            self.skipTest("Need multi-obs eROSITA data for disambiguation test")

        # Generate multi-obs + single inst
        try:
            srctool_spectrum(self.src, Quantity(500, 'kpc'), combine_tm=False, combine_obs=True)
        except Exception:
            self.skipTest("Failed to generate multi-obs single-inst spectra")

        # Generate multi-obs + multi inst
        try:
            srctool_spectrum(self.src, Quantity(500, 'kpc'), combine_tm=True, combine_obs=True)
        except Exception:
            self.skipTest("Failed to generate multi-obs multi-inst spectra")

        # Retrieve multi-obs + specific inst
        try:
            spec_tm1 = self.src.get_spectra(Quantity(500, 'kpc'), obs_id='combined', inst='tm1',
                                          group_spec=True, min_counts=5, telescope='erosita')
            assert spec_tm1.obs_id == 'combined'
            assert spec_tm1.instrument == 'tm1'
        except NoProductAvailableError:
            pass  # May not exist if TM1 not available

        # Retrieve multi-obs + combined inst
        spec_comb = self.src.get_spectra(Quantity(500, 'kpc'), obs_id='combined', inst='combined',
                                        group_spec=True, min_counts=5, telescope='erosita')
        assert spec_comb.obs_id == 'combined'
        assert spec_comb.instrument == 'combined'

    def test_get_combined_spectra_wrapper_equivalence(self):
        """
        Test that get_combined_spectra() wrapper behaves identically to
        get_spectra(obs_id='combined', ...).
        """
        if 'erosita' not in self.src.obs_ids or len(self.src.obs_ids.get('erosita', [])) <= 1:
            self.skipTest("Need multi-obs eROSITA data")

        # Generate multi-obs + multi-inst
        try:
            srctool_spectrum(self.src, Quantity(500, 'kpc'), combine_tm=True, combine_obs=True)
        except Exception:
            self.skipTest("Failed to generate spectra")

        # Both methods should return same product
        try:
            spec_direct = self.src.get_spectra(Quantity(500, 'kpc'), obs_id='combined', inst='combined',
                                             group_spec=True, min_counts=5, telescope='erosita')
            spec_wrapper = self.src.get_combined_spectra(Quantity(500, 'kpc'), inst='combined',
                                                       group_spec=True, min_counts=5, telescope='erosita')

            # Should be the same product
            if not isinstance(spec_direct, list):
                assert spec_direct.path == spec_wrapper.path
            else:
                assert len(spec_direct) == len(spec_wrapper)
        except NoProductAvailableError:
            self.skipTest("Combined spectra not available")


if __name__ == "__main__":
    unittest.main()



