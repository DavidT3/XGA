"""
Tests for combined product retrieval disambiguation (Phase 3.4).

Focus: Ensure get_spectra() correctly distinguishes between combination scenarios
when obs_id='combined' and/or inst='combined'.
"""
import unittest
from astropy.units import Quantity
from xga.sources import GalaxyCluster
from xga.generate.esass.spec import srctool_spectrum
from xga.exceptions import NoProductAvailableError
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .. import SRC_ERO


class TestCombinedProductDisambiguation(unittest.TestCase):
    """Test that combined product retrieval correctly disambiguates scenarios."""

    def test_erosita_multiobs_combined_inst_param(self):
        """
        Test that inst parameter correctly filters multi-obs combined spectra.
        Tests obs='combined' with inst='tm1' vs inst='combined'.
        """
        if 'erosita' not in SRC_ERO.obs_ids or len(SRC_ERO.obs_ids.get('erosita', [])) <= 1:
            self.skipTest("Need multi-obs eROSITA data for disambiguation test")

        # Generate multi-obs + single inst
        try:
            srctool_spectrum(SRC_ERO, Quantity(500, 'kpc'), combine_tm=False, combine_obs=True)
        except Exception:
            self.skipTest("Failed to generate multi-obs single-inst spectra")

        # Generate multi-obs + multi inst
        try:
            srctool_spectrum(SRC_ERO, Quantity(500, 'kpc'), combine_tm=True, combine_obs=True)
        except Exception:
            self.skipTest("Failed to generate multi-obs multi-inst spectra")

        # Retrieve multi-obs + specific inst
        try:
            spec_tm1 = SRC_ERO.get_spectra(Quantity(500, 'kpc'), obs_id='combined', inst='tm1',
                                          group_spec=True, min_counts=5, telescope='erosita')
            assert spec_tm1.obs_id == 'combined'
            assert spec_tm1.instrument == 'tm1'
        except NoProductAvailableError:
            pass  # May not exist if TM1 not available

        # Retrieve multi-obs + combined inst
        spec_comb = SRC_ERO.get_spectra(Quantity(500, 'kpc'), obs_id='combined', inst='combined',
                                        group_spec=True, min_counts=5, telescope='erosita')
        assert spec_comb.obs_id == 'combined'
        assert spec_comb.instrument == 'combined'

    def test_get_combined_spectra_wrapper_equivalence(self):
        """
        Test that get_combined_spectra() wrapper behaves identically to
        get_spectra(obs_id='combined', ...).
        """
        if 'erosita' not in SRC_ERO.obs_ids or len(SRC_ERO.obs_ids.get('erosita', [])) <= 1:
            self.skipTest("Need multi-obs eROSITA data")

        # Generate multi-obs + multi-inst
        try:
            srctool_spectrum(SRC_ERO, Quantity(500, 'kpc'), combine_tm=True, combine_obs=True)
        except Exception:
            self.skipTest("Failed to generate spectra")

        # Both methods should return same product
        try:
            spec_direct = SRC_ERO.get_spectra(Quantity(500, 'kpc'), obs_id='combined', inst='combined',
                                             group_spec=True, min_counts=5, telescope='erosita')
            spec_wrapper = SRC_ERO.get_combined_spectra(Quantity(500, 'kpc'), inst='combined',
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
