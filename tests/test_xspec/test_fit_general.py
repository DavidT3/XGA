#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 10/28/26, 12:33 PM. Copyright (c) The Contributors.

import os
import unittest

import numpy as np
import numpy.testing as npt
from astropy.units import Quantity

from xga.samples import ClusterSample
from xga.xspec.fit.general import single_temp_apec

from .. import CLUSTER_SMP, MISC_OUTPUT_TESTS, get_test_source


class TestXSPECSingleSource(unittest.TestCase):
    """
    This class houses tests that use XGA's XSPEC functionality on source class instances, as opposed to
    sample class instances.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_src = get_test_source("all")

        cls.no_fits_src = get_test_source("all", shared=False, load_fits=False)

        cls.out_rad = Quantity(500, "kpc")

    def test_global_single_temp_apec(self) -> None:
        par_combs = [
            {"stacked_spectra": True},
            {"stacked_spectra": False},
        ]

        # Set a fit configuration
        fit_conf = {"start_temp": Quantity(2.97, "keV")}

        for cur_pc in par_combs:
            with self.subTest(msg=f"Test global single_temp_apec with {cur_pc}", **cur_pc):
                single_temp_apec(self.test_src, self.out_rad, **cur_pc, **fit_conf)

                for cur_tel in self.test_src.telescopes:
                    cur_tx = self.test_src.get_temperature(self.out_rad, cur_tel, **cur_pc, fit_conf=fit_conf)

    def test_global_single_temp_apec_diff_fitconf(self) -> None:
        """
        Fits several versions of a single temperature apec model, with different
        starting parameters, and checks that the results can be retrieved.
        """
        stacked_spectra = True

        single_temp_apec(self.no_fits_src, self.out_rad, stacked_spectra=stacked_spectra)
        single_temp_apec(self.no_fits_src, self.out_rad, stacked_spectra=stacked_spectra, start_temp=Quantity(5, "keV"))
        single_temp_apec(
            self.no_fits_src,
            self.out_rad,
            stacked_spectra=stacked_spectra,
            start_temp=Quantity(5, "keV"),
            start_met=0.5,
        )

        all_tel_res = {}
        for tel in self.no_fits_src.telescopes:
            # First off, check that we get an error if the source has had multiple fits of the same model
            #  AND the user has not passed a fit_conf entry for retrieval
            with self.assertRaises(ValueError):
                self.no_fits_src.get_temperature(self.out_rad, telescope=tel, stacked_spectra=stacked_spectra)

            default_tx = self.no_fits_src.get_temperature(
                self.out_rad, telescope=tel, stacked_spectra=stacked_spectra, fit_conf={}
            )

            diff_start_temp_tx = self.no_fits_src.get_temperature(
                self.out_rad,
                telescope=tel,
                stacked_spectra=stacked_spectra,
                fit_conf={"start_temp": Quantity(5, "keV")},
            )

            diff_start_temp_met_tx = self.no_fits_src.get_temperature(
                self.out_rad,
                telescope=tel,
                stacked_spectra=stacked_spectra,
                fit_conf={"start_temp": Quantity(5, "keV"), "start_met": 0.5},
            )

            # Now a quick test that default_tx is not identical to diff_start_temp_tx. Though the values are hopefully
            #  relatively close, and there is no absolute guarantee that these fits with different start
            #  parameters WON'T produce the exact same temperatures, it is unlikely in the extreme.
            # What we're really testing for here is that temperature results from one fit have not been assigned
            #  to another fit_conf fit in error.
            with self.assertRaises(AssertionError):
                npt.assert_array_equal(default_tx, diff_start_temp_tx)

            # And that diff_start_temp_tx is not identical to diff_start_temp_met_tx
            with self.assertRaises(AssertionError):
                npt.assert_array_equal(diff_start_temp_tx, diff_start_temp_met_tx)

            all_tel_res[tel] = [default_tx, diff_start_temp_tx, diff_start_temp_met_tx]

        # Finally, we perform a very similar test to above, but checking that the temperature values are not
        #  identical across different telescopes. Again, there is no actual guarantee that they won't be identical,
        #  but it is even less likely than the temperatures from different start parameter values being identical.
        for tel_ind, tel in enumerate(list(all_tel_res.keys())[:-1]):
            next_tel = list(all_tel_res.keys())[tel_ind + 1]

            with self.assertRaises(AssertionError):
                npt.assert_array_equal(all_tel_res[tel][0], all_tel_res[next_tel][0])
            with self.assertRaises(AssertionError):
                npt.assert_array_equal(all_tel_res[tel][1], all_tel_res[next_tel][1])
            with self.assertRaises(AssertionError):
                npt.assert_array_equal(all_tel_res[tel][2], all_tel_res[next_tel][2])

    def test_global_single_temp_apec_diff_lum_fitconf(self) -> None:
        """
        Fits several versions of a single temperature apec model, with different
        luminosity energy band settings passed, and checks that the luminosity results
        can be retrieved.
        """
        stacked_spectra = True

        # Run with default settings, 0.5-2.0 and 0.01-100.0 luminosities are passed, and the luminosity confidence
        #  level is set to 68.
        single_temp_apec(self.no_fits_src, self.out_rad, stacked_spectra=stacked_spectra)

        # Now we use the same default energy bands, but increase the lum_conf to 90 - we can check that
        #  the errors are larger, but values essentially the same (some small difference is expected due
        #  to random fitting processes).
        single_temp_apec(self.no_fits_src, self.out_rad, stacked_spectra=stacked_spectra, lum_conf=90.0)

        # Now we alter the luminosity energy bands, keeping just one of the two default energy bands
        single_temp_apec(
            self.no_fits_src, self.out_rad, stacked_spectra=stacked_spectra, lum_en=Quantity([[0.5, 2.0]], "keV")
        )

        # Finally, we test the case of passing many different energy bands for luminosity measurement.
        many_lum_ens = Quantity([[0.3, 1.3], [1.3, 2.0], [2.0, 2.9], [2.9, 3.8], [3.8, 9.0]], "keV")
        single_temp_apec(self.no_fits_src, self.out_rad, stacked_spectra=stacked_spectra, lum_en=many_lum_ens)

        all_tel_res = {}
        for tel in self.no_fits_src.telescopes:
            with self.assertRaises(ValueError):
                self.no_fits_src.get_luminosities(self.out_rad, telescope=tel, stacked_spectra=stacked_spectra)

            default_lxs = self.no_fits_src.get_luminosities(
                self.out_rad, telescope=tel, stacked_spectra=stacked_spectra, fit_conf={}
            )

            larger_conf_lxs = self.no_fits_src.get_luminosities(
                self.out_rad, telescope=tel, stacked_spectra=stacked_spectra, fit_conf={"lum_conf": 90.0}
            )

            single_band_lx = self.no_fits_src.get_luminosities(
                self.out_rad,
                telescope=tel,
                stacked_spectra=stacked_spectra,
                fit_conf={"lum_en": Quantity([[0.5, 2.0]], "keV")},
            )

            many_band_lx = self.no_fits_src.get_luminosities(
                self.out_rad, telescope=tel, stacked_spectra=stacked_spectra, fit_conf={"lum_en": many_lum_ens}
            )

            # First, we can test if each retrieval returned the number of luminosity entries we expected.
            self.assertEqual(len(default_lxs), 2)
            self.assertEqual(len(larger_conf_lxs), 2)
            self.assertEqual(len(single_band_lx), 1)
            self.assertEqual(len(many_band_lx), len(many_lum_ens))

            # Now we can check that the larger confidence interval Lx measurements for the default energy bands
            #  return luminosities that have larger uncertainties than the 'default_lxs'.
            self.assertLess(default_lxs["bound_0.5-2.0"][1], larger_conf_lxs["bound_0.5-2.0"][1])
            self.assertLess(default_lxs["bound_0.5-2.0"][2], larger_conf_lxs["bound_0.5-2.0"][2])

            self.assertLess(default_lxs["bound_0.01-100.0"][1], larger_conf_lxs["bound_0.01-100.0"][1])
            self.assertLess(default_lxs["bound_0.01-100.0"][2], larger_conf_lxs["bound_0.01-100.0"][2])

            all_tel_res[tel] = [default_lxs, larger_conf_lxs, many_band_lx]

        # Finally, we perform a like-for-like comparison of luminosities measured using the same configuration
        #  but from different telescope's data. We assert that they should NOT be identical arrays, because
        #  the possibility of that happening is vanishingly small. We are testing for whether one telescope's
        #  luminosity measurements have been assigned to another telescope as well.
        for tel_ind, tel in enumerate(list(all_tel_res.keys())[:-1]):
            next_tel = list(all_tel_res.keys())[tel_ind + 1]

            for cur_res_set_ind, cur_res_set in enumerate(all_tel_res[tel]):
                for cur_res_key, cur_res in cur_res_set.items():
                    next_tel_res = all_tel_res[next_tel][cur_res_set_ind][cur_res_key]

                    with self.assertRaises(AssertionError):
                        npt.assert_array_equal(cur_res, next_tel_res)

    def test_global_single_temp_apec_abslum_retrieval(self) -> None:
        """Tests that secondary, absorbed, luminosities can be retrieved from Spectrum instances."""
        # Fit a couple of versions of APEC
        fit_confs = [{"start_temp": Quantity(4.32, "keV")}, {"start_temp": Quantity(4.32, "keV"), "start_met": 0.5}]

        single_temp_apec(self.no_fits_src, self.out_rad, telescope="xmm", **fit_confs[0])
        single_temp_apec(self.no_fits_src, self.out_rad, telescope="xmm", **fit_confs[1])

        # TODO This test should really be in test_products, as it is a mechanism of the Spectrum class rather
        #  than an particular test of XSPEC, but this is being added very quickly
        specs = self.no_fits_src.get_spectra(self.out_rad, telescope="xmm")

        # There should definitely be multiple spectra returned for our test
        #  source, so they should be in a list
        self.assertIsInstance(specs, list)

        # We fit multiple versions of the model to these spectra, so if we try to retrieve a luminosity value from
        #  a spectrum instance WITHOUT passing a fit configuration, it should raise an error.
        with self.assertRaises(ValueError):
            specs[0].get_luminosities(absorbed_lum=True)

        # Now we iterate through the spectra, and the fit configurations, retrieving both absorbed and unabsorbed
        #  luminosities, and asserting that the absorbed luminosity MUST be less than the absorbed.
        for cur_sp in specs:
            for cur_fit_conf in fit_confs:
                cur_lx = cur_sp.get_luminosities(
                    fit_conf=cur_fit_conf, absorbed_lum=False, lo_en=Quantity(0.5, "keV"), hi_en=Quantity(2.0, "keV")
                )
                cur_abs_lx = cur_sp.get_luminosities(
                    fit_conf=cur_fit_conf, absorbed_lum=True, lo_en=Quantity(0.5, "keV"), hi_en=Quantity(2.0, "keV")
                )

                self.assertLess(cur_abs_lx[0], cur_lx[0])

    def test_global_single_temp_apec_view(self) -> None:
        """Tests the view method of the Spectrum class after fitting several versions of the
        single temperature apec model to XMM data.
        """
        single_temp_apec(self.no_fits_src, self.out_rad, telescope="xmm")
        single_temp_apec(self.no_fits_src, self.out_rad, telescope="xmm", start_temp=Quantity(5, "keV"))
        single_temp_apec(self.no_fits_src, self.out_rad, telescope="xmm", start_temp=Quantity(5, "keV"), start_met=0.5)

        specs = self.no_fits_src.get_spectra(self.out_rad, telescope="xmm")

        # There should definitely be multiple spectra returned for our test
        #  source, so they should be in a list
        self.assertIsInstance(specs, list)

        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        for cur_sp in specs:
            cur_save_path = os.path.join(test_out_path, os.path.basename(cur_sp.path).replace(".fits", "viz.png"))
            cur_sp.view(show_residuals=True, save_path=cur_save_path)

    def test_reload_global_xspec_fit_results(self) -> None:
        """Fits several single temperature apec models to non-stacked spectra from all available
        telescopes, declares a new instance of the source, and tests to see if the fit results are
        correctly reloaded.
        """
        single_temp_apec(self.test_src, self.out_rad)
        single_temp_apec(self.test_src, self.out_rad, start_temp=Quantity(5, "keV"))
        single_temp_apec(self.test_src, self.out_rad, start_temp=Quantity(5, "keV"), start_met=0.5)

        all_tel_res = {}
        for tel in self.no_fits_src.telescopes:
            default_tx = self.test_src.get_temperature(self.out_rad, telescope=tel, fit_conf={})

            diff_start_temp_tx = self.test_src.get_temperature(
                self.out_rad, telescope=tel, fit_conf={"start_temp": Quantity(5, "keV")}
            )

            diff_start_temp_met_tx = self.test_src.get_temperature(
                self.out_rad, telescope=tel, fit_conf={"start_temp": Quantity(5, "keV"), "start_met": 0.5}
            )
            all_tel_res[tel] = [default_tx, diff_start_temp_tx, diff_start_temp_met_tx]

        redef_src = get_test_source("all", shared=False)

        for tel in all_tel_res:
            with self.subTest(msg=f"Comparing original and reloaded temperatures [{tel}; default fit]"):
                npt.assert_array_equal(
                    all_tel_res[tel][0], redef_src.get_temperature(self.out_rad, telescope=tel, fit_conf={})
                )

            with self.subTest(msg=f"Comparing original and reloaded temperatures [{tel}; modified start_temp fit]"):
                npt.assert_array_equal(
                    all_tel_res[tel][1],
                    redef_src.get_temperature(self.out_rad, telescope=tel, fit_conf={"start_temp": Quantity(5, "keV")}),
                )

            with self.subTest(
                msg=f"Comparing original and reloaded temperatures [{tel}; modified start_temp/start_met fit]"
            ):
                npt.assert_array_equal(
                    all_tel_res[tel][2],
                    redef_src.get_temperature(
                        self.out_rad, telescope=tel, fit_conf={"start_temp": Quantity(5, "keV"), "start_met": 0.5}
                    ),
                )

    def test_reload_stacked_global_xspec_fit_results(self) -> None:
        """Fits several single temperature apec models to stacked spectra from all available
        telescopes, declares a new instance of the source, and tests to see if the fit results are
        correctly reloaded.
        """
        single_temp_apec(self.test_src, self.out_rad, stacked_spectra=True)
        single_temp_apec(self.test_src, self.out_rad, stacked_spectra=True, start_temp=Quantity(5, "keV"))
        single_temp_apec(
            self.test_src, self.out_rad, stacked_spectra=True, start_temp=Quantity(5, "keV"), start_met=0.5
        )

        all_tel_res = {}
        for tel in self.no_fits_src.telescopes:
            default_tx = self.test_src.get_temperature(self.out_rad, telescope=tel, stacked_spectra=True, fit_conf={})

            diff_start_temp_tx = self.test_src.get_temperature(
                self.out_rad, telescope=tel, stacked_spectra=True, fit_conf={"start_temp": Quantity(5, "keV")}
            )

            diff_start_temp_met_tx = self.test_src.get_temperature(
                self.out_rad,
                telescope=tel,
                stacked_spectra=True,
                fit_conf={"start_temp": Quantity(5, "keV"), "start_met": 0.5},
            )
            all_tel_res[tel] = [default_tx, diff_start_temp_tx, diff_start_temp_met_tx]

        redef_src = get_test_source("all", shared=False)

        for tel in all_tel_res:
            with self.subTest(msg=f"Comparing original and reloaded temperatures [{tel}; default fit]"):
                npt.assert_array_equal(
                    all_tel_res[tel][0],
                    redef_src.get_temperature(self.out_rad, stacked_spectra=True, telescope=tel, fit_conf={}),
                )

            with self.subTest(msg=f"Comparing original and reloaded temperatures [{tel}; modified start_temp fit]"):
                npt.assert_array_equal(
                    all_tel_res[tel][1],
                    redef_src.get_temperature(
                        self.out_rad, stacked_spectra=True, telescope=tel, fit_conf={"start_temp": Quantity(5, "keV")}
                    ),
                )

            with self.subTest(
                msg=f"Comparing original and reloaded temperatures [{tel}; modified start_temp/start_met fit]"
            ):
                npt.assert_array_equal(
                    all_tel_res[tel][2],
                    redef_src.get_temperature(
                        self.out_rad,
                        stacked_spectra=True,
                        telescope=tel,
                        fit_conf={"start_temp": Quantity(5, "keV"), "start_met": 0.5},
                    ),
                )


class TestBaseSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_smp = ClusterSample(
            CLUSTER_SMP["ra"].values,
            CLUSTER_SMP["dec"].values,
            CLUSTER_SMP["z"].values,
            CLUSTER_SMP["name"].values,
            r500=Quantity(CLUSTER_SMP["r500"].values, "kpc"),
            use_peak=False,
            search_distance={"erass": Quantity(3.6, "deg")},
        )

        cls.smp_odd_tels = ClusterSample(
            CLUSTER_SMP["ra"].values,
            CLUSTER_SMP["dec"].values,
            CLUSTER_SMP["z"].values,
            CLUSTER_SMP["name"].values,
            r500=Quantity(CLUSTER_SMP["r500"].values, "kpc"),
            use_peak=False,
            search_distance={"erass": Quantity(3.6, "deg")},
        )
        cls.smp_odd_tels[0].disassociate_obs("erass")

    def test_Lx_w_stacked_spectra(self) -> None:
        single_temp_apec(self.test_smp, "r500", stacked_spectra=True, spectrum_checking=False)
        Lx = self.test_smp.Lx("r500", "erass", stacked_spectra=True)

        assert len(Lx) == 2
        assert isinstance(Lx, Quantity)

    def test_Lx_Tx_w_odd_telescope_sample(self) -> None:
        """
        Testing that for samples where sources don't have the same telescopes assigned, the Lx and Tx
        can be retrieved.
        """
        single_temp_apec(self.smp_odd_tels, "r500", stacked_spectra=True, spectrum_checking=False)

        cur_lx = self.smp_odd_tels.Lx("r500", "erass", stacked_spectra=True)

        assert isinstance(cur_lx, Quantity)
        assert len(cur_lx) == 2
        assert np.isnan(cur_lx[0][0].value)
