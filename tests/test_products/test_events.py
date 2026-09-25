#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 9/25/26, 5:46 PM. Copyright (c) The Contributors.

import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS

from xga.exceptions import ProductGenerationError, ProductNotUsableError
from xga.products.events import EventList
from xga.products.phot import Image

from .. import EXTERNAL_TEST_DATA_PATH, MISC_OUTPUT_TESTS
from . import HTTPS_ROOT, S3_ROOT, TEST_EVTS


class TestEventListImageGeneration(unittest.TestCase):
    """
    Granular tests for image generation across all defined event lists.
    Asserts success for imaging missions and failure for non-imaging missions.
    """

    @classmethod
    def setUpClass(cls):
        # Event list to use for the more specific image tests.
        xmm_pn_test_info = TEST_EVTS["XMM_PN"]
        cls.evt = EventList(S3_ROOT + xmm_pn_test_info["path"])
        cls.test_evt_name = "XMM_PN"

    def check_missions_evt_init_image_gen(self, name) -> None:
        cur_info = TEST_EVTS[name]

        # We'll do a subtest of the base event list checks first - no sense
        #  having a separate test for this when we're loading them all anyway to attempt to
        #  make images.
        # Declare the event list from the S3 URI
        cur_test_evt = EventList(os.path.join(S3_ROOT, cur_info["path"]))
        with self.subTest(check=f"Telescope, instrument, ObsID checks of EventList for {name}"):
            # Check the telescope is what we expected
            self.assertEqual(
                cur_test_evt.telescope.lower(),
                cur_info["tele"].lower(),
                msg=f"Telescope mismatch for {name}: {cur_test_evt.telescope} vs {cur_info['tele']}",
            )

            # Then same deal for the instrument
            actual_inst = cur_test_evt.instrument.lower()
            expected_inst = cur_info["inst"].lower()
            self.assertTrue(
                actual_inst.startswith(expected_inst) or expected_inst.startswith(actual_inst),
                f"Instrument mismatch for {name}: {cur_test_evt.instrument} vs {cur_info['inst']}",
            )

            # Then finally for the ObsID
            actual_obsid = cur_test_evt.obs_id.lower() if isinstance(cur_test_evt.obs_id, str) else cur_test_evt.obs_id
            expected_obsid = cur_info["obsid"].lower() if isinstance(cur_info["obsid"], str) else cur_info["obsid"]
            self.assertEqual(
                actual_obsid, expected_obsid, f"ObsID mismatch for {name}: {cur_test_evt.obs_id} vs {cur_info['obsid']}"
            )

        if cur_info["imaging"]:
            # For imaging missions, we expect success.
            # We use loose limits or fallback logic to avoid crashes on missions with weird coordinate ranges.
            try:
                # Setting the bin_size to save a little memory, and execution time, during the tests. Some
                #  special cases may have a different binsize set in the TEST_EVTS dictionary (e.g. eROSITA
                #  because otherwise it gobbles a LOT of memory).
                cur_bin_size = cur_info.get("use_binsize", 10)
                img = cur_test_evt.generate_image(bin_size=cur_bin_size)
                self.assertIsInstance(img, Image)
                self.assertGreater(img.data.sum(), 0, f"Generated image for {name} has no counts")

                # Saving the generated image as a PNG following the pattern in TestProfileView
                test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
                os.makedirs(test_out_path, exist_ok=True)
                img.save_view(os.path.join(test_out_path, f"{name}_binsize{cur_bin_size}.png"))

            except Exception as e:
                raise e
        else:
            # For non-imaging missions, we expect a ValueError.
            with self.assertRaises(
                (ValueError, ProductGenerationError),
                msg=f"Image generation should have failed for non-imaging mission {name}",
            ):
                cur_test_evt.generate_image()

        # Explicitly unload to free data memory immediately
        cur_test_evt.unload()
        if "img" in locals():
            img.unload()

    def test_image_save_fits(self) -> None:
        """Test that the EventList image generation function can write the image to a FITS file."""
        # Set the approximate eV/chan of XMM PN

        test_out_dir = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_dir, exist_ok=True)

        test_out_path = os.path.join(test_out_dir, f"{self.test_evt_name}_image.fits")

        cur_test_im = self.evt.generate_image(save_path=test_out_path)
        self.assertIsInstance(cur_test_im, Image)
        self.assertGreater(cur_test_im.data.sum(), 0, f"Generated image for {self.test_evt_name} has no counts.")

    def test_image_gen_en_bounds(self) -> None:
        """Test the generation of an image within specified energy bounds."""
        # Set the approximate eV/chan of XMM PN
        self.evt.ev_per_channel = Quantity(1, "eV/chan")

        lo_en = Quantity(0.5, "keV")
        hi_en = Quantity(2.0, "keV")

        cur_test_im = self.evt.generate_image(lo_en=lo_en, hi_en=hi_en)
        self.assertIsInstance(cur_test_im, Image)
        self.assertGreater(
            cur_test_im.data.sum(),
            0,
            f"Generated image for {self.test_evt_name} within {lo_en.value}{hi_en.value} keV has no counts.",
        )

        # Saving the generated image as a PNG following the pattern in TestProfileView
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)
        cur_test_im.save_view(
            os.path.join(test_out_path, f"{self.test_evt_name}_lo_en{lo_en.value}-lo_en{hi_en.value}keV.png")
        )

        self.assertIn(
            "LO_EN",
            cur_test_im.header,
            f"Generated image for {self.test_evt_name} within {lo_en.value}{hi_en.value} keV does not have a LO_EN header entry.",
        )
        self.assertIn(
            "HI_EN",
            cur_test_im.header,
            f"Generated image for {self.test_evt_name} within {lo_en.value}{hi_en.value} keV does not have a HI_EN header entry.",
        )

    def test_image_gen_en_bounds_failure(self) -> None:
        """Check that EventList image generation fails when energy bounds are specified, but ev_per_channel is not set."""
        # Make sure the ev_per_channel property is set to None (another test can modify this).
        self.evt.ev_per_channel = None

        # Run the image generation attempt - should fail because the necessary information isn't available.
        with self.assertRaises(
            NotImplementedError,
            msg="Energy bounded image generation should have failed for EventList with no energy-per-channel information.",
        ):
            self.evt.generate_image(lo_en=Quantity(0.5, "keV"), hi_en=Quantity(2.0, "keV"))

    def test_image_gen_angular_binsize(self) -> None:
        """Test passing a binsize in angular units (e.g. arcsec) to EventList.generate_image(...)"""
        ang_bin_size = Quantity(4.35, "arcsec")

        expec_size = 600
        acc_pix_diff = 10

        cur_test_im = self.evt.generate_image(bin_size=ang_bin_size)

        self.assertIsInstance(cur_test_im, Image)

        # Saving the generated image as a PNG following the pattern in TestProfileView
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)
        cur_test_im.save_view(
            os.path.join(test_out_path, f"{self.test_evt_name}_binsize{ang_bin_size.value}arcsec.png")
        )

        self.assertGreater(
            cur_test_im.data.sum(),
            0,
            f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec has no counts.",
        )

        with self.subTest(check="X size"):
            self.assertAlmostEqual(
                cur_test_im.shape[1],
                expec_size,
                delta=acc_pix_diff,
                msg=f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec has an X shape ({cur_test_im.shape[0]}) more than {acc_pix_diff} different from expected ({expec_size}) .",
            )
        with self.subTest(check="Y size"):
            self.assertAlmostEqual(
                cur_test_im.shape[0],
                expec_size,
                delta=acc_pix_diff,
                msg=f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec has a Y shape ({cur_test_im.shape[1]}) more than {acc_pix_diff} different from expected ({expec_size}) .",
            )

        self.assertIn(
            "CDELT1",
            cur_test_im.header,
            f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec does not have a CDELT1 header entry.",
        )
        self.assertIn(
            "CDELT2",
            cur_test_im.header,
            f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec does not have a CDELT2 header entry.",
        )

        self.assertEqual(
            cur_test_im.header["CDELT1"],
            -0.0012083333333333,
            f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec has incorrect ({cur_test_im.header['CDELT1']}) CDELT1 header entry (expected -0.0012083333333333).",
        )
        self.assertEqual(
            cur_test_im.header["CDELT2"],
            0.0012083333333333,
            f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec has incorrect ({cur_test_im.header['CDELT2']}) CDELT2 header entry (expected 0.0012083333333333).",
        )

    def test_image_gen_ang_lims(self) -> None:
        """Test passing a binsize in angular units (e.g. arcsec) to EventList.generate_image(...)"""
        ang_bin_size = Quantity(4.35, "arcsec")

        ang_x_lims = Quantity([149.6768191, 149.5104306], "deg")
        ang_y_lims = Quantity([-10.9981898, -11.1170894], "deg")

        expec_x_size = np.ceil((np.abs(ang_x_lims.diff()) / ang_bin_size).to("").value)
        expec_y_size = np.ceil((np.abs(ang_y_lims.diff()) / ang_bin_size).to("").value)
        acc_pix_diff = 10

        cur_test_im = self.evt.generate_image(bin_size=ang_bin_size, x_lims=ang_x_lims, y_lims=ang_y_lims)

        # Saving the generated image as a PNG following the pattern in TestProfileView
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)
        cur_test_im.save_view(
            os.path.join(
                test_out_path,
                f"{self.test_evt_name}_xlims{ang_x_lims.value}deg_ylims{ang_y_lims.value}deg_binsize{ang_bin_size.value}arcsec.png",
            )
        )

        self.assertIsInstance(cur_test_im, Image)
        self.assertGreater(
            cur_test_im.data.sum(),
            0,
            f"Generated image for {self.test_evt_name} with X-limits of {ang_x_lims}, Y-limits of {ang_y_lims}, and angular binsize {ang_bin_size.value} arcsec has no counts.",
        )

        with self.subTest(check="X size"):
            self.assertAlmostEqual(
                cur_test_im.shape[1],
                expec_x_size,
                delta=acc_pix_diff,
                msg=f"Generated image for {self.test_evt_name} with X-limits of {ang_x_lims} and angular binsize {ang_bin_size.value} arcsec has an X shape ({cur_test_im.shape[0]}) more than {acc_pix_diff} different from expected ({expec_x_size}) .",
            )

        with self.subTest(check="Y size"):
            self.assertAlmostEqual(
                cur_test_im.shape[0],
                expec_y_size,
                delta=acc_pix_diff,
                msg=f"Generated image for {self.test_evt_name} with Y-limits of {ang_y_lims} and angular binsize {ang_bin_size.value} arcsec has a Y shape ({cur_test_im.shape[1]}) more than {acc_pix_diff} different from expected ({expec_y_size}) .",
            )

    def test_image_gen_pix_lims(self) -> None:
        """Test passing a binsize in sky pixel units to EventList.generate_image(...)"""
        sky_bin_size = 100

        pix_x_lims = Quantity([13560.5, 27560.5], "pix")
        pix_y_lims = Quantity([23720.5, 30040.5], "pix")

        expec_x_size = np.ceil(np.abs(pix_x_lims.diff()).value / sky_bin_size)
        expec_y_size = np.ceil(np.abs(pix_y_lims.diff()).value / sky_bin_size)
        acc_pix_diff = 2

        cur_test_im = self.evt.generate_image(bin_size=sky_bin_size, x_lims=pix_x_lims, y_lims=pix_y_lims)

        # Saving the generated image as a PNG following the pattern in TestProfileView
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)
        cur_test_im.save_view(
            os.path.join(
                test_out_path,
                f"{self.test_evt_name}_xlims{pix_x_lims.value}_ylims{pix_y_lims.value}_binsize{sky_bin_size}skypix.png",
            )
        )

        self.assertIsInstance(cur_test_im, Image)
        self.assertGreater(
            cur_test_im.data.sum(),
            0,
            f"Generated image for {self.test_evt_name} with X-limits of {pix_x_lims}, Y-limits of {pix_y_lims}, and angular binsize {sky_bin_size} has no counts.",
        )

        with self.subTest(check="X size"):
            self.assertAlmostEqual(
                cur_test_im.shape[1],
                expec_x_size,
                delta=acc_pix_diff,
                msg=f"Generated image for {self.test_evt_name} with X-limits of {pix_x_lims} and sky pixel binsize {sky_bin_size} has an X shape ({cur_test_im.shape[0]}) more than {acc_pix_diff} different from expected ({expec_x_size}) .",
            )

        with self.subTest(check="Y size"):
            self.assertAlmostEqual(
                cur_test_im.shape[0],
                expec_y_size,
                delta=acc_pix_diff,
                msg=f"Generated image for {self.test_evt_name} with Y-limits of {pix_y_lims} and sky pixel binsize {sky_bin_size} has a Y shape ({cur_test_im.shape[1]}) more than {acc_pix_diff} different from expected ({expec_y_size}) .",
            )

    def test_donor_image_generation(self) -> None:
        """Tests generating an image using another image as a donor for the WCS grid."""
        rosat_path = os.path.join(S3_ROOT, "rosat/data/pspc/processed_data/900000/rp900029a02/rp900029a02_bas.fits.Z")
        xmm_path = os.path.join(S3_ROOT, "xmm/data/rev0/0147511701/PPS/P0147511701PNS003PIEVLI0000.FTZ")

        rosat_evt = EventList(rosat_path)
        xmm_evt = EventList(xmm_path)

        # Create ROSAT donor image
        rosat_img = rosat_evt.generate_image(bin_size=30)

        # Create XMM image using donor
        xmm_img = xmm_evt.generate_image(donor_image=rosat_img)

        # Save views as PNGs
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)
        rosat_img.save_view(os.path.join(test_out_path, "rosat_donor.png"))
        xmm_img.save_view(os.path.join(test_out_path, "xmm_from_donor.png"))

        # Assertions
        self.assertEqual(xmm_img.shape, rosat_img.shape, "XMM image shape does not match donor image shape.")
        # The WCS should be identical
        self.assertEqual(
            xmm_img.radec_wcs.to_header().tostring(),
            rosat_img.radec_wcs.to_header().tostring(),
            "XMM image WCS does not match donor image WCS.",
        )

        # Memory management
        rosat_evt.unload()
        xmm_evt.unload()
        rosat_img.unload()
        xmm_img.unload()

    def test_donor_image_diff_frame_epoch(self) -> None:
        """Tests generating an XMM image using a legacy Einstein image as a donor (FK4/B1950)."""
        einstein_img_path = os.path.join(HTTPS_ROOT, "einstein/data/hri/images/h0039n40.xia.Z")
        xmm_evt_path = os.path.join(S3_ROOT, "xmm/data/rev0/0727960401/PPS/P0727960401PNS003PIEVLI0000.FTZ")

        # We download and decompress the Einstein image to a local file first, as XGA's Image class
        #  does not currently support streaming compressed remote files directly.
        test_ext_data_dir = os.path.join(EXTERNAL_TEST_DATA_PATH, self.id())
        os.makedirs(test_ext_data_dir, exist_ok=True)

        loc_einstein_img_path = os.path.join(test_ext_data_dir, "h0039n40-xia.fits")
        if not os.path.exists(loc_einstein_img_path):
            with fits.open(einstein_img_path) as einsteino:
                einsteino.writeto(loc_einstein_img_path)

        einstein_img = Image(
            loc_einstein_img_path,
            "h0039n40",
            "HRI",
            "",
            "",
            "",
            Quantity(0.5, "keV"),
            Quantity(2.0, "keV"),
            telescope="einstein",
        )
        xmm_evt = EventList(xmm_evt_path)

        # Create XMM image using Einstein donor
        xmm_img = xmm_evt.generate_image(donor_image=einstein_img)

        # Save views as PNGs
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)
        einstein_img.save_view(os.path.join(test_out_path, "einstein_legacy_donor.png"))
        xmm_img.save_view(os.path.join(test_out_path, "xmm_from_legacy_donor.png"))

        # Comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        einstein_img.get_view(ax1)
        xmm_img.get_view(ax2)
        plt.tight_layout()
        plt.savefig(os.path.join(test_out_path, "comparison_einstein_xmm.png"))
        plt.close(fig)

        # Assertions
        self.assertEqual(xmm_img.shape, einstein_img.shape, "XMM image shape does not match legacy donor shape.")
        # The WCS should match exactly
        self.assertEqual(
            xmm_img.radec_wcs.to_header().tostring(),
            einstein_img.radec_wcs.to_header().tostring(),
            "XMM image WCS does not match legacy donor WCS.",
        )

        # Memory management
        xmm_evt.unload()
        einstein_img.unload()
        xmm_img.unload()


# Dynamically attach init and generation tests for every mission
# This avoids manual repetition while providing granular results for each mission
for mission_name in TEST_EVTS:
    # All the tests that check that an EventList can be declared and an image can be generated from them
    gen_method = f"test_evt_init_im_gen_{mission_name}"

    def create_gen_test(m_name):
        return lambda self: self.check_missions_evt_init_image_gen(m_name)

    setattr(TestEventListImageGeneration, gen_method, create_gen_test(mission_name))


class TestEventListFunctionality(unittest.TestCase):
    """General functionality tests using XMM PN as a representative standard spectro-imaging mission."""

    @classmethod
    def setUpClass(cls):
        info = TEST_EVTS["XMM_PN"]
        cls.evt = EventList(S3_ROOT + info["path"])

    def test_get_filtered_data_str(self) -> None:
        """Tests string events filtering logic."""
        times = self.evt.get_columns_from_data(["TIME"])["TIME"]
        t_start, t_end = float(times.min()), float(times.min() + 100)

        filt_ops = {"TIME": [f"> {t_start}", f"< {t_end}"]}
        # We need to explicitly convert back to pandas for the check
        filtered = self.evt.get_filtered_data(["TIME", "X", "Y"], filt_ops)

        self.assertTrue(all(filtered["TIME"] > t_start))
        self.assertTrue(all(filtered["TIME"] < t_end))
        self.assertIn("X", filtered.columns)

    def test_get_filtered_data_callable(self) -> None:
        """Tests callable events filtering logic."""
        filt_ops = {"X": lambda x: x > 100, "Y": lambda y: y < 200}
        # We need to explicitly convert back to pandas for the check
        filtered = self.evt.get_filtered_data(["TIME", "X", "Y"], filt_ops)

        self.assertTrue(all(filtered["X"] > 100))
        self.assertTrue(all(filtered["Y"] < 200))
        self.assertIn("X", filtered.columns)

    def test_memory_management(self) -> None:
        """Tests lazy loading and explicit unloading."""
        # Force a state where data is loaded
        _ = self.evt.data
        self.assertIsNotNone(self.evt._data)

        # Unload data only
        self.evt.unload(unload_data=True, unload_header=False)
        self.assertIsNone(self.evt._data)
        self.assertIsNotNone(self.evt._header)

    def test_wcs_construction(self) -> None:
        """Tests that a valid WCS is built from remote headers."""
        w = self.evt.radec_sky_wcs
        self.assertIsInstance(w, WCS)
        self.assertTrue(w.has_celestial)


class TestEventListRemoteProtocols(unittest.TestCase):
    """Verifies support for different protocols on a subset."""

    def test_https_access(self) -> None:
        """Checks that an event list can be loaded from a remote location specified by a HTTPS URL."""
        info = TEST_EVTS["Chandra_ACIS"]
        evt = EventList(HTTPS_ROOT + info["path"])
        self.assertEqual(evt.telescope.lower(), "chandra")


class TestEventListLocalLoad(unittest.TestCase):
    """Verifies that EventList behaviours with local files."""

    @classmethod
    def setUpClass(cls):
        rel_info = TEST_EVTS["Suzaku_HXD_PIN"]
        rel_url = os.path.join(HTTPS_ROOT, rel_info["path"])

        # We download and decompress the Einstein image to a local file first, as XGA's Image class
        #  does not currently support streaming compressed remote files directly.
        test_ext_data_dir = os.path.join(EXTERNAL_TEST_DATA_PATH, "TestEventListLocalLoad")
        os.makedirs(test_ext_data_dir, exist_ok=True)

        cls.loc_evt_path = os.path.join(test_ext_data_dir, os.path.basename(rel_url))
        if not os.path.exists(cls.loc_evt_path):
            with fits.open(rel_url) as hxdo:
                hxdo.writeto(cls.loc_evt_path)

    def test_local_load(self) -> None:
        """Simply tests that a locally stored FITS event list can be loaded."""
        cur_test_evt = EventList(self.loc_evt_path)
        self.assertEqual(cur_test_evt.telescope.lower(), "suzaku")
        self.assertEqual(cur_test_evt.instrument.lower(), "hxd")

        self.assertGreater(len(cur_test_evt.data), 1)

    def test_wrong_path_local_load(self) -> None:
        """Checks that an EventList notices when the file path it has been pointed at does not exist."""
        # Grab the real file path
        cur_test_path = self.loc_evt_path
        # Break the real file path
        cur_test_path += ".notrealextension"

        # Set up the EventList, with default `check_exists=True`
        with self.assertRaises(
            ProductNotUsableError,
            msg="EventList does not raise ProductNotUsableError when the file path does not exist.",
        ):
            cur_test_evt = EventList(cur_test_path, check_exists=True)

        # self.assertEqual(cur_test_evt.usable, False, msg="EventList does not have .usable set to False when the file path does not exist.")
        # self.assertEqual(cur_test_evt.not_usable_reasons, ["ProductPathDoesNotExist"], msg="EventList does not have .not_usable_reasons set to ['ProductPathDoesNotExist'] when the file path does not exist.")


if __name__ == "__main__":
    unittest.main()
