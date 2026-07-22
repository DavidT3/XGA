#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/22/26, 1:36 PM. Copyright (c) The Contributors.

import os
import unittest

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS

from xga.exceptions import ProductGenerationError
from xga.products.events import EventList
from xga.products.phot import Image
from .. import MISC_OUTPUT_TESTS, EXTERNAL_TEST_DATA_PATH

# These are a selection of event lists, with expected telescope/instrument, to test with
TEST_EVTS = {
    "ASCA_GIS": {"path": "asca/data/rev2/87036000/screened/ad87036000g300370m.evt.gz",
                 "tele": "asca", "inst": "gis3", "imaging": True},
    "ASCA_SIS": {"path": "asca/data/rev2/87036000/screened/ad87036000s000302m.evt.gz",
                 "tele": "asca", "inst": "sis0", "imaging": True},
    "BBXRT": {"path": "bbxrt/events/a2256i.evt.gz", "tele": "bbxrt", "inst": "A0-B4", "imaging": False},
    "Calet": {"path": "calet/data/cgbm/obs/2025/20250318/events/cgbm_20250318_hx2_113151.evt.gz",
              "tele": "calet", "inst": "cgbm", "imaging": False},
    "Chandra_ACIS": {"path": "chandra/data/byobsid/2/12812/primary/acisf12812N003_evt2.fits.gz",
                     "tele": "chandra", "inst": "acis", "imaging": True},
    "Chandra_HRC": {"path": "chandra/data/byobsid/2/22642/primary/hrcf22642N003_evt2.fits.gz",
                    "tele": "chandra", "inst": "hrc", "imaging": True},
    "Einstein_HRI": {"path": "einstein/data/hri/events/h0039n40.xpa.Z", "tele": "einstein", "inst": "hri", "imaging": True},
    "Einstein_IPC": {"path": "einstein/data/ipc/events/i2030n40.xpb.Z", "tele": "einstein", "inst": "ipc", "imaging": True},
    "HaloSat": {"path": "halosat/data/obs/101601/products/hs101601_s14_cl.evt.gz",
                "tele": "halosat", "inst": "sdd14", "imaging": False},
    "IXPE": {"path": "ixpe/data/obs/03/03005001/event_l2/ixpe03005001_det1_evt2_v02.fits.gz",
             "tele": "ixpe", "inst": "DU1", "imaging": True},
    "MAXI_GSC_Low": {"path": "maxi/data/obs/MJD57000/MJD57115/events/gsc_low/mx_mjd57115_gsc_low_078.evt.gz",
                     "tele": "maxi", "inst": "gsc", "imaging": True},
    "MAXI_GSC_Med": {"path": "maxi/data/obs/MJD57000/MJD57115/events/gsc_med/mx_mjd57115_gsc_med_126.evt.gz",
                     "tele": "maxi", "inst": "gsc", "imaging": True},
    "MAXI_SSC_Med": {"path": "maxi/data/obs/MJD57000/MJD57115/events/ssc_med/mx_mjd57115_ssch_med_167.evt.gz",
                     "tele": "maxi", "inst": "ssc", "imaging": True},
    "NICER_XTI": {"path": "nicer/data/obs/2023_06/6060040431/xti/event_cl/ni6060040431_0mpu7_cl.evt.gz",
                  "tele": "nicer", "inst": "xti", "imaging": False},
    "NuSTAR": {"path": "nustar/data/obs/10/7/71010003002/event_cl/nu71010003002A01_cl.evt.gz",
               "tele": "nustar", "inst": "fpma", "imaging": True},
    "ROSAT_HRI": {"path": "rosat/data/hri/processed_data/800000/rh800446a01/rh800446a01_bas.fits.Z",
                  "tele": "rosat", "inst": "hri", "imaging": True},
    "ROSAT_PSPC": {"path": "rosat/data/pspc/processed_data/500000/rp500211n00/rp500211n00_bas.fits.Z",
                   "tele": "rosat", "inst": "pspcb", "imaging": True},
    "BeppoSAX_LECS": {"path": "sax/data/events/20309003/event_files/LECS_20309003.evt.gz",
                      "tele": "sax", "inst": "lecs", "imaging": True},
    "BeppoSAX_MECS": {"path": "sax/data/events/20309003/event_files/MECS2_20309003.evt.gz",
                      "tele": "sax", "inst": "mecs2", "imaging": True},
    "SRG_eROSITA": {"path": "srg/data/erosita/erass1/obs/141/053/EXP_010/em01_053141_020_EventList_c010.fits.gz",
                    "tele": "erosita", "inst": "merged", "imaging": True, "use_binsize": 500},
    "Suzaku_XIS": {"path": "suzaku/data/obs/7/704015010/xis/event_cl/ae704015010xi1_0_3x3n069b_cl.evt.gz",
                   "tele": "suzaku", "inst": "xis1", "imaging": True},
    "Suzaku_HXD_GSO": {"path": "suzaku/data/obs/7/704015010/hxd/event_cl/ae704015010hxd_0_gsono_cl.evt.gz",
                       "tele": "suzaku", "inst": "hxd", "imaging": False},
    "Suzaku_HXD_PIN": {"path": "suzaku/data/obs/7/704015010/hxd/event_cl/ae704015010hxd_0_pinno_cl.evt.gz",
                       "tele": "suzaku", "inst": "hxd", "imaging": False},
    "Swift_XRT": {"path": "swift/data/obs/2010_12/00020153006/xrt/event/sw00020153006xpcw3po_cl.evt.gz",
                  "tele": "swift", "inst": "xrt", "imaging": True},
    "XMM_PN": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501PNS003PIEVLI0000.FTZ",
               "tele": "xmm", "inst": "epn", "imaging": True},
    "XMM_MOS1": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501M1S001MIEVLI0000.FTZ",
                 "tele": "xmm", "inst": "emos1", "imaging": True},
    "XMM_MOS2": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501M2S002MIEVLI0000.FTZ",
                 "tele": "xmm", "inst": "emos2", "imaging": True},
    "XMM_RGS1": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501R1S004EVENLI0000.FTZ",
                 "tele": "xmm", "inst": "rgs1", "imaging": False},
    "XMM_RGS2": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501R2S005EVENLI0000.FTZ",
                 "tele": "xmm", "inst": "rgs2", "imaging": False},
    "XRISM_XTEND": {"path": "xrism/data/obs/3/300049010/xtend/event_cl/xa300049010xtd_p032000010_cl.evt.gz",
                    "tele": "xrism", "inst": "xtend", "imaging": True},
    "XRISM_Resolve": {"path": "xrism/data/obs/3/300049010/resolve/event_cl/xa300049010rsl_p0px5000_cl.evt.gz",
                      "tele": "xrism", "inst": "resolve", "imaging": True},
}

S3_ROOT = "s3://nasa-heasarc/"
HTTPS_ROOT = "https://heasarc.gsfc.nasa.gov/FTP/"


class TestEventListInitialization(unittest.TestCase):
    """
    Granular tests for mission initialization across all defined event lists.
    """
    def check_mission_init(self, name):

        cur_info = TEST_EVTS[name]
        evt = EventList(os.path.join(S3_ROOT, cur_info['path']))

        self.assertEqual(evt.telescope.lower(), cur_info['tele'].lower())

        actual_inst = evt.instrument.lower()
        expected_inst = cur_info['inst'].lower()
        self.assertTrue(actual_inst.startswith(expected_inst) or expected_inst.startswith(actual_inst),
                        f"Instrument mismatch for {name}: {evt.instrument} vs {cur_info['inst']}")


class TestEventListImageGeneration(unittest.TestCase):
    """
    Granular tests for image generation across all defined event lists.
    Asserts success for imaging missions and failure for non-imaging missions.
    """
    @classmethod
    def setUpClass(cls):
        # Event list to use for the more specific image tests.
        xmm_pn_test_info = TEST_EVTS["XMM_PN"]
        cls.evt = EventList(S3_ROOT + xmm_pn_test_info['path'])
        cls.test_evt_name = "XMM_PN"

    def check_missions_image_gen(self, name):
        cur_info = TEST_EVTS[name]
        evt = EventList(os.path.join(S3_ROOT, cur_info['path']))

        if cur_info['imaging']:
            # For imaging missions, we expect success.
            # We use loose limits or fallback logic to avoid crashes on missions with weird coordinate ranges.
            try:
                # Setting the bin_size to save a little memory, and execution time, during the tests. Some
                #  special cases may have a different binsize set in the TEST_EVTS dictionary (e.g. eROSITA
                #  because otherwise it gobbles a LOT of memory).
                cur_bin_size = cur_info.get('use_binsize', 10)
                img = evt.generate_image(bin_size=cur_bin_size)
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
            with self.assertRaises((ValueError, ProductGenerationError),
                                   msg=f"Image generation should have failed for non-imaging mission {name}"):
                evt.generate_image()

        # Explicitly unload to free data memory immediately
        evt.unload()
        if 'img' in locals():
            img.unload()

    def test_image_gen_en_bounds(self):
        """Test the generation of an image within specified energy bounds."""
        # Set the approximate eV/chan of XMM PN
        self.evt.ev_per_channel = Quantity(1, 'eV/chan')

        lo_en = Quantity(0.5, 'keV')
        hi_en = Quantity(2., 'keV')

        cur_test_im = self.evt.generate_image(lo_en=lo_en, hi_en=hi_en)
        self.assertIsInstance(cur_test_im, Image)
        self.assertGreater(cur_test_im.data.sum(), 0, f"Generated image for {self.test_evt_name} within {lo_en.value}{hi_en.value} keV has no counts.")

        # Saving the generated image as a PNG following the pattern in TestProfileView
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)
        cur_test_im.save_view(os.path.join(test_out_path, f"{self.test_evt_name}_lo_en{lo_en.value}-lo_en{hi_en.value}keV.png"))

        self.assertIn("LO_EN", cur_test_im.header, f"Generated image for {self.test_evt_name} within {lo_en.value}{hi_en.value} keV does not have a LO_EN header entry.")
        self.assertIn("HI_EN", cur_test_im.header, f"Generated image for {self.test_evt_name} within {lo_en.value}{hi_en.value} keV does not have a HI_EN header entry.")

    def test_image_gen_en_bounds_failure(self):
        """Check that EventList image generation fails when energy bounds are specified, but ev_per_channel is not set."""
        # Make sure the ev_per_channel property is set to None (another test can modify this).
        self.evt.ev_per_channel = None

        # Run the image generation attempt - should fail because the necessary information isn't available.
        with self.assertRaises(NotImplementedError,
                       msg=f"Energy bounded image generation should have failed for EventList with no energy-per-channel information."):
            self.evt.generate_image(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(2., 'keV'))

    def test_image_gen_angular_binsize(self):
        """Test passing a binsize in angular units (e.g. arcsec) to EventList.generate_image(...)"""
        ang_bin_size = Quantity(4.35, 'arcsec')

        expec_size = 512
        acc_pix_diff = 5

        cur_test_im = self.evt.generate_image(bin_size=ang_bin_size)

        self.assertIsInstance(cur_test_im, Image)

        self.assertAlmostEqual(cur_test_im.shape[0], expec_size, delta=acc_pix_diff, msg=f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec has an X shape ({cur_test_im.shape[0]}) more than {acc_pix_diff} different from expected ({expec_size}) .")
        self.assertAlmostEqual(cur_test_im.shape[1], expec_size, delta=acc_pix_diff, msg=f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec has a Y shape ({cur_test_im.shape[1]}) more than {acc_pix_diff} different from expected ({expec_size}) .")

        self.assertGreater(cur_test_im.data.sum(), 0, f"Generated image for {self.test_evt_name} with angular binsize {ang_bin_size.value} arcsec has no counts.")

        # Saving the generated image as a PNG following the pattern in TestProfileView
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)
        cur_test_im.save_view(os.path.join(test_out_path, f"{self.test_evt_name}_binsize{ang_bin_size.value}arcsec.png"))

        print(cur_test_im.header)

        # self.assertIn("LO_EN", cur_test_im.header, f"Generated image for {self.test_evt_name} within {lo_en.value}{hi_en.value} keV does not have a LO_EN header entry.")
        # self.assertIn("HI_EN", cur_test_im.header, f"Generated image for {self.test_evt_name} within {lo_en.value}{hi_en.value} keV does not have a HI_EN header entry.")

    def test_donor_image_generation(self):
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
        self.assertEqual(xmm_img.radec_wcs.to_header().tostring(), rosat_img.radec_wcs.to_header().tostring(),
                         "XMM image WCS does not match donor image WCS.")

        # Memory management
        rosat_evt.unload()
        xmm_evt.unload()
        rosat_img.unload()
        xmm_img.unload()

    def test_donor_image_diff_frame_epoch(self):
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

        einstein_img = Image(loc_einstein_img_path, "h0039n40", "HRI", "", "", "", Quantity(0.5, 'keV'),
                             Quantity(2.0, 'keV'), telescope='einstein')
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
        self.assertEqual(xmm_img.radec_wcs.to_header().tostring(), einstein_img.radec_wcs.to_header().tostring(),
                         "XMM image WCS does not match legacy donor WCS.")

        # Memory management
        xmm_evt.unload()
        einstein_img.unload()
        xmm_img.unload()

# Dynamically attach init and generation tests for every mission
# This avoids manual repetition while providing granular results for each mission
for mission_name in TEST_EVTS:
    # All the tests that check that an EventList can be declared
    init_method = f"test_init_{mission_name}"
    def create_init_test(m_name):
        return lambda self: self.check_mission_init(m_name)
    setattr(TestEventListInitialization, init_method, create_init_test(mission_name))

    # And all the generic generation of image tests
    gen_method = f"test_gen_{mission_name}"
    def create_gen_test(m_name):
        return lambda self: self.check_missions_image_gen(m_name)
    setattr(TestEventListImageGeneration, gen_method, create_gen_test(mission_name))


class TestEventListFunctionality(unittest.TestCase):
    """General functionality tests using XMM PN as a representative standard spectro-imaging mission."""
    @classmethod
    def setUpClass(cls):
        info = TEST_EVTS["XMM_PN"]
        cls.evt = EventList(S3_ROOT + info['path'])

    def test_get_filtered_data_str(self):
        """Tests string events filtering logic."""
        times = self.evt.get_columns_from_data(['TIME'])['TIME']
        t_start, t_end = float(times.min()), float(times.min() + 100)

        filt_ops = {'TIME': [f'> {t_start}', f'< {t_end}']}
        # We need to explicitly convert back to pandas for the check
        filtered = self.evt.get_filtered_data(['TIME', 'X', 'Y'], filt_ops)

        self.assertTrue(all(filtered['TIME'] > t_start))
        self.assertTrue(all(filtered['TIME'] < t_end))
        self.assertIn('X', filtered.columns)

    def test_get_filtered_data_callable(self):
        """Tests callable events filtering logic."""
        filt_ops = {"X": lambda x: x > 100, "Y": lambda y: y < 200}
        # We need to explicitly convert back to pandas for the check
        filtered = self.evt.get_filtered_data(['TIME', 'X', 'Y'], filt_ops)

        self.assertTrue(all(filtered['X'] > 100))
        self.assertTrue(all(filtered['Y'] < 200))
        self.assertIn('X', filtered.columns)

    def test_memory_management(self):
        """Tests lazy loading and explicit unloading."""
        # Force a state where data is loaded
        _ = self.evt.data
        self.assertIsNotNone(self.evt._data)

        # Unload data only
        self.evt.unload(unload_data=True, unload_header=False)
        self.assertIsNone(self.evt._data)
        self.assertIsNotNone(self.evt._header)

    def test_wcs_construction(self):
        """Tests that a valid WCS is built from remote headers."""
        w = self.evt.radec_sky_wcs
        self.assertIsInstance(w, WCS)
        self.assertTrue(w.has_celestial)


class TestEventListRemoteProtocols(unittest.TestCase):
    """Verifies support for different protocols on a subset."""
    def test_https_access(self):
        info = TEST_EVTS["Chandra_ACIS"]
        evt = EventList(HTTPS_ROOT + info['path'])
        self.assertEqual(evt.telescope.lower(), 'chandra')


if __name__ == "__main__":
    unittest.main()