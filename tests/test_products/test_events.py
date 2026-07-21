#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/20/26, 10:10 PM. Copyright (c) The Contributors.

import os
import unittest

from astropy.wcs import WCS

from xga.exceptions import ProductGenerationError
from xga.products.events import EventList
from xga.products.phot import Image
from .. import MISC_OUTPUT_TESTS

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
                     "tele": "maxi", "inst": "gsc", "imaging": False},
    "MAXI_GSC_Med": {"path": "maxi/data/obs/MJD57000/MJD57115/events/gsc_med/mx_mjd57115_gsc_med_126.evt.gz",
                     "tele": "maxi", "inst": "gsc", "imaging": False},
    "MAXI_SSC_Med": {"path": "maxi/data/obs/MJD57000/MJD57115/events/ssc_med/mx_mjd57115_ssch_med_167.evt.gz",
                     "tele": "maxi", "inst": "ssc", "imaging": False},
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
                    "tele": "erosita", "inst": "merged", "imaging": True},
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
    def check_mission_gen(self, name):
        cur_info = TEST_EVTS[name]
        evt = EventList(os.path.join(S3_ROOT, cur_info['path']))

        if cur_info['imaging']:
            # For imaging missions, we expect success.
            # We use loose limits or fallback logic to avoid crashes on missions with weird coordinate ranges.
            try:
                img = evt.generate_image()
                self.assertIsInstance(img, Image)
                self.assertGreater(img.data.sum(), 0, f"Generated image for {name} has no counts")

                # Saving the generated image as a PNG following the pattern in TestProfileView
                test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
                os.makedirs(test_out_path, exist_ok=True)
                img.save_view(os.path.join(test_out_path, f"{name}_default.png"))

            except Exception as e:
                raise e
        else:
            # For non-imaging missions, we expect a ValueError.
            with self.assertRaises((ValueError, ProductGenerationError),
                                   msg=f"Image generation should have failed for non-imaging mission {name}"):
                evt.generate_image()


# Dynamically attach init and generation tests for every mission
# This avoids manual repetition while providing granular results for each mission
for mission_name in TEST_EVTS:
    # 1. Initialization tests
    init_method = f"test_init_{mission_name}"
    def create_init_test(m_name):
        return lambda self: self.check_mission_init(m_name)
    setattr(TestEventListInitialization, init_method, create_init_test(mission_name))

    # 2. Generation tests
    gen_method = f"test_gen_{mission_name}"
    def create_gen_test(m_name):
        return lambda self: self.check_mission_gen(m_name)
    setattr(TestEventListImageGeneration, gen_method, create_gen_test(mission_name))


class TestEventListFunctionality(unittest.TestCase):
    """General functionality tests using XMM PN as a representative standard imaging mission."""
    @classmethod
    def setUpClass(cls):
        info = TEST_EVTS["XMM_PN"]
        cls.evt = EventList(S3_ROOT + info['path'])

    def test_get_filtered_data(self):
        """Tests both string and callable filtering logic."""
        times = self.evt.get_columns_from_data(['TIME'])['TIME']
        t_start, t_end = float(times.min()), float(times.min() + 100)

        filt_ops = {'TIME': [f'> {t_start}', f'< {t_end}']}
        # We need to explicitly convert back to pandas for the check
        filtered = self.evt.get_filtered_data(['TIME', 'X', 'Y'], filt_ops)

        self.assertTrue(all(filtered['TIME'] > t_start))
        self.assertTrue(all(filtered['TIME'] < t_end))
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

    def test_ev_per_channel_not_implemented(self):
        """Verifies that the TODO state for ev_per_channel is captured correctly."""
        with self.assertRaises(NotImplementedError):
            _ = self.evt.ev_per_channel


class TestEventListRemoteProtocols(unittest.TestCase):
    """Verifies support for different protocols on a subset."""
    def test_https_access(self):
        info = TEST_EVTS["Chandra_ACIS"]
        evt = EventList(HTTPS_ROOT + info['path'])
        self.assertEqual(evt.telescope.lower(), 'chandra')


if __name__ == "__main__":
    unittest.main()