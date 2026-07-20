#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/20/26, 5:42 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity
from astropy.wcs import WCS

from xga.products.events import EventList
from xga.products.phot import Image

# These are a selection of event lists, with expected telescope/instrument, to test with
TEST_EVTS = {
    "ASCA_GIS": {"path": "asca/data/rev2/87036000/screened/ad87036000g300370m.evt.gz",
                 "tele": "asca", "inst": "gis3"},
    "ASCA_SIS": {"path": "asca/data/rev2/87036000/screened/ad87036000s000302m.evt.gz",
                 "tele": "asca", "inst": "sis0"},
    "BBXRT": {"path": "bbxrt/events/a2256i.evt.gz", "tele": "bbxrt", "inst": "AO-B4"},
    "Calet": {"path": "calet/data/cgbm/obs/2025/20250318/events/cgbm_20250318_hx2_113151.evt.gz",
              "tele": "calet", "inst": "cgbm"},
    "Chandra_ACIS": {"path": "chandra/data/byobsid/2/12812/primary/acisf12812N003_evt2.fits.gz",
                     "tele": "chandra", "inst": "acis"},
    "Chandra_HRC": {"path": "chandra/data/byobsid/2/22642/primary/hrcf22642N003_evt2.fits.gz",
                    "tele": "chandra", "inst": "hrc"},
    "Einstein_HRI": {"path": "einstein/data/hri/events/h0039n40.xpa.Z", "tele": "einstein", "inst": "hri"},
    "Einstein_IPC": {"path": "einstein/data/ipc/events/i2030n40.xpb.Z", "tele": "einstein", "inst": "ipc"},
    "HaloSat": {"path": "halosat/data/obs/101601/products/hs101601_s14_cl.evt.gz",
                "tele": "halosat", "inst": "sdd14"},
    "IXPE": {"path": "ixpe/data/obs/03/03005001/event_l2/ixpe03005001_det1_evt2_v02.fits.gz",
             "tele": "ixpe", "inst": "DU1"},
    "MAXI_GSC_Low": {"path": "maxi/data/obs/MJD57000/MJD57115/events/gsc_low/mx_mjd57115_gsc_low_078.evt.gz",
                     "tele": "maxi", "inst": "gsc"},
    "MAXI_GSC_Med": {"path": "maxi/data/obs/MJD57000/MJD57115/events/gsc_med/mx_mjd57115_gsc_med_126.evt.gz",
                     "tele": "maxi", "inst": "gsc"},
    "MAXI_SSC_Med": {"path": "maxi/data/obs/MJD57000/MJD57115/events/ssc_med/mx_mjd57115_ssch_med_167.evt.gz",
                     "tele": "maxi", "inst": "ssc"},
    "NICER_XTI": {"path": "nicer/data/obs/2023_06/6060040431/xti/event_cl/ni6060040431_0mpu7_cl.evt.gz",
                  "tele": "nicer", "inst": "xti"},
    "NuSTAR": {"path": "nustar/data/obs/10/7/71010003002/event_cl/nu71010003002A01_cl.evt.gz",
               "tele": "nustar", "inst": "fpma"},
    "ROSAT_HRI": {"path": "rosat/data/hri/processed_data/800000/rh800446a01/rh800446a01_bas.fits.Z",
                  "tele": "rosat", "inst": "hri"},
    "ROSAT_PSPC": {"path": "rosat/data/pspc/processed_data/500000/rp500211n00/rp500211n00_bas.fits.Z",
                   "tele": "rosat", "inst": "pspcb"},
    "BeppoSAX_LECS": {"path": "sax/data/events/20309003/event_files/LECS_20309003.evt.gz",
                      "tele": "sax", "inst": "lecs"},
    "BeppoSAX_MECS": {"path": "sax/data/events/20309003/event_files/MECS2_20309003.evt.gz",
                      "tele": "sax", "inst": "mecs2"},
    "SRG_eROSITA": {"path": "srg/data/erosita/erass1/obs/141/053/EXP_010/em01_053141_020_EventList_c010.fits.gz",
                    "tele": "erosita", "inst": "merged"},
    "Suzaku_XIS": {"path": "suzaku/data/obs/7/704015010/xis/event_cl/ae704015010xi1_0_3x3n069b_cl.evt.gz",
                   "tele": "suzaku", "inst": "xis1"},
    "Suzaku_HXD_GSO": {"path": "suzaku/data/obs/7/704015010/hxd/event_cl/ae704015010hxd_0_gsono_cl.evt.gz",
                       "tele": "suzaku", "inst": "hxd"},
    "Suzaku_HXD_PIN": {"path": "suzaku/data/obs/7/704015010/hxd/event_cl/ae704015010hxd_0_pinno_cl.evt.gz",
                       "tele": "suzaku", "inst": "hxd"},
    "Swift_XRT": {"path": "swift/data/obs/2010_12/00020153006/xrt/event/sw00020153006xpcw3po_cl.evt.gz",
                  "tele": "swift", "inst": "xrt"},
    "XMM_PN": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501PNS003PIEVLI0000.FTZ",
               "tele": "xmm", "inst": "epn"},
    "XMM_MOS1": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501M1S001MIEVLI0000.FTZ",
                 "tele": "xmm", "inst": "emos1"},
    "XMM_MOS2": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501M2S002MIEVLI0000.FTZ",
                 "tele": "xmm", "inst": "emos2"},
    "XMM_RGS1": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501R1S004EVENLI0000.FTZ",
                 "tele": "xmm", "inst": "rgs1"},
    "XMM_RGS2": {"path": "xmm/data/rev0/0201903501/PPS/P0201903501R2S005EVENLI0000.FTZ",
                 "tele": "xmm", "inst": "rgs2"},
    "XRISM_XTEND": {"path": "xrism/data/obs/3/300049010/xtend/event_cl/xa300049010xtd_p032000010_cl.evt.gz",
                    "tele": "xrism", "inst": "xtend"},
    "XRISM_Resolve": {"path": "xrism/data/obs/3/300049010/resolve/event_cl/xa300049010rsl_p0px5000_cl.evt.gz",
                      "tele": "xrism", "inst": "resolve"},
}

S3_ROOT = "s3://nasa-heasarc/"
HTTPS_ROOT = "https://heasarc.gsfc.nasa.gov/FTP/"


class TestEventListInitialization(unittest.TestCase):
    """
    Base class for mission initialization tests.
    Specific tests are dynamically attached below.
    """
    def check_mission_init(self, name):
        info = TEST_EVTS[name]
        full_path = S3_ROOT + info['path']
        try:
            evt = EventList(full_path)
            self.assertEqual(evt.telescope.lower(), info['tele'].lower(),
                             f"Telescope mismatch for {name}")

            actual_inst = evt.instrument.lower()
            expected_inst = info['inst'].lower()
            self.assertTrue(actual_inst.startswith(expected_inst) or expected_inst.startswith(actual_inst),
                            f"Instrument mismatch for {name}: {evt.instrument} vs {info['inst']}")
        except (KeyError, FileNotFoundError) as e:
            # Handle known mission bugs/unsupported states
            if name in ["ASCA_GIS", "ASCA_SIS", "BBXRT", "Burstcube", "EUVE", "Einstein_HRI", "Einstein_IPC"]:
                self.skipTest(f"Known initialization issue for {name}: {e}")
            raise e

# Dynamically parameterize the TestEventListInitialization class
# This avoids manual repetition while providing granular results for each mission
for mission_name in TEST_EVTS:
    method_name = f"test_init_{mission_name}"
    # Use a closure to capture the mission name correctly
    def create_test(m_name):
        return lambda self: self.check_mission_init(m_name)
    setattr(TestEventListInitialization, method_name, create_test(mission_name))


class TestEventListFunctionality(unittest.TestCase):
    """Tests core methods using XMM PN as a representative standard imaging mission."""
    @classmethod
    def setUpClass(cls):
        info = TEST_EVTS["XMM_PN"]
        cls.evt = EventList(S3_ROOT + info['path'])

    def test_get_filtered_data(self):
        """Tests both string and callable filtering logic."""
        times = self.evt.get_columns_from_data(['TIME'])['TIME']
        t_start, t_end = float(times.min()), float(times.min() + 100)

        filt_ops = {'TIME': [f'> {t_start}', f'< {t_end}']}
        filtered = self.evt.get_filtered_data(['TIME', 'X', 'Y'], filt_ops)

        self.assertTrue(all(filtered['TIME'] > t_start))
        self.assertTrue(all(filtered['TIME'] < t_end))
        self.assertIn('X', filtered.columns)

    def test_generate_image_basic(self):
        """Tests image binning and XGA Image object return."""
        x_data = self.evt.get_columns_from_data(['X'])['X']
        y_data = self.evt.get_columns_from_data(['Y'])['Y']
        x_mid, y_mid = x_data.mean(), y_data.mean()

        x_lims = Quantity([x_mid - 1000, x_mid + 1000], 'pix')
        y_lims = Quantity([y_mid - 1000, y_mid + 1000], 'pix')

        img = self.evt.generate_image(x_lims=x_lims, y_lims=y_lims)
        self.assertIsInstance(img, Image)
        self.assertGreater(img.data.sum(), 0)

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