#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/11/26, 3:25 PM. Copyright (c) The Contributors.

"""Tests for BaseSource observation assignment, observation mode inspection, and existing product loading."""

import unittest
import warnings
from typing import Any
from unittest.mock import patch

from astropy.units import Quantity

from xga.exceptions import NoProductAvailableError
from xga.generate.esass import evtool_image, srctool_spectrum
from xga.products.events import EventList
from xga.products.phot import Image
from xga.products.spec import Spectrum
from xga.sources.base import BaseSource

from .. import EXPECTED_ERO_OBS, EXPECTED_XMM_OBS, SRC_INFO, get_test_source


class TestBaseSource(unittest.TestCase):
    """Test cases for core BaseSource properties and product loading."""

    src: Any

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise the shared multi-mission test GalaxyCluster instance."""
        cls.src = get_test_source("all")

    def test_obs_ids_assigned(self) -> None:
        """Verify that the expected XMM and eROSITA ObsIDs are associated with the test source."""
        obs = self.src.obs_ids

        xmm_obs = set(obs["xmm"])
        ero_obs = set(obs["erass"])

        self.assertEqual(EXPECTED_ERO_OBS, ero_obs)
        self.assertEqual(EXPECTED_XMM_OBS, xmm_obs)

    def test_obs_modes(self) -> None:
        """Verify that BaseSource.obs_modes returns expected OBS_MODE, DATAMODE, and SUBMODE values per instrument."""
        modes = self.src.obs_modes

        self.assertEqual(set(modes.keys()), set(self.src.instruments.keys()))

        # Check XMM observation modes, data modes, and sub-modes across all associated ObsIDs and instruments
        expected_xmm_submodes = {
            "0401040101": {
                "mos1": "PrimeFullWindow",
                "mos2": "PrimeFullWindow",
                "pn": "PrimeFullWindowExtended",
            },
            "0840580101": {
                "mos1": "PrimeFullWindow",
                "mos2": "PrimeFullWindow",
                "pn": "PrimeFullWindow",
            },
            "0840580201": {
                "mos1": "PrimeFullWindow",
                "mos2": "PrimeFullWindow",
                "pn": "PrimeFullWindow",
            },
        }
        self.assertEqual(set(modes["xmm"].keys()), set(expected_xmm_submodes.keys()))
        for obs_id, inst_map in expected_xmm_submodes.items():
            self.assertEqual(set(modes["xmm"][obs_id].keys()), set(inst_map.keys()))
            for inst, exp_submode in inst_map.items():
                self.assertEqual(
                    modes["xmm"][obs_id][inst],
                    {
                        "OBS_MODE": "POINTING",
                        "DATAMODE": "IMAGING",
                        "SUBMODE": exp_submode,
                    },
                )

        # Check eROSITA observation modes (combined event list read per TM instrument)
        self.assertEqual(set(modes["erass"].keys()), set(self.src.instruments["erass"].keys()))
        for obs_id, inst_list in self.src.instruments["erass"].items():
            self.assertEqual(set(modes["erass"][obs_id].keys()), set(inst_list))
            for inst in inst_list:
                self.assertEqual(
                    modes["erass"][obs_id][inst],
                    {
                        "OBS_MODE": "SURVEY",
                        "DATAMODE": "",
                        "SUBMODE": None,
                    },
                )

        # Ensure EventList table data was cleared from memory after header inspection
        for evts in self.src.get_products("events"):
            self.assertIsNone(evts._data)

    def test_existing_prods_loaded_in_img(self) -> None:
        """Testing _existing_xga_products() in BaseSource for images."""
        # Make a new local reference to the source class
        src = self.src
        # We make some images that won't already exist - separate ObsID but
        #  combined instruments
        evtool_image(src, Quantity(0.3, "keV"), Quantity(3, "keV"), combine_obs=False)

        # Use the get_test_source function we defined in the test init, but have it
        #  define a new instance of the test GalaxyCluster, so that it will
        #  reload the new products
        redef_src: Any = get_test_source("all", shared=False)

        try:
            img = redef_src.get_images(
                lo_en=Quantity(0.3, "keV"),
                hi_en=Quantity(3, "keV"),
                telescope="erass",
                inst="combined",
            )
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        # For this test source there should be multiple images retrieved, so if
        #  this isn't a list we have a problem.
        self.assertIsInstance(img, list)
        self.assertTrue(all(isinstance(im, Image) for im in img))

    def test_existing_prods_loaded_in_img_combined(self) -> None:
        """Testing _existing_xga_products() in BaseSource for combined images."""
        # Make a new local reference to the source class
        src = self.src
        # We make some COMBINED images that won't already exist
        evtool_image(src, Quantity(0.3, "keV"), Quantity(3, "keV"), combine_obs=True)

        # Use the get_test_source function we defined in the test init, but have it
        #  define a new instance of the test GalaxyCluster, so that it will
        #  reload the new products
        redef_src: Any = get_test_source("all", shared=False)

        try:
            img = redef_src.get_combined_images(
                lo_en=Quantity(0.3, "keV"),
                hi_en=Quantity(3, "keV"),
                telescope="erass",
            )
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        self.assertIsInstance(img, Image)
        self.assertEqual(img.obs_id, "combined")

    def test_existing_prods_loaded_in_spectra(self) -> None:
        """Testing _existing_xga_products() in BaseSource for spectra."""
        src = self.src
        srctool_spectrum(src, "r500", combine_obs=False, combine_tm=True)

        redef_src: Any = get_test_source("all", shared=False)

        try:
            spec = redef_src.get_spectra(
                "r500",
                inst="combined",
                telescope="erass",
            )
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        self.assertIsInstance(spec, list)
        self.assertTrue(all(isinstance(sp, Spectrum) for sp in spec))

    def test_existing_prods_loaded_in_comb_spectra(self) -> None:
        """Testing _existing_xga_products() in BaseSource for spectra made from combined obs."""
        src = self.src
        srctool_spectrum(src, "r500", combine_obs=True)

        redef_src: Any = get_test_source("all", shared=False)

        try:
            spec = redef_src.get_combined_spectra(
                "r500",
                inst="combined",
                telescope="erass",
            )
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        self.assertIsInstance(spec, Spectrum)

    def test_existing_prods_loaded_in_idv_inst_comb_obs_spectra(self) -> None:
        """Testing _existing_xga_products() in BaseSource for spectra made from combined obs."""
        src = self.src
        srctool_spectrum(src, "r500", combine_obs=True, combine_tm=False)

        redef_src: Any = get_test_source("all", shared=False)

        try:
            spec = redef_src.get_combined_spectra(
                "r500",
                telescope="erass",
                inst="tm1",
            )
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        self.assertIsInstance(spec, Spectrum)
        self.assertEqual(spec.instrument, "tm1")

    def test_removed_obs_warnings(self) -> None:
        """Verify removed-observation warning suppression for valid COMBINED_INSTS and emission on missing files."""
        # Prefix used by BaseSource.__init__ when warning that one or more matched observations or instruments
        #  had missing/unusable foundational files (e.g. event lists) and were dropped during _initial_products
        removed_prefix = "The following observations/instruments could not be read in and have been removed: "
        ra_val = float(SRC_INFO["ra"])  # type: ignore[arg-type]
        dec_val = float(SRC_INFO["dec"])  # type: ignore[arg-type]
        z_val = float(SRC_INFO["z"])  # type: ignore[arg-type]
        name_val = str(SRC_INFO["name"])
        # All 5 eRASS sky tiles matched within 3.6 deg at BaseSource initialization (before ExtendedSource.obs_check
        #  disassociates '227096' and '230093' based on aperture coverage to leave EXPECTED_ERO_OBS). We instantiate
        #  BaseSource directly (with load_products=False) so we test BaseSource.__init__ and _initial_products in
        #  isolation without triggering ExtendedSource combined-image generation when an ObsID subset is altered.
        init_ero_obs = {"227093", "227090", "227096", "224093", "230093"}

        # 1. Valid multi-mission declaration: for COMBINED_INSTS telescopes (eRASS), _initial_products loads the
        #    single combined event list under the 'combined' key and populates empty placeholder dicts for tm1..tm7.
        #    Ensure BaseSource.__init__ sees those instrument keys and does NOT emit a false-positive removal warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            valid_src = BaseSource(
                ra_val,
                dec_val,
                redshift=z_val,
                name=name_val,
                search_distance={"erass": Quantity(3.6, "deg")},
                load_products=False,
            )
        # Filter captured warnings to only those from the removed-observations check
        removed_warns = [str(w.message) for w in caught if removed_prefix in str(w.message)]
        self.assertEqual(removed_warns, [])
        # Confirm all 5 eRASS ObsIDs and all 7 TM instruments per ObsID are registered on the source
        self.assertEqual(set(valid_src.obs_ids["erass"]), init_ero_obs)
        for obs_id in init_ero_obs:
            self.assertEqual(
                valid_src.instruments["erass"][obs_id],
                ["tm1", "tm2", "tm3", "tm4", "tm5", "tm6", "tm7"],
            )

        # 2. Partial COMBINED_INSTS removal: simulate a missing/corrupt event list for a single eRASS ObsID ("227093")
        #    by intercepting EventList instantiation in xga.sources.base and marking that ObsID's product unusable.
        #    Previously, dropping one COMBINED_INSTS ObsID in _initial_products caused a KeyError in BaseSource.__init__
        #    when populating per-instrument placeholder dicts over the unpruned initial obs dictionary.
        def fail_single_erass_evt(
            path: str,
            obs_id: str,
            instrument: str,
            stdout_str: str = "",
            stderr_str: str = "",
            gen_cmd: str = "",
            telescope: str | None = None,
            check_exists: bool = True,
        ) -> EventList:
            evt = EventList(
                path,
                obs_id=obs_id,
                instrument=instrument,
                stdout_str=stdout_str,
                stderr_str=stderr_str,
                gen_cmd=gen_cmd,
                telescope=telescope,
                check_exists=check_exists,
            )
            if obs_id == "227093":
                evt._usable = False
            return evt

        # Patch EventList only within xga.sources.base for the duration of this declaration
        with (
            patch("xga.sources.base.EventList", side_effect=fail_single_erass_evt),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            partial_src = BaseSource(
                ra_val,
                dec_val,
                redshift=z_val,
                name=name_val,
                search_distance={"erass": Quantity(3.6, "deg")},
                load_products=False,
            )

        # Verify that exactly one removal warning was emitted naming only ObsID 227093 (and all 7 of its TM instruments)
        removed_warns = [str(w.message) for w in caught if removed_prefix in str(w.message)]
        self.assertEqual(len(removed_warns), 1)
        self.assertIn("erass: 227093(tm1, tm2, tm3, tm4, tm5, tm6, tm7)", removed_warns[0])
        self.assertNotIn("227090", removed_warns[0])
        self.assertNotIn("xmm:", removed_warns[0])
        # Verify that 227093 was removed from obs_ids and instruments while the other 4 eRASS tiles remain intact
        self.assertEqual(set(partial_src.obs_ids["erass"]), init_ero_obs - {"227093"})
        self.assertNotIn("227093", partial_src.instruments["erass"])

        # 3. Whole-telescope removal: simulate all eRASS event lists being unusable while XMM event lists succeed.
        #    Previously, pruning a telescope with 0 surviving observations before constructing removed_obs caused a
        #    KeyError on new_prods[t]. Verify that the warning lists all failed eRASS ObsIDs and that "erass" is
        #    cleanly removed from the source's telescopes, obs_ids, and instruments.
        def fail_all_erass_evts(
            path: str,
            obs_id: str,
            instrument: str,
            stdout_str: str = "",
            stderr_str: str = "",
            gen_cmd: str = "",
            telescope: str | None = None,
            check_exists: bool = True,
        ) -> EventList:
            evt = EventList(
                path,
                obs_id=obs_id,
                instrument=instrument,
                stdout_str=stdout_str,
                stderr_str=stderr_str,
                gen_cmd=gen_cmd,
                telescope=telescope,
                check_exists=check_exists,
            )
            if telescope == "erass":
                evt._usable = False
            return evt

        with (
            patch("xga.sources.base.EventList", side_effect=fail_all_erass_evts),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            xmm_only_src = BaseSource(
                ra_val,
                dec_val,
                redshift=z_val,
                name=name_val,
                search_distance={"erass": Quantity(3.6, "deg")},
                load_products=False,
            )

        # Confirm the warning includes every matched eRASS tile and that only XMM remains on the initialized source
        removed_warns = [str(w.message) for w in caught if removed_prefix in str(w.message)]
        self.assertEqual(len(removed_warns), 1)
        for ero_oi in init_ero_obs:
            self.assertIn(f"{ero_oi}(tm1, tm2, tm3, tm4, tm5, tm6, tm7)", removed_warns[0])
        self.assertEqual(xmm_only_src.telescopes, ["xmm"])
        self.assertNotIn("erass", xmm_only_src.obs_ids)
        self.assertNotIn("erass", xmm_only_src.instruments)


if __name__ == "__main__":
    unittest.main()
