#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/9/26, 8:50 PM. Copyright (c) The Contributors.

"""Tests for BaseSource observation assignment, observation mode inspection, and existing product loading."""

import unittest
from typing import Any

from astropy.units import Quantity

from xga.exceptions import NoProductAvailableError
from xga.generate.esass import evtool_image, srctool_spectrum
from xga.products.phot import Image
from xga.products.spec import Spectrum

from .. import EXPECTED_ERO_OBS, EXPECTED_XMM_OBS, get_test_source


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


if __name__ == "__main__":
    unittest.main()
