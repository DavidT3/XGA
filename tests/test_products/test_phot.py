#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 20/04/2026, 20:00. Copyright (c) The Contributors.
"""Tests for photometric product classes (Image, ExpMap, RateMap) in xga.products.phot."""

import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units import Quantity, UnitConversionError
from astropy.wcs import WCS
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord

from xga.products import ExpMap, Image, RateMap

from .. import MISC_OUTPUT_TESTS


class TestPhotProducts(unittest.TestCase):
    """Test cases for Image, ExpMap, and RateMap coordinate lookups, region handling, and visualisation."""

    _tmp_dir: tempfile.TemporaryDirectory
    _im_path: str
    _ex_path: str
    _obs_id: str = "0201903501"
    _inst: str = "pn"
    _tel: str = "xmm"
    _lo_en: Quantity = Quantity(0.5, "keV")
    _hi_en: Quantity = Quantity(2.0, "keV")

    @classmethod
    def setUpClass(cls) -> None:
        """Create synthetic FITS image and exposure map files with a valid TAN WCS header."""
        cls._tmp_dir = tempfile.TemporaryDirectory()

        # Construct a simple 64x64 TAN WCS centred at RA=30.0 deg, Dec=-10.0 deg with 4 arcsec/pixel scale
        w = WCS(naxis=2)
        w.wcs.crpix = [32.0, 32.0]
        w.wcs.cdelt = np.array([-4.0 / 3600.0, 4.0 / 3600.0])
        w.wcs.crval = [30.0, -10.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        hdr = w.to_header()
        hdr["TELESCOP"] = "XMM"
        hdr["INSTRUME"] = "EPN"
        hdr["OBS_ID"] = cls._obs_id

        # Create deterministic non-zero image counts and exposure map values within an active detector sub-region
        im_data = np.zeros((64, 64), dtype=np.float64)
        im_data[10:54, 10:54] = 4.0
        im_data[20, 15] = 12.0  # y=20, x=15 -> 12 ct
        im_data[35, 40] = 30.0  # y=35, x=40 -> 30 ct

        ex_data = np.zeros((64, 64), dtype=np.float64)
        ex_data[10:54, 10:54] = 2.0
        ex_data[20, 15] = 3.0  # y=20, x=15 -> 3 s (rate = 4 ct/s)
        ex_data[35, 40] = 5.0  # y=35, x=40 -> 5 s (rate = 6 ct/s)

        cls._im_path = os.path.join(cls._tmp_dir.name, "synth_im.fits")
        cls._ex_path = os.path.join(cls._tmp_dir.name, "synth_ex.fits")

        fits.PrimaryHDU(data=im_data, header=hdr).writeto(cls._im_path, overwrite=True)
        fits.PrimaryHDU(data=ex_data, header=hdr).writeto(cls._ex_path, overwrite=True)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up the temporary directory holding synthetic FITS files."""
        cls._tmp_dir.cleanup()

    def _make_products(self) -> tuple[Image, ExpMap, RateMap]:
        """
        Instantiate fresh Image, ExpMap, and RateMap instances backed by the synthetic FITS files.

        :return: Tuple of (Image, ExpMap, RateMap).
        :rtype: tuple[Image, ExpMap, RateMap]
        """
        im = Image(
            self._im_path,
            self._obs_id,
            self._inst,
            "",
            "",
            "",
            self._lo_en,
            self._hi_en,
            telescope=self._tel,
        )
        ex = ExpMap(
            self._ex_path,
            self._obs_id,
            self._inst,
            "",
            "",
            "",
            self._lo_en,
            self._hi_en,
            telescope=self._tel,
        )
        rt = RateMap(im, ex)
        return im, ex, rt

    def test_coord_conv_and_point_lookups(self) -> None:
        """Test 1D and 2D coordinate conversions and get_count/get_exp/get_rate lookups."""
        im, ex, rt = self._make_products()

        single_pix = Quantity([15, 20], "pix")
        multi_pix = Quantity([[15, 20], [40, 35]], "pix")

        # 1D and 2D pixel -> deg -> pixel round-trip conversions
        single_deg = im.coord_conv(single_pix, "deg")
        self.assertEqual(single_deg.shape, (2,))
        np.testing.assert_array_equal(im.coord_conv(single_deg, "pix").value, single_pix.value)

        multi_deg = im.coord_conv(multi_pix, "deg")
        self.assertEqual(multi_deg.shape, (2, 2))
        np.testing.assert_array_equal(im.coord_conv(multi_deg, "pix").value, multi_pix.value)

        # Single-coordinate lookups on Image, ExpMap, and RateMap
        self.assertEqual(im.get_count(single_pix), Quantity(12.0, "ct"))
        self.assertEqual(ex.get_exp(single_pix), Quantity(3.0, "s"))
        self.assertEqual(rt.get_count(single_pix), Quantity(12.0, "ct"))
        self.assertEqual(rt.get_exp(single_pix), Quantity(3.0, "s"))
        self.assertEqual(rt.get_rate(single_pix), Quantity(4.0, "ct/s"))

        # Multi-coordinate (2D) lookups on Image, ExpMap, and RateMap
        np.testing.assert_allclose(im.get_count(multi_pix).value, np.array([12.0, 30.0]))
        np.testing.assert_allclose(ex.get_exp(multi_pix).value, np.array([3.0, 5.0]))
        np.testing.assert_allclose(rt.get_count(multi_pix).value, np.array([12.0, 30.0]))
        np.testing.assert_allclose(rt.get_exp(multi_pix).value, np.array([3.0, 5.0]))
        np.testing.assert_allclose(rt.get_rate(multi_pix).value, np.array([4.0, 6.0]))

        # Invalid coordinate array shapes should raise ValueError
        with self.assertRaises(ValueError):
            im.coord_conv(Quantity([15, 20, 25], "pix"), "deg")
        with self.assertRaises(ValueError):
            im.coord_conv(Quantity([[15, 20, 25], [40, 35, 30]], "pix"), "deg")

    def test_regions_and_matched_regions_handling(self) -> None:
        """Test regions and matched_regions initialisation, property setters, and RateMap->Image propagation."""
        im, ex, rt = self._make_products()

        pix_reg = CirclePixelRegion(center=PixCoord(x=25.0, y=25.0), radius=5.0)
        sky_reg = CircleSkyRegion(center=SkyCoord(ra=30.0, dec=-10.0, unit="deg"), radius=Quantity(16.0, "arcsec"))

        # 1. Assign a list of regions (including a None entry that should be filtered out)
        im.regions = [pix_reg, sky_reg, None]
        self.assertIn(self._obs_id, im.regions)
        self.assertEqual(len(im.regions[self._obs_id]), 2)
        self.assertTrue(all(isinstance(r, CirclePixelRegion) for r in im.regions[self._obs_id]))

        # 2. Assign a telescope-nested dictionary (like BaseSource._regions)
        nested_regs = {self._tel: {self._obs_id: [sky_reg]}}
        rt.regions = nested_regs
        self.assertIn(self._obs_id, rt.regions)
        self.assertEqual(len(rt.regions[self._obs_id]), 1)
        # Verify propagation to the underlying Image inside RateMap
        self.assertIn(self._obs_id, rt.image.regions)
        self.assertEqual(len(rt.image.regions[self._obs_id]), 1)

        # 3. Assign a dictionary where the ObsID entry is None
        rt.regions = {self._obs_id: None}
        self.assertEqual(rt.regions[self._obs_id], [])
        self.assertEqual(rt.image.regions[self._obs_id], [])

        # 4. Clearing regions with None resets to an empty dict
        rt.regions = None
        self.assertEqual(rt.regions, {})
        self.assertEqual(rt.image.regions, {})

        # 5. Matched regions: single SkyRegion, telescope-nested dict, and clearing with None
        rt.matched_regions = sky_reg
        self.assertIn(self._obs_id, rt.matched_regions)
        self.assertIsInstance(rt.matched_regions[self._obs_id], CirclePixelRegion)
        self.assertIn(self._obs_id, rt.image.matched_regions)

        rt.matched_regions = {self._tel: {self._obs_id: pix_reg}}
        self.assertIsInstance(rt.matched_regions[self._obs_id], CirclePixelRegion)
        self.assertIsInstance(rt.image.matched_regions[self._obs_id], CirclePixelRegion)

        rt.matched_regions = None
        self.assertEqual(rt.matched_regions, {})
        self.assertEqual(rt.image.matched_regions, {})

        # 6. RateMap.__init__ with explicit regs and matched_regs
        rt_init = RateMap(im, ex, regs=[pix_reg], matched_regs=sky_reg)
        self.assertEqual(len(rt_init.regions[self._obs_id]), 1)
        self.assertEqual(len(rt_init.image.regions[self._obs_id]), 1)
        self.assertIsInstance(rt_init.matched_regions[self._obs_id], CirclePixelRegion)
        self.assertIsInstance(rt_init.image.matched_regions[self._obs_id], CirclePixelRegion)

    def test_get_view_and_save_view(self) -> None:
        """Test Image and RateMap get_view/save_view with zoom_in, mask, cross_hair, and radial/background bins."""
        im, _, rt = self._make_products()

        mask = np.ones(im.shape, dtype=np.float64)
        mask[:15, :] = 0.0

        pix_reg = CirclePixelRegion(center=PixCoord(x=32.0, y=32.0), radius=6.0)
        rt.regions = [pix_reg]
        rt.matched_regions = pix_reg

        # Test get_view with zoom_in=True, mask, 1D cross_hair, and angular radial_bins_pix / back_bin_pix
        fig, ax = plt.subplots(figsize=(6, 6))
        ax_out = rt.get_view(
            ax,
            cross_hair=Quantity([30.0, -10.0], "deg"),
            mask=mask,
            zoom_in=True,
            radial_bins_pix=Quantity([12.0, 24.0], "arcsec"),
            back_bin_pix=Quantity([30.0, 40.0], "arcsec"),
            view_regions=True,
        )
        self.assertIs(ax_out, ax)
        plt.close(fig)

        # Test get_view with 2D cross_hair, manual zoom limits, and pixel radial_bins_pix / back_bin_pix
        fig, ax = plt.subplots(figsize=(6, 6))
        im.get_view(
            ax,
            cross_hair=Quantity([[28, 28], [36, 36]], "pix"),
            zoom_in=True,
            manual_zoom_xlims=(10.0, 54.0),
            manual_zoom_ylims=(10.0, 54.0),
            radial_bins_pix=Quantity([4, 8], "pix"),
            back_bin_pix=Quantity([10, 14], "pix"),
        )
        plt.close(fig)

        # Save a view to the test figure output directory
        save_path = os.path.join(MISC_OUTPUT_TESTS, f"{self.id()}.png")
        rt.save_view(save_path, zoom_in=True, view_regions=True)
        self.assertTrue(os.path.exists(save_path))

        # Incompatible units on radial_bins_pix or back_bin_pix must raise UnitConversionError
        fig, ax = plt.subplots(figsize=(6, 6))
        with self.assertRaises(UnitConversionError):
            im.get_view(
                ax,
                cross_hair=Quantity([32, 32], "pix"),
                radial_bins_pix=Quantity([5, 10], "keV"),
            )
        with self.assertRaises(UnitConversionError):
            im.get_view(
                ax,
                cross_hair=Quantity([32, 32], "pix"),
                back_bin_pix=Quantity([12, 16], "keV"),
            )
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
