#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 20/04/2026, 20:30. Copyright (c) The Contributors.

"""Tests for XGA LightCurve and AggregateLightCurve product classes and their view methods."""

import os
import unittest
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from xga.generate.esass.lightcurve import srctool_lightcurve
from xga.generate.sas.lightcurve import evselect_lightcurve
from xga.products.lightcurve import AggregateLightCurve, LightCurve

from .. import MISC_OUTPUT_TESTS, get_test_source
from ..utils import require_esass, require_sas


class TestLightCurveProducts(unittest.TestCase):
    """
    Tests for XGA LightCurve and AggregateLightCurve product methods, including get_view and one_tick_per_chunk
    behaviour across XMM and eROSITA light curves.
    """

    xmm_src: Any
    erass_src: Any
    lo_en: Quantity
    hi_en: Quantity

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise test sources and energy bounds for light curve view tests."""
        os.makedirs(MISC_OUTPUT_TESTS, exist_ok=True)
        cls.xmm_src = get_test_source("xmm")
        cls.erass_src = get_test_source("erass")
        cls.lo_en = Quantity(0.5, "keV")
        cls.hi_en = Quantity(2.0, "keV")

    def tearDown(self) -> None:
        """Close any open matplotlib figures after each test."""
        plt.close("all")

    def _check_lightcurve_and_aggregate_views(self, lcs: list[LightCurve], prefix: str) -> None:
        """
        Helper method to verify LightCurve.get_view and AggregateLightCurve.get_view (including one_tick_per_chunk
        options and validation) for a list of generated LightCurve instances.

        :param list[LightCurve] lcs: List of LightCurve objects for a source.
        :param str prefix: Filename prefix for saved test figures.
        """
        self.assertGreaterEqual(len(lcs), 2)
        lc = lcs[0]
        self.assertIsInstance(lc, LightCurve)

        # Single LightCurve.get_view test
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        out_ax = lc.get_view(ax, custom_title=f"{prefix} LightCurve Test")
        self.assertIsInstance(out_ax, Axes)
        lc_fig_path = os.path.join(MISC_OUTPUT_TESTS, f"{prefix}_single_lc_view.png")
        fig.savefig(lc_fig_path)
        self.assertTrue(os.path.exists(lc_fig_path))
        plt.close("all")

        # AggregateLightCurve instantiation and default get_view (one_tick_per_chunk=False)
        agg_lc = AggregateLightCurve(lcs)
        self.assertIsInstance(agg_lc, AggregateLightCurve)
        num_chunks = len(agg_lc.time_chunk_ids)
        self.assertGreaterEqual(num_chunks, 1)

        fig = plt.figure(figsize=(14, 5))
        axes_dict, out_fig = agg_lc.get_view(fig, one_tick_per_chunk=False)
        self.assertIsInstance(out_fig, Figure)
        self.assertEqual(len(axes_dict), num_chunks)
        agg_default_path = os.path.join(MISC_OUTPUT_TESTS, f"{prefix}_agg_lc_default.png")
        out_fig.savefig(agg_default_path, bbox_inches="tight")
        self.assertTrue(os.path.exists(agg_default_path))
        plt.close("all")

        # AggregateLightCurve.get_view with one_tick_per_chunk=True (broadcast across all chunks)
        fig = plt.figure(figsize=(14, 5))
        axes_dict, out_fig = agg_lc.get_view(fig, one_tick_per_chunk=True)
        for tc_ind in range(num_chunks):
            self.assertEqual(len(axes_dict[tc_ind].get_xticks()), 1)
        agg_single_tick_path = os.path.join(MISC_OUTPUT_TESTS, f"{prefix}_agg_lc_one_tick.png")
        out_fig.savefig(agg_single_tick_path, bbox_inches="tight")
        self.assertTrue(os.path.exists(agg_single_tick_path))
        plt.close("all")

        # AggregateLightCurve.get_view with per-chunk boolean list and numpy array
        per_chunk_flags = [i % 2 == 0 for i in range(num_chunks)]
        fig = plt.figure(figsize=(14, 5))
        axes_dict, _ = agg_lc.get_view(fig, one_tick_per_chunk=per_chunk_flags)
        for tc_ind, flag in enumerate(per_chunk_flags):
            if flag:
                self.assertEqual(len(axes_dict[tc_ind].get_xticks()), 1)
        plt.close("all")

        fig = plt.figure(figsize=(14, 5))
        axes_dict, _ = agg_lc.get_view(fig, one_tick_per_chunk=np.array(per_chunk_flags))
        for tc_ind, flag in enumerate(per_chunk_flags):
            if flag:
                self.assertEqual(len(axes_dict[tc_ind].get_xticks()), 1)
        plt.close("all")

        # Validation errors for invalid one_tick_per_chunk inputs
        fig = plt.figure(figsize=(14, 5))
        with self.assertRaises(TypeError):
            agg_lc.get_view(fig, one_tick_per_chunk="invalid")  # type: ignore[arg-type]
        plt.close("all")

        fig = plt.figure(figsize=(14, 5))
        with self.assertRaises(TypeError):
            agg_lc.get_view(fig, one_tick_per_chunk=[1] * num_chunks)  # type: ignore[list-item]
        plt.close("all")

        fig = plt.figure(figsize=(14, 5))
        with self.assertRaises(ValueError):
            agg_lc.get_view(fig, one_tick_per_chunk=[True] * (num_chunks + 1))
        plt.close("all")

    @require_sas
    def test_xmm_lightcurve_views(self) -> None:
        """Test LightCurve and AggregateLightCurve get_view methods using real XMM light curves generated with SAS."""
        evselect_lightcurve(self.xmm_src, "r500", lo_en=self.lo_en, hi_en=self.hi_en)
        lcs = self.xmm_src.get_lightcurves("r500", lo_en=self.lo_en, hi_en=self.hi_en, telescope="xmm")
        if isinstance(lcs, LightCurve):
            lcs = [lcs]
        self._check_lightcurve_and_aggregate_views(lcs, "xmm")

    @require_esass
    def test_erass_lightcurve_views(self) -> None:
        """Test LightCurve and AggregateLightCurve get_view methods using real eROSITA light curves from eSASS."""
        srctool_lightcurve(
            self.erass_src,
            "r500",
            lo_en=self.lo_en,
            hi_en=self.hi_en,
            combine_tm=False,
            combine_obs=False,
        )
        lcs = self.erass_src.get_lightcurves(
            "r500",
            lo_en=self.lo_en,
            hi_en=self.hi_en,
            telescope="erass",
            pattern=None,
        )
        if isinstance(lcs, LightCurve):
            lcs = [lcs]
        self._check_lightcurve_and_aggregate_views(lcs, "erass")


if __name__ == "__main__":
    unittest.main()
