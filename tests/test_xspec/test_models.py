#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/4/26, 9:50 PM. Copyright (c) The Contributors.
"""Unit tests for XSPEC model.dat parsing, model expression validation, and custom fit function generation."""

import inspect
import unittest
import warnings

from astropy.units import Quantity, Unit

from xga import utils
from xga.exceptions import XSPECFitError
from xga.xspec.fit import custom_xspec_fit, describe_xspec_model, general_model_fit, validate_and_parse_model
from xga.xspec.fitconfgen import fit_conf_from_function


class TestXSPECModelsParsing(unittest.TestCase):
    """Tests for dynamic XSPEC model.dat parsing, validation, and factory utilities."""

    def test_lazy_loaded_model_tables(self) -> None:
        """Checks that XSPEC_MODELS, MODEL_PARS, and MODEL_LIMITS are populated from HEASoft's model.dat."""
        self.assertIsInstance(utils.XSPEC_MODELS, dict)
        self.assertIn("apec", utils.XSPEC_MODELS)
        self.assertIn("tbabs", utils.XSPEC_MODELS)
        self.assertIn("powerlaw", utils.XSPEC_MODELS)
        self.assertIn("constant", utils.XSPEC_MODELS)

        self.assertEqual(utils.XSPEC_MODELS["apec"]["model_type"], "add")
        self.assertEqual(utils.XSPEC_MODELS["tbabs"]["model_type"], "mul")

        self.assertIn("apec", utils.MODEL_PARS)
        self.assertEqual(utils.MODEL_PARS["apec"], ["kT", "Abundanc", "Redshift", "norm"])
        self.assertIn("apec", utils.MODEL_UNITS)
        self.assertEqual(utils.MODEL_UNITS["apec"][0], Unit("keV"))

    def test_validate_and_parse_standard_models(self) -> None:
        """Tests model expression normalisation, parameter disambiguation, and absorption nH detection."""
        clean_mod, flat_pars, nh_to_zero = validate_and_parse_model("TBabs * apec", norm_scale=True)
        self.assertEqual(clean_mod, "constant*tbabs*apec")
        self.assertEqual([p["par_key"] for p in flat_pars], ["factor", "nH", "kT", "Abundanc", "Redshift", "norm"])
        self.assertEqual(nh_to_zero, [2])

        # Verify norm_scale=False does not prepend constant*
        clean_no_const, flat_no_const, nh_no_const = validate_and_parse_model("tbabs*apec", norm_scale=False)
        self.assertEqual(clean_no_const, "tbabs*apec")
        self.assertEqual([p["par_key"] for p in flat_no_const], ["nH", "kT", "Abundanc", "Redshift", "norm"])
        self.assertEqual(nh_no_const, [1])

        # Verify top-level addition is wrapped in parentheses when norm_scale=True
        clean_add, _, _ = validate_and_parse_model("apec + powerlaw", norm_scale=True)
        self.assertEqual(clean_add, "constant*(apec+powerlaw)")

        # Verify duplicate parameter names across components use _1, _2 suffixes matching general_xspec_fit.xcm
        clean_2t, flat_2t, nh_2t = validate_and_parse_model("tbabs*(apec+apec)", norm_scale=True)
        self.assertEqual(clean_2t, "constant*tbabs*(apec+apec)")
        self.assertEqual(
            [p["par_key"] for p in flat_2t],
            [
                "factor",
                "nH",
                "kT",
                "Abundanc",
                "Redshift",
                "norm",
                "kT_1",
                "Abundanc_1",
                "Redshift_1",
                "norm_1",
            ],
        )
        self.assertEqual(nh_2t, [2])

    def test_plasma_nh_vs_absorption_nh(self) -> None:
        """Checks that additive plasma density nH (e.g. mekal) is not misidentified as absorption nH."""
        _, flat_pars, nh_to_zero = validate_and_parse_model("tbabs*mekal", norm_scale=True)
        # Only tbabs nH (par_idx 2) is an absorption column density; mekal's nH_1 (par_idx 4, cm^-3) is not
        self.assertEqual(nh_to_zero, [2])
        par_by_key = {p["par_key"]: p for p in flat_pars}
        self.assertTrue(par_by_key["nH"]["is_galactic_nh"])
        self.assertFalse(par_by_key["nH_1"]["is_galactic_nh"])
        self.assertFalse(par_by_key["nH_1"]["is_abs_nh"])

    def test_multiple_or_redshifted_absorbers_warn(self) -> None:
        """Checks that models with redshifted or secondary absorbers emit a warning and preserve defaults."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, flat_pars, nh_to_zero = validate_and_parse_model("tbabs*ztbabs*apec", norm_scale=True)
            self.assertEqual(len(caught), 1)
            self.assertIn("redshifted or multiple absorption components", str(caught[0].message))

        self.assertEqual(nh_to_zero, [2, 3])
        par_by_key = {p["par_key"]: p for p in flat_pars}
        self.assertTrue(par_by_key["nH"]["is_galactic_nh"])
        self.assertFalse(par_by_key["nH_1"]["is_galactic_nh"])
        # ztbabs Redshift should NOT be auto-populated with the source redshift by default
        self.assertFalse(par_by_key["Redshift"]["is_redshift"])
        # apec Redshift_1 SHOULD be auto-populated with the source redshift
        self.assertTrue(par_by_key["Redshift_1"]["is_redshift"])

    def test_invalid_model_expressions(self) -> None:
        """Checks that malformed or non-physical model expressions raise appropriate exceptions."""
        with self.assertRaises(ValueError):
            validate_and_parse_model("")

        with self.assertRaises(ValueError):
            validate_and_parse_model("tbabs(apec+apec)")

        with self.assertRaises(ValueError):
            validate_and_parse_model("tbabs*(apec+apec")

        with self.assertRaises(XSPECFitError):
            validate_and_parse_model("tbabs*not_a_real_xspec_model")

        with self.assertRaises(XSPECFitError):
            validate_and_parse_model("tbabs*constant")

    def test_describe_xspec_model(self) -> None:
        """Checks that describe_xspec_model returns editable start, freeze, and link dictionaries."""
        start_pars, freeze_pars, link_pars = describe_xspec_model("tbabs*apec", print_table=False)
        self.assertEqual(set(start_pars.keys()), {"nH", "kT", "Abundanc", "Redshift", "norm"})
        self.assertIsNone(start_pars["nH"])
        self.assertIsNone(start_pars["Redshift"])
        self.assertEqual(start_pars["kT"], Quantity(1.0, "keV"))
        self.assertTrue(freeze_pars["nH"])
        self.assertFalse(freeze_pars["kT"])
        self.assertTrue(freeze_pars["Abundanc"])
        self.assertTrue(all(link_pars.values()))

    def test_custom_xspec_fit_signature(self) -> None:
        """Checks that custom_xspec_fit constructs a callable with explicit per-parameter arguments."""
        fit_fn = custom_xspec_fit("tbabs*(apec+powerlaw)")
        sig = inspect.signature(fit_fn)
        for expected_arg in (
            "sources",
            "outer_radius",
            "inner_radius",
            "start_nH",
            "start_kT",
            "start_PhoIndex",
            "freeze_Abundanc",
            "link_norm",
            "link_norm_1",
        ):
            self.assertIn(expected_arg, sig.parameters)
        self.assertIsNotNone(fit_fn.__doc__)
        assert fit_fn.__doc__ is not None
        self.assertIn("constant*tbabs*(apec+powerlaw)", fit_fn.__doc__)

    def test_fit_conf_from_general_model_fit(self) -> None:
        """Checks that fit_conf_from_function handles dict, list, and None overrides for general_model_fit."""
        key_default = fit_conf_from_function(general_model_fit, {})
        key_override = fit_conf_from_function(
            general_model_fit,
            {
                "start_pars": {"kT": Quantity(4.5, "keV"), "Abundanc": 0.3},
                "freeze_pars": {"Abundanc": False},
            },
        )
        self.assertNotEqual(key_default, key_override)
        # Key generation must be independent of dictionary insertion order
        key_override_reordered = fit_conf_from_function(
            general_model_fit,
            {
                "start_pars": {"Abundanc": 0.3, "kT": Quantity(4.5, "keV")},
                "freeze_pars": {"Abundanc": False},
            },
        )
        self.assertEqual(key_override, key_override_reordered)

        # Custom fit functions must produce flat fit_conf keys identical in format to standard fit functions,
        #  and accept both flat kwargs (e.g. freeze_Velocity=False) and dict overrides (e.g. freeze_pars={...}),
        #  including disambiguated multi-component parameter keys (e.g. norm, norm_1).
        bapec_pl_fn = custom_xspec_fit("tbabs*(bapec+powerlaw)")
        flat_default = fit_conf_from_function(bapec_pl_fn)
        self.assertIn("freezeVelocityTrue", flat_default)
        self.assertIn("linknormTrue", flat_default)
        self.assertIn("linknorm1True", flat_default)
        flat_override = fit_conf_from_function(
            bapec_pl_fn,
            {"freeze_Velocity": False, "start_kT": Quantity(4.5, "keV"), "start_norm_1": 1e-4},
        )
        self.assertIn("freezeVelocityFalse", flat_override)
        self.assertIn("startkT4.5keV", flat_override)
        self.assertIn("startnorm10.0001", flat_override)
        dict_override = fit_conf_from_function(
            bapec_pl_fn,
            {"freeze_pars": {"Velocity": False}, "start_pars": {"kT": 4.5, "norm_1": 1e-4}},
        )
        self.assertEqual(flat_override, dict_override)


if __name__ == "__main__":
    unittest.main()
