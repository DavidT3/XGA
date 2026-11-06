#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/4/26, 9:50 PM. Copyright (c) The Contributors.
"""Validation, tokenisation, and parameter inspection utilities for XSPEC model expressions."""

import os
import re
import shlex
from collections.abc import Callable
from inspect import Parameter, Signature
from typing import Any
from warnings import warn

import numpy as np
from astropy.units import Quantity, Unit
from tabulate import tabulate

from ... import NUM_CORES, utils
from ...exceptions import XSPECFitError, XSPECNotFoundError
from ...samples.base import BaseSample
from ...sources import BaseSource
from ..fitconfgen import FIT_FUNC_ARGS

# Valid characters in an XSPEC model expression (model names, '+', '*', and parentheses)
_VALID_XSPEC_EXPR_RE = re.compile(r"^[a-zA-Z0-9_.*+()]+$")
# Matches either an operator/parenthesis or a model component identifier
_XSPEC_TOKEN_RE = re.compile(r"\(|\)|\*|\+|[a-zA-Z0-9_.]+")

# Mapping of non-standard or shorthand unit strings used in XSPEC's model.dat file onto valid astropy unit
#  representations. XSPEC's model.dat was written for human display in terminal prompts rather than strict machine
#  parsing, so many entries include parenthetical notes (e.g. 'keV(-th)'), omit the column-density area dimension
#  (e.g. '10^22' for hydrogen column density in 10^22 cm^-2), or use text labels for dimensionless quantities
#  (e.g. 'solar', 'z', '0/1').
_XSPEC_UNIT_MAP: dict[str, str] = {
    "": "",
    "0/1": "",
    "0off/1on": "",
    "real": "",
    "solar": "",
    "Solar": "",
    "z": "",
    "(-disk)": "",
    "(-selfg)": "",
    "(_calc)": "",
    "(_disk)": "",
    "(_selfg)": "",
    "10^22": "10^22 cm^-2",
    "10^16": "10^16 cm^-2",
    "1e18": "10^18 cm^-2",
    "10^-4": "1e-4",
    "10^9": "1e9",
    "1e12G": "1e12 G",
    "1e17g/s": "1e17 g s^-1",
    "A": "AA",
    "Gau": "G",
    "KeV": "keV",
    "keV(-nt)": "keV",
    "keV(-pl)": "keV",
    "keV(-sc)": "keV",
    "keV(-th)": "keV",
    "keV(_pl)": "keV",
    "keV(_sc)": "keV",
    "M_sun": "Msun",
    "Msun": "Msun",
    "M0yr": "Msun yr^-1",
    "cm-3": "cm^-3",
    "cm^-3": "cm^-3",
    "cm**-3": "cm^-3",
    "days": "d",
    "mum": "um",
    "ergcm/s": "erg cm s^-1",
    "erg cm/s": "erg cm s^-1",
}


def _xspec_unit_to_astropy(raw_unit: str) -> tuple[Unit, bool]:
    """
    Internal helper to convert an XSPEC parameter unit string (as defined in HEASoft's model.dat) into an
    astropy Unit object. If a unit string represents a dimensionless quantity, or uses an XSPEC-specific notation
    (such as 'R_g' or 'L_Edd') that cannot be represented as a standard physical astropy unit, a dimensionless
    Unit('') is returned rather than raising an error, along with a boolean flag indicating whether the unit was
    recognised so that callers (such as get_results with with_units=True) can warn the user if needed.

    :param str raw_unit: The raw unit string extracted from XSPEC's model.dat file.
    :return: A tuple of (astropy Unit object, boolean indicating whether the unit was recognised).
    :rtype: Tuple[Unit, bool]
    """
    cleaned = raw_unit.strip()
    if cleaned in _XSPEC_UNIT_MAP:
        return Unit(_XSPEC_UNIT_MAP[cleaned]), True

    # Strip any trailing parenthetical qualifiers (e.g. 'keV(-th)' or future annotations of that form) before
    #  asking astropy to parse the unit string.
    stripped_parens = re.sub(r"\([^)]*\)$", "", cleaned).strip()
    if stripped_parens in _XSPEC_UNIT_MAP:
        return Unit(_XSPEC_UNIT_MAP[stripped_parens]), True

    for candidate in (cleaned, stripped_parens):
        if not candidate:
            continue
        try:
            return Unit(candidate), True
        except ValueError:
            continue

    return Unit(""), False


def _parse_xspec_model_dat(model_dat_path: str) -> tuple[dict[str, dict], dict[str, list[str]], dict[str, list[Unit]]]:
    """
    Internal helper that reads the local HEASoft XSPEC model.dat descriptor file and extracts all built-in models,
    their types ('add', 'mul', 'con', 'acn'), and full parameter metadata (names, units, default values, hard/soft
    limits, and default freeze states).

    We parse model.dat directly from the user's active HEASoft installation rather than bundling static model lists
    with XGA because XGA does not mandate a single XSPEC version, and new models (or updated parameter definitions)
    are added across HEASoft releases.

    :param str model_dat_path: Path to the model.dat file in the local HEASoft installation.
    :return: A tuple of three dictionaries keyed by lowercase model name:
        1. Full model metadata dictionary (XSPEC_MODELS).
        2. Mapping of model name to ordered list of parameter names (MODEL_PARS).
        3. Mapping of model name to ordered list of astropy Unit objects (MODEL_UNITS).
    :rtype: Tuple[dict[str, dict], dict[str, list[str]], dict[str, list[Unit]]]
    """
    xspec_models: dict[str, dict] = {}
    model_pars: dict[str, list[str]] = {}
    model_units: dict[str, list[Unit]] = {}

    if not os.path.exists(model_dat_path):
        return xspec_models, model_pars, model_units

    with open(model_dat_path, encoding="utf-8", errors="replace") as model_file:
        lines = [line.strip() for line in model_file if line.strip()]

    line_idx = 0
    num_lines = len(lines)
    while line_idx < num_lines:
        hdr_tokens = shlex.split(lines[line_idx])
        # A valid model header line in model.dat has at least 6 tokens:
        #  <model_name> <num_pars> <lo_energy> <hi_energy> <func_name> <model_type> [flags...]
        if len(hdr_tokens) < 6:
            line_idx += 1
            continue

        xspec_name = hdr_tokens[0]
        model_key = xspec_name.lower()
        num_pars = int(hdr_tokens[1])
        model_type = hdr_tokens[5].lower()

        pars_meta: list[dict] = []
        for par_offset in range(1, num_pars + 1):
            if line_idx + par_offset >= num_lines:
                break
            par_tokens = shlex.split(lines[line_idx + par_offset])
            if not par_tokens:
                continue

            raw_par_name = par_tokens[0]
            # In model.dat, a leading '$' denotes an integer switch parameter (e.g. '$switch 1' in mekal),
            #  and a leading '*' denotes a fixed scale/pivot parameter (e.g. '*pivotE keV 1.0' in logpar).
            #  When XSPEC loads the model (and when tcloutr pinfo is queried in TCL), those prefix characters
            #  are stripped from the parameter name, and the parameter cannot be freely fit by default.
            is_switch = raw_par_name.startswith("$")
            is_scale = raw_par_name.startswith("*")
            par_name = raw_par_name.lstrip("$*")

            if is_switch:
                # Most switch lines are 2 tokens ('$switch 1'), though a few include an empty unit string column
                #  before the default integer value (e.g. '$lflag " " 1.'), so the default value is always the final
                #  token on the line.
                default_val = float(par_tokens[-1]) if len(par_tokens) > 1 else 1.0
                pars_meta.append(
                    {
                        "name": par_name,
                        "raw_unit": "",
                        "unit": Unit(""),
                        "unit_recognised": True,
                        "default": default_val,
                        "hard_min": None,
                        "soft_min": None,
                        "soft_max": None,
                        "hard_max": None,
                        "delta": -1.0,
                        "frozen": True,
                        "switch_or_scale": True,
                    }
                )
            elif is_scale and len(par_tokens) < 8:
                raw_unit = par_tokens[1].strip() if len(par_tokens) > 1 else ""
                par_unit, unit_rec = _xspec_unit_to_astropy(raw_unit)
                default_val = float(par_tokens[2]) if len(par_tokens) > 2 else 1.0
                pars_meta.append(
                    {
                        "name": par_name,
                        "raw_unit": raw_unit,
                        "unit": par_unit,
                        "unit_recognised": unit_rec,
                        "default": default_val,
                        "hard_min": None,
                        "soft_min": None,
                        "soft_max": None,
                        "hard_max": None,
                        "delta": -1.0,
                        "frozen": True,
                        "switch_or_scale": True,
                    }
                )
            else:
                # Standard 8-column parameter entry:
                #  <name> <unit> <default> <hard_min> <soft_min> <soft_max> <hard_max> <delta>
                # A negative or zero step size (delta) means XSPEC freezes the parameter by default.
                raw_unit = par_tokens[1].strip()
                par_unit, unit_rec = _xspec_unit_to_astropy(raw_unit)
                default_val = float(par_tokens[2])
                hard_min = float(par_tokens[3])
                soft_min = float(par_tokens[4])
                soft_max = float(par_tokens[5])
                hard_max = float(par_tokens[6])
                delta = float(par_tokens[7])
                pars_meta.append(
                    {
                        "name": par_name,
                        "raw_unit": raw_unit,
                        "unit": par_unit,
                        "unit_recognised": unit_rec,
                        "default": default_val,
                        "hard_min": hard_min,
                        "soft_min": soft_min,
                        "soft_max": soft_max,
                        "hard_max": hard_max,
                        "delta": delta,
                        "frozen": (delta <= 0.0) or is_scale,
                        "switch_or_scale": is_scale,
                    }
                )

        # Additive ('add') models in XSPEC always have an implicit 'norm' parameter appended as their final
        #  parameter, even though it is not listed as a parameter line in model.dat.
        if model_type == "add":
            pars_meta.append(
                {
                    "name": "norm",
                    "raw_unit": "",
                    "unit": Unit(""),
                    "unit_recognised": True,
                    "default": 1.0,
                    "hard_min": 0.0,
                    "soft_min": 0.0,
                    "soft_max": 1e24,
                    "hard_max": 1e24,
                    "delta": 0.01,
                    "frozen": False,
                    "switch_or_scale": False,
                }
            )

        xspec_models[model_key] = {
            "xspec_name": xspec_name,
            "model_type": model_type,
            "pars": pars_meta,
        }
        model_pars[model_key] = [p["name"] for p in pars_meta]
        model_units[model_key] = [p["unit"] for p in pars_meta]

        line_idx += num_pars + 1

    return xspec_models, model_pars, model_units


def _strip_outer_parens(expr: str) -> str:
    """
    Removes redundant outermost parentheses that wrap an entire model expression (e.g. '(tbabs*apec)' -> 'tbabs*apec'),
    while preserving parentheses that group sub-expressions (e.g. '(apec+apec)*(tbabs)').

    :param str expr: Whitespace-stripped model expression string with balanced parentheses.
    :return: Expression with redundant outer parentheses stripped.
    :rtype: str
    """
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        wraps_all = True
        # Walk through the expression and check whether the opening parenthesis at index 0 closes before the final
        #  character; if it does (e.g. '(apec+apec)*(tbabs)'), the outer parentheses do not wrap the whole expression
        for idx, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and idx < len(expr) - 1:
                wraps_all = False
                break
        if wraps_all:
            expr = expr[1:-1]
        else:
            break
    return expr


def _has_top_level_addition(expr: str) -> bool:
    """
    Checks whether a model expression contains a '+' operator at parenthesis depth 0.

    We need this when prepending 'constant*' to a multi-component model: if the user passes 'apec+powerlaw',
    simply prepending 'constant*' would produce 'constant*apec+powerlaw' (which in XSPEC only multiplies 'apec'
    by the constant), whereas 'constant*(apec+powerlaw)' scales the entire model as intended.

    :param str expr: Whitespace-stripped model expression string.
    :return: True if a '+' operator appears outside any parentheses, False otherwise.
    :rtype: bool
    """
    depth = 0
    # Track parenthesis nesting depth and return True immediately if we encounter '+' at depth 0
    for char in expr:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "+" and depth == 0:
            return True
    return False


def _is_absorption_nh_par(model_type: str, par_meta: dict[str, Any]) -> bool:
    """
    Determines whether a parameter represents a photoelectric absorption hydrogen column density (nH) on a
    multiplicative model component, as opposed to an intrinsic plasma density parameter (such as mekal's nH).

    In HEASoft's model.dat, multiplicative absorption models (e.g. tbabs, ztbabs, phabs, wabs, pcfabs) have
    model_type == 'mul', parameter name 'nH', raw_unit == '10^22', and a hard lower limit of 0.0. Plasma emission
    models with an 'nH' parameter (e.g. mekal, vmekal, bcie) are additive ('add'), have units of 'cm-3', and
    have a non-zero hard lower limit (1e-6).

    :param str model_type: The XSPEC model type ('add', 'mul', 'con', 'acn').
    :param dict par_meta: Parameter metadata dictionary from XSPEC_MODELS.
    :return: True if the parameter is a multiplicative absorption column density, False otherwise.
    :rtype: bool
    """
    return (
        model_type == "mul"
        and par_meta["name"].lower() == "nh"
        and par_meta["raw_unit"] == "10^22"
        and par_meta["hard_min"] == 0.0
    )


def _component_has_absorption_nh(comp_info: dict[str, Any]) -> bool:
    """
    Checks whether a model component is a multiplicative absorption model with a column density nH parameter.

    :param dict comp_info: Component metadata dictionary from XSPEC_MODELS.
    :return: True if the component is a multiplicative absorber with an nH parameter, False otherwise.
    :rtype: bool
    """
    m_type = comp_info["model_type"]
    return any(_is_absorption_nh_par(m_type, p) for p in comp_info["pars"])


def _component_has_redshift(comp_info: dict[str, Any]) -> bool:
    """
    Checks whether a model component defines a cosmological redshift parameter ('Redshift' or 'z').

    :param dict comp_info: Component metadata dictionary from XSPEC_MODELS.
    :return: True if the component has a redshift parameter, False otherwise.
    :rtype: bool
    """
    return any(p["name"].lower() in ("redshift", "z") and not p["switch_or_scale"] for p in comp_info["pars"])


def _is_redshift_par(par_meta: dict[str, Any]) -> bool:
    """
    Determines whether a model parameter represents cosmological redshift (so that XGA can automatically
    populate it from source.redshift unless overridden by the user).

    :param dict par_meta: Parameter metadata dictionary from XSPEC_MODELS.
    :return: True if the parameter is a redshift parameter, False otherwise.
    :rtype: bool
    """
    return par_meta["name"].lower() in ("redshift", "z") and not par_meta["switch_or_scale"]


def validate_and_parse_model(
    model: str,
    norm_scale: bool = True,
) -> tuple[str, list[dict[str, Any]], list[int]]:
    """
    Validates an XSPEC model expression against the locally installed XSPEC model.dat definitions, normalises
    its formatting, optionally prepends the 'constant*' cross-calibration factor used in simultaneous fits, and
    returns the ordered list of parameter descriptors for a single spectrum/data group alongside the 1-based
    parameter indices of any absorption nH parameters to zero when computing unabsorbed luminosities.

    Duplicate parameter names across multiple components (e.g. 'kT' in 'tbabs*(apec+apec)') are disambiguated
    using the exact same suffix convention ('kT', 'kT_1', 'kT_2', ...) used internally by XGA's
    general_xspec_fit.xcm script.

    :param str model: The XSPEC model expression (e.g. 'tbabs*apec', 'tbabs*(apec+powerlaw)').
    :param bool norm_scale: If True, prepends the multiplicative 'constant*' model component to the expression (if
        not already present) to account for cross-calibration normalisation offsets when simultaneously fitting
        multiple observations/instruments (linking the normalisation parameters of additive emission components such as
        'apec' or 'powerlaw' across spectra while allowing the multiplicative constant factor to vary independently per
        spectrum). Default is True.
    :return: A tuple of:
        1. The normalised XSPEC model string (lowercase, whitespace removed, 'constant*' prepended if requested).
        2. Ordered list of parameter metadata dictionaries for one data group.
        3. List of 1-based parameter indices corresponding to multiplicative absorption nH parameters.
    :rtype: Tuple[str, list[dict[str, Any]], list[int]]
    """
    if utils.XSPEC_VERSION is None or not utils.XSPEC_MODELS:
        raise XSPECNotFoundError(
            "XSPEC is either not installed or its model.dat descriptor file could not be located on this system."
        )

    if not isinstance(model, str) or not model.strip():
        raise ValueError("The XSPEC model expression must be a non-empty string.")

    # Remove all whitespace and convert to lowercase to match XGA's model storage naming conventions
    clean_model = "".join(model.split()).lower()

    if not _VALID_XSPEC_EXPR_RE.match(clean_model):
        raise ValueError(
            f"Model expression '{model}' contains invalid characters. Only alphanumeric model names, "
            f"'+', '*', '(', and ')' are permitted."
        )

    # Tokenise and verify balanced parentheses and valid operator placement before checking model names
    tokens = _XSPEC_TOKEN_RE.findall(clean_model)
    if "".join(tokens) != clean_model:
        raise ValueError(f"Model expression '{model}' could not be cleanly tokenised.")

    depth = 0
    prev_tok: str | None = None
    for tok in tokens:
        if tok == "(":
            if prev_tok is not None and prev_tok not in ("(", "*", "+"):
                # In XSPEC syntax, convolution/multiplicative grouping like 'tbabs(apec+apec)' without '*' is
                #  sometimes typed by hand, but XGA's naming convention requires explicit '*' operators.
                raise ValueError(
                    f"Missing '*' or '+' operator before '(' in model expression '{model}'. "
                    f"Please write multiplication explicitly (e.g. 'tbabs*(apec+apec)')."
                )
            depth += 1
        elif tok == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced closing parenthesis ')' in model expression '{model}'.")
            if prev_tok in ("(", "*", "+"):
                raise ValueError(f"Empty or incomplete parenthesised group in model expression '{model}'.")
        elif tok in ("*", "+"):
            if prev_tok is None or prev_tok in ("(", "*", "+"):
                raise ValueError(f"Invalid placement of operator '{tok}' in model expression '{model}'.")
        else:
            # Component name token
            if prev_tok is not None and prev_tok not in ("(", "*", "+"):
                raise ValueError(f"Missing operator before component '{tok}' in model expression '{model}'.")
        prev_tok = tok

    if depth != 0:
        raise ValueError(f"Unbalanced opening parenthesis '(' in model expression '{model}'.")
    if prev_tok in ("*", "+", "("):
        raise ValueError(f"Model expression '{model}' cannot end with '{prev_tok}'.")

    clean_model = _strip_outer_parens(clean_model)

    # Prepend 'constant*' if norm_scale is enabled. If the user already included a leading 'constant*' (for
    #  example 'constant*apec+powerlaw'), strip it first and re-apply so top-level additions are properly
    #  parenthesised as 'constant*(apec+powerlaw)'.
    if norm_scale:
        if clean_model.startswith("constant*"):
            clean_model = _strip_outer_parens(clean_model[len("constant*") :])
        if _has_top_level_addition(clean_model):
            clean_model = f"constant*({clean_model})"
        else:
            clean_model = f"constant*{clean_model}"
        tokens = _XSPEC_TOKEN_RE.findall(clean_model)

    components = [tok for tok in tokens if tok not in ("(", ")", "*", "+")]
    if not components:
        raise ValueError(f"Model expression '{model}' does not contain any model components.")

    # Verify that all component names exist in the user's installed XSPEC model.dat
    unknown_comps = [comp for comp in components if comp not in utils.XSPEC_MODELS]
    if unknown_comps:
        raise XSPECFitError(
            f"The following model component(s) are not recognised in your XSPEC installation's model.dat: "
            f"{', '.join(unknown_comps)}."
        )

    # XSPEC requires at least one additive ('add') model component to define a physical spectrum
    if not any(utils.XSPEC_MODELS[comp]["model_type"] == "add" for comp in components):
        raise XSPECFitError(
            f"Model expression '{clean_model}' does not contain any additive ('add') model components. "
            f"XSPEC requires at least one additive component."
        )

    # Flatten the parameters across all components in order and disambiguate duplicate parameter names
    #  using the exact same '<name>_1', '<name>_2' convention as general_xspec_fit.xcm.
    flat_pars: list[dict[str, Any]] = []
    nh_to_zero: list[int] = []
    seen_par_keys: set[str] = set()
    par_idx = 1

    # Track whether we have already assigned the Galactic column density (source.nH) to a non-redshifted
    #  multiplicative absorption component, and record any secondary or redshifted absorption components so we
    #  can warn the user about how their parameters are being treated.
    galactic_nh_assigned_key: str | None = None
    extra_abs_nh_keys: list[str] = []
    redshifted_abs_comps: list[str] = []

    for comp_idx, comp_name in enumerate(components, start=1):
        comp_info = utils.XSPEC_MODELS[comp_name]
        m_type = comp_info["model_type"]
        is_redshifted_absorber = (
            m_type == "mul" and _component_has_absorption_nh(comp_info) and _component_has_redshift(comp_info)
        )
        if is_redshifted_absorber:
            redshifted_abs_comps.append(comp_name)

        for par_meta in comp_info["pars"]:
            # Disambiguate parameter names that appear more than once across the model's components by appending
            #  _1, _2, etc. (matching the exact suffix convention used in general_xspec_fit.xcm)
            base_name = par_meta["name"]
            par_key = base_name
            dup_count = 1
            while par_key in seen_par_keys:
                par_key = f"{base_name}_{dup_count}"
                dup_count += 1
            seen_par_keys.add(par_key)

            # Identify whether this parameter is the leading cross-calibration constant factor or an absorption nH
            is_leading_norm_scale = norm_scale and comp_idx == 1 and comp_name == "constant" and par_idx == 1
            is_abs_nh = _is_absorption_nh_par(m_type, par_meta)

            # Only the first non-redshifted multiplicative absorption component (e.g. tbabs, phabs, wabs) is
            #  automatically populated with the Galactic column density (source.nH) and frozen by default.
            #  Redshifted absorption models (e.g. ztbabs, zphabs, zwabs) or secondary absorption components
            #  likely represent intrinsic/additional absorption whose column density and redshift should not be
            #  silently overwritten with Galactic nH or the cluster's redshift.
            if is_abs_nh and not is_redshifted_absorber and galactic_nh_assigned_key is None:
                is_galactic_nh = True
                galactic_nh_assigned_key = par_key
            else:
                is_galactic_nh = False
                if is_abs_nh:
                    extra_abs_nh_keys.append(f"{comp_name}:{par_key} (par {par_idx})")

            # Cosmological redshift parameters on non-redshifted-absorber components default to source.redshift
            is_z = _is_redshift_par(par_meta) and not is_redshifted_absorber

            # Determine default freeze and link states: the leading constant factor is thawed and unlinked across
            #  spectra (since general_xspec_fit.xcm freezes data group 1's constant at 1.0 and thaws the rest),
            #  Galactic nH is frozen and linked, and all other parameters follow model.dat freeze defaults and link
            if is_leading_norm_scale:
                default_frozen = False
                default_linked = False
            elif is_galactic_nh:
                default_frozen = True
                default_linked = True
            else:
                default_frozen = bool(par_meta["frozen"])
                default_linked = True

            # Record the 1-based parameter index of any absorption column density so xga_extract can zero it when
            #  calculating unabsorbed luminosities
            if is_abs_nh:
                nh_to_zero.append(par_idx)

            flat_pars.append(
                {
                    "par_idx": par_idx,
                    "comp_idx": comp_idx,
                    "component": comp_name,
                    "model_type": m_type,
                    "par_name": base_name,
                    "par_key": par_key,
                    "raw_unit": par_meta["raw_unit"],
                    "unit": par_meta["unit"],
                    "unit_recognised": par_meta["unit_recognised"],
                    "default": par_meta["default"],
                    "hard_min": par_meta["hard_min"],
                    "soft_min": par_meta["soft_min"],
                    "soft_max": par_meta["soft_max"],
                    "hard_max": par_meta["hard_max"],
                    "frozen": default_frozen,
                    "linked": default_linked,
                    "switch_or_scale": par_meta["switch_or_scale"],
                    "is_leading_norm_scale": is_leading_norm_scale,
                    "is_abs_nh": is_abs_nh,
                    "is_galactic_nh": is_galactic_nh,
                    "is_redshift": is_z,
                }
            )
            par_idx += 1

    if extra_abs_nh_keys:
        gal_msg = (
            f"Only '{galactic_nh_assigned_key}' will default to source.nH (and be frozen)"
            if galactic_nh_assigned_key is not None
            else "No absorption component will automatically be assigned source.nH"
        )
        z_msg = (
            f" Redshifted absorption component(s) ({', '.join(redshifted_abs_comps)}) will also keep their "
            f"model.dat default redshift unless overridden."
            if redshifted_abs_comps
            else ""
        )
        warn(
            f"Model '{clean_model}' contains redshifted or multiple absorption components. {gal_msg}; "
            f"additional absorption nH parameter(s) ({', '.join(extra_abs_nh_keys)}) will use their model.dat "
            f"defaults unless overridden.{z_msg} By default, all absorption nH parameters (indices {nh_to_zero}) "
            f"will be zeroed when calculating unabsorbed luminosities unless 'nh_to_zero' is specified.",
            stacklevel=2,
        )

    return clean_model, flat_pars, nh_to_zero


def describe_xspec_model(
    model: str,
    norm_scale: bool = True,
    print_table: bool = True,
) -> tuple[dict[str, Quantity | float | None], dict[str, bool], dict[str, bool]]:
    """
    Validates an XSPEC model expression, optionally prints a summary table of all component parameters, units,
    default starting values, freeze/link states, and hard limits, and returns pre-populated dictionaries of
    starting values, freeze flags, and linking flags that can be modified and passed directly to
    ``general_model_fit``.

    Parameters that are automatically populated per source by XGA (such as Galactic absorption ``nH`` from
    ``source.nH`` and cosmological ``Redshift`` from ``source.redshift``) are given a value of ``None`` in the
    returned ``start_pars`` dictionary; leaving them as ``None`` instructs ``general_model_fit`` to substitute the
    source's own ``nH`` or ``redshift`` value, whereas replacing ``None`` with a float or Astropy ``Quantity``
    overrides that automatic substitution.

    :param str model: The XSPEC model expression (e.g. 'tbabs*apec', 'tbabs*(apec+powerlaw)').
    :param bool norm_scale: Whether 'constant*' cross-calibration scaling is prepended. Default is True.
    :param bool print_table: Whether to print a formatted parameter summary table to stdout. Default is True.
    :return: A tuple of three dictionaries ``(start_pars, freeze_pars, link_pars)`` keyed by disambiguated
        parameter name (``par_key``).
    :rtype: Tuple[dict[str, Quantity | float | None], dict[str, bool], dict[str, bool]]
    """
    clean_model, flat_pars, nh_to_zero = validate_and_parse_model(model, norm_scale=norm_scale)

    start_pars: dict[str, Quantity | float | None] = {}
    freeze_pars: dict[str, bool] = {}
    link_pars: dict[str, bool] = {}
    table_rows: list[list[Any]] = []

    for p in flat_pars:
        par_key = p["par_key"]
        unit_obj: Unit = p["unit"]
        raw_unit: str = p["raw_unit"]

        # Galactic nH and cosmological redshift parameters default to None in start_pars so general_model_fit knows
        #  to substitute source.nH and source.redshift automatically unless the user replaces None with a value
        if p["is_galactic_nh"]:
            default_display = f"<source.nH> ({p['default']} {raw_unit})"
            start_val: Quantity | float | None = None
        elif p["is_redshift"]:
            default_display = f"<source.redshift> ({p['default']})"
            start_val = None
        else:
            # Wrap parameters with recognised physical units in an Astropy Quantity
            if unit_obj != Unit(""):
                start_val = Quantity(p["default"], unit_obj)
                default_display = f"{p['default']} {unit_obj}"
            else:
                start_val = float(p["default"])
                default_display = f"{p['default']}"

        # Populate the user-editable dictionaries for all parameters except the auto-managed leading constant factor
        if not p["is_leading_norm_scale"]:
            start_pars[par_key] = start_val
            freeze_pars[par_key] = p["frozen"]
            link_pars[par_key] = p["linked"]

        # Format hard parameter limits for display in the summary table
        if p["hard_min"] is not None and p["hard_max"] is not None:
            limits_str = f"[{p['hard_min']}, {p['hard_max']}]"
        else:
            limits_str = "Fixed / Switch"

        # Build any contextual notes (norm_scale auto-management, unabsorbed luminosity zeroing, or unparsed units)
        notes = []
        if p["is_leading_norm_scale"]:
            notes.append("norm_scale factor (auto-managed)")
        if p["par_idx"] in nh_to_zero:
            notes.append("zeroed for unabs Lx")
        if not p["unit_recognised"] and raw_unit:
            notes.append(f"unparsed unit '{raw_unit}'")

        table_rows.append(
            [
                p["par_idx"],
                p["component"],
                par_key,
                raw_unit if raw_unit else "-",
                default_display,
                p["frozen"],
                p["linked"],
                limits_str,
                ", ".join(notes),
            ]
        )

    if print_table:
        headers = [
            "Idx",
            "Component",
            "Par Key",
            "XSPEC Unit",
            "Default Start",
            "Frozen",
            "Linked",
            "Hard Limits",
            "Notes",
        ]
        print(f"XSPEC Model: {clean_model}")
        print(tabulate(table_rows, headers=headers, tablefmt="simple"))

    return start_pars, freeze_pars, link_pars


def custom_xspec_fit(model: str, norm_scale: bool = True) -> Callable[..., BaseSource | BaseSample | list[BaseSource]]:
    """
    A factory function that validates an XSPEC model expression and dynamically generates a tailored XGA fitting
    function with explicit ``start_<par>``, ``freeze_<par>``, and ``link_<par>`` keyword arguments (and corresponding
    Sphinx docstring entries) for every parameter of the model.

    This allows users working in Jupyter notebooks or IDEs (such as PyCharm or VSCode) to inspect the exact
    parameter names, units, default values, and hard/soft limits via ``Shift+Tab`` or hover tooltips, while
    delegating the underlying spectral fit execution to :func:`~xga.xspec.fit.general_model_fit`.

    :param str model: The XSPEC model expression (e.g. 'tbabs*apec', 'tbabs*(apec+powerlaw)').
    :param bool norm_scale: Whether 'constant*' cross-calibration scaling is prepended to the model expression.
        Default is True.
    :return: A dynamically constructed fitting function with an explicit signature and docstring for the requested
        XSPEC model.
    :rtype: Callable[..., BaseSource | BaseSample | list[BaseSource]]
    """
    clean_model, flat_pars, auto_nh_to_zero = validate_and_parse_model(model, norm_scale=norm_scale)

    # Filter out the leading cross-calibration constant factor (if norm_scale=True), as that is managed
    #  automatically by general_xspec_fit.xcm just like in single_temp_apec
    user_pars = [p for p in flat_pars if not p["is_leading_norm_scale"]]

    sig_params: list[Parameter] = [
        Parameter(
            "sources",
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BaseSource | BaseSample | list[BaseSource],
        ),
        Parameter("outer_radius", Parameter.POSITIONAL_OR_KEYWORD, annotation=str | Quantity),
        Parameter(
            "inner_radius",
            Parameter.POSITIONAL_OR_KEYWORD,
            default=Quantity(0, "arcsec"),
            annotation=str | Quantity,
        ),
    ]

    default_start_map: dict[str, Quantity | float | None] = {}
    default_freeze_map: dict[str, bool] = {}
    default_link_map: dict[str, bool] = {}
    doc_param_lines: list[str] = []

    # 1. Add start_<par> parameters
    for p in user_pars:
        pk = p["par_key"]
        comp = p["component"]
        unit_obj: Unit = p["unit"]
        raw_unit: str = p["raw_unit"]

        if p["hard_min"] is not None and p["hard_max"] is not None:
            lims_desc = (
                f"soft limits [{p['soft_min']}, {p['soft_max']}], hard limits [{p['hard_min']}, {p['hard_max']}]"
            )
        else:
            lims_desc = "fixed/switch parameter"

        if p["is_galactic_nh"]:
            def_val: Quantity | float | None = None
            ann: Any = float | int | Quantity | list | np.ndarray | None
            doc_param_lines.append(
                f"    :param float/int/Quantity/list/np.ndarray start_{pk}: Initial value for the {comp} '{pk}' "
                f"parameter. Default is None, which automatically substitutes source.nH (in 10^22 cm^-2; "
                f"model.dat default is {p['default']} {raw_unit}, {lims_desc})."
            )
        elif p["is_redshift"]:
            def_val = None
            ann = float | int | list | np.ndarray | None
            doc_param_lines.append(
                f"    :param float/int/list/np.ndarray start_{pk}: Initial value for the {comp} '{pk}' "
                f"parameter. Default is None, which automatically substitutes source.redshift "
                f"(model.dat default is {p['default']}, {lims_desc})."
            )
        elif unit_obj != Unit(""):
            def_val = Quantity(p["default"], unit_obj)
            ann = Quantity | float | int | list | np.ndarray
            doc_param_lines.append(
                f"    :param Quantity/float/int/list/np.ndarray start_{pk}: Initial value for the {comp} '{pk}' "
                f"parameter in units convertible to {unit_obj} (default is Quantity({p['default']}, '{unit_obj}'), "
                f"{lims_desc})."
            )
        else:
            def_val = float(p["default"])
            ann = float | int | list | np.ndarray
            unit_note = f" (raw XSPEC unit '{raw_unit}')" if raw_unit else ""
            doc_param_lines.append(
                f"    :param float/int/list/np.ndarray start_{pk}: Initial value for the {comp} '{pk}' "
                f"parameter{unit_note} (default is {def_val}, {lims_desc})."
            )

        default_start_map[pk] = def_val
        sig_params.append(Parameter(f"start_{pk}", Parameter.POSITIONAL_OR_KEYWORD, default=def_val, annotation=ann))

    # 2. Add freeze_<par> parameters
    for p in user_pars:
        pk = p["par_key"]
        comp = p["component"]
        def_fr = bool(p["frozen"])
        default_freeze_map[pk] = def_fr
        sig_params.append(Parameter(f"freeze_{pk}", Parameter.POSITIONAL_OR_KEYWORD, default=def_fr, annotation=bool))
        doc_param_lines.append(
            f"    :param bool freeze_{pk}: Whether the {comp} '{pk}' parameter should be frozen during the fit. "
            f"Default is {def_fr}."
        )

    # 3. Add link_<par> parameters
    for p in user_pars:
        pk = p["par_key"]
        comp = p["component"]
        def_lk = bool(p["linked"])
        default_link_map[pk] = def_lk
        sig_params.append(Parameter(f"link_{pk}", Parameter.POSITIONAL_OR_KEYWORD, default=def_lk, annotation=bool))
        doc_param_lines.append(
            f"    :param bool link_{pk}: Whether the {comp} '{pk}' parameter should be linked across spectra "
            f"during the fit. Default is {def_lk}."
        )

    # 4. Add standard XGA fitting configuration arguments
    standard_params = [
        Parameter("nh_to_zero", Parameter.POSITIONAL_OR_KEYWORD, default=None, annotation=list[str | int] | None),
        Parameter(
            "lum_en",
            Parameter.POSITIONAL_OR_KEYWORD,
            default=Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
            annotation=Quantity,
        ),
        Parameter("lo_en", Parameter.POSITIONAL_OR_KEYWORD, default=Quantity(0.3, "keV"), annotation=Quantity),
        Parameter("hi_en", Parameter.POSITIONAL_OR_KEYWORD, default=Quantity(7.9, "keV"), annotation=Quantity),
        Parameter("par_fit_stat", Parameter.POSITIONAL_OR_KEYWORD, default=2.706, annotation=float),
        Parameter("lum_conf", Parameter.POSITIONAL_OR_KEYWORD, default=68.0, annotation=float),
        Parameter("abund_table", Parameter.POSITIONAL_OR_KEYWORD, default="angr", annotation=str),
        Parameter("fit_method", Parameter.POSITIONAL_OR_KEYWORD, default="leven", annotation=str),
        Parameter("group_spec", Parameter.POSITIONAL_OR_KEYWORD, default=True, annotation=bool),
        Parameter("min_counts", Parameter.POSITIONAL_OR_KEYWORD, default=5, annotation=int),
        Parameter("min_sn", Parameter.POSITIONAL_OR_KEYWORD, default=None, annotation=int | float | None),
        Parameter("over_sample", Parameter.POSITIONAL_OR_KEYWORD, default=None, annotation=float | None),
        Parameter("one_rmf", Parameter.POSITIONAL_OR_KEYWORD, default=True, annotation=bool),
        Parameter("num_cores", Parameter.POSITIONAL_OR_KEYWORD, default=NUM_CORES, annotation=int),
        Parameter("spectrum_checking", Parameter.POSITIONAL_OR_KEYWORD, default=False, annotation=bool),
        Parameter("check_pars", Parameter.POSITIONAL_OR_KEYWORD, default=None, annotation=str | list[str] | None),
        Parameter(
            "check_lo_lims",
            Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=float | int | Quantity | list[float | int | Quantity] | None,
        ),
        Parameter(
            "check_hi_lims",
            Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=float | int | Quantity | list[float | int | Quantity] | None,
        ),
        Parameter(
            "check_err_lims",
            Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=float | int | Quantity | list[float | int | Quantity] | None,
        ),
        Parameter("timeout", Parameter.POSITIONAL_OR_KEYWORD, default=Quantity(1, "hr"), annotation=Quantity),
        Parameter("stacked_spectra", Parameter.POSITIONAL_OR_KEYWORD, default=False, annotation=bool),
        Parameter("telescope", Parameter.POSITIONAL_OR_KEYWORD, default=None, annotation=str | list[str] | None),
        Parameter("force_gen", Parameter.POSITIONAL_OR_KEYWORD, default=False, annotation=bool),
    ]
    sig_params.extend(standard_params)

    custom_sig = Signature(parameters=sig_params, return_annotation=BaseSource | BaseSample | list[BaseSource])

    def _values_equal(
        v1: float | int | Quantity | list | np.ndarray | None,
        v2: float | int | Quantity | list | np.ndarray | None,
    ) -> bool:
        """
        Safely compare a user-supplied parameter value against its default (including ``None``, Astropy ``Quantity``
        instances, lists, and numpy arrays) without triggering array-truth-value ambiguity errors.
        """
        if v1 is None and v2 is None:
            return True
        if v1 is None or v2 is None:
            return False
        if type(v1) is not type(v2):
            return False
        if isinstance(v1, Quantity) and isinstance(v2, Quantity):
            if v1.unit != v2.unit or v1.shape != v2.shape:
                return False
            return bool(np.all(v1.value == v2.value))
        # Per-source lists/arrays are never equal to scalar default values
        if isinstance(v1, (list, np.ndarray)):
            return False
        return bool(v1 == v2)

    def _custom_fit_wrapper(*args: object, **kwargs: object) -> BaseSource | BaseSample | list[BaseSource]:
        # Lazy import avoids a circular import between xga.xspec.fit.models and xga.xspec.fit.general
        from .general import general_model_fit

        # Bind the caller's positional and keyword arguments against our dynamically synthesized signature
        bound = custom_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        b_args = bound.arguments

        # Collect only those start_<par>, freeze_<par>, and link_<par> arguments that differ from the model's
        #  defaults so they can be forwarded as override dictionaries to general_model_fit
        start_overrides: dict[str, Any] = {}
        freeze_overrides: dict[str, bool] = {}
        link_overrides: dict[str, bool] = {}

        for p in user_pars:
            pk = p["par_key"]
            cur_start = b_args[f"start_{pk}"]
            if not _values_equal(cur_start, default_start_map[pk]):
                start_overrides[pk] = cur_start

            cur_freeze = bool(b_args[f"freeze_{pk}"])
            if cur_freeze != default_freeze_map[pk]:
                freeze_overrides[pk] = cur_freeze

            cur_link = bool(b_args[f"link_{pk}"])
            if cur_link != default_link_map[pk]:
                link_overrides[pk] = cur_link

        # Delegate execution of the spectral fit to general_model_fit
        res: BaseSource | BaseSample | list[BaseSource] = general_model_fit(
            b_args["sources"],
            b_args["outer_radius"],
            model,
            inner_radius=b_args["inner_radius"],
            norm_scale=norm_scale,
            start_pars=start_overrides if start_overrides else None,
            freeze_pars=freeze_overrides if freeze_overrides else None,
            link_pars=link_overrides if link_overrides else None,
            nh_to_zero=b_args["nh_to_zero"],
            lum_en=b_args["lum_en"],
            lo_en=b_args["lo_en"],
            hi_en=b_args["hi_en"],
            par_fit_stat=b_args["par_fit_stat"],
            lum_conf=b_args["lum_conf"],
            abund_table=b_args["abund_table"],
            fit_method=b_args["fit_method"],
            group_spec=b_args["group_spec"],
            min_counts=b_args["min_counts"],
            min_sn=b_args["min_sn"],
            over_sample=b_args["over_sample"],
            one_rmf=b_args["one_rmf"],
            num_cores=b_args["num_cores"],
            spectrum_checking=b_args["spectrum_checking"],
            check_pars=b_args["check_pars"],
            check_lo_lims=b_args["check_lo_lims"],
            check_hi_lims=b_args["check_hi_lims"],
            check_err_lims=b_args["check_err_lims"],
            timeout=b_args["timeout"],
            stacked_spectra=b_args["stacked_spectra"],
            telescope=b_args["telescope"],
            force_gen=b_args["force_gen"],
        )
        return res

    # Attach the dynamically constructed inspect.Signature and function metadata to the wrapper callable.
    #  Standard Python functions do not declare __signature__ or _xga_user_pars on their static Callable type,
    #  so we silence mypy's attr-defined check on those two dynamic attribute assignments.
    _custom_fit_wrapper.__signature__ = custom_sig  # type: ignore[attr-defined]
    safe_func_suffix = re.sub(r"[^a-zA-Z0-9]+", "_", clean_model).strip("_")
    _custom_fit_wrapper.__name__ = f"fit_{safe_func_suffix}"
    _custom_fit_wrapper.__qualname__ = f"fit_{safe_func_suffix}"
    _custom_fit_wrapper.__annotations__ = {
        param.name: param.annotation for param in sig_params if param.annotation is not Parameter.empty
    }
    _custom_fit_wrapper.__annotations__["return"] = BaseSource | BaseSample | list[BaseSource]

    # Register the flat parameter relevance map in FIT_FUNC_ARGS so fit_conf_from_function and
    #  Spectrum.fitted_model_changed_From_default treat custom fit functions identically to built-in fit functions
    rel_args: dict[str, bool] = {"inner_radius": False}
    for p in user_pars:
        rel_args[f"start_{p['par_key']}"] = True
    for p in user_pars:
        rel_args[f"freeze_{p['par_key']}"] = True
    for p in user_pars:
        rel_args[f"link_{p['par_key']}"] = True
    rel_args.update(
        {
            "nh_to_zero": True,
            "lum_en": True,
            "lo_en": True,
            "hi_en": True,
            "par_fit_stat": True,
            "lum_conf": True,
            "abund_table": True,
            "fit_method": True,
            "group_spec": False,
            "min_counts": False,
            "min_sn": False,
            "over_sample": False,
            "one_rmf": False,
            "num_cores": False,
            "spectrum_checking": True,
            "check_pars": True,
            "check_lo_lims": True,
            "check_hi_lims": True,
            "check_err_lims": True,
            "timeout": False,
            "stacked_spectra": False,
            "telescope": False,
            "force_gen": False,
        }
    )
    FIT_FUNC_ARGS[_custom_fit_wrapper.__name__] = rel_args
    # Store the parsed parameter descriptors on the wrapper so fit_conf_from_function can normalise units and
    #  dict-based overrides (e.g. start_pars={...}) when given this custom fit function
    _custom_fit_wrapper._xga_user_pars = user_pars  # type: ignore[attr-defined]

    nh_zero_doc = (
        f"Default is None, which automatically zeroes parameter index(es) {auto_nh_to_zero}."
        if auto_nh_to_zero
        else "Default is None (no absorption nH parameters identified to zero)."
    )
    dynamic_params_doc = "\n".join(doc_param_lines)
    _custom_fit_wrapper.__doc__ = (
        f"Dynamically generated XGA convenience function for fitting the XSPEC model ``{clean_model}`` "
        f"(``norm_scale={norm_scale}``) to source spectra.\n\n"
        f"    If there are no existing spectra with the passed settings, then they will be generated automatically.\n"
        f"    This function delegates execution to :func:`~xga.xspec.fit.general_model_fit`.\n\n"
        f"    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.\n"
        f"    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the\n"
        f"        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,\n"
        f"        or Quantity(1000, 'kpc')).\n"
        f"    :param str/Quantity inner_radius: The name or value of the inner radius of the region that the\n"
        f"        desired spectrum covers. By default this is zero arcseconds, resulting in a circular spectrum.\n"
        f"{dynamic_params_doc}\n"
        f"    :param list nh_to_zero: List of parameter names or 1-based indices to zero when computing unabsorbed\n"
        f"        luminosities. {nh_zero_doc}\n"
        f"    :param Quantity lum_en: Energy bands in which to measure luminosity.\n"
        f"    :param Quantity lo_en: The lower energy limit for the data to be fitted.\n"
        f"    :param Quantity hi_en: The upper energy limit for the data to be fitted.\n"
        f"    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 2.706.\n"
        f"    :param float lum_conf: The confidence level for XSPEC luminosity measurements.\n"
        f"    :param str abund_table: The abundance table to use for the fit.\n"
        f"    :param str fit_method: The XSPEC fit method to use.\n"
        f"    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.\n"
        f"    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per "
        f"channel.\n"
        f"    :param int/float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each "
        f"channel.\n"
        f"    :param float over_sample: The minimum energy resolution for each group, set to None to disable.\n"
        f"    :param bool one_rmf: Whether to generate only one RMF per ObsID-instrument combination.\n"
        f"    :param int num_cores: The number of cores to use (if running locally).\n"
        f"    :param bool spectrum_checking: Whether the individual spectrum checking step should be run.\n"
        f"    :param str/list[str] check_pars: Parameter name(s) to check if spectrum_checking is enabled.\n"
        f"    :param float/int/Quantity/list check_lo_lims: Lower acceptable limit(s) for check_pars.\n"
        f"    :param float/int/Quantity/list check_hi_lims: Upper acceptable limit(s) for check_pars.\n"
        f"    :param float/int/Quantity/list check_err_lims: Upper acceptable uncertainty limit(s) for check_pars.\n"
        f"    :param Quantity timeout: The amount of time each individual fit is allowed to run for.\n"
        f"    :param bool stacked_spectra: Whether stacked spectra should be used where supported.\n"
        f"    :param str/List[str] telescope: Telescope(s) to perform the XSPEC operations for.\n"
        f"    :param bool force_gen: Whether to force regeneration of spectra even if they already exist.\n"
        f"    :return: The source object(s) passed in.\n"
        f"    :rtype: Union[BaseSource, BaseSample, List[BaseSource]]\n"
    )

    return _custom_fit_wrapper
