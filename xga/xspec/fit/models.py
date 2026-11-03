#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/03/2026, 20:25. Copyright (c) The Contributors
"""Validation, tokenisation, and parameter inspection utilities for XSPEC model expressions."""

import re
from typing import Any
from warnings import warn

from astropy.units import Quantity, Unit
from tabulate import tabulate

from ... import utils
from ...exceptions import XSPECFitError, XSPECNotFoundError

# Valid characters in an XSPEC model expression (model names, '+', '*', and parentheses)
_VALID_EXPR_RE = re.compile(r"^[a-zA-Z0-9_.*+()]+$")
# Matches either an operator/parenthesis or a model component identifier
_TOKEN_RE = re.compile(r"\(|\)|\*|\+|[a-zA-Z0-9_.]+")


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
    :param bool norm_scale: If True, prepends 'constant*' to the model expression (if not already present) so
        that simultaneous fits across multiple spectra link additive normalisations and scale via a multiplicative
        constant. Default is True.
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

    if not _VALID_EXPR_RE.match(clean_model):
        raise ValueError(
            f"Model expression '{model}' contains invalid characters. Only alphanumeric model names, "
            f"'+', '*', '(', and ')' are permitted."
        )

    # Tokenise and verify balanced parentheses and valid operator placement before checking model names
    tokens = _TOKEN_RE.findall(clean_model)
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
        tokens = _TOKEN_RE.findall(clean_model)

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
            base_name = par_meta["name"]
            par_key = base_name
            dup_count = 1
            while par_key in seen_par_keys:
                par_key = f"{base_name}_{dup_count}"
                dup_count += 1
            seen_par_keys.add(par_key)

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

            is_z = _is_redshift_par(par_meta) and not is_redshifted_absorber

            if is_leading_norm_scale:
                default_frozen = False
                default_linked = False
            elif is_galactic_nh:
                default_frozen = True
                default_linked = True
            else:
                default_frozen = bool(par_meta["frozen"])
                default_linked = True

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

        if p["is_galactic_nh"]:
            default_display = f"<source.nH> ({p['default']} {raw_unit})"
            start_val: Quantity | float | None = None
        elif p["is_redshift"]:
            default_display = f"<source.redshift> ({p['default']})"
            start_val = None
        else:
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

        if p["hard_min"] is not None and p["hard_max"] is not None:
            limits_str = f"[{p['hard_min']}, {p['hard_max']}]"
        else:
            limits_str = "Fixed / Switch"

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
