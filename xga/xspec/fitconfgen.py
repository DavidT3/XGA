#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/4/26, 9:50 PM. Copyright (c) The Contributors.
"""Functions and constants for generating deterministic XSPEC fit configuration keys."""

from collections.abc import Callable
from inspect import Parameter, signature
from types import FunctionType

import numpy as np
from astropy.units import Quantity, Unit

# This constant very importantly defines whether each argument to the XGA XSPEC fitting functions should be included
#  in the fit configuration storage key - we define it here so that this information can be accessed outside the
#  actual fit function. We will also check the arguments of each fit function against these entries to see whether
#  an argument has been added that is not accounted for here.
FIT_FUNC_ARGS = {
    "single_temp_apec": {
        "inner_radius": False,
        "start_temp": True,
        "start_met": True,
        "lum_en": True,
        "freeze_nh": True,
        "freeze_met": True,
        "freeze_temp": True,
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
        "timeout": False,
        "stacked_spectra": False,
        "telescope": False,
        "force_gen": False,
    },
    "double_temp_apec": {
        "inner_radius": False,
        "start_temp_one": True,
        "start_temp_two": True,
        "start_met_one": True,
        "start_met_two": True,
        "lum_en": True,
        "freeze_nh": True,
        "freeze_met_one": True,
        "freeze_met_two": True,
        "freeze_temp_one": True,
        "freeze_temp_two": True,
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
        "timeout": False,
        "stacked_spectra": False,
        "telescope": False,
    },
    "single_temp_mekal": {
        "inner_radius": False,
        "start_temp": True,
        "start_met": True,
        "lum_en": True,
        "freeze_nh": True,
        "freeze_met": True,
        "freeze_temp": True,
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
        "timeout": False,
        "stacked_spectra": False,
        "telescope": False,
    },
    "multi_temp_dem_apec": {
        "inner_radius": False,
        "start_max_temp": True,
        "start_met": True,
        "start_t_rat": True,
        "start_inv_em_slope": True,
        "lum_en": True,
        "freeze_nh": True,
        "freeze_met": True,
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
        "timeout": False,
        "stacked_spectra": False,
        "telescope": False,
    },
    "power_law": {
        "inner_radius": False,
        "redshifted": False,
        "lum_en": True,
        "start_pho_index": True,
        "freeze_nh": True,
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
        "timeout": False,
        "stacked_spectra": False,
        "telescope": False,
    },
    "blackbody": {
        "inner_radius": False,
        "redshifted": False,
        "lum_en": True,
        "start_temp": True,
        "freeze_nh": True,
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
        "timeout": False,
        "stacked_spectra": False,
        "telescope": False,
    },
    "single_temp_apec_profile": {
        "start_temp": True,
        "start_met": True,
        "lum_en": True,
        "freeze_nh": True,
        "freeze_met": True,
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
        "timeout": False,
        "use_cross_arf": True,
        "first_fit_start_pars": True,
        "detmap_bin": True,
        "stacked_spectra": False,
        "telescope": False,
    },
    "general_model_fit": {
        "inner_radius": False,
        "norm_scale": True,
        "start_pars": True,
        "freeze_pars": True,
        "link_pars": True,
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
    },
}


def fit_conf_from_function(fit_func: FunctionType | Callable, changed_pars: dict | str | None = None) -> str:
    """
    This function is a convenient way to assemble a fit configuration key without adding together all the function
    arguments yourself, and is used in various parts of XGA to make it easier to retrieve non-default model fits. It
    takes the default parameter values of the model fit, knowledge of which parameters are to be included in the
    fit configuration storage key, and (optionally) a dictionary of the parameters which were changed from the
    default fit (in the case where this changed parameter dictionary is not supplied, the default key will be made).

    If a string is passed for 'changed_pars', we will assume that it is already a fully formed fit configuration
    key, and it will be passed back.

    :param FunctionType fit_func: The XGA XSPEC function that was run, and for which a fit configuration storage
        key is to be generated.
    :param dict/str changed_pars: A dictionary containing parameters that were altered from the default values when the
        fit function was called, and the values that they were altered too. This is to make it easier to assemble a
        fit configuration key for a non-default model run. If a string value is passed, we will assume that it is
        already the full fit configuration key, and it will be returned.
    :return: The full fit configuration storage key.
    :rtype: str
    """
    # This is a little convenience thing for functions in XGA were either a dictionary or a full key can
    #  be passed by the user - if a string is passed we're just going to assume it is the already assembled
    #  fit configuration key and pass it right back
    if isinstance(changed_pars, str):
        fit_conf = changed_pars
    else:
        # This reads the signature (i.e. the first line of the function definition with all the arguments) - we shall
        #  need it in order to know what the default values are
        sig = signature(fit_func)
        # This gets the dictionary that describes whether an argument is relevant to the fit configuration key for
        #  the function that has been passed
        rel_args = FIT_FUNC_ARGS[fit_func.__name__]

        # This snippet uses the read-in signature of the function to create a dictionary of keyword arguments and
        #  default values - we shall need this in order to make it easier to construct the key, as the user will only
        #  need to supply values for the parameters that they changed from default.
        def_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

        if changed_pars is not None and not isinstance(changed_pars, dict):
            raise TypeError("'changed_pars' argument must be a dictionary of the values that were changed from default")

        if changed_pars is not None and hasattr(fit_func, "_xga_user_pars"):
            user_pars = fit_func._xga_user_pars
            norm_changed: dict = {}
            for ch_k, ch_v in changed_pars.items():
                if ch_k in ("start_pars", "freeze_pars", "link_pars"):
                    if ch_v is None:
                        continue
                    if not isinstance(ch_v, dict):
                        raise TypeError(f"'{ch_k}' in 'changed_pars' must be a dictionary of parameter overrides.")
                    prefix = ch_k.split("_")[0]
                    for pk, pv in ch_v.items():
                        norm_changed[f"{prefix}_{pk}"] = pv
                else:
                    norm_changed[ch_k] = ch_v

            for p in user_pars:
                pk = p["par_key"]
                p_unit = p["unit"]
                start_k = f"start_{pk}"
                if start_k in norm_changed and norm_changed[start_k] is not None:
                    raw_v = norm_changed[start_k]
                    if isinstance(raw_v, Quantity):
                        conv_v = (
                            float(raw_v.to(p_unit).value) if p_unit != Unit("") else float(raw_v.to(Unit("")).value)
                        )
                    else:
                        conv_v = float(raw_v)
                    norm_changed[start_k] = Quantity(conv_v, p_unit) if p_unit != Unit("") else conv_v
                freeze_k = f"freeze_{pk}"
                if freeze_k in norm_changed:
                    norm_changed[freeze_k] = bool(norm_changed[freeze_k])
                link_k = f"link_{pk}"
                if link_k in norm_changed:
                    norm_changed[link_k] = bool(norm_changed[link_k])
            changed_pars = norm_changed

        if changed_pars is not None and any([not rel_args.get(ch_par_key, False) for ch_par_key in changed_pars]):
            not_pres = ", ".join([ch_par_key for ch_par_key in changed_pars if not rel_args.get(ch_par_key, False)])
            all_args = ". ".join([kn for kn in rel_args if rel_args[kn]])
            raise KeyError(
                f"Some entries in 'changed_pars' ({not_pres}) do not correspond to a keyword argument that is "
                f"included in the fit configuration key for {fit_func.__name__}; the keyword arguments are "
                f"{all_args}."
            )

        # Here we set up the dictionary that will make the default key - if the user passed information on parameters
        #  they changed then we're going to replace them in this dictionary, but if they didn't pass anything then
        #  this will stay as it is
        in_fit_conf = {kn: def_args[kn] for kn in rel_args if rel_args[kn]}

        if changed_pars is not None:
            for kn in changed_pars:
                # TODO Need to handle any unit conversions I think
                in_fit_conf[kn] = changed_pars[kn]

        # Use the fit_conf function to generate the required key
        fit_conf = _gen_fit_conf(in_fit_conf)

    return fit_conf


def _format_fit_conf_value(val: int | float | bool | str | Quantity | np.ndarray | dict | list | tuple | None) -> str:
    """
    Convert an individual value into a deterministic, compact string for a fit configuration key.

    This is used for formatting parameter dictionaries, lists, or None values passed to general_model_fit. Any
    underscores in dictionary keys or string values are replaced with '~' so that downstream methods which split
    fit configuration keys on '_' do not fracture a single parameter entry, while still distinguishing disambiguated
    parameter names (e.g. 'kT_1' -> 'kT~1') from native parameter names (e.g. 'kT1').

    :param int/float/bool/str/Quantity/np.ndarray/dict/list/tuple val: The value to format as a string.
    :return: The formatted string representation of the value.
    :rtype: str
    """
    if isinstance(val, Quantity):
        if val.isscalar:
            return f"{val.value}{val.unit.to_string()}"
        else:
            vals_str = ";".join(str(v) for v in np.asarray(val.value).flatten())
            return f"[{vals_str}]{val.unit.to_string()}"
    elif isinstance(val, np.ndarray):
        return "[" + ";".join(_format_fit_conf_value(elem) for elem in val.tolist()) + "]"
    elif isinstance(val, dict):
        return (
            "{"
            + ";".join(
                f"{str(k).replace('_', '~')}:{_format_fit_conf_value(v)}"
                for k, v in sorted(val.items(), key=lambda x: str(x[0]))
            )
            + "}"
        )
    elif isinstance(val, (list, tuple)):
        return "[" + ";".join(_format_fit_conf_value(elem) for elem in val) + "]"
    elif isinstance(val, str):
        return val.replace("_", "~")
    else:
        return str(val)


def _gen_fit_conf(key_comps: dict) -> str:
    """
    A very simple internal function that is called by XGA XSPEC fit functions in order to construct
    a 'fit configuration' key, which allows us to differentiate between fits of the same model run with different
    configurations (e.g. different start parameters, settings, abundance tables). This particular function should
    only be called from XGA XSPEC fit functions, another function will be created to make the construction of fit
    configuration keys easier for the user (so that they can just pass a dictionary of the parameters they changed
    from default).

    :param dict key_comps: The arguments that should be included in the fit configuration key.
    :return: The generated fit configuration key.
    :rtype: str
    """
    # Firstly I sort the arguments to make sure they're in a predictable order
    key_names = list(key_comps.keys())
    sorted(key_names)

    # Then we simply cycle through the arguments that will make up the contents of the key
    fit_conf_key_parts = []
    for kn in key_names:
        cur_val = key_comps[kn]
        # Some argument names will have underscores in, which could mess up our file names, so we replace them
        mod_kn = kn.replace("_", "")

        # Depending on the data type of the argument in question we will treat it differently
        if kn == "lum_en":
            bands = cur_val.to_value("keV")
            cur_val = mod_kn + ";".join(":".join(str(float(en)) for en in band) for band in bands) + "keV"
        elif kn == "lum_conf":
            cur_val = mod_kn + str(float(cur_val))
        elif isinstance(cur_val, (int, float, bool)):
            cur_val = mod_kn + str(cur_val)
        # We make our own string version of quantities because the version you get when applying str() to a quantity
        #  instance has a space between value and unit and we don't want that
        elif isinstance(cur_val, Quantity):
            cur_val = mod_kn + str(cur_val.value) + cur_val.unit.to_string()
        # General model fits accept optional dictionaries and lists (or None when using model.dat defaults) for
        #  parameter overrides and spectrum checking limits, so we serialise them deterministically here
        elif cur_val is None or isinstance(cur_val, (dict, list, tuple, np.ndarray)):
            cur_val = mod_kn + _format_fit_conf_value(cur_val)

        # The components of the fit configuration key are appended to a string
        fit_conf_key_parts.append(cur_val)

    # Then eventually the components are joined into a single string, separated by underscores, as we usually do in XGA
    return "_".join(fit_conf_key_parts)
