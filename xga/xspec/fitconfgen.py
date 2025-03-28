#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/03/2025, 22:31. Copyright (c) The Contributors

from inspect import signature, Parameter
from types import FunctionType
from typing import Union

from astropy.units import Quantity

# This constant very importantly defines whether each argument to the XGA XSPEC fitting functions should be included
#  in the fit configuration storage key - we define it here so that this information can be accessed outside the
#  actual fit function. We will also check the arguments of each fit function against these entries to see whether
#  an argument has been added that is not accounted for here.
FIT_FUNC_ARGS = {
    'single_temp_apec': {'inner_radius': False, 'start_temp': True, 'start_met': True, 'lum_en': False,
                         'freeze_nh': True, 'freeze_met': True, 'freeze_temp': True, 'lo_en': True, 'hi_en': True,
                         'par_fit_stat': True, 'lum_conf': False, 'abund_table': True, 'fit_method': True,
                         'group_spec': False, 'min_counts': False, 'min_sn': False, 'over_sample': False,
                         'one_rmf': False, 'num_cores': False, 'spectrum_checking': False, 'timeout': False},

    'double_temp_apec': {'inner_radius': False, 'start_temp_one': True, 'start_temp_two': True,
                         'start_met_one': True, 'start_met_two': True, 'lum_en': False, 'freeze_nh': True,
                         'freeze_met_one': True, 'freeze_met_two': True, 'freeze_temp_one': True,
                         'freeze_temp_two': True, 'lo_en': True, 'hi_en': True, 'par_fit_stat': True,
                         'lum_conf': False, 'abund_table': True, 'fit_method': True,
                         'group_spec': False, 'min_counts': False, 'min_sn': False, 'over_sample': False,
                         'one_rmf': False, 'num_cores': False, 'spectrum_checking': False, 'timeout': False},

    'single_temp_mekal': {'inner_radius': False, 'start_temp': True, 'start_met': True, 'lum_en': False,
                          'freeze_nh': True, 'freeze_met': True, 'freeze_temp': True, 'lo_en': True, 'hi_en': True,
                          'par_fit_stat': True, 'lum_conf': False, 'abund_table': True, 'fit_method': True,
                          'group_spec': False, 'min_counts': False, 'min_sn': False, 'over_sample': False,
                          'one_rmf': False, 'num_cores': False, 'spectrum_checking': False, 'timeout': False},

    'multi_temp_dem_apec': {'inner_radius': False, 'start_max_temp': True, 'start_met': True, 'start_t_rat': True,
                            'start_inv_em_slope': True, 'lum_en': False, 'freeze_nh': True, 'freeze_met': True,
                            'lo_en': True, 'hi_en': True, 'par_fit_stat': True, 'lum_conf': False,
                            'abund_table': True, 'fit_method': True, 'group_spec': False, 'min_counts': False,
                            'min_sn': False, 'over_sample': False, 'one_rmf': False, 'num_cores': False,
                            'spectrum_checking': False, 'timeout': False},

    'power_law': {'inner_radius': False, 'redshifted': False, 'lum_en': False, 'start_pho_index': True,
                  'freeze_nh': True, 'lo_en': True, 'hi_en': True, 'par_fit_stat': True, 'lum_conf': False,
                  'abund_table': True, 'fit_method': True, 'group_spec': False, 'min_counts': False, 'min_sn': False,
                  'over_sample': False, 'one_rmf': False, 'num_cores': False, 'timeout': False},

    'blackbody': {'inner_radius': False, 'redshifted': False, 'lum_en': False, 'start_temp': True,
                  'freeze_nh': True, 'lo_en': True, 'hi_en': True, 'par_fit_stat': True, 'lum_conf': False,
                  'abund_table': True, 'fit_method': True, 'group_spec': False, 'min_counts': False, 'min_sn': False,
                  'over_sample': False, 'one_rmf': False, 'num_cores': False, 'timeout': False},

    'single_temp_apec_profile': {'start_temp': True, 'start_met': True, 'lum_en': False,
                                 'freeze_nh': True, 'freeze_met': True, 'lo_en': True, 'hi_en': True,
                                 'par_fit_stat': True, 'lum_conf': False, 'abund_table': True, 'fit_method': True,
                                 'group_spec': False, 'min_counts': False, 'min_sn': False, 'over_sample': False,
                                 'one_rmf': False, 'num_cores': False, 'spectrum_checking': False, 'timeout': False,
                                 'use_cross_arf': True, 'first_fit_start_pars': True, 'detmap_bin': True}
}


def fit_conf_from_function(fit_func: FunctionType, changed_pars: Union[dict, str] = None) -> str:
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
        elif changed_pars is not None and any([ch_par_key not in rel_args for ch_par_key in changed_pars]):
            not_pres = ", ".join([ch_par_key for ch_par_key in changed_pars if ch_par_key not in def_args])
            all_args = ". ".join([kn for kn in rel_args if rel_args[kn]])
            raise KeyError("Some entries in 'changed_pars' ({be}) do not correspond to a keyword argument that is "
                           "included in the fit configuration key for {f}; the keyword arguments are "
                           "{kw}.".format(be=not_pres, kw=all_args, f=fit_func.__name__))

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
        if isinstance(cur_val, (int, float, bool)):
            cur_val = mod_kn+str(cur_val)
        # We make our own string version of quantities because the version you get when applying str() to a quantity
        #  instance has a space between value and unit and we don't want that
        elif isinstance(cur_val, Quantity):
            cur_val = mod_kn+str(cur_val.value) + cur_val.unit.to_string()

        # The components of the fit configuration key are appended to a string
        fit_conf_key_parts.append(cur_val)

    # Then eventually the components are joined into a single string, separated by underscores, as we usually do in XGA
    return "_".join(fit_conf_key_parts)
