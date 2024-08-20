#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/08/2024, 12:10. Copyright (c) The Contributors

from inspect import signature, Parameter
from types import FunctionType

from astropy.units import Quantity

# 'freeze_nh': True,

# This constant very importantly defined whether each argument to the XGA XSPEC fitting functions should be included
#  in the fit configuration storage key - we define it here so that this information can be accessed outside the
#  actual fit function. We will also check the arguments of each fit function against these entries to see whether
#  an argument has been added that is not accounted for here.
FIT_FUNC_ARGS = {
    'single_temp_apec': {'inner_radius': False, 'start_temp': True, 'start_met': True, 'lum_en': False,
                         'freeze_met': True, 'freeze_temp': True, 'lo_en': True, 'hi_en': True,
                         'par_fit_stat': True, 'lum_conf': False, 'abund_table': True, 'fit_method': True,
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

    'single_temp_apec_profile': {'inner_radius': False, 'start_temp': True, 'start_met': True, 'lum_en': False,
                                 'freeze_nh': True, 'freeze_met': True, 'lo_en': True, 'hi_en': True,
                                 'par_fit_stat': True, 'lum_conf': False, 'abund_table': True, 'fit_method': True,
                                 'group_spec': False, 'min_counts': False, 'min_sn': False, 'over_sample': False,
                                 'one_rmf': False, 'num_cores': False, 'spectrum_checking': False, 'timeout': False}
}


def fit_conf_from_function(fit_func: FunctionType, changed_pars: dict = None):
    sig = signature(fit_func)

    def_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}
    print(def_args)

    if changed_pars is not None and not isinstance(changed_pars, dict):
        raise TypeError("'changed_pars' argument must be a dictionary of the values that were changed from default")
    elif changed_pars is not None and any([ch_par_key not in def_args for ch_par_key in changed_pars]):
        not_pres = ", ".join([ch_par_key for ch_par_key in changed_pars if ch_par_key not in def_args])
        all_args = ". ".join(list(def_args.keys()))
        raise KeyError("Some entries in 'changed_pars' ({be}) do not correspond to any keyword argument for the "
                       "passed function; the keyword arguments are {kw}.".format(be=not_pres, kw=all_args))

    # TODO Need to handle any unit conversions I think



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
