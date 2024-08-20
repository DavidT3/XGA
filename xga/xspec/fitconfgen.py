#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/08/2024, 09:36. Copyright (c) The Contributors

from astropy.units import Quantity


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
