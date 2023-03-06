#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

import inspect
from types import FunctionType

# Doing star imports just because its more convenient, and there won't ever be enough code in these that
#  it becomes a big inefficiency
from .density import *
from .misc import *
from .sb import *
from .temperature import *

# This dictionary is meant to provide pretty versions of model/function names to go in plots
# This method of merging dictionaries only works in Python 3.5+, but that should be fine
MODEL_PUBLICATION_NAMES = {**DENS_MODELS_PUB_NAMES, **MISC_MODELS_PUB_NAMES, **SB_MODELS_PUB_NAMES,
                           **TEMP_MODELS_PUB_NAMES}
MODEL_PUBLICATION_PAR_NAMES = {**DENS_MODELS_PAR_NAMES, **MISC_MODELS_PAR_NAMES, **SB_MODELS_PAR_NAMES,
                               **TEMP_MODELS_PAR_NAMES}
# These dictionaries tell the profile fitting function what models, start pars, and priors are allowed
PROF_TYPE_MODELS = {"brightness": SB_MODELS, "gas_density": DENS_MODELS, "gas_temperature": TEMP_MODELS}


def convert_to_odr_compatible(model_func: FunctionType, new_par_name: str = 'β', new_data_name: str = 'x_values') \
        -> FunctionType:
    """
    This is a bit of a weird one; its meant to convert model functions from the standard XGA setup
    (i.e. pass x values, then parameters as individual variables), into the form expected by Scipy's ODR.
    I'd recommend running a check to compare results from the original and converted functions where-ever
    this function is called - I don't completely trust it.

    :param FunctionType model_func: The original model function to be converted.
    :param str new_par_name: The name we want to use for the new list/array of fit parameters.
    :param str new_data_name: The new name we want to use for the x_data.
    :return: A successfully converted model function (hopefully) which can be used with ODR.
    :rtype: FunctionType
    """
    # This is not at all perfect, but its a bodge that will do for now. If type hints are included in
    #  the signature (as they should be in all XGA models), then np.ndarray will be numpy.ndarray in the
    #  signature I extract. This dictionary will be used to swap that out, along with any similar problems I encounter
    common_conversions = {'numpy': 'np'}

    # This reads out the function signature - which should be structured as x_values, par1, par2, par3 etc.
    mod_sig = inspect.signature(model_func)
    # Convert that signature into a string
    str_mod_sig = str(mod_sig)

    # Go through the conversion dictionary and 'correct' the signature
    for conv in common_conversions:
        str_mod_sig = str_mod_sig.replace(conv, common_conversions[conv])

    # For ODR I've decided that β is the name of the new fit parameter array, and x_values the name of the
    #  x data. This will replace the current signature of the function.
    new_mod_sig = '({np}, {nd})'.format(np=new_par_name, nd=new_data_name)
    # I find the current names of the parameters in the signature, excluding the x value name in the original function
    #  and reading that into a separate variable
    mod_sig_pars = list(mod_sig.parameters.keys())
    par_names = mod_sig_pars[1:]
    # Store the name of the x data here
    data_name = mod_sig_pars[0]

    # This gets the source code of the function as a string
    mod_code = inspect.getsource(model_func)
    # I swap in the new signature
    new_mod_code = mod_code.replace(str_mod_sig, new_mod_sig)

    # And now I know the exact form of the whole def line I can define that as a variable and then temporarily
    #  remove it from the source code
    known_def = 'def {mn}'.format(mn=model_func.__name__) + new_mod_sig + ':'
    new_mod_code = new_mod_code.replace(known_def, '')

    # Then I swing through all the original parameter names and replace them with accessing elements of our
    #  new beta parameter list/array.
    for par_ind, par_name in enumerate(par_names):
        new_mod_code = new_mod_code.replace(par_name, '{np}[{i}]'.format(np=new_par_name, i=par_ind))

    # Then I do the same thing for the new x data variable name
    new_mod_code = new_mod_code.replace(data_name, new_data_name)

    # Adds the def SIGNATURE line back in
    new_mod_code = known_def + new_mod_code

    # This compiles the code and creates a new function
    new_model_func_code = compile(new_mod_code, '<string>', 'exec')
    new_model_func = FunctionType(new_model_func_code.co_consts[0], globals(), model_func.__name__)

    return new_model_func





