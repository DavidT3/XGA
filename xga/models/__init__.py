#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/12/2020, 13:21. Copyright (c) David J Turner

import inspect
from types import FunctionType

# Doing star imports just because its more convenient, and there won't ever be enough code in these that
#  it becomes a big inefficiency
from .density import *
from .misc import *
from .sb import *
from .temperature import *

# This dictionary is meant to provide pretty versions of model/function names to go in plots
MODEL_PUBLICATION_NAMES = {'power_law': 'Power Law'}


def convert_to_odr_compatible(model_func: FunctionType):
    # This is not at all perfect, but its a bodge that will do for now
    common_conversions = {'numpy': 'np'}

    # This reads out the function signature - which should be structured as x_values, par1, par2, par3 etc.
    mod_sig = inspect.signature(model_func)
    str_mod_sig = str(mod_sig)
    for conv in common_conversions:
        str_mod_sig = str_mod_sig.replace(conv, common_conversions[conv])
    new_mod_sig = '(β, x)'
    mod_sig_pars = mod_sig.parameters
    par_names = list(mod_sig_pars.keys())[1:]

    mod_code = str(inspect.getsource(model_func))
    new_mod_code = mod_code.replace(str_mod_sig, new_mod_sig)
    known_def = 'def {mn}'.format(mn=model_func.__name__) + new_mod_sig + ':'
    new_mod_code = new_mod_code.replace(known_def, '')

    for par_ind, par_name in enumerate(par_names):
        new_mod_code = new_mod_code.replace(par_name, 'β[{}]'.format(par_ind))

    new_mod_code = known_def + new_mod_code

    new_model_func_code = compile(new_mod_code, '<string>', 'exec')
    new_model_func = FunctionType(new_model_func_code.co_consts[0], globals(), model_func.__name__)

    return new_model_func





