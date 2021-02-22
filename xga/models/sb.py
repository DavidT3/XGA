#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 22/02/2021, 14:10. Copyright (c) David J Turner

from typing import Union

import numpy as np


# Here we define models that can be used to describe surface brightness profiles of Galaxy Clusters
def beta_profile(r_values: Union[np.ndarray, float], beta: float, r_core: float, norm: float) \
        -> Union[np.ndarray, float]:
    """
    The famous (among a certain circle) beta profile. This is a projected model so can be used to fit/describe
    a surface brightness profile of a cluster. Obviously assumes a radial symmetry as it only depends on radius.

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float/int beta: The beta slope parameter of the model.
    :param float/int r_core: The core radius.
    :param float/int norm: The normalisation of the model.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    return norm * np.power((1 + (np.power(r_values / r_core, 2))), ((-3 * beta) + 0.5))


def double_beta_profile(r_values: Union[np.ndarray, float], norm_one: float, beta_one: float, r_core_one: float,
                        norm_two: float, beta_two: float, r_core_two: float, ) -> Union[np.ndarray, float]:
    """
    A summation of two single beta models. Often thought to deal better with peaky cluster cores that you might
    get from a cool-core cluster.

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float/int norm_one: The normalisation of the first beta profile.
    :param float/int beta_one: The beta slope parameter of the first component beta profile.
    :param float/int r_core_one: The core radius of the first component beta profile.
    :param float/int norm_two: The normalisation of the second beta profile.
    :param float/int beta_two:  The beta slope parameter of the second component beta profile.
    :param float/int r_core_two: The core radius of the second component beta profile.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    return beta_profile(r_values, beta_one, r_core_one, norm_one) + beta_profile(r_values, beta_two, r_core_two,
                                                                                 norm_two)


# So that things like fitting functions can be written generally to support different models
SB_MODELS = {"beta": beta_profile, "double_beta": double_beta_profile}

# For curve_fit type fitters where a initial value is important
SB_MODELS_STARTS = {"beta": [1, 50, 1], "double_beta": [1, 1, 400, 1, 1, 100]}

SB_MODELS_PRIORS = {"beta": [[0, 3], [0, 300], [0, 100]],
                    "double_beta": [[0, 100], [0, 3], [1, 2000], [0, 100], [0, 3], [1, 2000]]}

SB_MODELS_PUB_NAMES = {"beta": "Beta Profile", 'double_beta': 'Double Beta Profile'}

SB_MODELS_PAR_NAMES = {"beta": [r"$\beta$", r"R$_{\rm{core}}$", "Norm"],
                       "double_beta": [r"S$_{01}$", r"$\beta_{1}$", r"R$_{\rm{core},1}$", r"S$_{02}$", r"$\beta_{2}$",
                                       r"R$_{\rm{core},2}$"]}
