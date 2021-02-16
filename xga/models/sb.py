#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 12/02/2021, 10:49. Copyright (c) David J Turner

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


def double_beta_profile(r_values: Union[np.ndarray, float], beta_one: float, r_core_one: float, beta_two: float,
                        r_core_two: float, weight: float, norm: float) -> Union[np.ndarray, float]:
    """
    A summation of two single beta models. Often thought to deal better with peaky cluster cores that you might
    get from a cool-core cluster.

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float/int beta_one: The beta slope parameter of the first component beta profile.
    :param float/int r_core_one: The core radius of the first component beta profile.
    :param float/int beta_two:  The beta slope parameter of the second component beta profile.
    :param float/int r_core_two: The core radius of the second component beta profile.
    :param float/int weight: The weight of the second profile compared to the first.
    :param float/int norm: The normalisation of the whole model.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    return norm*(beta_profile(r_values, beta_one, r_core_one, 1) + (weight * beta_profile(r_values, beta_two,
                                                                                          r_core_two, 1)))


# So that things like fitting functions can be written generally to support different models
SB_MODELS = {"beta_profile": beta_profile, "double_beta_profile": double_beta_profile}

# For curve_fit type fitters where a initial value is important
SB_MODELS_STARTS = {"beta_profile": [1, 50, 1], "double_beta_profile": [1, 400, 1, 100, 0.5, 0.5]}

SB_MODELS_PRIORS = {"beta_profile": [[0, 3], [0, 300], [0, 10]],
                    "double_beta_profile": [[0, 1000], [0, 2000], [0, 1000], [0, 2000], [-100, 100], [0, 100]]}

SB_MODELS_PUB_NAMES = {"beta_profile": "Beta Profile", 'double_beta_profile': 'Double Beta Profile'}

SB_MODELS_PAR_NAMES = {"beta_profile": [r"$\beta$", r"R$_{\rm{core}}$", "Norm"],
                       "double_beta_profile": [r"$\beta_{1}$", r"R$_{\rm{core},1}$", r"$\beta_{2}$",
                                               r"R$_{\rm{core},2}$", "Weight", "Norm"]}
