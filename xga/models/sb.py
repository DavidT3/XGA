#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 02/10/2020, 15:31. Copyright (c) David J Turner

from typing import Union

import numpy as np


# Here we define models that can be used to describe surface brightness profiles of Galaxy Clusters


def beta_profile(r: np.ndarray, beta: Union[float, int], r_core: Union[float, int]) -> np.ndarray:
    """
    The famous (among a certain circle) beta profile. This is a projected model so can be used to fit/describe
    a surface brightness profile of a cluster. Obviously assumes a radial symmetry as it only depends on radius
    :param np.ndarray r:
    :param Union[float, int] beta:
    :param Union[float, int] r_core: The core radius
    :return:
    :rtype:
    """
    return (1 + (r**2/r_core**2))**((-3*beta) + 0.5)


def double_beta_profile(r: np.ndarray, beta_one: Union[float, int], r_core_one: Union[float, int],
                        beta_two: Union[float, int], r_core_two: Union[float, int], weight: Union[float, int]):

    return beta_profile(r, beta_one, r_core_one) + (weight * beta_profile(r, beta_two, r_core_two))


def simple_vikhlinin(r, beta, r_core, alpha, r_s, epsilon, gamma, norm):
    """
    Used relatively recently in https://doi.org/10.1051/0004-6361/201833325 by Ghirardini et al., a
    simplified form of Vikhlinin's full model, which can be found in https://doi.org/10.1086/500288.
    """
    return norm * ((r/r_core)**(-alpha)) * ((1 + (r/r_core)**2)**((-3*beta) + (alpha/2))) * \
        ((1 + (r/r_s)**gamma)**(-epsilon/gamma))


# So that things like fitting functions can be written generally to support different models
SB_MODELS = {"beta_profile": beta_profile, "double_beta_profile": double_beta_profile,
             "simple_vikhlinin": simple_vikhlinin}
SB_MODELS_STARTS = {"beta_profile": [1, 100], "double_beta_profile": [1, 100, 1, 300, 0.5],
                    "simple_vikhlinin": [1, 100, 1, 300, 3, 0.1, 1]}


