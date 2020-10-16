#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 16/10/2020, 15:29. Copyright (c) David J Turner

from typing import Union

import numpy as np


# Here we define models that can be used to describe surface brightness profiles of Galaxy Clusters


def beta_profile(r: np.ndarray, beta: float, r_core: float, norm: float) -> np.ndarray:
    """
    The famous (among a certain circle) beta profile. This is a projected model so can be used to fit/describe
    a surface brightness profile of a cluster. Obviously assumes a radial symmetry as it only depends on radius
    :param np.ndarray r:
    :param Union[float, int] beta:
    :param Union[float, int] r_core: The core radius
    :param Union[float, int] norm:
    :return:
    :rtype:
    """
    return norm*(1 + (r**2/r_core**2))**((-3*beta) + 0.5)


def double_beta_profile(r: np.ndarray, beta_one: float, r_core_one: float, beta_two: float, r_core_two: float,
                        weight: float, norm: float):
    return norm*(beta_profile(r, beta_one, r_core_one, 1) + (weight * beta_profile(r, beta_two, r_core_two, 1)))


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
# For curve_fit type fitters where a initial value is really important
SB_MODELS_STARTS = {"beta_profile": [1, 50, 1], "double_beta_profile": [1, 400, 1, 100, 0.5, 0.5],
                    "simple_vikhlinin": [1, 100, 1, 300, 3, 0.1, 1]}

SB_MODELS_PRIORS = {"beta_profile": [[0, 3], [0, 1000], [0, 100]],
                    "double_beta_profile": [[0, 1000], [0, 2000], [0, 1000], [0, 2000], [-100, 100], [0, 100]]}


