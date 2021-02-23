#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 19/02/2021, 08:30. Copyright (c) David J Turner

from typing import Union

import numpy as np


def king_profile(r_values: Union[np.ndarray, float], beta: float, r_core: float, norm: float) \
        -> Union[np.ndarray, float]:
    """
    The unprojected version of the beta profile, suitable for a simple fit to 3D density distributions.

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float/int beta: The beta slope parameter of the model.
    :param float/int r_core: The core radius.
    :param float/int norm: The normalisation of the model.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    return norm * np.power((1 + (np.power(r_values / r_core, 2))), (-3 * beta))


# I have set the gamma parameter to 3 in this implementation, though I may allow it as a parameter again
#  in the future
def simple_vikhlinin_dens(r_values: Union[np.ndarray, float], beta: float, r_core: float, alpha: float, r_s: float,
                          epsilon: float, norm: float) -> Union[np.ndarray, float]:
    """
    Used relatively recently in https://doi.org/10.1051/0004-6361/201833325 by Ghirardini et al., a
    simplified form of Vikhlinin's full density model, which can be found in https://doi.org/10.1086/500288.

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float beta: The beta parameter of the model.
    :param float r_core: The core radius of the model.
    :param float alpha: The alpha parameter of the model.
    :param float r_s: The radius near where a change of slope by epsilon occurs.
    :param float epsilon: The epsilon parameter of the model.
    :param float norm: The overall normalisation of the model.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    # Calculates the ratio of the r_values to the r_core parameter
    rc_rat = r_values / r_core
    # Calculates the ratio of the r_values to the r_s parameter
    rs_rat = r_values / r_s

    first_term = np.power(rc_rat, -alpha) / np.power((1 + np.power(rc_rat, 2)), ((3 * beta) - (alpha / 2)))
    second_term = 1 / np.power(1 + np.power(rs_rat, 3), epsilon / 3)
    result = norm * np.sqrt(first_term * second_term)
    return result


def full_vikhlinin_dens(r_values: Union[np.ndarray, float], beta_one: float, r_core_one: float, alpha: float,
                        r_s: float, epsilon: float, gamma: float, norm_one: float, beta_two: float,
                        r_core_two: float, norm_two: float):
    """
    The full model for cluster density profiles described in https://doi.org/10.1086/500288. This model has MANY
    free parameters which can be very hard to get constraints on, and as such many people would use the simplified
    version which is implemented as 'simple_vikhlinin_dens' in XGA.

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float beta_one: The beta parameter of the model.
    :param float r_core_one: The core radius of the model.
    :param float alpha: The alpha parameter of the model.
    :param float r_s: The radius near where a change of slope by epsilon occurs.
    :param float epsilon: The epsilon parameter of the model.
    :param float gamma: Width of slope change transition region.
    :param float norm_one: The normalisation of the model first part of the model.
    :param beta_two: The beta parameter slope of the small core part of the model.
    :param r_core_two:The core radius of the small core part of the model.
    :param norm_two: The normalisation of the additive, small core part of the model.
    """
    # Calculates the ratio of the r_values to the r_core_one parameter
    rc1_rat = r_values / r_core_one
    # Calculates the ratio of the r_values to the r_core_two parameter
    rc2_rat = r_values / r_core_two
    # Calculates the ratio of the r_values to the r_s parameter
    rs_rat = r_values / r_s

    first_term = np.power(rc1_rat, -alpha) / np.power((1 + np.power(rc1_rat, 2)), ((3 * beta_one) - (alpha / 2)))
    second_term = 1 / np.power(1 + np.power(rs_rat, gamma), epsilon / gamma)
    additive_term = 1 / np.power(1 + np.power(rc2_rat, 2), 3*beta_two)

    return np.sqrt(np.power(norm_one, 2)*first_term*second_term + np.power(norm_two, 2)*additive_term)


# So that things like fitting functions can be written generally to support different models
DENS_MODELS = {"simple_vikhlinin_dens": simple_vikhlinin_dens, 'king': king_profile,
               'vikhlinin_dens': full_vikhlinin_dens}

DENS_MODELS_STARTS = {"simple_vikhlinin_dens": [1, 100, 1, 300, 2, 1e+13],
                      "king": [1, 100, 1e+13],
                      "vikhlinin_dens": [1, 100, 1, 300, 2, 3, 1e+13, 1, 50, 1e+13]}

DENS_MODELS_PRIORS = {"simple_vikhlinin_dens": [[-3, 3], [1, 1000], [-3, 3], [1, 2000], [0, 5], [1e+12, 1e+16]],
                      "king": [[0, 3], [1, 1000], [1e+12, 1e+16]],
                      "vikhlinin_dens": [[-3, 3], [1, 1000], [-3, 3], [1, 2000], [0, 5], [-5, 5], [1e+12, 1e+16],
                                         [-3, 3], [1, 1000], [1e+12, 1e+16]]}

DENS_MODELS_PAR_NAMES = {"simple_vikhlinin_dens": [r"$\beta$", r"R$_{\rm{core}}$", r"$\alpha$", r"R$_{\rm{s}}$",
                                                   r"$\epsilon$", r"$\rho_{0}$"],
                         "king": [r"$\beta$", r"R$_{\rm{core}}$", r"$\rho_{0}$"],
                         "vikhlinin_dens": [r"$\beta_{1}$", r"R$_{\rm{core,1}}$", r"$\alpha$", r"R$_{\rm{s}}$",
                                            r"$\epsilon$", r"$\gamma$", r"$\rho_{01}$", r"$\beta_{2}$",
                                            r"R$_{\rm{core,2}}$", r"$\rho_{02}$"]}

DENS_MODELS_PUB_NAMES = {'simple_vikhlinin_dens': 'Simplified Vikhlinin', 'king': 'King Function',
                         'vikhlinin_dens': 'Full Vikhlinin'}



