#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/02/2021, 16:05. Copyright (c) David J Turner

from typing import Union

import numpy as np


def central_region(r_values: Union[np.ndarray, float], r_cool: float, a_cool: float, t_min: float, t_zero: float) \
        -> Union[np.ndarray, float]:
    """
    A model that should describe the decline in the 3D temperature profile in the central region of clusters, as
    taken from the Vikhlinin 2006 paper (https://doi.org/10.1086/500288), though there it is cited as being from
    https://doi.org/10.1046/j.1365-8711.2001.05079.x

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float r_cool: Parameter describing the radius of the cooling region (I THINK - NOT CERTAIN YET).
    :param float a_cool: Power law parameter for the cooling region (I THINK - NOT CERTAIN YET).
    :param float t_min: A minimum temperature parameter for the model (I THINK - NOT CERTAIN YET).
    :param float t_zero: A normalising temperature parameter for the model (I THINK - NOT CERTAIN YET).
    :return: The temperature value of this model, corresponding to the input radius value.
    :rtype: Union[np.ndarray, float]
    """
    # Separated out just because its used twice in the expression, the ratio of the radius value(s) to the cool region
    #  radius, raised to a power
    power_rad_ratio = np.power((r_values/r_cool), a_cool)
    # The rest of the model expression
    t_cool = (power_rad_ratio + (t_min/t_zero)) / (power_rad_ratio + 1)
    return t_cool


def outer_region(r_values: Union[np.ndarray, float], r_transition: float, a_power: float, b_power: float,
                 c_power: float, t_zero: float = 1) -> Union[np.ndarray, float]:
    """
    A model that should describe the 3D temperature profile outside the central region of a cluster, essentially a
    broken power law. This was defined in the Vikhlinin 2006 paper (https://doi.org/10.1086/500288), where they
    state that 'Outside the central cooling region, the temperature profile can be adequately represented as a
    broken power law with a transition region'.

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float r_transition: The radius of the transition region of this broken power law model.
    :param float a_power: The first power law index.
    :param float b_power: The second power law index.
    :param float c_power: The third power law index.
    :param float t_zero: A normalising temperature value, not present in the original Vikhlinin model, but added
        here to allow this model to be fit independently of the whole Vikhlinin model. When full_vikhlinin_temp is
        used however this will be set to one.
    :return: The temperature value(s) of this model, corresponding to the input radius(ii) value.
    :rtype: Union[np.ndarray, float]
    """
    # The ratio of the input radius (or radii) to the transition radius
    rad_ratio = r_values / r_transition

    return np.power(rad_ratio, -a_power) / np.power((1 + np.power(rad_ratio, b_power)), (c_power/b_power))


def full_vikhlinin_temp(r_values: Union[np.ndarray, float], r_cool: float, a_cool: float, t_min: float, t_zero: float,
                        r_transition: float, a_power: float, b_power: float, c_power: float) \
        -> Union[np.ndarray, float]:
    """
    The full 3D temperature model proposed in the Vikhlinin 2006 paper (https://doi.org/10.1086/500288), it combines
    the central_region and outer_region (as they have been called in XGA).

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float r_cool: Parameter describing the radius of the cooling region (I THINK - NOT CERTAIN YET).
    :param float a_cool: Power law parameter for the cooling region (I THINK - NOT CERTAIN YET).
    :param float t_min: A minimum temperature parameter for the model (I THINK - NOT CERTAIN YET).
    :param float t_zero: A normalising temperature parameter for the model (I THINK - NOT CERTAIN YET).
    :param float r_transition: The radius of the transition region of this broken power law model.
    :param float a_power: The first power law index.
    :param float b_power: The second power law index.
    :param float c_power: the third power law index.
    :return: The temperature value(s) of this model, corresponding to the input radius(ii) value.
    :rtype: Union[np.ndarray, float]
    """
    return t_zero * central_region(r_values, r_cool, a_cool, t_min, t_zero) * outer_region(r_values, r_transition,
                                                                                           a_power, b_power, c_power)


def simplified_vikhlinin_temp(r_values: Union[np.ndarray, float], r_cool: float, a_cool: float,
                              t_min: float, t_zero: float, r_transition: float, c_power: float) \
        -> Union[np.ndarray, float]:
    """
    A simplified, 'functional', form of Vikhlinin's temperature model. This model has 6  free parameters rather
    than the 9 free parameters of the original, and was used in this (https://doi.org/10.1051/0004-6361/201833325)
    X-COP study of the thermodynamic properties of their sample. In that analysis they fit temperature profiles
    which have been scaled by the particular cluster's R500 and T500 value, but the default start parameters and
    priors of this implementation are geared toward directly fitting the original sample, with radius units of kpc.
    Honestly the X-COP way of doing is probably better, and there's no reason you couldn't do the same with XGA.

    :param np.ndarray/float r_values: The radii to calculate y values for.
    :param float r_cool: Parameter describing the radius of the cooling region (I THINK - NOT CERTAIN YET).
    :param float a_cool: Power law parameter for the cooling region (I THINK - NOT CERTAIN YET).
    :param float t_min: A minimum temperature parameter for the model (I THINK - NOT CERTAIN YET).
    :param float t_zero: A normalising temperature parameter for the model (I THINK - NOT CERTAIN YET).
    :param float r_transition: The radius of the transition region of this broken power law model.
    :param float c_power: The power law index for the part of the model which describes the outer region of
        the cluster.
    :return: The temperature value(s) of this model, corresponding to the input radius(ii) value.
    :rtype: Union[np.ndarray, float]
    """
    cool_expr = ((t_min/t_zero) + np.power(r_values/r_cool, a_cool)) / (1 + np.power(r_values/r_cool, a_cool))
    out_expr = 1 / np.power(1 + np.power(r_values/r_transition, 2), c_power/2)

    return t_zero * cool_expr * out_expr


# So that things like fitting functions can be written generally to support different models
TEMP_MODELS = {"central_region": central_region, "outer_region": outer_region, "vikhlinin_temp": full_vikhlinin_temp,
               "simple_vikhlinin_temp": simplified_vikhlinin_temp}

TEMP_MODELS_STARTS = {"central_region": [100, 1, 1, 1],
                      "outer_region": [400, 1, 2, 1, 1, 1],
                      "vikhlinin_temp": [100, 1, 1, 1, 400, 1, 2, 1],
                      "simple_vikhlinin_temp": [100, 1, 1, 1, 400, 1]}

TEMP_MODELS_PRIORS = {"central_region": [[0, 400], [0, 3], [0, 3], [0, 2]],
                      "outer_region": [[0, 1000], [0, 3], [0, 3], [0, 3], [0, 10]],
                      "vikhlinin_temp": [[0, 400], [0, 3], [0, 3], [0, 2], [0, 1000], [0, 3], [0, 3], [0, 3]],
                      "simple_vikhlinin_temp": [[1, 1000], [0, 10], [0, 10], [0, 10], [1, 1000], [0, 10]]
                      }
TEMP_MODELS_PUB_NAMES = {"central_region": "Central Region Cooling", "outer_region": "Outer Region",
                         "vikhlinin_temp": "Full Vikhlinin", "simple_vikhlinin_temp": "Simplified Vikhlinin"}
