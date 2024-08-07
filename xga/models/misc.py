#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

from typing import Union

import numpy as np


def straight_line(x_values: Union[np.ndarray, float], gradient: float, intercept: float) -> Union[np.ndarray, float]:
    """
    As simple a model as you can get, a straight line. Possible uses include fitting very simple scaling relations.

    :param np.ndarray/float x_values: The x_values to retrieve corresponding y values for.
    :param float gradient: The gradient of the straight line.
    :param float intercept: The intercept of the straight line.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    return (gradient * x_values) + intercept


def power_law(x_values: Union[np.ndarray, float], slope: float, norm: float) -> Union[np.ndarray, float]:
    """
    A simple power law model, with slope and normalisation parameters. This is the standard model for fitting cluster
    scaling relations in XGA.

    :param np.ndarray/float x_values: The x_values to retrieve corresponding y values for.
    :param float slope: The slope parameter of the power law.
    :param float norm: The normalisation parameter of the power law.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    return np.power(x_values, slope) * norm


# So that things like fitting functions can be written generally to support different models
MISC_MODELS = {'straight_line': straight_line, 'power_law': power_law}
# The default start parameters for these models
MISC_MODELS_STARTS = {'straight_line': [1, 1], 'power_law': [1, 1]}
# The default priors for these models, for MCMC type fitters. THESE PARTICULAR ONES DON'T HAVE MUCH MEANING, and MCMC
#  really shouldn't be necessary to fit such simple models.
MISC_MODELS_PRIORS = {"straight_line": [[0, 100], [0, 100]], "power_law": [[0, 100], [0, 100]]}

MISC_MODELS_PUB_NAMES = {'power_law': 'Power Law', 'straight_line': "Straight Line"}
MISC_MODELS_PAR_NAMES = {"straight_line": ['m', 'c'], "power_law": ['Slope', 'Norm']}



