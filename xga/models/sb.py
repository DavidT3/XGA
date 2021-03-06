#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 06/03/2021, 19:11. Copyright (c) David J Turner

from typing import Union

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg

from .base import BaseModel1D
from ..utils import r500_unit, r200_unit, r2500_unit


class BetaProfile1D(BaseModel1D):
    """
    An XGA model implementation of the beta profile, essentially a projected isothermal king profile, it can be
    used to describe a simple galaxy cluster radial surface brightness profile.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('ct/(s*arcmin**2)')):

        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('ct/(s*arcmin**2)'), Unit('ct/(s*kpc**2)'), Unit('ct/(s*pix**2)')]
        y_convertible = [u.is_equivalent(y_unit) for u in poss_y_units]
        if not any(y_convertible):
            allowed = ", ".join([u.to_string() for u in poss_y_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=y_unit.to_string(), a=allowed))
        else:
            yu_ind = y_convertible.index(True)

        poss_x_units = [kpc, deg, r200_unit, r500_unit, r2500_unit]
        x_convertible = [u.is_equivalent(x_unit) for u in poss_x_units]
        if not any(x_convertible):
            allowed = ", ".join([u.to_string() for u in poss_x_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=x_unit.to_string(), a=allowed))
        else:
            xu_ind = x_convertible.index(True)

        r_core_starts = [Quantity(50, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200_unit), Quantity(0.1, r500_unit),
                         Quantity(0.5, r2500_unit)]
        # TODO MAKE THE NEW START PARAMETERS MORE SENSIBLE
        norm_starts = [Quantity(1, 'ct/(s*arcmin**2)'), Quantity(1, 'ct/(s*kpc**2)'), Quantity(1, 'ct/(s*pix**2)')]

        start_pars = [Quantity(1, ''), r_core_starts[xu_ind], norm_starts[yu_ind]]

        # TODO ALSO MAKE THESE MORE SENSIBLE
        r_core_priors = [{'prior': Quantity([0, 300], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200_unit), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500_unit), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500_unit), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([0, 100], 'ct/(s*arcmin**2)'), 'type': 'uniform'},
                       {'prior': Quantity([0, 100], 'ct/(s*kpc**2)'), 'type': 'uniform'},
                       {'prior': Quantity([0, 100], 'ct/(s*pix**2)'), 'type': 'uniform'}]

        priors = [{'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind], norm_priors[yu_ind]]

        nice_pars = [r"$\beta$", r"R$_{\rm{core}}$", "S$_{0}$"]
        info_dict = {'author': 'placeholder', 'year': 'placeholder', 'reference': 'placeholder',
                     'general': 'Essentially a projected isothermal king profile, it can be\n'
                                'used to describe a simple galaxy cluster radial surface brightness profile.'}
        super().__init__(x_unit, y_unit, start_pars, priors, 'beta', 'Beta Profile', nice_pars, 'Surface Brightness',
                         info_dict)

    @staticmethod
    def model(x: Quantity, beta: Quantity, r_core: Quantity, norm: Quantity) -> Quantity:
        """
        The model function for the beta profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity beta: The beta slope parameter of the model.
        :param Quantity r_core: The core radius.
        :param Quantity norm: The normalisation of the model.
        :return: The y values corresponding to the input x values.
        :rtype: Union[np.ndarray, float]
        """
        return norm * np.power((1 + (np.power(x / r_core, 2))), ((-3 * beta) + 0.5))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, '')) -> Quantity:
        """
        Calculates the gradient of the beta profile at a given point, overriding the numerical method implemented
        in the BaseModel1D class, as this simple model has an easily derivable first derivative.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        beta, r_core, norm = self._model_pars
        return ((2*x)/np.power(r_core, 2))*(-3*beta + 0.5)*norm*np.power((1+(np.power(x/r_core, 2))), ((-3*beta)-0.5))


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
