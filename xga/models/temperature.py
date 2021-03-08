#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/03/2021, 14:01. Copyright (c) David J Turner

from typing import Union

import numpy as np
from astropy.constants import k_B
from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg

from .base import BaseModel1D
from ..utils import r500, r200, r2500


class SimpleVikhlininTemperature1D(BaseModel1D):
    """
    An XGA model implementation of the simplified version of Vikhlinin's temperature model. This is for the
    description of 3D temperature profiles of galaxy clusters.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('keV')):
        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('keV'), Unit('K')]
        y_convertible = [u.is_equivalent(y_unit) for u in poss_y_units]
        if not any(y_convertible):
            allowed = ", ".join([u.to_string() for u in poss_y_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=y_unit.to_string(), a=allowed))
        else:
            yu_ind = y_convertible.index(True)

        poss_x_units = [kpc, deg, r200, r500, r2500]
        x_convertible = [u.is_equivalent(x_unit) for u in poss_x_units]
        if not any(x_convertible):
            allowed = ", ".join([u.to_string() for u in poss_x_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=x_unit.to_string(), a=allowed))
        else:
            xu_ind = x_convertible.index(True)

        r_cool_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                         Quantity(0.5, r2500)]
        r_tran_starts = [Quantity(400, 'kpc'), Quantity(0.6, 'deg'), Quantity(0.2, r200), Quantity(0.4, r500),
                         Quantity(1, r2500)]
        t_min_starts = [Quantity(1, 'keV'), (Quantity(1, 'keV')/k_B).to('K')]
        t_zero_starts = [Quantity(5, 'keV'), (Quantity(5, 'keV')/k_B).to('K')]

        start_pars = [r_cool_starts[xu_ind], Quantity(1, ''), t_min_starts[yu_ind], t_zero_starts[yu_ind],
                      r_tran_starts[xu_ind], Quantity(1, '')]

        r_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                    {'prior': Quantity([0, 2], 'deg'), 'type': 'uniform'},
                    {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                    {'prior': Quantity([0, 1.5], r500), 'type': 'uniform'},
                    {'prior': Quantity([0, 3], r2500), 'type': 'uniform'}]
        t_priors = [{'prior': Quantity([0, 15], 'keV'), 'type': 'uniform'},
                    {'prior': (Quantity([0, 15], 'keV')/k_B).to('K'), 'type': 'uniform'}]

        priors = [r_priors[xu_ind], Quantity([-10, 10]), t_priors[yu_ind], t_priors[yu_ind], r_priors[xu_ind],
                  Quantity([-10, 10])]

        nice_pars = [r"R$_{\rm{cool}}$", r"a$_{\rm{cool}}$", r"T$_{\rm{min}}$", "T$_{0}$", r"R$_{\rm{T}}$", "c"]
        info_dict = {'author': 'Ghirardini et al.', 'year': 2019,
                     'reference': 'https://doi.org/10.1051/0004-6361/201833325',
                     'general': "A simplified, 'functional', form of Vikhlinin's temperature model.\n"
                                " This model has 6 free parameters rather than the 9 free parameters\n"
                                " of the original"}
        super().__init__(x_unit, y_unit, start_pars, priors, 'simple_vikhlinin_temp',
                         'Simplified Vikhlinin Profile', nice_pars, 'Gas Temperature', info_dict)

    @staticmethod
    def model(x: Quantity, r_cool: Quantity, a_cool: Quantity, t_min: Quantity, t_zero: Quantity,  r_tran: Quantity,
              c_power: Quantity) -> Quantity:
        """
        The model function for the simplified Vikhlinin temperature profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity r_cool: Parameter describing the radius of the cooler core region.
        :param Quantity a_cool: Power law parameter for the cooler core region.
        :param Quantity t_min: A minimum temperature parameter for the model.
        :param Quantity t_zero: A normalising temperature parameter for the model.
        :param Quantity r_tran: The radius of the transition region of this broken power law model.
        :param Quantity c_power: The power law index for the part of the model which describes the outer region of
            the cluster.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        cool_expr = ((t_min / t_zero) + np.power(x / r_cool, a_cool)) / (1 + np.power(x / r_cool, a_cool))
        out_expr = 1 / np.power(1 + np.power(x / r_tran, 2), c_power / 2)

        return t_zero * cool_expr * out_expr

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, '')) -> Quantity:
        """
        Calculates the gradient of the simple Vikhlinin temperature profile at a given point, overriding the
        numerical method implemented in the BaseModel1D class.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        r_c, a, t_m, t_0, r_t, c = self._model_pars
        p1 = (((x/r_t)**2)+1)**(-c/2)*((a*-(t_m-t_0))*((x**2)+(r_t**2))*((x/r_c)**a)
                                       - c*x**2*(((x/r_c)**a)+1)*(t_m+(t_0*((x/r_c)**a))))
        p2 = x*(x**2+r_t**2)*(((x/r_c)**a) + 1)**2

        return p1/p2


class VikhlininTemperature1D(BaseModel1D):
    """
    An XGA model implementation of the full version of Vikhlinin's temperature model. This is for the
    description of 3D temperature profiles of galaxy clusters.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('keV')):
        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('keV'), Unit('K')]
        y_convertible = [u.is_equivalent(y_unit) for u in poss_y_units]
        if not any(y_convertible):
            allowed = ", ".join([u.to_string() for u in poss_y_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=y_unit.to_string(), a=allowed))
        else:
            yu_ind = y_convertible.index(True)

        poss_x_units = [kpc, deg, r200, r500, r2500]
        x_convertible = [u.is_equivalent(x_unit) for u in poss_x_units]
        if not any(x_convertible):
            allowed = ", ".join([u.to_string() for u in poss_x_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=x_unit.to_string(), a=allowed))
        else:
            xu_ind = x_convertible.index(True)

        r_cool_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                         Quantity(0.5, r2500)]
        r_tran_starts = [Quantity(400, 'kpc'), Quantity(0.6, 'deg'), Quantity(0.2, r200), Quantity(0.4, r500),
                         Quantity(1, r2500)]
        t_min_starts = [Quantity(1, 'keV'), (Quantity(1, 'keV')/k_B).to('K')]
        t_zero_starts = [Quantity(5, 'keV'), (Quantity(5, 'keV')/k_B).to('K')]

        start_pars = [r_cool_starts[xu_ind], Quantity(1, ''), t_min_starts[yu_ind], t_zero_starts[yu_ind],
                      r_tran_starts[xu_ind], Quantity(1, ''), Quantity(1, ''), Quantity(1, '')]

        r_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                    {'prior': Quantity([0, 2], 'deg'), 'type': 'uniform'},
                    {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                    {'prior': Quantity([0, 1.5], r500), 'type': 'uniform'},
                    {'prior': Quantity([0, 3], r2500), 'type': 'uniform'}]
        t_priors = [{'prior': Quantity([0, 15], 'keV'), 'type': 'uniform'},
                    {'prior': (Quantity([0, 15], 'keV')/k_B).to('K'), 'type': 'uniform'}]

        priors = [r_priors[xu_ind], Quantity([-10, 10]), t_priors[yu_ind], t_priors[yu_ind], r_priors[xu_ind],
                  Quantity([-10, 10]), Quantity([-10, 10]), Quantity([-10, 10])]

        nice_pars = [r"R$_{\rm{cool}}$", r"a$_{\rm{cool}}$", r"T$_{\rm{min}}$", "T$_{0}$", r"R$_{\rm{T}}$", "a", "b",
                     "c"]
        info_dict = {'author': 'Vikhlinin et al.', 'year': 2005,
                     'reference': 'https://doi.org/10.1086/500288',
                     'general': "The full form of Vikhlinin's temperature model, describes a\n"
                                "cluster temperature profile from the core to the outskirts."}
        super().__init__(x_unit, y_unit, start_pars, priors, 'vikhlinin_temp',
                         'Vikhlinin Profile', nice_pars, 'Gas Temperature', info_dict)

    @staticmethod
    def model(x: Quantity, r_cool: Quantity, a_cool: Quantity, t_min: Quantity, t_zero: Quantity, r_tran: Quantity,
              a_power: Quantity, b_power: Quantity, c_power: Quantity) -> Quantity:
        """
        The model function for the full Vikhlinin temperature profile.

        :param Quantity x: The radii to calculate y values for.
        :param float r_cool: Parameter describing the radius of the cooling region (I THINK - NOT CERTAIN YET).
        :param float a_cool: Power law parameter for the cooling region (I THINK - NOT CERTAIN YET).
        :param float t_min: A minimum temperature parameter for the model (I THINK - NOT CERTAIN YET).
        :param float t_zero: A normalising temperature parameter for the model (I THINK - NOT CERTAIN YET).
        :param float r_tran: The radius of the transition region of this broken power law model.
        :param float a_power: The first power law index.
        :param float b_power: The second power law index.
        :param float c_power: the third power law index.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        power_rad_ratio = np.power((x / r_cool), a_cool)
        # The rest of the model expression
        t_cool = (power_rad_ratio + (t_min / t_zero)) / (power_rad_ratio + 1)

        # The ratio of the input radius (or radii) to the transition radius
        rad_ratio = x / r_tran
        t_outer = np.power(rad_ratio, -a_power) / np.power((1 + np.power(rad_ratio, b_power)),
                                                           (c_power / b_power))

        return t_zero * t_cool * t_outer


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

    return np.power(rad_ratio, -a_power) / np.power((1 + np.power(rad_ratio, b_power)), (c_power/b_power)) * t_zero


def simplified_vikhlinin_temp(r_values: Union[np.ndarray, float], r_cool: float, a_cool: float, t_min: float,
                              t_zero: float, r_transition: float, c_power: float) \
        -> Union[np.ndarray, float]:
    """
    A simplified, 'functional', form of Vikhlinin's temperature model. This model has 6 free parameters rather
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


# So that things like fitting functions can be written generally to support different models
TEMP_MODELS = {"central_region": central_region, "outer_region": outer_region, "vikhlinin_temp": full_vikhlinin_temp,
               "simple_vikhlinin_temp": simplified_vikhlinin_temp}

TEMP_MODELS_STARTS = {"central_region": [100, 1, 1, 1],
                      "outer_region": [400, 1, 2, 1, 1],
                      "vikhlinin_temp": [100, 1, 1, 1, 400, 1, 2, 1],
                      "simple_vikhlinin_temp": [100, 1, 1, 1, 400, 1]}

TEMP_MODELS_PRIORS = {"central_region": [[0, 400], [0, 3], [0, 3], [0, 2]],
                      "outer_region": [[0, 1000], [0, 3], [0, 3], [0, 3], [0, 10]],
                      "vikhlinin_temp": [[0, 400], [0, 3], [0, 3], [0, 2], [0, 1000], [0, 3], [0, 3], [0, 3]],
                      "simple_vikhlinin_temp": [[1, 1000], [0, 10], [0, 10], [0, 10], [1, 1000], [0, 10]]
                      }
TEMP_MODELS_PUB_NAMES = {"central_region": "Central Region Cooling", "outer_region": "Outer Region",
                         "vikhlinin_temp": "Full Vikhlinin", "simple_vikhlinin_temp": "Simplified Vikhlinin"}

TEMP_MODELS_PAR_NAMES = {"central_region": [r"R$_{\rm{cool}}$", r"a$_{\rm{cool}}$", r"T$_{\rm{min}}$", "T$_{0}$"],
                         "outer_region": [r"R$_{\rm{T}}$", "a", "b", "c", "T$_{0}$"],
                         "vikhlinin_temp": [r"R$_{\rm{cool}}$", r"a$_{\rm{cool}}$", r"T$_{\rm{min}}$", "T$_{0}$",
                                            r"R$_{\rm{T}}$", "a", "b", "c"],
                         "simple_vikhlinin_temp": [r"R$_{\rm{cool}}$", r"a$_{\rm{cool}}$", r"T$_{\rm{min}}$", "T$_{0}$",
                                                   r"R$_{\rm{T}}$", "c"]}




