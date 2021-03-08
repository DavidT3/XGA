#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/03/2021, 10:48. Copyright (c) David J Turner

from typing import Union

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg

from .base import BaseModel1D
from ..utils import r500, r200, r2500


class KingProfile1D (BaseModel1D):
    """
    An XGA model implementation of the King profile, describing an isothermal sphere. This describes a
    radial density profile and assumes spherical symmetry.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('Msun/Mpc^3')):

        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('Msun/Mpc^3'), Unit('1/cm^3')]
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

        r_core_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                         Quantity(0.5, r2500)]
        # TODO MAKE THE NEW START PARAMETERS MORE SENSIBLE
        norm_starts = [Quantity(1e+13, 'Msun/Mpc^3'), Quantity(1e-3, '1/cm^3')]
        start_pars = [Quantity(1, ''), r_core_starts[xu_ind], norm_starts[yu_ind]]

        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([[1e+12, 1e+16]], 'Msun/Mpc^3'), 'type': 'uniform'},
                       {'prior': Quantity([0, 10], '1/cm^3'), 'type': 'uniform'}]

        priors = [{'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind], norm_priors[yu_ind]]

        nice_pars = [r"$\beta$", r"R$_{\rm{core}}$", "S$_{0}$"]
        info_dict = {'author': 'placeholder', 'year': 'placeholder', 'reference': 'placeholder',
                     'general': 'The unprojected version of the beta profile, suitable for a simple fit\n'
                                ' to 3D density distributions. Describes a simple isothermal sphere.'}
        super().__init__(x_unit, y_unit, start_pars, priors, 'king', 'King Profile', nice_pars, 'Gas Density',
                         info_dict)

    @staticmethod
    def model(x: Quantity, beta: Quantity, r_core: Quantity, norm: Quantity) -> Quantity:
        """
        The model function for the king profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity beta: The beta slope parameter of the model.
        :param Quantity r_core: The core radius.
        :param Quantity norm: The normalisation of the model.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        return norm * np.power((1 + (np.power(x / r_core, 2))), (-3 * beta))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, '')) -> Quantity:
        """
        Calculates the gradient of the king profile at a given point, overriding the numerical method implemented
        in the BaseModel1D class, as this simple model has an easily derivable first derivative.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        beta, r_core, norm = self._model_pars
        return (-6*beta*norm*x/np.power(r_core, 2))*np.power((1+np.power(x/r_core, 2)), (-3*beta) - 1)


class SimpleVikhlininDensity1D (BaseModel1D):
    """
    An XGA model implementation of a simplified version of Vikhlinin's full density model. Used relatively recently
    in https://doi.org/10.1051/0004-6361/201833325 by Ghirardini et al., a simplified form of Vikhlinin's full
    density model, which can be found in https://doi.org/10.1086/500288.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('Msun/Mpc^3')):

        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('Msun/Mpc^3'), Unit('1/cm^3')]
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

        r_core_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                         Quantity(0.5, r2500)]
        r_s_starts = [Quantity(300, 'kpc'), Quantity(0.35, 'deg'), Quantity(0.15, r200), Quantity(0.3, r500),
                      Quantity(1.0, r2500)]

        norm_starts = [Quantity(1e+13, 'Msun/Mpc^3'), Quantity(1e-3, '1/cm^3')]
        start_pars = [Quantity(1, ''), r_core_starts[xu_ind], Quantity(1, ''), r_s_starts[xu_ind], Quantity(2, ''),
                      norm_starts[yu_ind]]

        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([[1e+12, 1e+16]], 'Msun/Mpc^3'), 'type': 'uniform'},
                       {'prior': Quantity([0, 10], '1/cm^3'), 'type': 'uniform'}]

        priors = [{'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind],
                  {'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind],
                  {'prior': Quantity([0, 5]), 'type': 'uniform'}, norm_priors[yu_ind]]

        nice_pars = [r"$\beta$", r"R$_{\rm{core}}$", r"$\alpha$", r"R$_{\rm{s}}$", r"$\epsilon$", r"$\rho_{0}$"]
        info_dict = {'author': 'Ghirardini et al.', 'year': 2019,
                     'reference': 'https://doi.org/10.1051/0004-6361/201833325',
                     'general': "A simplified form of Vikhlinin's full density model, a type of broken\n"
                                " power law that deals well with most galaxy cluster density profile."}
        super().__init__(x_unit, y_unit, start_pars, priors, 'simple_vikhlinin_dens', 'Simplified Vikhlinin Profile',
                         nice_pars, 'Gas Density', info_dict)

    @staticmethod
    def model(x: Quantity, beta: Quantity, r_core: Quantity, alpha: Quantity, r_s: Quantity, epsilon: Quantity,
              norm: Quantity) -> Quantity:
        """
        The model function for the simplified Vikhlinin density profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity beta: The beta parameter of the model.
        :param Quantity r_core: The core radius of the model.
        :param Quantity alpha: The alpha parameter of the model.
        :param Quantity r_s: The radius near where a change of slope by epsilon occurs.
        :param Quantity epsilon: The epsilon parameter of the model.
        :param Quantity norm: The overall normalisation of the model.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        # Calculates the ratio of the r_values to the r_core parameter
        rc_rat = x / r_core
        # Calculates the ratio of the r_values to the r_s parameter
        rs_rat = x / r_s

        first_term = np.power(rc_rat, -alpha) / np.power((1 + np.power(rc_rat, 2)), ((3 * beta) - (alpha / 2)))
        second_term = 1 / np.power(1 + np.power(rs_rat, 3), epsilon / 3)
        result = norm * np.sqrt(first_term * second_term)
        return result

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, '')) -> Quantity:
        """
        Calculates the gradient of the simple Vikhlinin density profile at a given point, overriding the
        numerical method implemented in the BaseModel1D class.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        # TODO DOUBLE CHECK THIS WHEN I'M LESS TIRED
        beta, r_core, alpha, r_s, epsilon, norm = self.model_pars

        first_term = -1*norm*np.sqrt(np.power(x/r_core, -alpha)*np.power((x**3/r_s**3) + 1, -epsilon/3)
                                     *np.power((x**2/r_core**2) + 1, 0.5*(alpha-(6*beta))))
        second_term = 1/(2*x*(x**2 + r_core**2)*(x**3 + r_s**3))
        third_term = (x**3 + r_s**3)*(6*beta*x**2 + alpha*r_core**2) + x**3*epsilon*(x**2 + r_core**2)

        return first_term*second_term*third_term


class VikhlininDensity1D (BaseModel1D):
    """
    An XGA model implementation of Vikhlinin's full density model for galaxy cluster intra-cluster medium,
    which can be found in https://doi.org/10.1086/500288. It is a radial profile, so an assumption
    of spherical symmetry is baked in.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('Msun/Mpc^3')):

        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('Msun/Mpc^3'), Unit('1/cm^3')]
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

        r_core_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                         Quantity(0.5, r2500)]
        r_s_starts = [Quantity(300, 'kpc'), Quantity(0.35, 'deg'), Quantity(0.15, r200), Quantity(0.3, r500),
                      Quantity(1.0, r2500)]
        r_core_two_starts = [Quantity(50, 'kpc'), Quantity(0.1, 'deg'), Quantity(0.03, r200), Quantity(0.05, r500),
                             Quantity(0.25, r2500)]

        norm_starts = [Quantity(1e+13, 'Msun/Mpc^3'), Quantity(1e-3, '1/cm^3')]
        norm_two_starts = [Quantity(5e+12, 'Msun/Mpc^3'), Quantity(5e-4, '1/cm^3')]
        start_pars = [Quantity(1, ''), r_core_starts[xu_ind], Quantity(1, ''), r_s_starts[xu_ind], Quantity(2, ''),
                      Quantity(3, ''), norm_starts[yu_ind], Quantity(1, ''), r_core_two_starts[xu_ind],
                      norm_two_starts[yu_ind]]

        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([[1e+12, 1e+16]], 'Msun/Mpc^3'), 'type': 'uniform'},
                       {'prior': Quantity([0, 10], '1/cm^3'), 'type': 'uniform'}]

        priors = [{'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind],
                  {'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind],
                  {'prior': Quantity([0, 5]), 'type': 'uniform'}, {'prior': Quantity([-5, 5]), 'type': 'uniform'},
                  norm_priors[yu_ind], {'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind],
                  norm_priors[yu_ind]]

        nice_pars = [r"$\beta_{1}$", r"R$_{\rm{core,1}}$", r"$\alpha$", r"R$_{\rm{s}}$", r"$\epsilon$", r"$\gamma$",
                     r"$\rho_{01}$", r"$\beta_{2}$", r"R$_{\rm{core,2}}$", r"$\rho_{02}$"]
        info_dict = {'author': 'Vikhlinin et al.', 'year': 2006,
                     'reference': 'https://doi.org/10.1086/500288',
                     'general': "The full model for cluster density profiles created by Vikhlinin et al.\n"
                                "This model has MANY free parameters which can be very hard to get constraints\n"
                                " on, and as such many people would use the simplified version which is implemented\n"
                                " as 'simple_vikhlinin_dens' in XGA."}
        super().__init__(x_unit, y_unit, start_pars, priors, 'vikhlinin_dens', 'Vikhlinin Profile',
                         nice_pars, 'Gas Density', info_dict)

    @staticmethod
    def model(x: Quantity, beta_one: Quantity, r_core_one: Quantity, alpha: Quantity, r_s: Quantity, epsilon: Quantity,
              gamma: Quantity, norm_one: Quantity, beta_two: Quantity, r_core_two: Quantity, norm_two: Quantity):
        """
        The model function for the full Vikhlinin density profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity beta_one: The beta parameter of the model.
        :param Quantity r_core_one: The core radius of the model.
        :param Quantity alpha: The alpha parameter of the model.
        :param Quantity r_s: The radius near where a change of slope by epsilon occurs.
        :param Quantity epsilon: The epsilon parameter of the model.
        :param Quantity gamma: Width of slope change transition region.
        :param Quantity norm_one: The normalisation of the model first part of the model.
        :param Quantity beta_two: The beta parameter slope of the small core part of the model.
        :param Quantity r_core_two:The core radius of the small core part of the model.
        :param Quantity norm_two: The normalisation of the additive, small core part of the model.
        """
        # Calculates the ratio of the r_values to the r_core_one parameter
        rc1_rat = x / r_core_one
        # Calculates the ratio of the r_values to the r_core_two parameter
        rc2_rat = x / r_core_two
        # Calculates the ratio of the r_values to the r_s parameter
        rs_rat = x / r_s

        first_term = np.power(rc1_rat, -alpha) / np.power((1 + np.power(rc1_rat, 2)), ((3 * beta_one) - (alpha / 2)))
        second_term = 1 / np.power(1 + np.power(rs_rat, gamma), epsilon / gamma)
        additive_term = 1 / np.power(1 + np.power(rc2_rat, 2), 3 * beta_two)

        return np.sqrt((np.power(norm_one, 2) * first_term * second_term) + (np.power(norm_two, 2) * additive_term))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, '')) -> Quantity:
        """
        Calculates the gradient of the full Vikhlinin density profile at a given point, overriding the
        numerical method implemented in the BaseModel1D class.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        b, rc, a, rs, e, g, n, b2, rc2, n2 = self.model_pars

        # Its horrible I know...
        p1 = (-6*b2*(n2**2)*x*(((x/rc2)**2) + 1)**((-3*b2)-1)) / rc2**2
        p2 = (-a*(n**2)*((x/rc)**(-a-1))*((((x/rc)**2)+1)**((a/2)-3*b))*((((x/rs)**g) + 1)**(-e/g)))/rc
        p3 = (2*(n**2)*x*((a/2)-(3*b))*((x/rc)**(-a))*((((x/rc)**2)+1)**((a/2)-(3*b)-1))*((((x/rs)**g) + 1)**(-e/g)))/rc**2
        p4 = -(n**2)*e*(x**(g-1))*(rs**(-g))*((x/rc)**(-a))*((((x/rc)**2)+1)**((a/2)-(3*b)))*((((x/rs)**g)+1)**(-e/g-1))
        p5 = 2*np.sqrt((n2**2)*((((x/rc2)**2)+1)**(-3*b2)) + (n**2)*((x/rc)**(-a))*((((x/rc)**2)+1)**((a/2)-(3*b)))*((((x/rs)**g)+1)**(-e/g)))

        return (p1 + p2 + p3 + p4) / p5


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



