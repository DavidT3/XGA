#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 04/06/2025, 13:31. Copyright (c) The Contributors

from typing import Union, List

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg

from .base import BaseModel1D
from ..utils import r500, r200, r2500


class KingProfile1D(BaseModel1D):
    """
    An XGA model implementation of the King profile, describing an isothermal sphere. This describes a
    radial density profile and assumes spherical symmetry.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('Msun/Mpc^3'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing a basic model for galaxy cluster gas
        density, the king profile.
        """
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
        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([1e+12, 1e+16], 'Msun/Mpc^3'), 'type': 'uniform'},
                       {'prior': Quantity([0, 10], '1/cm^3'), 'type': 'uniform'}]

        priors = [{'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind], norm_priors[yu_ind]]

        nice_pars = [r"$\beta$", r"R$_{\rm{core}}$", "N$_{0}$"]
        info_dict = {'author': 'placeholder', 'year': 'placeholder', 'reference': 'placeholder',
                     'general': 'The un-projected version of the beta profile, suitable for a simple fit\n'
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
        return norm * ((1 + (x / r_core)**2)**(-3 * beta))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, ''), use_par_dist: bool = False) -> Quantity:
        """
        Calculates the gradient of the king profile at a given point, overriding the numerical method implemented
        in the BaseModel1D class, as this simple model has an easily derivable first derivative.

        :param Quantity x: The point(s) at which the slope of the model should be measured. If multiple,
            non-distribution, radii are to be used, make sure to pass them as an (M,), single dimension, astropy
            quantity, where M is the number of separate radii to generate realisations for. To marginalise over a
            radius distribution when generating realisations, pass a multi-dimensional astropy quantity; i.e. for
            a single set of realisations pass a (1, N) quantity, where N is the number of samples in the parameter
            posteriors, for realisations for M different radii pass a (M, N) quantity.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a derivative
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        # This makes sure that the input radius, or radii, are being used properly. For a single x value, or a set of
        #  x values, we use [..., None] to ensure that (if the user has decided to use the parameter distribution) we
        #  get M distributions of the derivative. In the case where the user has passed a distribution of radii, really
        #  representative of a single radius but with uncertainty, then this does not trigger, and rather than an N
        #  by N (where N is the number of samples in each posterior) array, you get an M x N array
        if x.isscalar or (not x.isscalar and x.ndim == 1):
            x = x[..., None]
        if not use_par_dist:
            beta, r_core, norm = self._model_pars
        else:
            beta, r_core, norm = self.par_dists

        return (-6*beta*norm*x/np.power(r_core, 2))*np.power((1+np.power(x/r_core, 2)), (-3*beta) - 1)


class DoubleKingProfile1D(BaseModel1D):
    """
    An XGA model implementation of the double King profile, simply the sum of two King profiles. This describes a
    radial density profile and assumes spherical symmetry.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('Msun/Mpc^3'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing a basic model for galaxy cluster gas
        density, the king profile.
        """
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
        start_pars = [Quantity(1, ''), r_core_starts[xu_ind], norm_starts[yu_ind],
                      Quantity(1, ''), r_core_starts[xu_ind], norm_starts[yu_ind]]
        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        # TODO MAYBE ADJUST ALL OF THE PRIORS ETC AS THEY'RE JUST COPIED FROM KING
        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([1e+12, 1e+16], 'Msun/Mpc^3'), 'type': 'uniform'},
                       {'prior': Quantity([0, 10], '1/cm^3'), 'type': 'uniform'}]

        priors = [{'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind], norm_priors[yu_ind],
                  {'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind], norm_priors[yu_ind]]

        nice_pars = [r"$\beta_{1}$", r"R$_{\rm{core}, 1}$", "N$_{0, 1}$", r"$\beta_{2}$", r"R$_{\rm{core}, 2}$",
                     "N$_{0, 2}$"]
        info_dict = {'author': 'placeholder', 'year': 'placeholder', 'reference': 'placeholder',
                     'general': 'placeholder'}
        super().__init__(x_unit, y_unit, start_pars, priors, 'double_king', 'Double King Profile', nice_pars,
                         'Gas Density', info_dict)

    @staticmethod
    def model(x: Quantity, beta_one: Quantity, r_core_one: Quantity, norm_one: Quantity, beta_two: Quantity,
              r_core_two: Quantity, norm_two: Quantity) -> Quantity:
        """
        The model function for the double King profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity beta_one: The beta slope parameter of the first King model.
        :param Quantity r_core_one: The core radius of the first King model.
        :param Quantity norm_one: The normalisation of the first King model.
        :param Quantity beta_two: The beta slope parameter of the second King model.
        :param Quantity r_core_two: The core radius of the second King model.
        :param Quantity norm_two: The normalisation of the second King model.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        return (norm_one * ((1 + (x / r_core_one)**2)**(-3 * beta_one))) + \
               (norm_two * ((1 + (x / r_core_two)**2)**(-3 * beta_two)))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, ''), use_par_dist: bool = False) -> Quantity:
        """
        Calculates the gradient of the double King profile at a given point, overriding the numerical method implemented
        in the BaseModel1D class, as this simple model has an easily derivable first derivative.

        :param Quantity x: The point(s) at which the slope of the model should be measured. If multiple,
            non-distribution, radii are to be used, make sure to pass them as an (M,), single dimension, astropy
            quantity, where M is the number of separate radii to generate realisations for. To marginalise over a
            radius distribution when generating realisations, pass a multi-dimensional astropy quantity; i.e. for
            a single set of realisations pass a (1, N) quantity, where N is the number of samples in the parameter
            posteriors, for realisations for M different radii pass a (M, N) quantity.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a derivative
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        # This makes sure that the input radius, or radii, are being used properly. For a single x value, or a set of
        #  x values, we use [..., None] to ensure that (if the user has decided to use the parameter distribution) we
        #  get M distributions of the derivative. In the case where the user has passed a distribution of radii, really
        #  representative of a single radius but with uncertainty, then this does not trigger, and rather than an N
        #  by N (where N is the number of samples in each posterior) array, you get an M x N array
        if x.isscalar or (not x.isscalar and x.ndim == 1):
            x = x[..., None]

        if not use_par_dist:
            beta_one, r_core_one, norm_one, beta_two, r_core_two, norm_two = self._model_pars
        else:
            beta_one, r_core_one, norm_one, beta_two, r_core_two, norm_two = self.par_dists
        p1 = (-6*beta_one*norm_one*x/np.power(r_core_one, 2))*np.power((1+np.power(x/r_core_one, 2)), (-3*beta_one) - 1)
        p2 = (-6*beta_two*norm_two*x/np.power(r_core_two, 2))*np.power((1+np.power(x/r_core_two, 2)), (-3*beta_two) - 1)
        return p1 + p2


class SimpleVikhlininDensity1D(BaseModel1D):
    """
    An XGA model implementation of a simplified version of Vikhlinin's full density model. Used relatively recently
    in https://doi.org/10.1051/0004-6361/201833325 by Ghirardini et al., a simplified form of Vikhlinin's full
    density model, which can be found in https://doi.org/10.1086/500288.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('Msun/Mpc^3'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing a simplified version of Vikhlinin et al.'s
        model for the gas density profile of a galaxy cluster.
        """
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
        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([1e+12, 1e+16], 'Msun/Mpc^3'), 'type': 'uniform'},
                       {'prior': Quantity([0, 10], '1/cm^3'), 'type': 'uniform'}]

        priors = [{'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind],
                  {'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind],
                  {'prior': Quantity([0, 5]), 'type': 'uniform'}, norm_priors[yu_ind]]

        nice_pars = [r"$\beta$", r"R$_{\rm{core}}$", r"$\alpha$", r"R$_{\rm{s}}$", r"$\epsilon$", r"N$_{0}$"]
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
        try:
            # Calculates the ratio of the r_values to the r_core parameter
            rc_rat = x / r_core
            # Calculates the ratio of the r_values to the r_s parameter
            rs_rat = x / r_s

            first_term = rc_rat**(-alpha) / ((1+rc_rat**2)**((3 * beta) - (alpha / 2)))
            second_term = 1 / ((1 + rs_rat**3)**(epsilon / 3))
            result = norm * np.sqrt(first_term * second_term)
        except ZeroDivisionError:
            result = np.nan

        return result

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, ''), use_par_dist: bool = False) -> Quantity:
        """
        Calculates the gradient of the simple Vikhlinin density profile at a given point, overriding the
        numerical method implemented in the BaseModel1D class.

        :param Quantity x: The point(s) at which the slope of the model should be measured. If multiple,
            non-distribution, radii are to be used, make sure to pass them as an (M,), single dimension, astropy
            quantity, where M is the number of separate radii to generate realisations for. To marginalise over a
            radius distribution when generating realisations, pass a multi-dimensional astropy quantity; i.e. for
            a single set of realisations pass a (1, N) quantity, where N is the number of samples in the parameter
            posteriors, for realisations for M different radii pass a (M, N) quantity.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a derivative
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        # This makes sure that the input radius, or radii, are being used properly. For a single x value, or a set of
        #  x values, we use [..., None] to ensure that (if the user has decided to use the parameter distribution) we
        #  get M distributions of the derivative. In the case where the user has passed a distribution of radii, really
        #  representative of a single radius but with uncertainty, then this does not trigger, and rather than an N
        #  by N (where N is the number of samples in each posterior) array, you get an M x N array
        if x.isscalar or (not x.isscalar and x.ndim == 1):
            x = x[..., None]

        if not use_par_dist:
            beta, r_core, alpha, r_s, epsilon, norm = self.model_pars
        else:
            beta, r_core, alpha, r_s, epsilon, norm = self.par_dists

        # TODO DOUBLE CHECK THIS WHEN I'M LESS TIRED
        first_term = -1*norm*np.sqrt(np.power(x/r_core, -alpha)*np.power((x**3/r_s**3) + 1, -epsilon/3)
                                     *np.power((x**2/r_core**2) + 1, 0.5*(alpha-(6*beta))))
        second_term = 1/(2*x*(x**2 + r_core**2)*(x**3 + r_s**3))
        third_term = (x**3 + r_s**3)*(6*beta*x**2 + alpha*r_core**2) + x**3*epsilon*(x**2 + r_core**2)

        return first_term*second_term*third_term


class VikhlininDensity1D(BaseModel1D):
    """
    An XGA model implementation of Vikhlinin's full density model for galaxy cluster intra-cluster medium,
    which can be found in https://doi.org/10.1086/500288. It is a radial profile, so an assumption
    of spherical symmetry is baked in.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('Msun/Mpc^3'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing the full version of Vikhlinin et al.'s
        model for the gas density profile of a galaxy cluster.
        """

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
        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

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
                     r"N$_{01}$", r"$\beta_{2}$", r"R$_{\rm{core,2}}$", r"N$_{02}$"]
        info_dict = {'author': 'Vikhlinin et al.', 'year': 2006,
                     'reference': 'https://doi.org/10.1086/500288',
                     'general': "The full model for cluster density profiles created by Vikhlinin et al.\n"
                                "This model has MANY free parameters which can be very hard to get constraints\n"
                                " on, and as such many people would use the simplified version which is implemented\n"
                                " as the SimpleVikhlininDensity1D class in XGA."}
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
        try:
            # Calculates the ratio of the r_values to the r_core_one parameter
            rc1_rat = x / r_core_one
            # Calculates the ratio of the r_values to the r_core_two parameter
            rc2_rat = x / r_core_two
            # Calculates the ratio of the r_values to the r_s parameter
            rs_rat = x / r_s

            first_term = rc1_rat**(-alpha) / ((1 + rc1_rat**2)**((3 * beta_one) - (alpha / 2)))
            second_term = 1 / ((1 + rs_rat**gamma)**(epsilon / gamma))
            additive_term = 1 / ((1 + rc2_rat**2)**(3 * beta_two))
        except ZeroDivisionError:
            return np.nan

        return np.sqrt((np.power(norm_one, 2) * first_term * second_term) + (np.power(norm_two, 2) * additive_term))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, ''), use_par_dist: bool = False) -> Quantity:
        """
        Calculates the gradient of the full Vikhlinin density profile at a given point, overriding the
        numerical method implemented in the BaseModel1D class.

        :param Quantity x: The point(s) at which the slope of the model should be measured. If multiple,
            non-distribution, radii are to be used, make sure to pass them as an (M,), single dimension, astropy
            quantity, where M is the number of separate radii to generate realisations for. To marginalise over a
            radius distribution when generating realisations, pass a multi-dimensional astropy quantity; i.e. for
            a single set of realisations pass a (1, N) quantity, where N is the number of samples in the parameter
            posteriors, for realisations for M different radii pass a (M, N) quantity.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a derivative
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        # This makes sure that the input radius, or radii, are being used properly. For a single x value, or a set of
        #  x values, we use [..., None] to ensure that (if the user has decided to use the parameter distribution) we
        #  get M distributions of the derivative. In the case where the user has passed a distribution of radii, really
        #  representative of a single radius but with uncertainty, then this does not trigger, and rather than an N
        #  by N (where N is the number of samples in each posterior) array, you get an M x N array
        if x.isscalar or (not x.isscalar and x.ndim == 1):
            x = x[..., None]
        if not use_par_dist:
            b, rc, a, rs, e, g, n, b2, rc2, n2 = self.model_pars
        else:
            b, rc, a, rs, e, g, n, b2, rc2, n2 = self.par_dists

        # Its horrible I know...
        p1 = (-6*b2*(n2**2)*x*(((x/rc2)**2) + 1)**((-3*b2)-1)) / rc2**2
        p2 = (-a*(n**2)*((x/rc)**(-a-1))*((((x/rc)**2)+1)**((a/2)-3*b))*((((x/rs)**g) + 1)**(-e/g)))/rc
        p3 = (2*(n**2)*x*((a/2)-(3*b))*((x/rc)**(-a))*((((x/rc)**2)+1)**((a/2)-(3*b)-1))*((((x/rs)**g) + 1)**(-e/g)))/rc**2
        p4 = -(n**2)*e*(x**(g-1))*(rs**(-g))*((x/rc)**(-a))*((((x/rc)**2)+1)**((a/2)-(3*b)))*((((x/rs)**g)+1)**(-e/g-1))
        p5 = 2*np.sqrt((n2**2)*((((x/rc2)**2)+1)**(-3*b2)) + (n**2)*((x/rc)**(-a))*((((x/rc)**2)+1)**((a/2)-(3*b)))*((((x/rs)**g)+1)**(-e/g)))

        return (p1 + p2 + p3 + p4) / p5


DENS_MODELS = {"simple_vikhlinin_dens": SimpleVikhlininDensity1D, 'king': KingProfile1D,
               'double_king': DoubleKingProfile1D, 'vikhlinin_dens': VikhlininDensity1D}
DENS_MODELS_PAR_NAMES = {n: m().par_publication_names for n, m in DENS_MODELS.items()}
DENS_MODELS_PUB_NAMES = {n: m().publication_name for n, m in DENS_MODELS.items()}



