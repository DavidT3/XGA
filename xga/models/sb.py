#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 08/06/2023, 22:40. Copyright (c) The Contributors

from typing import Union, List

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg
from scipy.special import gamma

from .base import BaseModel1D
from ..exceptions import XGAFitError
from ..utils import r500, r200, r2500


class BetaProfile1D(BaseModel1D):
    """
    An XGA model implementation of the beta profile, essentially a projected isothermal king profile, it can be
    used to describe a simple galaxy cluster radial surface brightness profile.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('ct/(s*arcmin**2)'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing the surface brightness beta profile model.
        """
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
        norm_starts = [Quantity(1, 'ct/(s*arcmin**2)'), Quantity(1, 'ct/(s*kpc**2)'), Quantity(1, 'ct/(s*pix**2)')]

        start_pars = [Quantity(1, ''), r_core_starts[xu_ind], norm_starts[yu_ind]]
        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        # TODO ALSO MAKE THESE MORE SENSIBLE
        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([0, 3], 'ct/(s*arcmin**2)'), 'type': 'uniform'},
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
        :rtype: Quantity
        """
        return norm * ((1 + ((x / r_core)**2))**((-3 * beta) + 0.5))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, ''), use_par_dist: bool = False) -> Quantity:
        """
        Calculates the gradient of the beta profile at a given point, overriding the numerical method implemented
        in the BaseModel1D class, as this simple model has an easily derivable first derivative.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a derivative
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        # Just makes sure that if there are multiple x values then the broadcasting will go to the correct shape of
        #  numpy array
        # This makes sure that the input radius, or radii, are being used properly. For a single x value, or a set of
        #  x values, we use [..., None] to ensure that (if the user has decided to use the parameter distribution) we
        #  get M distributions of the derivative. In the case where the user has passed a distribution of radii, really
        #  representative of a single radius but with uncertainty, then this does not trigger, and rather than an N
        #  by N (where N is the number of samples in each posterior) array, you get an M x N array
        if x.isscalar or (not x.isscalar and x.ndim == 1):
            x = x[..., None]

        # Generates a distribution of derivatives using the parameter distributions
        if not use_par_dist:
            beta, r_core, norm = self._model_pars
        else:
            beta, r_core, norm = self.par_dists
        return ((2*x)/np.power(r_core, 2))*((-3*beta) + 0.5)*norm*np.power((1+(np.power(x/r_core, 2))), ((-3*beta)-0.5))

    def inverse_abel(self, x: Quantity, use_par_dist: bool = False, method='analytical') -> Quantity:
        """
        This overrides the inverse abel method of the model superclass, as there is an analytical solution to the
        inverse abel transform of the single beta model. The form of the inverse abel transform is that of the
        king profile, but with an extra transformation applied to the normalising parameter. This method can either
        return a single value calculated using the current model parameters, or a distribution of values using
        the parameter distributions (assuming that this model has had a fit run on it).

        :param Quantity x: The x location(s) at which to calculate the value of the inverse abel transform.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a inverse abel transform
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :param str method: The method that should be used to calculate the values of this inverse abel transform.
            Default for this overriding method is 'analytical', in which case the analytical solution is used.
            You  may pass 'direct', 'basex', 'hansenlaw', 'onion_bordas', 'onion_peeling', 'two_point', or
            'three_point' to calculate the transform numerically.
        :return: The inverse abel transform result.
        :rtype: Quantity
        """
        def transform(x_val: Quantity, beta: Quantity, r_core: Quantity, norm: Quantity):
            """
            The function that calculates the inverse abel transform of this beta profile.

            :param Quantity x_val: The x location(s) at which to calculate the value of the inverse abel transform.
            :param Quantity beta: The beta parameter of the beta profile.
            :param Quantity r_core: The core radius parameter of the beta profile.
            :param Quantity norm: The normalisation of the beta profile.
            :return:
            """
            # We calculate the new normalisation parameter
            new_norm = norm / ((gamma((3 * beta) - 0.5) * np.sqrt(np.pi) * r_core) / gamma(3 * beta))

            # Then return the value of the transformed beta profile
            return new_norm * np.power((1 + (np.power(x_val / r_core, 2))), (-3 * beta))

        # Checking x units to make sure that they are valid
        if not x.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("The input x coordinates cannot be converted to units of "
                                      "{}".format(self._x_unit.to_string()))
        else:
            x = x.to(self._x_unit)

        if method == 'analytical':
            # The way the calculation is called depends on whether the user wants to use the parameter distributions
            #  or just the current model parameter values to calculate the inverse abel transform.
            if not use_par_dist:
                transform_res = transform(x, *self.model_pars)
            elif use_par_dist and len(self._par_dists[0]) != 0:
                transform_res = transform(x[..., None], *self.par_dists)
            elif use_par_dist and len(self._par_dists[0]) == 0:
                raise XGAFitError("No fit has been performed with this model, so there are no parameter distributions"
                                  " available.")
        else:
            transform_res = super().inverse_abel(x, use_par_dist, method)

        return transform_res


class DoubleBetaProfile1D(BaseModel1D):
    """
    An XGA model implementation of the double beta profile, a summation of two single beta models. Often thought
    to deal better with peaky cluster cores that you might get from a cool-core cluster, this model can be used
    to describe a galaxy cluster radial surface brightness profile.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('ct/(s*arcmin**2)'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing the surface brightness double-beta
        profile model.
        """

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

        poss_x_units = [kpc, deg, r200, r500, r2500]
        x_convertible = [u.is_equivalent(x_unit) for u in poss_x_units]
        if not any(x_convertible):
            allowed = ", ".join([u.to_string() for u in poss_x_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=x_unit.to_string(), a=allowed))
        else:
            xu_ind = x_convertible.index(True)

        # TODO MAKE THE NEW START PARAMETERS MORE SENSIBLE
        r_core1_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                          Quantity(0.5, r2500)]
        norm_starts = [Quantity(1, 'ct/(s*arcmin**2)'), Quantity(1, 'ct/(s*kpc**2)'), Quantity(1, 'ct/(s*pix**2)')]
        r_core2_starts = [Quantity(400, 'kpc'), Quantity(0.5, 'deg'), Quantity(0.2, r200), Quantity(0.4, r500),
                          Quantity(1, r2500)]

        start_pars = [Quantity(1, ''), r_core1_starts[xu_ind], norm_starts[yu_ind], Quantity(0.5, ''),
                      r_core2_starts[xu_ind], norm_starts[yu_ind]*0.5]
        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        # TODO ALSO MAKE THESE MORE SENSIBLE
        r_core_priors = [{'prior': Quantity([1, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([0, 3], 'ct/(s*arcmin**2)'), 'type': 'uniform'},
                       {'prior': Quantity([0, 100], 'ct/(s*kpc**2)'), 'type': 'uniform'},
                       {'prior': Quantity([0, 100], 'ct/(s*pix**2)'), 'type': 'uniform'}]

        priors = [{'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind], norm_priors[yu_ind],
                  {'prior': Quantity([0, 3]), 'type': 'uniform'}, r_core_priors[xu_ind], norm_priors[yu_ind]]

        nice_pars = [r"$\beta_{1}$", r"R$_{\rm{core},1}$", r"S$_{01}$", r"$\beta_{2}$", r"R$_{\rm{core},2}$",
                     r"S$_{02}$"]
        info_dict = {'author': 'placeholder', 'year': 'placeholder', 'reference': 'placeholder',
                     'general': 'The double beta profile, a summation of two single beta models. Often\n '
                                'thought to deal better with peaky cluster cores that you might get from a\n'
                                ' cool-core cluster, this model can be used to describe a galaxy cluster\n'
                                ' radial surface brightness profile.'}
        super().__init__(x_unit, y_unit, start_pars, priors, 'double_beta', 'Double Beta Profile', nice_pars,
                         'Surface Brightness', info_dict)

    @staticmethod
    def model(x: Quantity, beta_one: Quantity, r_core_one: Quantity, norm_one: Quantity, beta_two: Quantity,
              r_core_two: Quantity, norm_two: Quantity) -> Quantity:
        """
        The model function for the double beta profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity norm_one: The normalisation of the first beta profile.
        :param Quantity beta_one: The beta slope parameter of the first component beta profile.
        :param Quantity r_core_one: The core radius of the first component beta profile.
        :param Quantity norm_two: The normalisation of the second beta profile.
        :param Quantity beta_two:  The beta slope parameter of the second component beta profile.
        :param Quantity r_core_two: The core radius of the second component beta profile.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        p1 = norm_one * ((1 + ((x / r_core_one) ** 2)) ** ((-3 * beta_one) + 0.5))
        p2 = norm_two * ((1 + ((x / r_core_two) ** 2)) ** ((-3 * beta_two) + 0.5))
        return p1 + p2

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, ''), use_par_dist: bool = False) -> Quantity:
        """
        Calculates the gradient of the double beta profile at a given point, overriding the numerical method
        implemented in the BaseModel1D class, as this simple model has an easily derivable first derivative.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
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

        p1 = ((2*x)/np.power(r_core_one, 2))*((-3*beta_one) + 0.5)*norm_one*np.power((1+(np.power(x/r_core_one, 2))),
                                                                                     ((-3*beta_one)-0.5))
        p2 = ((2*x)/np.power(r_core_two, 2))*((-3*beta_two)+0.5)*norm_two*np.power((1+(np.power(x/r_core_two, 2))),
                                                                                   ((-3*beta_two)-0.5))
        return p1 + p2

    def inverse_abel(self, x: Quantity, use_par_dist: bool = False, method='analytical') -> Quantity:
        """
        This overrides the inverse abel method of the model superclass, as there is an analytical solution to the
        inverse abel transform of the double beta model. The form of the inverse abel transform is that of two summed
        king profiles, but with extra transformations applied to the normalising parameters. This method can either
        return a single value calculated using the current model parameters, or a distribution of values using
        the parameter distributions (assuming that this model has had a fit run on it).

        :param Quantity x: The x location(s) at which to calculate the value of the inverse abel transform.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a inverse abel transform
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :param str method: The method that should be used to calculate the values of this inverse abel transform.
            Default for this overriding method is 'analytical', in which case the analytical solution is used.
            You  may pass 'direct', 'basex', 'hansenlaw', 'onion_bordas', 'onion_peeling', 'two_point', or
            'three_point' to calculate the transform numerically.
        :return: The inverse abel transform result.
        :rtype: Quantity
        """
        def transform(x_val: Quantity, beta: Quantity, r_core: Quantity, norm: Quantity, beta_two: Quantity,
                      r_core_two: Quantity, norm_two: Quantity):
            """
            The function that calculates the inverse abel transform of this double beta profile.

            :param Quantity x_val: The x location(s) at which to calculate the value of the inverse abel transform.
            :param Quantity beta: The beta parameter of the first beta profile.
            :param Quantity r_core: The core radius parameter of the first beta profile.
            :param Quantity norm: The normalisation of the first beta profile.
            :param Quantity beta_two: The beta parameter of the second beta profile.
            :param Quantity r_core_two: The core radius parameter of the second beta profile.
            :param Quantity norm_two: The normalisation of the second beta profile.
            :return:
            """
            # We calculate the new normalisation parameter
            new_norm = norm / ((gamma((3 * beta) - 0.5) * np.sqrt(np.pi) * r_core) / gamma(3 * beta))
            new_norm_two = norm_two / ((gamma((3 * beta_two) - 0.5) * np.sqrt(np.pi)
                                        * r_core_two) / gamma(3 * beta_two))

            # Then return the value of the transformed beta profile
            return new_norm * np.power((1 + (np.power(x_val / r_core, 2))), (-3 * beta)) + \
                   new_norm_two * np.power((1 + (np.power(x_val / r_core_two, 2))), (-3 * beta_two))

        # Checking x units to make sure that they are valid
        if not x.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("The input x coordinates cannot be converted to units of "
                                      "{}".format(self._x_unit.to_string()))
        else:
            x = x.to(self._x_unit)

        if method == 'analytical':
            # The way the calculation is called depends on whether the user wants to use the parameter distributions
            #  or just the current model parameter values to calculate the inverse abel transform.
            if not use_par_dist:
                transform_res = transform(x, *self.model_pars)
            elif use_par_dist and len(self._par_dists[0]) != 0:
                transform_res = transform(x[..., None], *self.par_dists)
            elif use_par_dist and len(self._par_dists[0]) == 0:
                raise XGAFitError("No fit has been performed with this model, so there are no parameter distributions"
                                  " available.")
        else:
            transform_res = super().inverse_abel(x, use_par_dist, method)

        return transform_res


# So that things like fitting functions can be written generally to support different models
SB_MODELS = {"beta": BetaProfile1D, "double_beta": DoubleBetaProfile1D}
SB_MODELS_PUB_NAMES = {n: m().publication_name for n, m in SB_MODELS.items()}
SB_MODELS_PAR_NAMES = {n: m().par_publication_names for n, m in SB_MODELS.items()}
