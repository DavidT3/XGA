#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 10/03/2021, 12:47. Copyright (c) David J Turner

from typing import Union, List

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg

from .base import BaseModel1D
from ..utils import r500, r200, r2500


class BetaProfile1D(BaseModel1D):
    """
    An XGA model implementation of the beta profile, essentially a projected isothermal king profile, it can be
    used to describe a simple galaxy cluster radial surface brightness profile.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('ct/(s*arcmin**2)'),
                 cust_start_pars: List[Quantity] = None):

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
        :rtype: Quantity
        """
        return norm * np.power((1 + (np.power(x / r_core, 2))), ((-3 * beta) + 0.5))
        # return norm * (1 + ((x / r_core)**2)**((-3 * beta) + 0.5))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, '')) -> Quantity:
        """
        Calculates the gradient of the beta profile at a given point, overriding the numerical method implemented
        in the BaseModel1D class, as this simple model has an easily derivable first derivative.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        beta, r_core, norm = self._model_pars
        return ((2*x)/np.power(r_core, 2))*((-3*beta) + 0.5)*norm*np.power((1+(np.power(x/r_core, 2))), ((-3*beta)-0.5))


class DoubleBetaProfile1D(BaseModel1D):
    """
    An XGA model implementation of the double beta profile, a summation of two single beta models. Often thought
    to deal better with peaky cluster cores that you might get from a cool-core cluster, this model can be used
    to describe a galaxy cluster radial surface brightness profile.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('ct/(s*arcmin**2)'),
                 cust_start_pars: List[Quantity] = None):

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
        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([0, 100], 'ct/(s*arcmin**2)'), 'type': 'uniform'},
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
        return (norm_one * np.power((1 + (np.power(x / r_core_one, 2))), ((-3 * beta_one) + 0.5))) + \
               (norm_two * np.power((1 + (np.power(x / r_core_two, 2))), ((-3 * beta_two) + 0.5)))

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, '')) -> Quantity:
        """
        Calculates the gradient of the double beta profile at a given point, overriding the numerical method
        implemented in the BaseModel1D class, as this simple model has an easily derivable first derivative.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: This makes no difference here, as this is an analytical derivative. It has
            been left in so that the inputs for this method don't vary between models.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        beta_one, r_core_one, norm_one, beta_two, r_core_two, norm_two = self._model_pars
        p1 = ((2*x)/np.power(r_core_one, 2))*((-3*beta_one) + 0.5)*norm_one*np.power((1+(np.power(x/r_core_one, 2))),
                                                                                     ((-3*beta_one)-0.5))
        p2 = ((2*x)/np.power(r_core_two, 2))*((-3*beta_two)+0.5)*norm_two*np.power((1+(np.power(x/r_core_two, 2))),
                                                                                   ((-3*beta_two)-0.5))
        return p1 + p2


# So that things like fitting functions can be written generally to support different models
SB_MODELS = {"beta": BetaProfile1D, "double_beta": DoubleBetaProfile1D}
SB_MODELS_PUB_NAMES = {n: m().publication_name for n, m in SB_MODELS.items()}
SB_MODELS_PAR_NAMES = {n: m().par_publication_names for n, m in SB_MODELS.items()}
