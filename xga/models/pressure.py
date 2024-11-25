#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 25/11/2024, 09:53. Copyright (c) The Contributors

from typing import Union, List

from astropy.units import Quantity, Unit, UnitConversionError, kpc

from .base import BaseModel1D
from ..utils import r200, r500, r2500


class SimpleGNFWThermalPressure(BaseModel1D):
    """
    A model to fit galaxy cluster radial thermal pressure profiles, based on the generalized NFW profile equation
    proposed by https://ui.adsabs.harvard.edu/abs/2007ApJ...668....1N/abstract and used
    in https://ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A/abstract - this one has had several parameters removed
    or frozen.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV/cm^3 for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('keV cm^-3'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing a model of how thermal pressure changes
        with radius.
        """
        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('keV cm^-3')]
        y_convertible = [u.is_equivalent(y_unit) for u in poss_y_units]
        if not any(y_convertible):
            allowed = ", ".join([u.to_string() for u in poss_y_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=y_unit.to_string(), a=allowed))
        else:
            yu_ind = y_convertible.index(True)

        poss_x_units = [kpc]
        x_convertible = [u.is_equivalent(x_unit) for u in poss_x_units]
        if not any(x_convertible):
            allowed = ", ".join([u.to_string() for u in poss_x_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=x_unit.to_string(), a=allowed))
        else:
            xu_ind = x_convertible.index(True)

        p_zero_starts = [Quantity(0.1, 'keV cm^-3')]
        r_scale_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                          Quantity(0.5, r2500)]
        alpha = Quantity(1, '')
        beta = Quantity(1, '')

        start_pars = [p_zero_starts[yu_ind], r_scale_starts[xu_ind], alpha, beta]

        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        pz_priors = [{'prior': Quantity([0.0001, 10], 'keV cm^-3'), 'type': 'uniform'}]
        r_scale_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        alpha_priors = {'prior': Quantity([0, 10], ''), 'type': 'uniform'}
        beta_priors = {'prior': Quantity([0, 10], ''), 'type': 'uniform'}

        priors = [pz_priors[yu_ind], r_scale_priors[xu_ind], alpha_priors, beta_priors]

        nice_pars = [r"P$_{0}$", r"$R_{s}$", r"$\alpha$", r"$\beta$"]
        info_dict = {'author': 'Arnaud et al.', 'year': 2010,
                     'reference': 'https://ui.adsabs.harvard.edu/abs/2007ApJ...668....1N/abstract',
                     'general': "A model to fit galaxy cluster radial thermal pressure profiles, \n"
                                "based on the generalized NFW profile equation, but with gamma fixed."}

        super().__init__(x_unit, y_unit, start_pars, priors, 'simple_gnfw_pressure',
                         'Simplified gNFW Pressure Profile', nice_pars, 'Thermal Pressure', info_dict)

    @staticmethod
    def model(x: Quantity, p_zero: Quantity, r_scale: Quantity, alpha: Quantity, beta: Quantity) -> Quantity:
        """
        The model function for the simplified generalized NFW pressure profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity p_zero: The pressure normalization for the model.
        :param Quantity r_scale: The scale radius for the radial pressure profile.
        :param Quantity alpha: The alpha slope parameter, for the intermediate slope.
        :param Quantity beta: The beta slope parameter, for the outer slope.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        gamma = 0.31
        rad_rat = x/r_scale

        result = p_zero * (1/(rad_rat**gamma * (1 + rad_rat**alpha)**((beta - gamma)/alpha)))

        return result


class GNFWThermalPressure(BaseModel1D):
    """
    The full generalized NFW profile equation used to fit galaxy cluster radial thermal pressure profiles, based on
    https://ui.adsabs.harvard.edu/abs/2007ApJ...668....1N/abstract, but first used for pressure in
    https://ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A/abstract.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV/cm^3 for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('keV cm^-3'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing a model of how thermal pressure changes
        with radius.
        """
        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('keV cm^-3')]
        y_convertible = [u.is_equivalent(y_unit) for u in poss_y_units]
        if not any(y_convertible):
            allowed = ", ".join([u.to_string() for u in poss_y_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=y_unit.to_string(), a=allowed))
        else:
            yu_ind = y_convertible.index(True)

        poss_x_units = [kpc]
        x_convertible = [u.is_equivalent(x_unit) for u in poss_x_units]
        if not any(x_convertible):
            allowed = ", ".join([u.to_string() for u in poss_x_units])
            raise UnitConversionError("{p} is not convertible to any of the allowed units; "
                                      "{a}".format(p=x_unit.to_string(), a=allowed))
        else:
            xu_ind = x_convertible.index(True)

        p_zero_starts = [Quantity(0.1, 'keV cm^-3')]
        r_scale_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                          Quantity(0.5, r2500)]
        alpha = Quantity(1, '')
        beta = Quantity(1, '')
        gamma = Quantity(1, '')

        start_pars = [p_zero_starts[yu_ind], r_scale_starts[xu_ind], alpha, beta, gamma]

        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        pz_priors = [{'prior': Quantity([0.0001, 10], 'keV cm^-3'), 'type': 'uniform'}]
        r_scale_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        alpha_priors = {'prior': Quantity([0, 10], ''), 'type': 'uniform'}
        beta_priors = {'prior': Quantity([0, 10], ''), 'type': 'uniform'}
        gamma_priors = {'prior': Quantity([0, 10], ''), 'type': 'uniform'}

        priors = [pz_priors[yu_ind], r_scale_priors[xu_ind], alpha_priors, beta_priors, gamma_priors]

        nice_pars = [r"P$_{0}$", r"$R_{s}$", r"$\alpha$", r"$\beta$", r"$\gamma"]
        info_dict = {'author': 'Arnaud et al.', 'year': 2010,
                     'reference': 'https://ui.adsabs.harvard.edu/abs/2007ApJ...668....1N/abstract',
                     'general': "A model to fit galaxy cluster radial thermal pressure profiles, \n"
                                "based on the full generalized NFW profile equation."}

        super().__init__(x_unit, y_unit, start_pars, priors, 'gnfw_pressure',
                         'gNFW Pressure Profile', nice_pars, 'Thermal Pressure', info_dict)

    @staticmethod
    def model(x: Quantity, p_zero: Quantity, r_scale: Quantity, alpha: Quantity, beta: Quantity,
              gamma: Quantity) -> Quantity:
        """
        The model function for the simplified generalized NFW pressure profile.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity p_zero: The pressure normalization for the model.
        :param Quantity r_scale: The scale radius for the radial pressure profile.
        :param Quantity alpha: The alpha slope parameter, for the intermediate slope.
        :param Quantity beta: The beta slope parameter, for the outer slope.
        :param Quantity gamma: The beta slope parameter, for the inner slope.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        rad_rat = x/r_scale

        result = p_zero * (1/(rad_rat**gamma * (1 + rad_rat**alpha)**((beta - gamma)/alpha)))

        return result

# So that things like fitting functions can be written generally to support different models
PRESSURE_MODELS = {"simple_gnfw_pressure": SimpleGNFWThermalPressure, "gnfw_pressure": GNFWThermalPressure}
PRESSURE_MODELS_PUB_NAMES = {n: m().publication_name for n, m in PRESSURE_MODELS.items()}
PRESSURE_MODELS_PAR_NAMES = {n: m().par_publication_names for n, m in PRESSURE_MODELS.items()}