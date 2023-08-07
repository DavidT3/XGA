#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 08/06/2023, 22:40. Copyright (c) The Contributors

from typing import Union, List

import numpy as np
from astropy.constants import k_B
from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg

from .base import BaseModel1D
from ..utils import r500, r200, r2500


class SimpleVikhlininTemperature1D(BaseModel1D):
    """
    An XGA model implementation of the simplified version of Vikhlinin's temperature model. This is for the
    description of 3D temperature profiles of galaxy clusters.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('keV'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing a simple version of the galaxy cluster
        temperature profile model created by Vikhlinin et al.
        """
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

        r_cool_starts = [Quantity(50, 'kpc'), Quantity(0.01, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                         Quantity(0.5, r2500)]
        r_tran_starts = [Quantity(200, 'kpc'), Quantity(0.015, 'deg'), Quantity(0.2, r200), Quantity(0.4, r500),
                         Quantity(0.7, r2500)]
        t_min_starts = [Quantity(3, 'keV'), (Quantity(3, 'keV')/k_B).to('K')]
        t_zero_starts = [Quantity(6, 'keV'), (Quantity(6, 'keV')/k_B).to('K')]

        start_pars = [r_cool_starts[xu_ind], Quantity(1, ''), t_min_starts[yu_ind], t_zero_starts[yu_ind],
                      r_tran_starts[xu_ind], Quantity(1, '')]

        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        rc_priors = [{'prior': Quantity([10, 500], 'kpc'), 'type': 'uniform'},
                     {'prior': Quantity([0.0, 0.032951243], 'deg'), 'type': 'uniform'},
                     {'prior': Quantity([0, 0.5], r200), 'type': 'uniform'},
                     {'prior': Quantity([0, 0.3], r500), 'type': 'uniform'},
                     {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        rt_priors = [{'prior': Quantity([100, 500], 'kpc'), 'type': 'uniform'},
                     {'prior': Quantity([0.001, 0.032951243], 'deg'), 'type': 'uniform'},
                     {'prior': Quantity([0.1, 0.5], r200), 'type': 'uniform'},
                     {'prior': Quantity([0.07, 0.3], r500), 'type': 'uniform'},
                     {'prior': Quantity([0.2, 1], r2500), 'type': 'uniform'}]
        t0_priors = [{'prior': Quantity([0.5, 15], 'keV'), 'type': 'uniform'},
                     {'prior': (Quantity([0.5, 15], 'keV')/k_B).to('K'), 'type': 'uniform'}]
        tm_priors = [{'prior': Quantity([0.1, 6], 'keV'), 'type': 'uniform'},
                     {'prior': (Quantity([0.1, 6], 'keV') / k_B).to('K'), 'type': 'uniform'}]

        priors = [rc_priors[xu_ind], {'prior': Quantity([0, 5]), 'type': 'uniform'}, tm_priors[yu_ind],
                  t0_priors[yu_ind], rt_priors[xu_ind], {'prior': Quantity([0, 5]), 'type': 'uniform'}]

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
        try:
            cool_expr = ((t_min / t_zero) + (x / r_cool)**a_cool) / (1 + (x / r_cool)**a_cool)
            out_expr = 1 / ((1 + (x / r_tran)**2)**(c_power / 2))
            result = t_zero * cool_expr * out_expr
        except ZeroDivisionError:
            result = np.NaN

        return result

    def derivative(self, x: Quantity, dx: Quantity = Quantity(0, ''), use_par_dist: bool = False) -> Quantity:
        """
        Calculates the gradient of the simple Vikhlinin temperature profile at a given point, overriding the
        numerical method implemented in the BaseModel1D class.

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
            r_c, a, t_m, t_0, r_t, c = self._model_pars
        else:
            r_c, a, t_m, t_0, r_t, c = self.par_dists

        p1 = (((x/r_t)**2)+1)**(-c/2)*((a*-(t_m-t_0))*((x**2)+(r_t**2))*((x/r_c)**a)
                                       - c*x**2*(((x/r_c)**a)+1)*(t_m+(t_0*((x/r_c)**a))))
        p2 = x*(x**2+r_t**2)*(((x/r_c)**a) + 1)**2

        return p1/p2


class VikhlininTemperature1D(BaseModel1D):
    """
    An XGA model implementation of the full version of Vikhlinin's temperature model. This is for the
    description of 3D temperature profiles of galaxy clusters.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('keV'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing the full version of the galaxy cluster
        temperature profile model created by Vikhlinin et al.
        """
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

        r_cool_starts = [Quantity(50, 'kpc'), Quantity(0.01, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                         Quantity(0.5, r2500)]
        r_tran_starts = [Quantity(200, 'kpc'), Quantity(0.015, 'deg'), Quantity(0.2, r200), Quantity(0.4, r500),
                         Quantity(0.7, r2500)]
        t_min_starts = [Quantity(3, 'keV'), (Quantity(3, 'keV') / k_B).to('K')]
        t_zero_starts = [Quantity(6, 'keV'), (Quantity(6, 'keV') / k_B).to('K')]

        start_pars = [r_cool_starts[xu_ind], Quantity(1, ''), t_min_starts[yu_ind], t_zero_starts[yu_ind],
                      r_tran_starts[xu_ind], Quantity(1, ''), Quantity(1, ''), Quantity(1, '')]

        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        rc_priors = [{'prior': Quantity([10, 500], 'kpc'), 'type': 'uniform'},
                     {'prior': Quantity([0.0, 0.032951243], 'deg'), 'type': 'uniform'},
                     {'prior': Quantity([0, 0.5], r200), 'type': 'uniform'},
                     {'prior': Quantity([0, 0.3], r500), 'type': 'uniform'},
                     {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        rt_priors = [{'prior': Quantity([100, 500], 'kpc'), 'type': 'uniform'},
                     {'prior': Quantity([0.001, 0.032951243], 'deg'), 'type': 'uniform'},
                     {'prior': Quantity([0.1, 0.5], r200), 'type': 'uniform'},
                     {'prior': Quantity([0.07, 0.3], r500), 'type': 'uniform'},
                     {'prior': Quantity([0.2, 1], r2500), 'type': 'uniform'}]
        t0_priors = [{'prior': Quantity([0.5, 15], 'keV'), 'type': 'uniform'},
                     {'prior': (Quantity([0.5, 15], 'keV') / k_B).to('K'), 'type': 'uniform'}]
        tm_priors = [{'prior': Quantity([0.1, 6], 'keV'), 'type': 'uniform'},
                     {'prior': (Quantity([0.1, 6], 'keV') / k_B).to('K'), 'type': 'uniform'}]

        priors = [rc_priors[xu_ind], {'prior': Quantity([0, 5]), 'type': 'uniform'}, tm_priors[yu_ind],
                  t0_priors[yu_ind], rt_priors[xu_ind], {'prior': Quantity([0, 5]), 'type': 'uniform'},
                  {'prior': Quantity([0, 5]), 'type': 'uniform'}, {'prior': Quantity([0, 5]), 'type': 'uniform'}]

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
        try:
            power_rad_ratio = (x / r_cool)**a_cool
            # The rest of the model expression
            t_cool = (power_rad_ratio + (t_min / t_zero)) / (power_rad_ratio + 1)

            # The ratio of the input radius (or radii) to the transition radius
            rad_ratio = x / r_tran
            t_outer = rad_ratio**(-a_power) / (1 + rad_ratio**b_power)**(c_power / b_power)
            result = t_zero * t_cool * t_outer

        except ZeroDivisionError:
            result = np.NaN

        return result


# So that things like fitting functions can be written generally to support different models
TEMP_MODELS = {"vikhlinin_temp": VikhlininTemperature1D, "simple_vikhlinin_temp": SimpleVikhlininTemperature1D}
TEMP_MODELS_PUB_NAMES = {n: m().publication_name for n, m in TEMP_MODELS.items()}
TEMP_MODELS_PAR_NAMES = {n: m().par_publication_names for n, m in TEMP_MODELS.items()}




