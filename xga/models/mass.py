#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/11/2024, 22:32. Copyright (c) The Contributors

from typing import Union, List

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg

from .base import BaseModel1D
from ..utils import r500, r200, r2500


class NFW(BaseModel1D):
    """
    A simple model to fit galaxy cluster mass profiles (https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract)
    - a cumulative mass version of the Navarro-Frenk-White profile. Typically, the NFW is formulated in terms of mass
    density, but one can derive a mass profile from it (https://ui.adsabs.harvard.edu/abs/2006MNRAS.368..518V/abstract).

    The NFW is extremely widely used, though generally for dark matter mass profiles, but will act as a handy
    functional form to fit to data-driven mass profiles derived from X-ray observations of clusters.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, Msun for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('Msun'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing the shape of cumulative mass profiles for
        a galaxy cluster based on the NFW mass density profile.
        """

        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('Msun')]
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

        r_scale_starts = [Quantity(100, 'kpc'), Quantity(0.2, 'deg'), Quantity(0.05, r200), Quantity(0.1, r500),
                         Quantity(0.5, r2500)]
        # We will implement the NFW mass profile with a rho_0 normalization parameter, a density - and leave in the
        #  volume integration terms - rather than fitting for some mass normalization
        norm_starts = [Quantity(1e+13, 'Msun/Mpc^3')]

        start_pars = [r_scale_starts[xu_ind], norm_starts[yu_ind]]
        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        r_core_priors = [{'prior': Quantity([0, 2000], 'kpc'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], 'deg'), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r200), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r500), 'type': 'uniform'},
                         {'prior': Quantity([0, 1], r2500), 'type': 'uniform'}]
        norm_priors = [{'prior': Quantity([1e+12, 1e+16], 'Msun/Mpc^3'), 'type': 'uniform'}]

        priors = [r_core_priors[xu_ind], norm_priors[yu_ind]]

        nice_pars = [r"R$_{\rm{s}}$", r"$\rho_{0}$"]
        info_dict = {'author': 'Navarro J, Frenk C, White S', 'year': '1997',
                     'reference': 'https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract',
                     'general': 'The cumulative mass version of the NFW mass-density profile for galaxy \n'
                                'clusters - normally used to describe dark matter profiles.'}

        super().__init__(x_unit, y_unit, start_pars, priors, 'nfw', 'NFW Profile', nice_pars, 'Mass',
                         info_dict)

    @staticmethod
    def model(x: Quantity, r_scale: Quantity, rho_zero: Quantity) -> Quantity:
        """
        The model function for the constant-core and power-law entropy model.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity r_scale: The scale radius parameter.
        :param Quantity rho_zero: A density normalization parameter.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """

        norm_rad = x / r_scale
        result = 4*np.pi*rho_zero*np.power(r_scale, 3)*(np.log(1 + norm_rad) - (norm_rad / (1 + norm_rad)))
        return result


# So that things like fitting functions can be written generally to support different models
MASS_MODELS = {"nfw": NFW}
MASS_MODELS_PUB_NAMES = {n: m().publication_name for n, m in MASS_MODELS.items()}
MASS_MODELS_PAR_NAMES = {n: m().par_publication_names for n, m in MASS_MODELS.items()}