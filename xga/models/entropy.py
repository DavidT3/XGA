#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 29/07/2024, 22:28. Copyright (c) The Contributors

from typing import Union, List

from astropy.units import Quantity, Unit, UnitConversionError, kpc, deg

from .base import BaseModel1D
from ..utils import r500, r200, r2500


class CoreConstPowerEntropy(BaseModel1D):
    """


    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] cust_start_pars: The start values of the model parameters for any fitting function that
        used start values. The units are checked against default start values.
    """
    def __init__(self, x_unit: Union[str, Unit] = 'kpc', y_unit: Union[str, Unit] = Unit('keV cm^2'),
                 cust_start_pars: List[Quantity] = None):
        """
        The init of a subclass of the XGA BaseModel1D class, describing a model of how entropy changes with radius -
        this one is almost constant in the core and behaves as a power-law in the outskirts.
        """
        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        poss_y_units = [Unit('keV cm^2')]
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

        k_zero_starts = [Quantity(75, 'keV cm^2')]
        k_100kpc = [Quantity(100, 'keV cm^2')]
        alpha = Quantity(1)

        start_pars = [k_zero_starts[yu_ind], k_100kpc[yu_ind], alpha]

        if cust_start_pars is not None:
            # If the custom start parameters can run this gauntlet without tripping an error then we're all good
            # This method also returns the custom start pars converted to exactly the same units as the default
            start_pars = self.compare_units(cust_start_pars, start_pars)

        kz_priors = [{'prior': Quantity([1, 300], 'keV cm^2'), 'type': 'uniform'}]
        k100_priors = [{'prior': Quantity([1, 500], 'keV cm^2'), 'type': 'uniform'}]
        alpha_priors = {'prior': Quantity([-4, 4]), 'type': 'uniform'}

        priors = [kz_priors[yu_ind], k100_priors[yu_ind], alpha_priors]

        nice_pars = [r"K$_{0}$", r"K$_{100\rm{kpc}}$", r"\alpha"]
        info_dict = {'author': 'Cavagnolo et al.', 'year': 2009,
                     'reference': 'https://doi.org/10.1088/0067-0049/182/1/12',
                     'general': "A model to fit galaxy cluster radial entropy profiles, made up of a constant core "
                                "combined with a power law that dominates in the outskirts."
                     }

        super().__init__(x_unit, y_unit, start_pars, priors, 'coreconst_power_entropy',
                         'Constant Core and Power-Law Profile', nice_pars, 'Gas Entropy', info_dict)

    @staticmethod
    def model(x: Quantity, k_zero: Quantity, k_100kpc: Quantity, alpha: Quantity) -> Quantity:
        """
        The model function for the constant-core and power-law entropy model.

        :param Quantity x: The radii to calculate y values for.
        :param Quantity k_zero: Parameter quantifying the typical excess of core entropy above the best fitting
            power-law found at larger radii.
        :param Quantity k_100kpc: A normalization for entropy at 100kpc.
        :param Quantity alpha: The power law index.
        :return: The y values corresponding to the input x values.
        :rtype: Quantity
        """
        result = k_zero + (k_100kpc * (x / Quantity(100, 'kpc'))**alpha)

        return result


# So that things like fitting functions can be written generally to support different models
# "power":
ENTROPY_MODELS = {"coreconst_power_entropy": CoreConstPowerEntropy}
ENTROPY_MODELS_PUB_NAMES = {n: m().publication_name for n, m in ENTROPY_MODELS.items()}
ENTROPY_MODELS_PAR_NAMES = {n: m().par_publication_names for n, m in ENTROPY_MODELS.items()}