#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 05/03/2021, 16:39. Copyright (c) David J Turner

from typing import Union, List, Dict
from warnings import warn

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError
from scipy.misc import derivative


class BaseModel1D:
    """
    The superclass of XGA's 1D models, with base functionality implemented, including the numerical methods for
    calculating derivatives and abel transforms which can be overwritten by subclasses if analytical solutions
    are available. The BaseModel class shouldn't be instantiated by itself, as it won't do anything.
    """
    def __init__(self, x_unit: Union[Unit, str], y_unit: Union[Unit, str], xlim: Quantity = None):
        """
        Initialisation method for the base model class, just sets up all the necessary attributes and does some
        checks on the passed in parameters.

        :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
            representation or an astropy unit object.
        :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
            representation or an astropy unit object.
        :param Quantity xlim: Upper and lower limits outside of which the model may not be valid, to be passed as
            a single non-scalar astropy quantity, with the first entry being the lower limit and the second entry
            being the upper limit. Default is None.
        """
        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        # Just saving the expected units to attributes, they also have matching properties for the user
        #  to retrieve them
        self._x_unit = x_unit
        self._y_unit = y_unit

        # The expected number of parameters of this model.
        self._num_pars = 0
        # This will be a list of the units that the model parameters have
        self._par_units = []

        # A list of starting values for the parameters of the model
        self._start_pars = []
        # A list of priors for the parameters of the model. Each entry in this list will be a dictionary with
        #  two keys 'prior' and 'type'. The value for prior will be an astropy quantity, and the value for type
        #  will be a prior type (so uniform, gaussian, etc.)
        self._par_priors = []

        # These will be set AFTER a fit has been performed to a model, and the model class instance will then
        #  describe that fit
        self._model_pars = []
        self._model_par_errs = []

        # Just checking that the units and shape of xlim make sense
        if xlim is not None and not xlim.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("You have passed x-limits with a unit ({p}) that cannot be converted to the "
                                      "set x-unit of this model ({e})".format(p=xlim.unit.to_string(),
                                                                              e=self._x_unit.to_string()))
        elif xlim is not None and xlim.isscalar:
            raise ValueError("The xlim argument should be a non-scalar astropy quantity, with a lower and upper limit.")
        elif xlim is not None and xlim.shape != (2,):
            raise ValueError("The quantity passed for xlim should have a lower and upper limit, with a shape of (2,)")
        elif xlim is not None:
            xlim = xlim.to(self._x_unit)

        # And setting xlim attribute, it is allowed to be None
        self._xlim = xlim

        # These are the prior types that XGA currently understands
        self._prior_types = ['uniform']

    def __call__(self, x: Quantity):
        if not x.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("You have passed an x value in units of {p}, but this model expects units of "
                                      "{e}".format(p=x.unit.to_string(), e=self._x_unit.to_string()))
        else:
            # Just to be sure its in exactly the right units
            x = x.to(self._x_unit)

        if self._xlim is not None and (np.any(x < self._xlim[0]) or np.any(x > self._xlim[1])):
            warn("Some x values are outside of the x-axis limits for this model, results may not be trustworthy.")

        return self.model_function(x, *self._pars)

    @staticmethod
    def model_function(x: Quantity, pars: List[Quantity]):
        return

    def derivative(self, x: Quantity, dx: Quantity):
        return self.nth_derivative(x, dx, 1)

    def nth_derivative(self, x: Quantity, dx: Quantity, order: int):
        if not x.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("You have passed an x value in units of {p}, but this model expects units of "
                                      "{e}".format(p=x.unit.to_string(), e=self._x_unit.to_string()))
        else:
            # Just to be sure its in exactly the right units
            x = x.to(self._x_unit)

        if not dx.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("You have passed a dx value in units of {p}, but this model expects units of "
                                      "{e}".format(p=dx.unit.to_string(), e=self._x_unit.to_string()))
        else:
            dx = dx.to(self._x_unit)

        return derivative(self.model_function, x, dx, order=order)

    def inverse_abel(self, x):
        pass

    def integral(self):
        pass

    @property
    def model_pars(self) -> List[Quantity]:
        return self._model_pars

    @model_pars.setter
    def model_pars(self, new_vals: List[Quantity]):
        if len(new_vals) != self._num_pars:
            raise ValueError("This model takes {t} parameters, the list you passed contains "
                             "{c}".format(t=self._num_pars, c=len(new_vals)))
        elif not all([p.unit == self._model_pars[p_ind].unit for p_ind, p in enumerate(new_vals)]):
            raise UnitConversionError("All new parameters must have the same unit as the old parameters.")
        self._model_pars = new_vals

    @property
    def start_pars(self) -> List[Quantity]:
        return self._start_pars

    @start_pars.setter
    def start_pars(self, new_vals: List[Quantity]):
        if len(new_vals) != self._num_pars:
            raise ValueError("This model takes {t} parameters, the list you passed contains "
                             "{c}".format(t=self._num_pars, c=len(new_vals)))
        elif not all([p.unit == self._start_pars[p_ind].unit for p_ind, p in enumerate(new_vals)]):
            raise UnitConversionError("All new start parameters must have the same unit as the old start parameters")

        self._start_pars = new_vals

    @property
    def par_priors(self) -> List[Dict[str, Union[Quantity, str]]]:
        return self._par_priors

    @par_priors.setter
    def par_priors(self, new_vals: List[Dict[str, Union[Quantity, str]]]):
        for prior_ind, prior in enumerate(new_vals):
            if len(prior) != self._num_pars:
                raise ValueError("This model takes {t} parameters, the list you passed contains "
                                 "{c}".format(t=self._num_pars, c=len(new_vals)))
            elif 'prior' not in prior or 'type' not in prior:
                raise KeyError("All entries in prior list must be dictionaries, with 'prior' and 'type' keys.")
            elif prior['type'] not in self._prior_types:
                allowed = ", ".format(self._prior_types)
                raise ValueError("Priors of type {t} are not supported, currently supported types are; "
                                 "{a}".format(t=prior['type'], a=allowed))
            elif not prior['prior'].unit.is_equivalent(self._par_priors[prior_ind]["prior"].unit):
                raise UnitConversionError("Cannot convert new prior's {n} to old prior's "
                                          "{o}".format(n=prior['prior'].unit.to_string(),
                                                       o=self._par_priors[prior_ind]["prior"].unit.to_string()))
            else:
                prior['prior'] = prior['prior'].to(self._par_priors[prior_ind]["prior"].unit)
                self._par_priors[prior_ind] = prior

    @property
    def x_unit(self) -> Unit:
        return self._x_unit

    @property
    def y_unit(self) -> Unit:
        return self._y_unit

    @property
    def xlim(self) -> Quantity:
        return self._xlim

    @property
    def par_units(self) -> List[Unit]:
        return self._par_units











