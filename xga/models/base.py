#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 06/03/2021, 09:41. Copyright (c) David J Turner

import inspect
from copy import deepcopy
from typing import Union, List, Dict
from warnings import warn

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError
from scipy.misc import derivative
from tabulate import tabulate


class BaseModel1D:
    """
    The superclass of XGA's 1D models, with base functionality implemented, including the numerical methods for
    calculating derivatives and abel transforms which can be overwritten by subclasses if analytical solutions
    are available. The BaseModel class shouldn't be instantiated by itself, as it won't do anything.
    """
    def __init__(self, x_unit: Union[Unit, str], y_unit: Union[Unit, str], start_pars: List[Quantity],
                 par_priors: List[Dict[str, Union[Quantity, str]]], model_name: str, model_pub_name: str,
                 par_pub_names: List[str], describes: str, info: dict, x_lims: Quantity = None):
        """
        Initialisation method for the base model class, just sets up all the necessary attributes and does some
        checks on the passed in parameters.

        :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
            representation or an astropy unit object.
        :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
            representation or an astropy unit object.
        :param List[Quantity] start_pars: The start values of the model parameters for any fitting function that
            used start values.
        :param List[Dict[str, Union[Quantity, str]]] par_priors: The priors on the model parameters, for any
            fitting function that uses them.
        :param str model_name: The simple name of the particular model, e.g. 'beta'.
        :param str model_pub_name: A smart name for the model that might be used in a publication
            plot, e.g. 'Beta Profile'
        :param List[str] par_pub_names: The names of the parameters of the model, as they should be used in plots
            for publication. As matplotlib supports LaTeX formatting for labels these should use $$ syntax.
        :param str describes: An identifier for the type of data this model describes, e.g. 'Surface Brightness'.
        :param dict info: A dictionary with information about the model, used by the info() method. Can hold
            a general description, reference, authors etc.
        :param Quantity x_lims: Upper and lower limits outside of which the model may not be valid, to be passed as
            a single non-scalar astropy quantity, with the first entry being the lower limit and the second entry
            being the upper limit. Default is None.
        """
        # These are the prior types that XGA currently understands
        self._prior_types = ['uniform']
        # This is used by the allowed_prior_types method to populate the table explaining what info should
        #  be supplied for each prior type
        self._prior_type_format = ['Quantity([LOWER, UPPER], UNIT)']

        # If a string representation of a unit was passed then we make it an astropy unit
        if isinstance(x_unit, str):
            x_unit = Unit(x_unit)
        if isinstance(y_unit, str):
            y_unit = Unit(y_unit)

        # Just saving the expected units to attributes, they also have matching properties for the user
        #  to retrieve them
        self._x_unit = x_unit
        self._y_unit = y_unit

        # A list of starting values for the parameters of the model
        self._start_pars = start_pars

        # These will be set AFTER a fit has been performed to a model, and the model class instance will then
        #  describe that fit. Initially however they are the same as the start pars
        self._model_pars = deepcopy(start_pars)
        self._model_par_errs = [Quantity(0, p.unit) for p in self._model_pars]

        # The expected number of parameters of this model.
        self._num_pars = len(self._model_pars)
        # This will be a list of the units that the model parameters have
        self._par_units = [p.unit for p in self._model_pars]

        # A list of priors for the parameters of the model. Each entry in this list will be a dictionary with
        #  two keys 'prior' and 'type'. The value for prior will be an astropy quantity, and the value for type
        #  will be a prior type (so uniform, gaussian, etc.)
        self._par_priors = par_priors

        # Just checking that the units and shape of xlim make sense
        if x_lims is not None and not x_lims.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("You have passed x-limits with a unit ({p}) that cannot be converted to the "
                                      "set x-unit of this model ({e})".format(p=x_lims.unit.to_string(),
                                                                              e=self._x_unit.to_string()))
        elif x_lims is not None and x_lims.isscalar:
            raise ValueError("The xlim argument should be a non-scalar astropy quantity, with a lower and upper limit.")
        elif x_lims is not None and x_lims.shape != (2,):
            raise ValueError("The quantity passed for xlim should have a lower and upper limit, with a shape of (2,)")
        elif x_lims is not None:
            xlim = x_lims.to(self._x_unit)

        # And setting xlim attribute, it is allowed to be None
        self._x_lims = x_lims

        # Setting up attributes to store the names of the model and its parameters
        self._name = model_name
        self._pretty_name = model_pub_name
        if len(par_pub_names) != self._num_pars:
            raise ValueError("The par_pub_names list should have an entry for every parameter of the model")
        self._pretty_par_names = par_pub_names

        # This sets up the attribute to store what this model describes (e.g. surface brightness)
        self._describes = describes
        # This dictionary gives information about the model, have to make sure required keys are present
        required = ['general', 'reference', 'author', 'year']
        if any([k not in info for k in required]):
            raise KeyError("The following keys must be present in the info dictionary: "
                           "{r}".format(r=', '.join(required)))
        else:
            self._info = info

    def __call__(self, x: Quantity) -> Quantity:
        """
        This method gets run when an instance of a particular model class gets called (i.e. an x-value is
        passed in). As the model stores parameter values it only needs an x-value at which to evaluate the
        output and return a value.

        :param Quantity x: The x-position at which the model should be evaluated.
        :return: The y-value of the model at x.
        :rtype: Quantity
        """
        if not x.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("You have passed an x value in units of {p}, but this model expects units of "
                                      "{e}".format(p=x.unit.to_string(), e=self._x_unit.to_string()))
        else:
            # Just to be sure its in exactly the right units
            x = x.to(self._x_unit)

        if self._x_lims is not None and (np.any(x < self._x_lims[0]) or np.any(x > self._x_lims[1])):
            warn("Some x values are outside of the x-axis limits for this model, results may not be trustworthy.")

        return self.model(x, *self._pars)

    @staticmethod
    def model(x: Quantity, pars: List[Quantity]) -> Quantity:
        """
        This is where the model function is actually defined, this MUST be overridden by every subclass model.

        :param Quantity x: The x-position at which the model should be evaluated.
        :param List[Quantity] pars: The parameters of model to be evaluated.
        :return: The y-value of the model at x.
        """
        return

    def derivative(self, x: Quantity, dx: Quantity) -> Quantity:
        """
        Calculates a numerical derivative of the model at the specified x value, using the specified dx
        value. This method will be overridden in models that have an analytical solution to their first
        derivative, in which case the dx value will become irrelevant.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: The dx value to use during the calculation.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        return self.nth_derivative(x, dx, 1)

    def nth_derivative(self, x: Quantity, dx: Quantity, order: int) -> Quantity:
        """
        A method to calculate the nth order derivative of the model using a numerical method.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: The dx value to use during the calculation.
        :param int order: The order of the desired derivative.
        :return: The value(s) of the nth order derivative of the model at x.
        :rtype: Quantity
        """
        # Just checking that the units of x and dx aren't silly
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

        return derivative(self.model, x, dx, order=order)

    def inverse_abel(self, x: Quantity) -> Quantity:
        """
        Calculates the inverse abel transform of the model using numerical methods. This method will be overridden
        in models that have an analytical solution to the inverse abel transform.

        :param Quantity x: The x value(s) at which to measure the value of the inverse abel transform of the model.
        :return: The value(s) of the inverse abel trans
        :rtype: Quantity
        """
        raise NotImplementedError("This method has not yet been written")

    def integral(self):
        raise NotImplementedError("This method has not yet been written, and it may never be, but I am considering"
                                  " adding this feature to this class.")

    def allowed_prior_types(self):
        """
        Simple method to display the allowed prior types and their expected formats.
        """
        table_data = [[self._prior_types[i], self._prior_type_format[i]] for i in range(0, len(self._prior_types))]
        headers = ["PRIOR TYPE", "EXPECTED PRIOR FORMAT"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    def info(self):
        """
        A method that gives some information about this particular model.
        """
        headers = [self.publication_name, '']
        ugly_pars = ", ".join([p.name for p in list(inspect.signature(self.model).parameters.values())[1:]])
        data = [['Describes:', self.describes], ['Parameters:', ugly_pars], ["Author:", self._info['author']],
                ["Year:", self._info['year']], ["Paper:", self._info['reference']], [self._info['info']]]
        tabulate(data, headers=headers, tablefmt='fancy_grid')

    @property
    def model_pars(self) -> List[Quantity]:
        """
        Property that returns the current parameters of the model, by default they are the same as the
        parameter start values.

        :return: A list of astropy quantities representing the values of the parameters of this model.
        :rtype: List[Quantity]
        """
        return self._model_pars

    @model_pars.setter
    def model_pars(self, new_vals: List[Quantity]):
        """
        Property that allows the current parameter values of the model to be set.

        :param List[Quantity] new_vals: A list of astropy quantities representing the new values
            of the parameters of this model.
        """
        # Need to check that units are the same as the original parameters
        if len(new_vals) != self._num_pars:
            raise ValueError("This model takes {t} parameters, the list you passed contains "
                             "{c}".format(t=self._num_pars, c=len(new_vals)))
        elif not all([p.unit == self._model_pars[p_ind].unit for p_ind, p in enumerate(new_vals)]):
            raise UnitConversionError("All new parameters must have the same unit as the old parameters.")
        self._model_pars = new_vals

    @property
    def start_pars(self) -> List[Quantity]:
        """
        Property that returns the current start parameters of the model, by which I mean the values that
        certain types of fitting function will use to start their fit.

        :return: A list of astropy quantities representing the values of the start parameters of this model.
        :rtype: List[Quantity]
        """
        return self._start_pars

    @start_pars.setter
    def start_pars(self, new_vals: List[Quantity]):
        """
        Property that allows the current start parameter values of the model to be set.

        :param List[Quantity] new_vals: A list of astropy quantities representing the new values
            of the start parameters of this model.
        """
        if len(new_vals) != self._num_pars:
            raise ValueError("This model takes {t} parameters, the list you passed contains "
                             "{c}".format(t=self._num_pars, c=len(new_vals)))
        elif not all([p.unit == self._start_pars[p_ind].unit for p_ind, p in enumerate(new_vals)]):
            raise UnitConversionError("All new start parameters must have the same unit as the old start parameters")

        self._start_pars = new_vals

    @property
    def par_priors(self) -> List[Dict[str, Union[Quantity, str]]]:
        """
        Property that returns the current priors on parameters of the model, these will be used by any
        fitting function that sets priors on parameters. Each entry in this list will be a dictionary with
        two keys 'prior' and 'type'. The value for prior will be an astropy quantity, and the value for type
        will be a prior type (so uniform, gaussian, etc.)

        :return: A list of astropy quantities representing the values of the start parameters of this model.
        :rtype: List[Quantity]
        """
        return self._par_priors

    @par_priors.setter
    def par_priors(self, new_vals: List[Dict[str, Union[Quantity, str]]]):
        """
        Property setter for the parameter priors of this model. The user should supply a list of dictionaries
        which describe the characteristic values of the prior and its type, for instance:
        [{'prior': Quantity([1, 1000], 'kpc'), 'type': 'uniform'}, {'prior': Quantity([0, 3]), 'type': 'uniform'}]

        Which describes a uniform prior between 1 and 1000kpc on the first prior, and a uniform prior
        between 0 and 3 on the second. The types of prior supported by XGA can be accessed using the
        allowed_prior_types() method.

        :param List[Dict[str, Union[Quantity, str]]] new_vals:
        """
        # new_vals should have one entry per parameter of the model
        if len(new_vals) != self._num_pars:
            raise ValueError("This model takes {t} parameters, the list you passed contains "
                             "{c}".format(t=self._num_pars, c=len(new_vals)))

        # Need to iterate through all the priors and perform checks to make sure they can be allowed
        for prior_ind, prior in enumerate(new_vals):
            # Need to make sure the dictionary describing the prior on a parameter has the entries we expect
            if 'prior' not in prior or 'type' not in prior:
                raise KeyError("All entries in prior list must be dictionaries, with 'prior' and 'type' keys.")
            # Check that the type of prior is currently supported
            elif prior['type'] not in self._prior_types:
                allowed = ", ".format(self._prior_types)
                raise ValueError("Priors of type {t} are not supported, currently supported types are; "
                                 "{a}".format(t=prior['type'], a=allowed))
            # And finally check the units of the characteristic values of the prior
            elif not prior['prior'].unit.is_equivalent(self._par_priors[prior_ind]["prior"].unit):
                raise UnitConversionError("Cannot convert new prior's {n} to old prior's "
                                          "{o}".format(n=prior['prior'].unit.to_string(),
                                                       o=self._par_priors[prior_ind]["prior"].unit.to_string()))
            else:
                # We make sure that if the prior values are in an equivelant unit to the current prior then
                #  they are converted to the original unit, to be consistent.
                prior['prior'] = prior['prior'].to(self._par_priors[prior_ind]["prior"].unit)
                self._par_priors[prior_ind] = prior

    @property
    def x_unit(self) -> Unit:
        """
        Property to access the expected x-unit of this model.

        :return: Astropy unit of the x values of the model.
        :rtype: Unit
        """
        return self._x_unit

    @property
    def y_unit(self) -> Unit:
        """
        Property to access the expected y-unit of this model.

        :return: Astropy unit of the y values of the model.
        :rtype: Unit
        """
        return self._y_unit

    @property
    def x_lims(self) -> Quantity:
        """
        Property to access the x limits within which the model is considered valid, the default is
        None if no x limits were set for the model on instantiation.

        :return: A non-scalar astropy quantity with two entries, the first is a lower limit, and the second an
            upper limit. The default is None if no x limits were set.
        :rtype: Quantity
        """
        return self._x_lims

    @property
    def par_units(self) -> List[Unit]:
        """
        A list of units for the parameters of this model.

        :return: A list of astropy units.
        :rtype: List[Unit]
        """
        return self._par_units

    @property
    def name(self) -> str:
        """
        Property getter for the simple name of the model, which the user would enter when requesting
        a particular model to be fit to a profile, for instance.

        :return: String representation of the simple name of the model.
        :rtype: str
        """
        return self._name

    @property
    def publication_name(self) -> str:
        """
        Property getter for the publication name of the model, which is what would be added in a plot
        meant for publication, for instance.

        :return: String representation of the publication (i.e. pretty) name of the model.
        :rtype: str
        """
        return self._pretty_name

    @property
    def par_publication_names(self) -> List[str]:
        """
        Property getter for the publication names of the model parameters. These would be used in a plot
        for instance, and so can make use of Matplotlib's ability to render LaTeX math code.

        :return: List of string representation of the publication (i.e. pretty) names of the model parameters.
        :rtype: List[str]
        """
        return self._pretty_par_names

    @property
    def describes(self) -> str:
        """
        A one or two word description of the type of data this model describes.

        :return: A string description.
        :rtype: str
        """
        return self._describes










