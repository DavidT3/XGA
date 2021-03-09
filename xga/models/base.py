#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 09/03/2021, 18:23. Copyright (c) David J Turner

import inspect
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Union, List, Dict
from warnings import warn

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError
from matplotlib import pyplot as plt
from scipy.misc import derivative
from tabulate import tabulate


class BaseModel1D(metaclass=ABCMeta):
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

        # In much the same way as I create storage keys for spectra, this should uniquely identify the model name
        #  and the start parameters (which cannot be changed after declaration) - when stored in profiles this will
        #  allow multiple fits to the same model but with different start points to be stored together
        self._storage_key = self._name + '_' + '_'.join([str(p) for p in start_pars])

        # Parameter distributions will be stored in this list, as generated by some external fitting process
        #  that way this model can generate random realisations of itself
        self._par_dists = [Quantity([], p.unit) for p in self._model_pars]

        # Any warnings that the user should be aware of from the fitting function can be stored in here
        self._fit_warnings = ''

        # If the fit was performed with an MCMC fitter than it can store an acceptance fraction in the model,
        #  its quite a useful diagnostic
        self._acc_frac = None

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

        return self.model(x, *self._model_pars).to(self._y_unit)

    @staticmethod
    @abstractmethod
    def model(x: Quantity, pars: List[Quantity]) -> Quantity:
        """
        This is where the model function is actually defined, this MUST be overridden by every subclass
        model, hence why I've used the abstract method decorator.

        :param Quantity x: The x-position at which the model should be evaluated.
        :param List[Quantity] pars: The parameters of model to be evaluated.
        :return: The y-value of the model at x.
        """
        raise NotImplementedError("Base Model doesn't have this implemented")

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

        return Quantity(derivative(lambda r: self(Quantity(r, self._x_unit)).value, x.value, dx.value, n=order),
                        self._y_unit / np.power(self._x_unit, order))

    # def inverse_abel(self, x: Quantity) -> Quantity:
    #     """
    #     Calculates the inverse abel transform of the model using numerical methods. This method will be overridden
    #     in models that have an analytical solution to the inverse abel transform.
    #
    #     :param Quantity x: The x value(s) at which to measure the value of the inverse abel transform of the model.
    #     :return: The value(s) of the inverse abel transformation.
    #     :rtype: Quantity
    #     """
    #     raise NotImplementedError("This method has not yet been written")
    #
    # def integral(self):
    #     raise NotImplementedError("This method has not yet been written, and it may never be, but I am considering"
    #                               " adding this feature to this class.")

    def allowed_prior_types(self):
        """
        Simple method to display the allowed prior types and their expected formats.
        """
        table_data = [[self._prior_types[i], self._prior_type_format[i]] for i in range(0, len(self._prior_types))]
        headers = ["PRIOR TYPE", "EXPECTED PRIOR FORMAT"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    @staticmethod
    def compare_units(check_pars: List[Quantity], good_pars: List[Quantity]) -> List[Quantity]:
        """
        Simple method that will be used in the inits of subclasses to make sure that any custom start
        values passed in by the user match the expected units of the default start parameters for that model.

        :param List[Quantity] check_pars: The first list of parameters, these are being checked.
        :param List[Quantity] good_pars: The second list of parameters, these are taken as having 'correct'
            units.
        :return: Only if the check pars pass the tests. We return the check pars list but with all elements
            converted to EXACTLY the same units as good_pars, not just equivelant.
        :rtype: List[Quantity]
        """
        if len(check_pars) != len(good_pars):
            raise ValueError("If you pass custom start parameters you must pass a list with one entry for"
                             " each parameter")

        # Check if custom par units are compatible with the correct default start parameter units
        unit_check = np.array([p.unit.is_equivalent(good_pars[p_ind].unit) for p_ind, p in enumerate(check_pars)])
        unit_strings = []
        # Putting together an error string in the case where there are incompatible units
        for uc_ind, uc in enumerate(unit_check):
            if not uc:
                unit_strings.append("{p} != {e}".format(p=check_pars[uc_ind].unit.to_string(),
                                                        e=good_pars[uc_ind].unit.to_string()))
        which_units = ', '.join(unit_strings)

        # If that string is not empty then some of the units are buggered and we throw an error
        if which_units != "":
            raise UnitConversionError("The custom start parameters which have been passed can't all be converted to "
                                      "the expected units; " + which_units)

        # If we get this far though then we know we're all good, so we just convert the parameters to
        #  exactly the same units and return them
        conv_check_pars = [p.to(good_pars[p_ind].unit) for p_ind, p in enumerate(check_pars)]

        return conv_check_pars

    def info(self):
        """
        A method that gives some information about this particular model.
        """
        headers = [self.publication_name, '']
        # ugly_pars = ", ".join([p.name for p in list(inspect.signature(self.model).parameters.values())[1:]])
        ugly_pars = ""
        cur_length = 0
        for p in list(inspect.signature(self.model).parameters.values())[1:]:
            if cur_length > 70:
                ugly_pars += '\n'
                cur_length = 0

            if ugly_pars == "":
                next_par = '{}'.format(p.name)
            else:
                next_par = ', {}'.format(p.name)
            cur_length += len(next_par)
            ugly_pars += next_par

        par_units = ", ".join([u.to_string() for u in self.par_units])

        data = [['DESCRIBES', self.describes], ['UNIT', self._y_unit.to_string()], ['PARAMETERS', ugly_pars],
                ['PARAMETER UNITS', par_units], ["AUTHOR", self._info['author']], ["YEAR", self._info['year']],
                ["PAPER", self._info['reference']], ['INFO', self._info['general']]]
        print(tabulate(data, headers=headers, tablefmt='fancy_grid'))

    def par_dist_view(self, bins: Union[str, int] = 'auto', colour: str = "lightslategrey"):
        """
        Very simple method that allows you to view the parameter distributions that have been added to this
        model. The model parameter and uncertainties are indicated with red lines, highlighting the value
        and enclosing the 1sigma confidence region.

        :param Union[str, int] bins: Equivelant to the plt.hist bins argument, set either the number of bins
            or the algorithm to decide on the number of bins.
        :param str colour: Set the colour of the histogram.
        """
        # Check if there are parameter distributions associated with this model
        if len(self._par_dists[0] != 0):
            # Set up the figure
            figsize = (6, 5*self.num_pars)
            fig, ax_arr = plt.subplots(ncols=1, nrows=self.num_pars, figsize=figsize)

            # Iterate through the axes and plot the histograms
            for ax_ind, ax in enumerate(ax_arr):
                # Add histogram
                ax.hist(self.par_dists[ax_ind].value, bins=bins, color=colour)
                # Add parameter value as a solid red line
                ax.axvline(self.model_pars[ax_ind].value, color='red')
                # Read out the errors
                err = self.model_par_errs[ax_ind]
                # Depending how many entries there are per parameter in the error quantity depends how we plot them
                if err.isscalar:
                    ax.axvline(self.model_pars[ax_ind].value-err.value, color='red', linestyle='dashed')
                    ax.axvline(self.model_pars[ax_ind].value+err.value, color='red', linestyle='dashed')
                elif not err.isscalar and len(err) == 2:
                    ax.axvline(self.model_pars[ax_ind].value - err[0].value, color='red', linestyle='dashed')
                    ax.axvline(self.model_pars[ax_ind].value + err[1].value, color='red', linestyle='dashed')
                else:
                    raise ValueError("Parameter error has three elements in it!")
                ax.set_xlabel(self.par_publication_names[ax_ind])

            # And show the plot
            plt.tight_layout()
            plt.show()
        else:
            warn("You have not added parameter distributions to this model")

    def view(self, radii: Quantity = None, xscale: str = 'log', yscale: str = 'log', figsize: tuple = (8, 8),
             colour: str = "black"):
        """
        Very simple view method to visualise XGA models with the current parameters.

        :param Quantity radii: Radii at which to calculate points to plot, doesn't need to be set if the model has
            x limits defined.
        :param str xscale: The scale to apply to the x-axis, default is log.
        :param str yscale: The scale to apply to the y-axis, default is log.
        :param tuple figsize: The size of figure to be set up.
        :param str colour: The colour that the line in the plot should be.
        """
        if radii is None and self.x_lims is not None:
            radii = Quantity(np.linspace(*self._x_lims.value, 100), self._x_unit)
        elif radii is None and self._x_lims is None:
            raise ValueError("You did not set x-limits for this model, so you must pass radii values to the"
                             " view method.")

        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        plt.plot(radii, self(radii), color=colour)

        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xlim([radii.value.min(), radii.value.max()])

        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + self._x_unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + self._y_unit.to_string("latex").strip("$") + r"\right]$"

        plt.xlabel('x {}'.format(x_unit), fontsize=12)
        y_lab = self.describes + ' {}'.format(y_unit)
        plt.ylabel(y_lab, fontsize=12)
        plt.title(self.publication_name, fontsize=16)
        plt.tight_layout()
        plt.show()

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
    def model_par_errs(self) -> List[Quantity]:
        """
        Property that returns the uncertainties on the current parameters of the model, by default these will
        be zero as the default model_pars are the same as the start_pars.

        :return: A list of astropy quantities representing the uncertainties on the parameters of this model.
        :rtype: List[Quantity]
        """
        return self._model_par_errs

    @model_par_errs.setter
    def model_par_errs(self, new_vals: List[Quantity]):
        """
        Property that allows the current parameter uncertainties of the model to be set. Quantities representing
        uncertainties may have one or two entries, with single element quantities assumed to represent
        1σ gaussian errors, and double element quantities representing confidence limits in the 68th
        percentile region.

        :param List[Quantity] new_vals: A list of astropy quantities representing the new uncertainties
            on the parameters of this model.
        """
        if len(new_vals) != self._num_pars:
            raise ValueError("This model takes {t} parameters, the list you passed contains "
                             "{c}".format(t=self._num_pars, c=len(new_vals)))
        elif not all([p.unit == self._model_pars[p_ind].unit for p_ind, p in enumerate(new_vals)]):
            raise UnitConversionError("All new parameter uncertainties must have the same unit as the "
                                      "old parameters.")

        self._model_par_errs = new_vals

    @property
    def start_pars(self) -> List[Quantity]:
        """
        Property that returns the current start parameters of the model, by which I mean the values that
        certain types of fitting function will use to start their fit.

        :return: A list of astropy quantities representing the values of the start parameters of this model.
        :rtype: List[Quantity]
        """
        return self._start_pars

    @property
    def unitless_start_pars(self) -> np.ndarray:
        """
        Returns sanitised start parameters which are floats rather than astropy quantities, sometimes necessary
        for fitting methods.

        :return: Array of floats representing model start parameters.
        :rtype: np.ndarray
        """
        return np.array([p.value for p in self.start_pars])

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

    @x_lims.setter
    def x_lims(self, new_val: Quantity):
        """
        Property to set the x limits within which the model is considered valid

        :param Quantity new_val: The new x-limits, first element lower, second element upper.
        """
        if not new_val.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("The x-axis unit of this model is {e}, you have passed limits in "
                                      "{p}.".format(e=self._x_unit.to_string(), p=new_val.unit.to_string()))
        elif len(new_val) != 2:
            raise ValueError("The new quantity for the x-axis limits must have two entries.")

        self.x_lims = new_val

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

    @property
    def storage_key(self) -> str:
        """
        Returns a storage key for this model that is based upon the model name and start parameters, used
        for placing a model instance in a profile storage structure.

        :return: String storage key.
        :rtype: str
        """
        return self._storage_key

    @property
    def num_pars(self) -> int:
        """
        Property getter for the number of parameters associated with this model.

        :return: Number of parameters.
        :rtype: int
        """
        return self._num_pars

    @property
    def par_dists(self) -> List[Quantity]:
        """
        A property that returns the currently stored distributions for the model parameters, by default these
        will be empty quantities as no fitting will have occurred. Once a fit has been performed involving the
        model however, the distributions can be set externally.

        :return: A list of astropy quantities containing parameter distributions for all model parameters.
        :rtype: List[Quantity]
        """
        return self._par_dists

    @par_dists.setter
    def par_dists(self, new_vals: List[Quantity]):
        if len(new_vals) != len(self._par_dists):
            raise ValueError("The new list of parameter distributions must have an entry for each "
                             "model parameter.")
        elif not all([p.unit.is_equivalent(self._par_units[p_ind]) for p_ind, p in enumerate(new_vals)]):
            raise UnitConversionError("Some of the new par distributions do not have the correct units.")
        elif any([p.isscalar for p in new_vals]):
            raise ValueError("Parameter distributions cannot be scalar astropy quantities, that doesn't make sense.")
        else:
            # Just making absolutely sure they're in exactly the units we expect
            new_vals = [p.to(self._par_units[p_ind]) for p_ind, p in enumerate(new_vals)]

        self._par_dists = new_vals

    @property
    def fit_warning(self) -> str:
        """
        Returns any warnings generated by a fitting function that acted upon this model.

        :return: A string containing warnings.
        :rtype: str
        """
        return self._fit_warnings

    @fit_warning.setter
    def fit_warning(self, new_val: str):
        """
        Property setter to add warnings from a fit to the model.

        :param str new_val: Fit warning to add.
        """
        # If there is already a warning then just add a separator before the new one
        if self._fit_warnings != "":
            self._fit_warnings += ", "

        self._fit_warnings += new_val

    @property
    def acceptance_fraction(self) -> int:
        """
        Property getter for the acceptance fraction of an MCMC fit (if one has been associated with this model).

        :return: The acceptance fraction.
        :rtype: int
        """
        return self._acc_frac

    @acceptance_fraction.setter
    def acceptance_fraction(self, new_val: int):
        """
        Setter for the acceptance fraction of an MCMC fit run on this model. If an ensemble sampler
        (like emcee) was used then this will be the average acceptance fraction of all the chains.

        :param int new_val: The new acceptance fraction to add.
        """
        self._acc_frac = new_val








