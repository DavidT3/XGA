#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 08/06/2023, 22:40. Copyright (c) The Contributors

import inspect
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Union, List, Dict
from warnings import warn

import emcee as em
import numpy as np
from abel.basex import basex_transform
from abel.dasch import onion_peeling_transform, two_point_transform, three_point_transform
from abel.direct import direct_transform
from abel.hansenlaw import hansenlaw_transform
from abel.onion_bordas import onion_bordas_transform
from astropy.units import Quantity, Unit, UnitConversionError
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.misc import derivative
from tabulate import tabulate

from ..exceptions import XGAFitError


class BaseModel1D(metaclass=ABCMeta):
    """
    The superclass of XGA's 1D models, with base functionality implemented, including the numerical methods for
    calculating derivatives and abel transforms which can be overwritten by subclasses if analytical solutions
    are available. The BaseModel class shouldn't be instantiated by itself, as it won't do anything.

    :param Unit/str x_unit: The unit of the x-axis of this model, kpc for instance. May be passed as a string
        representation or an astropy unit object.
    :param Unit/str y_unit: The unit of the output of this model, keV for instance. May be passed as a string
        representation or an astropy unit object.
    :param List[Quantity] start_pars: The start values of the model parameters for any fitting function that
        used start values.
    :param List[Dict[str, Quantity/str]] par_priors: The priors on the model parameters, for any
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

    def __init__(self, x_unit: Union[Unit, str], y_unit: Union[Unit, str], start_pars: List[Quantity],
                 par_priors: List[Dict[str, Union[Quantity, str]]], model_name: str, model_pub_name: str,
                 par_pub_names: List[str], describes: str, info: dict, x_lims: Quantity = None):
        """
        Initialisation method for the base model class, just sets up all the necessary attributes and does some
        checks on the passed in parameters.
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

        # I also add this attribute to store the parameter names as they appear in the model function, though
        #  they are only found if someone accesses the par_names property
        self._par_names = None

        # This sets up the attribute to store what this model describes (e.g. surface brightness)
        self._describes = describes
        # This dictionary gives information about the model, have to make sure required keys are present
        required = ['general', 'reference', 'author', 'year']
        if any([k not in info for k in required]):
            raise KeyError("The following keys must be present in the info dictionary: "
                           "{r}".format(r=', '.join(required)))
        else:
            self._info = info

        # Parameter distributions will be stored in this list, as generated by some external fitting process
        #  that way this model can generate random realisations of itself
        self._par_dists = [Quantity([], p.unit) for p in self._model_pars]

        # Any warnings that the user should be aware of from the fitting function can be stored in here
        self._fit_warnings = ''

        # And another attribute to hold whether the external fit method considers the fit run using
        #  this model to be successful or not. This is also reflected in how the model is stored in a profile
        #  object, but it feels useful to have the information here as well.
        self._success = None

        # If the fit was performed with an MCMC fitter than it can store an acceptance fraction in the model,
        #  its quite a useful diagnostic
        self._acc_frac = None

        # If an emcee sampler is used to do the fitting than that sampler can be stored in the model instance
        self._emcee_sampler = None
        # And the number of steps the fitting method decided on for a burn-in region
        self._cut_off = None

        # Allows the storage of the method used to fit the data in the model instance
        self._fit_method = None

        # I'm going to store any volume integral results within the model itself
        self._vol_ints = {'pars': {}, 'par_dists': {}}

        # This attribute stores a reference to a profile that a model instance has been used to fit. If the model
        #  exists in isolation and hasn't been fit through a profile fit() method then it will remain None, but
        #  otherwise I'm storing the profile to avoid one model being fit to two different profiles accidentally.
        #  The BaseProfile1D internal method _model_allegiance sets this
        self._profile = None

    def __call__(self, x: Quantity, use_par_dist: bool = False) -> Quantity:
        """
        This method gets run when an instance of a particular model class gets called (i.e. an x-value is
        passed in). As the model stores parameter values it only needs an x-value at which to evaluate the
        output and return a value. By default it will use the best fit parameters stored in this model, but can
        use the parameter distributions to produce a distribution of values at x.

        :param Quantity x: The x-position at which the model should be evaluated.
        :param bool use_par_dist: Should the parameter distributions be used to calculate model values; this can
            only be used if a fit has been performed using the model instance. Default is False, in which case
            the current parameters will be used to calculate a single value.
        :return: The y-value of the model at x.
        :rtype: Quantity
        """
        if not x.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("You have passed an x value in units of {p}, but this model expects units of "
                                      "{e}".format(p=x.unit.to_string(), e=self._x_unit.to_string()))
        else:
            # Just to be sure it's in exactly the right units
            x = x.to(self._x_unit)

        if self._x_lims is not None and (np.any(x < self._x_lims[0]) or np.any(x > self._x_lims[1])):
            warn("Some x values are outside of the x-axis limits for this model, results may not be trustworthy.")

        # Check whether parameter distributions have been added to this model
        if use_par_dist and len(self._par_dists[0]) == 0:
            raise XGAFitError("No fit has been performed with this model, so there are no parameter distributions"
                              " available.")

        if use_par_dist:
            val = self.get_realisations(x)
        else:
            val = self.model(x[..., None], *self._model_pars).to(self._y_unit)

        return val

    def get_realisations(self, x: Quantity) -> Quantity:
        """
        This method uses the parameter distributions added to this model by a fitting process to generate
        random realisations of this model at a given x-position (or positions).

        :param Quantity x: The x-position(s) at which realisations of the model should be generated from the
            associated parameter distributions. If multiple, non-distribution, radii are to be used, make sure
            to pass them as an (M,), single dimension, astropy quantity, where M is the number of separate radii to
            generate realisations for. To marginalise over a radius distribution when generating realisations, pass
            a multi-dimensional astropy quantity; i.e. for a single set of realisations pass a (1, N) quantity, where
            N is the number of samples in the parameter posteriors, for realisations for M different radii pass a
            (M, N) quantity.
        :return: The model realisations, in a Quantity with shape (len(x), num_samples) if x has multiple
            radii in it (num_samples is the number of samples in the parameter distributions), and (num_samples,) if
            only a single x value is passed.
        :rtype: Quantity
        """
        if not x.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("You have passed an x value in units of {p}, but this model expects units of "
                                      "{e}".format(p=x.unit.to_string(), e=self._x_unit.to_string()))
        else:
            # Just to be sure it's in exactly the right units
            x = x.to(self._x_unit)

        if self._x_lims is not None and (np.any(x < self._x_lims[0]) or np.any(x > self._x_lims[1])):
            warn("Some x values are outside of the x-axis limits for this model, results may not be trustworthy.")

        if x.isscalar or (not x.isscalar and x.ndim == 1):
            realisations = self.model(x[..., None], *self._par_dists)
        else:
            # This case is for marginalising over a radius distribution (or distributions), so in other words we want
            #  distribution(s) of N values out (where N is the number of values in the model parameter posterior
            #  distributions).
            # Note the lack of [..., None] on x here, this is what makes it different to the first part of the if
            #  statement
            realisations = self.model(x, *self._par_dists)

        return realisations

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

    def derivative(self, x: Quantity, dx: Quantity, use_par_dist: bool = False) -> Quantity:
        """
        Calculates a numerical derivative of the model at the specified x value, using the specified dx
        value. This method will be overridden in models that have an analytical solution to their first
        derivative, in which case the dx value will become irrelevant.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: The dx value to use during the calculation.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a derivative
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :return: The calculated slope of the model at the supplied x position(s).
        :rtype: Quantity
        """
        return self.nth_derivative(x, dx, 1, use_par_dist)

    def nth_derivative(self, x: Quantity, dx: Quantity, order: int, use_par_dist: bool = False) -> Quantity:
        """
        A method to calculate the nth order derivative of the model using a numerical method.

        :param Quantity x: The point(s) at which the slope of the model should be measured.
        :param Quantity dx: The dx value to use during the calculation.
        :param int order: The order of the desired derivative.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a derivative
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :return: The value(s) of the nth order derivative of the model at x, either calculated from the current
            best fit parameters, or a distribution.
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

        der_val = Quantity(derivative(lambda r: self(Quantity(r, self._x_unit), use_par_dist).value, x.value,
                                      dx.value, n=order), self._y_unit / np.power(self._x_unit, order))
        return der_val

    def inverse_abel(self, x: Quantity, use_par_dist: bool = False, method: str = 'direct') -> Quantity:
        """
        This method uses numerical methods to generate the inverse abel transform of the model. It may be
        overridden by models that have analytical solutions to the inverse abel transform. All numerical inverse
        abel transform methods are from the pyabel module, and please be aware that in my (limited) experience the
        numerical solutions tend to diverge from analytical solutions at large radii.

        :param Quantity x: The x location(s) at which to calculate the value of the inverse abel transform.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a inverse abel transform
            distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value.
        :param str method: The method that should be used to calculate the values of this inverse abel transform. You
            may pass 'direct', 'basex', 'hansenlaw', 'onion_bordas', 'onion_peeling', 'two_point', or 'three_point'.
        :return: The inverse abel transform result.
        :rtype: Quantity
        """

        # Checking x units to make sure that they are valid
        if not x.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("The input x coordinates cannot be converted to units of "
                                      "{}".format(self._x_unit.to_string()))
        else:
            x = x.to(self._x_unit)

        # Sets up the resolution of the radial spatial sampling
        force_change = False
        if len(set(np.diff(x))) != 1:
            warn("Most numerical methods for the abel transform require uniformly sampled radius values, setting "
                 "the method to 'direct'")
            method = 'direct'
            force_change = True
        else:
            dr = (x[1] - x[0]).value

        # If the user just wants to use the current values of the model parameters then this is what happens
        if not use_par_dist:
            if method == 'direct' and force_change:
                transform_res = direct_transform(self(x).value, r=x.value, backend='python')
            elif method == 'direct' and not force_change:
                transform_res = direct_transform(self(x).value, dr=dr)
            elif method == 'basex':
                transform_res = basex_transform(self(x).value, dr=dr)
            elif method == 'hansenlaw':
                transform_res = hansenlaw_transform(self(x).value, dr=dr)
            elif method == 'onion_bordas':
                transform_res = onion_bordas_transform(self(x).value, dr=dr)
            elif method == 'onion_peeling':
                transform_res = onion_peeling_transform(self(x).value, dr=dr)
            elif method == 'two_point':
                transform_res = two_point_transform(self(x).value, dr=dr)
            elif method == 'three_point':
                transform_res = three_point_transform(self(x).value, dr=dr)
            else:
                raise ValueError("{} is not a recognised inverse abel transform type".format(method))

        # This uses the parameter distributions of this module
        elif use_par_dist:
            realisations = self.get_realisations(x).value
            transform_res = np.zeros(realisations.shape)
            for t_ind in range(0, realisations.shape[1]):
                if method == 'direct' and force_change:
                    transform_res[:, t_ind] = direct_transform(realisations[:, t_ind], r=x.value, backend='python')
                elif method == 'direct' and not force_change:
                    transform_res[:, t_ind] = direct_transform(realisations[:, t_ind], dr=dr)
                elif method == 'basex':
                    transform_res[:, t_ind] = basex_transform(realisations[:, t_ind], dr=dr)
                elif method == 'hansenlaw':
                    transform_res[:, t_ind] = hansenlaw_transform(realisations[:, t_ind], dr=dr)
                elif method == 'onion_bordas':
                    transform_res[:, t_ind] = onion_bordas_transform(realisations[:, t_ind], dr=dr)
                elif method == 'onion_peeling':
                    transform_res[:, t_ind] = onion_peeling_transform(realisations[:, t_ind], dr=dr)
                elif method == 'two_point':
                    transform_res[:, t_ind] = two_point_transform(realisations[:, t_ind], dr=dr)
                elif method == 'three_point':
                    transform_res[:, t_ind] = three_point_transform(realisations[:, t_ind], dr=dr)
                else:
                    raise ValueError("{} is not a recognised inverse abel transform type".format(method))

        transform_res = Quantity(transform_res, self._y_unit / self._x_unit)

        return transform_res

    def volume_integral(self, outer_radius: Quantity, inner_radius: Quantity = None,
                        use_par_dist: bool = False) -> Quantity:
        """
        Calculates a numerical value for the volume integral of the function over a sphere of
        radius outer_radius. The scipy quad function is used. This method can either return a single value
        calculated using the current model parameters, or a distribution of values using the parameter
        distributions (assuming that this model has had a fit run on it).

        This method may be overridden if there is an analytical solution to a particular model's volume
        integration over a sphere.

        The results of calculations with single values of outer and inner radius are stored in the model object
        to reduce processing time if they are needed again, but if a distribution of radii are passed then
        the results will not be stored and will be re-calculated each time.

        :param Quantity outer_radius: The radius to integrate out to. Either a single value or, if you want to
            marginalise over a radius distribution when 'use_par_dist=True', a non-scalar quantity of the same
            length as the number of samples in the parameter posteriors.
        :param Quantity inner_radius: The inner bound of the radius integration. Default is None, which results
            in an inner radius of 0 in the units of outer_radius being used.
        :param bool use_par_dist: Should the parameter distributions be used to calculate a volume
            integral distribution; this can only be used if a fit has been performed using the model instance.
            Default is False, in which case the current parameters will be used to calculate a single value
        :return: The result of the integration, either a single value or a distribution.
        :rtype: Quantity
        """

        def integrand(x: float, pars: List[float]):
            """
            Internal function to wrap the model function.

            :param float x: The x-position currently being evaluated.
            :param List[float] pars: The model parameters
            :return: The integrand value.
            :rtype: float
            """

            return x ** 2 * self.model(x, *pars)

        # This variable just tells the rest of the function whether either the inner or outer radii are actually
        #  a distribution rather than a single value.
        if not inner_radius.isscalar or not outer_radius.isscalar:
            rad_dist = True
        else:
            rad_dist = False

        # This checks to see if inner radius is None (probably how it will be used most of the time), and if
        #  it is then creates a Quantity with the same units as outer_radius
        if inner_radius is None:
            inner_radius = Quantity(0, outer_radius.unit)
        elif inner_radius is not None and not inner_radius.unit.is_equivalent(outer_radius.unit):
            raise UnitConversionError("If an inner_radius Quantity is supplied, then it must be in the same units"
                                      " as the outer_radius Quantity.")

        if (not outer_radius.isscalar or not inner_radius.isscalar) and not use_par_dist:
            raise ValueError("Radius distributions can only be used with use_par_dist set to True.")
        elif not outer_radius.isscalar and len(outer_radius) != len(self.par_dists[0]):
            raise ValueError("The outer_radius distribution must have the same number of entries (currently {rd}) "
                             "as the model posterior distributions ({md}).".format(rd=len(outer_radius),
                                                                                   md=len(self.par_dists[0])))
        elif not inner_radius.isscalar and len(inner_radius) != len(self.par_dists[0]):
            raise ValueError("The inner_radius distribution must have the same number of entries (currently {rd}) "
                             "as the model posterior distributions ({md}).".format(rd=len(inner_radius),
                                                                                   md=len(self.par_dists[0])))

        # Do a basic sanity checks on the radii, they can't be below zero because that doesn't make any sense
        #  physically. Also make sure that outer_radius isn't less than inner_radius
        if (inner_radius.value < 0).any() or (not rad_dist and outer_radius < inner_radius):
            raise ValueError("Both inner_radius and outer_radius must be greater than zero (though inner_radius "
                             "may be None, which is equivalent to zero). Also, outer_radius must be greater than "
                             "inner_radius.")

        # Perform checks on the input outer radius units - don't need to explicitly check the inner radius units
        #  because I've already ensured that they're the same as outer_radius
        if not outer_radius.unit.is_equivalent(self._x_unit):
            raise UnitConversionError("Outer radius cannot be converted to units of "
                                      "{}".format(self._x_unit.to_string()))
        else:
            outer_radius = outer_radius.to(self._x_unit)
            # We already know that this conversion is possible, because I checked that inner_radius units are
            #  equivalent to outer_radius
            inner_radius = inner_radius.to(self._x_unit)

        # Here I just check to see whether this particular integral has been performed already, no sense repeating a
        #  costly-ish calculation if it has. Where the results are stored depends on whether the integral was performed
        #  using the median parameter values or the distributions
        if not use_par_dist and (outer_radius in self._vol_ints['pars'] and
                                 inner_radius in self._vol_ints['pars'][outer_radius]):
            # This makes sure the rest of the code in this function knows that this calculation has already been run
            already_run = True
            integral_res = self._vol_ints['pars'][outer_radius][inner_radius]

        # Equivalent to the above clause but for par distribution results rather than the median single values used
        #  to concisely represent the models
        elif use_par_dist and not rad_dist and (outer_radius in self._vol_ints['par_dists'] and
                                                inner_radius in self._vol_ints['par_dists'][outer_radius]):
            already_run = True
            integral_res = self._vol_ints['par_dists'][outer_radius][inner_radius]

        # Otherwise, this particular integral just hasn't been run
        elif not rad_dist:
            already_run = False
            # In this case I pre-emptively add the outer radius to the dictionary keys, for use later to store
            #  the result. I don't add the inner radius because it will be automatically added
            if use_par_dist:
                self._vol_ints['par_dists'][outer_radius] = {}
            else:
                self._vol_ints['pars'][outer_radius] = {}
        else:
            # In the case where we are using a radius distribution, we still need to set this parameter so
            #  that the calculation is actually run
            already_run = False

        # The user can either request a single value using the current model parameters, or a distribution
        #  using the current parameter distributions (if set)
        if not use_par_dist and not already_run:
            # Runs the volume integral for a sphere for the representative parameter values of this model
            integral_res = 4 * np.pi * quad(integrand, inner_radius.value, outer_radius.value,
                                            args=[p.value for p in self._model_pars])[0]
        elif use_par_dist and len(self._par_dists[0]) != 0 and not already_run:
            # Runs the volume integral for the parameter distributions (assuming there are any) of this model
            unitless_dists = [par_d.value for par_d in self.par_dists]
            integral_res = np.zeros(len(unitless_dists[0]))
            # An unfortunately unsophisticated way of doing this, but stepping through the parameter distributions
            #  one by one.
            for par_ind in range(len(unitless_dists[0])):
                if not outer_radius.isscalar:
                    out_rad = outer_radius[par_ind].value
                else:
                    out_rad = outer_radius.value

                if not inner_radius.isscalar:
                    inn_rad = inner_radius[par_ind].value
                else:
                    inn_rad = inner_radius.value
                integral_res[par_ind] = 4 * np.pi * quad(integrand, inn_rad, out_rad,
                                                         args=[par_d[par_ind] for par_d in unitless_dists])[0]
        elif use_par_dist and len(self._par_dists[0]) == 0 and not already_run:
            raise XGAFitError("No fit has been performed with this model, so there are no parameter distributions"
                              " available.")

        # If there wasn't already a result stored, the integration result is saved in a dictionary
        if not already_run:
            integral_res = Quantity(integral_res, self.y_unit * self.x_unit ** 3)

            if not rad_dist and use_par_dist:
                self._vol_ints['par_dists'][outer_radius][inner_radius] = integral_res
            elif not rad_dist:
                self._vol_ints['pars'][outer_radius][inner_radius] = integral_res

        return integral_res.copy()

    def allowed_prior_types(self, table_format: str = 'fancy_grid'):
        """
        Simple method to display the allowed prior types and their expected formats.
        :param str table_format: The desired format of the allowed models table. This is passed to the
            tabulate module (allowed formats can be found here - https://pypi.org/project/tabulate/), and
            alters the way the printed table looks.
        """
        table_data = [[self._prior_types[i], self._prior_type_format[i]] for i in range(0, len(self._prior_types))]
        headers = ["PRIOR TYPE", "EXPECTED PRIOR FORMAT"]
        print(tabulate(table_data, headers=headers, tablefmt=table_format))

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

    def info(self, table_format: str = 'fancy_grid'):
        """
        A method that gives some information about this particular model.
        :param str table_format: The desired format of the allowed models table. This is passed to the
            tabulate module (allowed formats can be found here - https://pypi.org/project/tabulate/), and
            alters the way the printed table looks.
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
        print(tabulate(data, headers=headers, tablefmt=table_format))

    def predicted_dist_view(self, radius: Quantity, bins: Union[str, int] = 'auto', colour: str = "lightslategrey",
                            figsize: tuple = (6, 5)):
        """
        A simple view method, to visualise the predicted value distribution at a particular radius. Only usable if
        this model has had parameter distributions assigned to it.

        :param Quantity radius: The radius at which you wish to evaluate this model and view the
            predicted distribution.
        :param Union[str, int] bins: Equivelant to the plt.hist bins argument, set either the number of bins
            or the algorithm to decide on the number of bins.
        :param str colour: Set the colour of the histogram.
        :param tuple figsize: The desired dimensions of the figure.
        """
        # Check if there are parameter distributions associated with this model
        if len(self._par_dists[0] != 0):
            # Set up the figure
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()

            ax.hist(self(radius, True).value, bins=bins, color=colour)
            # Add predicted value as a solid red line
            ax.axvline(self(radius).value, color='red')

            cur_unit = self.y_unit
            if cur_unit == Unit(''):
                par_unit_name = ""
            else:
                par_unit_name = r" $\left[" + cur_unit.to_string("latex").strip("$") + r"\right]$"

            ax.set_xlabel(self.describes + ' {}'.format(par_unit_name))

            # And show the plot
            plt.tight_layout()
            plt.show()
        else:
            warn("You have not added parameter distributions to this model")

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
            figsize = (6, 5 * self.num_pars)
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
                    ax.axvline(self.model_pars[ax_ind].value - err.value, color='red', linestyle='dashed')
                    ax.axvline(self.model_pars[ax_ind].value + err.value, color='red', linestyle='dashed')
                elif not err.isscalar and len(err) == 2:
                    ax.axvline(self.model_pars[ax_ind].value - err[0].value, color='red', linestyle='dashed')
                    ax.axvline(self.model_pars[ax_ind].value + err[1].value, color='red', linestyle='dashed')
                else:
                    raise ValueError("Parameter error has three elements in it!")

                cur_unit = err.unit
                if cur_unit == Unit(''):
                    par_unit_name = ""
                else:
                    par_unit_name = r" $\left[" + cur_unit.to_string("latex").strip("$") + r"\right]$"

                ax.set_xlabel(self.par_publication_names[ax_ind] + par_unit_name)

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

        # We also need to make sure any existing integrals are wiped from the model's storage, as they will
        #  no longer be valid
        self._vol_ints['pars'] = {}

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

        # Again must wipe any stored integrals as new parameter distributions will make them invalid
        self._vol_ints['par_dists'] = {}

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

    @property
    def emcee_sampler(self) -> em.EnsembleSampler:
        """
        Property getter for the emcee sampler used to fit this model, if applicable. By default this will be
        None, as the it has to be set externally, as the model won't necessarily be fit by emcee

        :return: The emcee sampler used to fit this model.
        :rtype: em.EnsembleSampler
        """
        return self._emcee_sampler

    @emcee_sampler.setter
    def emcee_sampler(self, new_val: em.EnsembleSampler):
        """
        Property setter for an emcee sampler to be added to this model.

        :param em.EnsembleSampler new_val: The emcee sampler used to fit this model
        """
        self._emcee_sampler = new_val

    @property
    def cut_off(self) -> int:
        """
        Property getter for the number of steps that an MCMC fitting method decided should be removed for burn-in.
         By default this will be None, as the it has to be set externally, as the model won't necessarily
         be fit by emcee

        :return: The number of steps to be removed for burn-in.
        :rtype: int
        """
        return self._cut_off

    @cut_off.setter
    def cut_off(self, new_val: int):
        """
        Property setter for the number of steps to removed from MCMC chains as burn-in.

        :param int new_val: The number of steps.
        """
        self._cut_off = new_val

    @property
    def fit_method(self) -> str:
        """
        Property getter for the method used to fit the model instance, this will be None if no fit has been
        run using this model.

        :return: The fit method.
        :rtype: str
        """
        return self._fit_method

    @fit_method.setter
    def fit_method(self, new_val: str):
        """
        Set the fit method used on this model.

        :param str new_val: The fit method.
        """
        self._fit_method = new_val

    @property
    def par_names(self) -> List[str]:
        """
        The names of the parameters as they appear in the signature of the model python function.

        :return: A list of parameter names.
        :rtype: List[str]
        """
        # We infer the parameter names from the signature of the model function
        if self._par_names is None:
            self._par_names = [p.name for p in list(inspect.signature(self.model).parameters.values())[1:]]

        return self._par_names

    @property
    def success(self) -> bool:
        """
        If an fit has been run using this model then this property will tell you whether the fit method considered
        it to be 'successful' or not. If no fit has been run using this model then the value is None.

        :return: Was the fit successful?
        :rtype: bool
        """
        return self._success

    @success.setter
    def success(self, new_val: bool):
        """
        This is for an external fit method to set whether the fit run using this model was successful or not

        :param bool new_val: True for successful, False for failed.
        """
        self._success = new_val

    @property
    def profile(self):
        """
        The profile that this model has been fit to.

        :return: The profile object that this has been fit to, if no fit has been performed then this property
            will return None.
        :rtype: BaseProfile1D
        """
        return self._profile

    @profile.setter
    def profile(self, new_val):
        """
        The property setter for the profile that this model has been fit to.

        :param BaseProfile1D new_val: A profile object that this model has been fit to.
        """
        # Have to do this here to avoid circular import errors
        from ..products import BaseProfile1D

        if new_val is not None and not isinstance(new_val, BaseProfile1D):
            raise TypeError("You may only set the profile property with an XGA profile object, or None.")
        else:
            self._profile = new_val
