#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/12/2020, 14:36. Copyright (c) David J Turner
import inspect
from types import FunctionType
from typing import Tuple
from warnings import warn

import numpy as np
import scipy.odr as odr
from astropy.units import Quantity, UnitConversionError
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit

from ..models import MODEL_PUBLICATION_NAMES
from ..models import convert_to_odr_compatible

# This is just to make some instances of astropy LaTeX units prettier for plotting
PRETTY_UNITS = {'solMass': r'$M_{\odot}$', 'erg / s': r"erg s$^{-1}$"}


def _fit_initialise(y_values: Quantity, y_errs: Quantity, x_values: Quantity, x_errs: Quantity = None,
                    y_norm: Quantity = None, x_norm: Quantity = None) \
        -> Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]:
    """
    A handy little function that prepares the data for fitting with the chosen method.
    :param Quantity y_values: The y data values to fit to.
    :param Quantity y_errs: The y error values of the data. These should be supplied as either a 1D Quantity with
    length N (where N is the length of y_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity x_values: The x data values to fit to.
    :param Quantity x_errs: The x error values of the data. These should be supplied as either a 1D Quantity with
    length N (where N is the length of x_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity y_norm: Quantity to normalise the y data by.
    :param Quantity x_norm: Quantity to normalise the x data by.
    :return: The x data, x errors, y data, and y errors. Also the x_norm, y_norm.
    :rtype: Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]
    """
    # Check the lengths of the value and uncertainty quantities
    if len(x_values) != len(y_values):
        raise ValueError("The x and y quantities must have the same number of entries!")
    elif len(y_errs) != len(y_values):
        raise ValueError("Uncertainty quantities must have the same number of entries as the value quantities.")
    elif x_errs is not None and len(x_errs) != len(x_values):
        raise ValueError("Uncertainty quantities must have the same number of entries as the value quantities.")
    elif y_errs.unit != y_values.unit:
        raise UnitConversionError("Uncertainty quantities must have the same units as value quantities.")
    elif x_errs is not None and x_errs.unit != x_values.unit:
        raise UnitConversionError("Uncertainty quantities must have the same units as value quantities.")

    # For my own sanity, we're going to make x_err a Quantity if it isn't one already
    if x_errs is None:
        # We want zero error
        x_errs = Quantity(np.zeros(len(x_values)), x_values.unit)

    # Need to do a cleaning stage, to remove any NaN values from the data
    # First have to identify which entries in both the x and y arrays are NaN
    x_not_nans = np.where(~np.isnan(x_values))[0]
    y_not_nans = np.where(~np.isnan(y_values))[0]
    all_not_nans = np.intersect1d(x_not_nans, y_not_nans)

    # We'll warn the user if some entries are being excluded
    thrown_away = len(x_values) - len(all_not_nans)
    if thrown_away != 0:
        warn("{} sources have NaN values and have been excluded".format(thrown_away))

    # Only values that aren't NaN will be permitted
    x_values = x_values[all_not_nans]
    y_values = y_values[all_not_nans]
    # We're not changing the error arrays here because I'll do that in the place were I ensure the error arrays
    #  are 1D

    # We need to see if the normalisation parameters have been set, and if not then make them a number
    #  And if they have been set we need to ensure they're the same units as the x and y values
    if x_norm is None:
        x_norm = Quantity(1, x_values.unit)
    elif x_norm is not None and x_norm.unit != x_values.unit:
        raise UnitConversionError("The x normalisation parameter must have the same units as the x values.")

    if y_norm is None:
        y_norm = Quantity(1, y_values.unit)
    elif y_norm is not None and y_norm.unit != y_values.unit:
        raise UnitConversionError("The y normalisation parameter must have the same units as the y values.")

    # We need to parse the errors that have been parsed in, because there are options here
    # The user can either pass a 1D quantity with the errors, in which case we assume that they are symmetrical
    if y_errs.ndim == 2:
        # Because people can't follow instructions, I'm going to try and automatically find which axis to
        #  average the errors over
        av_ax = y_errs.shape.index(2)
        y_errs = np.mean(y_errs, axis=av_ax)

    if x_errs.ndim == 2:
        # Because people can't follow instructions, I'm going to try and automatically find which axis to
        #  average the errors over
        av_ax = x_errs.shape.index(2)
        x_errs = np.mean(x_errs, axis=av_ax)

    y_errs = y_errs[all_not_nans]
    x_errs = x_errs[all_not_nans]

    # We divide through by the normalisation parameter, which makes the data unitless
    x_fit_data = x_values / x_norm
    y_fit_data = y_values / y_norm
    x_fit_err = x_errs / x_norm
    y_fit_err = y_errs / y_norm

    return x_fit_data, x_fit_err, y_fit_data, y_fit_err, x_norm, y_norm


def _generate_relation_plot(model_func: FunctionType, y_values: Quantity, y_errs: Quantity, x_values: Quantity,
                            x_errs: Quantity, y_norm: Quantity, x_norm: Quantity, model_pars: np.ndarray,
                            model_errs: np.ndarray, fit_method: str, y_name: str = 'Y', x_name: str = 'X',
                            log_scale: bool = True, plot_title: str = None, figsize: tuple = (8, 8),
                            data_colour: str = 'black', model_colour: str = 'grey', grid_on: bool = True,
                            conf_level: int = 90):
    """

    :param model_func:
    :param y_values:
    :param y_errs:
    :param x_values:
    :param x_errs:
    :param y_norm:
    :param x_norm:
    :param model_pars:
    :param model_errs:
    :param fit_method:
    :param y_name:
    :param x_name:
    :param log_scale:
    :param plot_title:
    :param figsize:
    :param data_colour:
    :param model_colour:
    :param grid_on:
    :param conf_level:
    """
    # Setting up the matplotlib figure
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    ax = plt.gca()

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Un-normalise the data for plotting
    x_values *= x_norm
    x_errs *= x_norm
    y_values *= y_norm
    y_errs *= y_norm

    # Setup the aesthetics of the axis
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

    # Plot the data, with uncertainties (I only plot the averaged uncertainties if upper and lower uncertainties
    #  were given by the user).
    ax.errorbar(x_values.value, y_values.value, xerr=x_errs.value, yerr=y_errs.value, fmt="x", color=data_colour,
                capsize=2, label="Data")

    # Need to randomly sample from the fitted model
    model_pars = np.repeat(model_pars[..., None], 300, axis=1).T
    model_par_errs = np.repeat(model_errs[..., None], 300, axis=1).T

    model_par_dists = np.random.normal(model_pars, model_par_errs)

    # Fractional factor for the amount to go above and below the max and min x values
    buffer_factor = 0.1
    # The indices of the maximum and minimum x values
    max_x_ind = np.argmax(x_values)
    min_x_ind = np.argmin(x_values)

    model_x = np.linspace((1-buffer_factor)*(x_values[min_x_ind].value - x_errs[min_x_ind].value) / x_norm.value,
                          (1+buffer_factor)*(x_values[max_x_ind].value + x_errs[max_x_ind].value) / x_norm.value, 100)

    model_xs = np.repeat(model_x[..., None], 300, axis=1)

    upper = 50 + (conf_level / 2)
    lower = 50 - (conf_level / 2)

    model_realisations = model_func(model_xs, *model_par_dists.T) * y_norm
    model_mean = np.mean(model_realisations, axis=1)
    model_lower = np.percentile(model_realisations, lower, axis=1)
    model_upper = np.percentile(model_realisations, upper, axis=1)

    # I want the name of the function to include in labels and titles, but if its one defined in XGA then
    #  I can grab the publication version of the name - it'll be prettier
    mod_name = model_func.__name__
    for m_name in MODEL_PUBLICATION_NAMES:
        mod_name = mod_name.replace(m_name, MODEL_PUBLICATION_NAMES[m_name])

    plt.plot(model_x * x_norm.value, model_func(model_x, *model_pars[0, :]) * y_norm.value, color=model_colour,
             label="{mn} - {cf}% Confidence".format(mn=mod_name, cf=conf_level))
    plt.plot(model_x * x_norm.value, model_upper, color=model_colour, linestyle="--")
    plt.plot(model_x * x_norm.value, model_lower, color=model_colour, linestyle="--")
    ax.fill_between(model_x * x_norm.value, model_lower, model_upper, where=model_upper >= model_lower,
                    facecolor=model_colour, alpha=0.6, interpolate=True)

    # I can dynamically grab the units in LaTeX formatting from the Quantity objects (thank you astropy)
    #  However I've noticed specific instances where the units can be made prettier
    x_unit = '[' + x_values.unit.to_string() + ']'
    y_unit = '[' + y_values.unit.to_string() + ']'
    for og_unit in PRETTY_UNITS:
        x_unit = x_unit.replace(og_unit, PRETTY_UNITS[og_unit])
        y_unit = y_unit.replace(og_unit, PRETTY_UNITS[og_unit])

    # Dimensionless quantities can be fitted too, and this make the axis label look nicer by not having empty
    #  square brackets
    if x_unit == '[]':
        x_unit = ''
    if y_unit == '[]':
        y_unit = ''

    # I use the passed x and y names
    plt.xlabel("{xn} {un}".format(xn=x_name, un=x_unit), fontsize=12)
    plt.ylabel("{yn} {un}".format(yn=y_name, un=y_unit), fontsize=12)

    # The user can also pass a plot title, but if they don't then I construct one automatically
    if plot_title is None:
        plot_title = 'Scaling Relation - {mod} fitted with {fm}'.format(mod=mod_name, fm=fit_method)

    plt.title(plot_title, fontsize=13)

    # Use the axis limits quite a lot in this next bit, so read them out into variables
    x_axis_lims = ax.get_xlim()
    y_axis_lims = ax.get_ylim()

    # This dynamically changes how tick labels are formatted depending on the values displayed
    if max(x_axis_lims) < 1000:
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
    if max(y_axis_lims) < 1000:
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

    # And this dynamically changes the grid depending on whether a whole order of magnitude is covered or not
    #  Though as I don't much like the look of the grid the user can disable the grid
    if grid_on and (max(x_axis_lims) / min(x_axis_lims)) < 10:
        ax.grid(which='minor', axis='x', linestyle='dotted', color='grey')
    elif grid_on:
        ax.grid(which='major', axis='x', linestyle='dotted', color='grey')
    else:
        ax.grid(which='both', axis='both', b=False)

    if grid_on and (max(y_axis_lims) / min(y_axis_lims)) < 10:
        ax.grid(which='minor', axis='y', linestyle='dotted', color='grey')
    elif grid_on:
        ax.grid(which='major', axis='y', linestyle='dotted', color='grey')
    else:
        ax.grid(which='both', axis='both', b=False)

    # I change the lengths of the tick lines, to make it look nicer (imo)
    ax.tick_params(length=7)
    ax.tick_params(which='minor', length=3)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def scaling_relation_curve_fit(model_func, y_values: Quantity, y_errs: Quantity, x_values: Quantity,
                               x_errs: Quantity = None, y_norm: Quantity = None, x_norm: Quantity = None,
                               start_pars: list = None, log_scale: bool = True, y_name: str = 'Y', x_name: str = 'X',
                               plot_title: str = None, figsize: tuple = (8, 8), data_colour: str = 'black',
                               model_colour: str = 'grey', grid_on: bool = True, conf_level: int = 90) \
        -> Tuple[np.ndarray, np.ndarray, Quantity, Quantity]:
    """

    :param model_func: The function object of the model you wish to fit. PLEASE NOTE, the function must be defined
    in the style used in xga.models.misc; i.e. powerlaw(x: np.ndarray, k: float, a: float), where the first argument
    is for x values, and the following arguments are all fit parameters.
    :param Quantity y_values: The y data values to fit to.
    :param Quantity y_errs: The y error values of the data. These should be supplied as either a 1D Quantity with
    length N (where N is the length of y_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity x_values: The x data values to fit to.
    :param Quantity x_errs: The x error values of the data. These should be supplied as either a 1D Quantity with
    length N (where N is the length of x_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity y_norm: Quantity to normalise the y data by.
    :param Quantity x_norm: Quantity to normalise the x data by.
    :param list start_pars:
    :param bool log_scale:
    :param str y_name:
    :param str x_name:
    :param str plot_title:
    :param tuple figsize:
    :param str data_colour:
    :param str model_colour:
    :param bool grid_on:
    :param int conf_level:
    :return: The fit parameter and their uncertainties, the x data normalisation, and the y data normalisation.
    :rtype: Tuple[np.ndarray, np.ndarray, Quantity, Quantity]
    """
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm = _fit_initialise(y_values, y_errs, x_values,
                                                                                     x_errs, y_norm, x_norm)

    fit_par, fit_cov = curve_fit(model_func, x_fit_data.value, y_fit_data.value, sigma=y_fit_errs.value,
                                 absolute_sigma=True, p0=start_pars)
    fit_par_err = np.sqrt(np.diagonal(fit_cov))
    _generate_relation_plot(model_func, y_fit_data, y_fit_errs, x_fit_data, x_fit_errs, y_norm, x_norm, fit_par,
                            fit_par_err, 'Curve Fit', y_name, x_name, log_scale, plot_title, figsize, data_colour,
                            model_colour, grid_on, conf_level)

    return fit_par, fit_par_err, x_norm, y_norm


def scaling_relation_lira():
    raise NotImplementedError("I'm working on it!")


def scaling_relation_odr(model_func, y_values: Quantity, y_errs: Quantity, x_values: Quantity, x_errs: Quantity = None,
                         y_norm: Quantity = None, x_norm: Quantity = None, start_pars: list = None,
                         log_scale: bool = True, y_name: str = 'Y', x_name: str = 'X', plot_title: str = None,
                         figsize: tuple = (8, 8), data_colour: str = 'black', model_colour: str = 'grey',
                         grid_on: bool = True, conf_level: int = 90) \
        -> Tuple[np.ndarray, np.ndarray, Quantity, Quantity, odr.Output]:
    """

    :param model_func:
    :param Quantity y_values: The y data values to fit to.
    :param Quantity y_errs: The y error values of the data. These should be supplied as either a 1D Quantity with
    length N (where N is the length of y_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity x_values: The x data values to fit to.
    :param Quantity x_errs: The x error values of the data. These should be supplied as either a 1D Quantity with
    length N (where N is the length of x_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity y_norm: Quantity to normalise the y data by.
    :param Quantity x_norm: Quantity to normalise the x data by.
    :param list start_pars:
    :param bool log_scale:
    :param str y_name:
    :param str x_name:
    :param str plot_title:
    :param tuple figsize:
    :param str data_colour:
    :param str model_colour:
    :param bool grid_on:
    :param int conf_level:
    :return: The fit parameter and their uncertainties, the x data normalisation, and the y data normalisation. This
    fit function also returns the orthogonal distance regression output object, which contains all information from
    the fit.
    :rtype: Tuple[np.ndarray, np.ndarray, Quantity, Quantity, odr.Output]
    """
    if start_pars is None:
        # Setting up the start_pars, as ODR doesn't have a default value like curve_fit
        num_par = len(list(inspect.signature(model_func).parameters.keys())) - 1
        start_pars = np.ones(num_par)

    # Standard data preparation
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm = _fit_initialise(y_values, y_errs, x_values,
                                                                                     x_errs, y_norm, x_norm)

    # Immediately faced with a problem, because scipy's ODR is naff and wants functions defined like
    #  blah(par_vector, x_values), which is completely different to my standard definition of models in this module.
    # I don't want the user to have to define things differently for this fit function, so I'm gonna try and
    #  dynamically redefine function models passed in here...
    converted_model_func = convert_to_odr_compatible(model_func)

    # Then we define a scipy odr model
    odr_model = odr.Model(converted_model_func)

    # And a RealData (which takes the errors as the proper standard deviations of the data)
    odr_data = odr.RealData(x_fit_data, y_fit_data, x_fit_errs, y_fit_errs)

    # Now we instantiate the ODR class with our model and data. This is currently basically ripped from the example
    #  given in the scipy ODR documentation
    odr_obj = odr.ODR(odr_data, odr_model, beta0=start_pars)

    # Actually run the fit, and grab the output
    fit_results = odr_obj.run()

    # And from here, with this output object, I just read out the parameter values, along with the standard dev
    fit_par = fit_results.beta
    fit_par_err = fit_results.sd_beta

    _generate_relation_plot(model_func, y_fit_data, y_fit_errs, x_fit_data, x_fit_errs, y_norm, x_norm, fit_par,
                            fit_par_err, 'ODR', y_name, x_name, log_scale, plot_title, figsize, data_colour,
                            model_colour, grid_on, conf_level)

    return fit_par, fit_par_err, x_norm, y_norm, fit_results


def scaling_relation_emcee():
    raise NotImplementedError("I'm working on it!")
