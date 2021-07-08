#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/07/2021, 11:16. Copyright (c) David J Turner
import inspect
from types import FunctionType
from typing import Tuple
from warnings import warn

import numpy as np
import scipy.odr as odr
from astropy.units import Quantity, UnitConversionError
from scipy.optimize import curve_fit

from ..exceptions import XGAFunctionConversionError, XGAOptionalDependencyError
from ..models import convert_to_odr_compatible
from ..models.misc import power_law
from ..products.relation import ScalingRelation

# This is just so the allowed fitting methods can be referenced from one place
ALLOWED_FIT_METHODS = ["curve_fit", "odr", "lira", "emcee"]


def _fit_initialise(y_values: Quantity, y_errs: Quantity, x_values: Quantity, x_errs: Quantity = None,
                    y_norm: Quantity = None, x_norm: Quantity = None, log_data: bool = False) \
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
    :param bool log_data: This parameter controls whether the data is logged before being returned. The
        default is False as it isn't likely to be used often - its included because LIRA wants logged data.
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
        # We're also going to set a flag indicating that a default value was set
        no_x_errs = True
    else:
        no_x_errs = False

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

    # TODO I'M STILL NOT SURE THAT THIS IS THE RIGHT WAY TO CONVERT ERRORS TO LOG
    if log_data:
        # We're logging here because the fitting package wants it
        x_fit_err = x_fit_err / x_fit_data / np.log(10)
        x_fit_data = np.log10(x_fit_data)

        y_fit_err = y_fit_err / y_fit_data / np.log(10)
        y_fit_data = np.log10(y_fit_data)

        if no_x_errs:
            # I know I'm setting something that already exists, but I want errors of 0 to be passed out
            x_fit_err = Quantity(np.zeros(len(x_values)), x_values.unit)

    return x_fit_data, x_fit_err, y_fit_data, y_fit_err, x_norm, y_norm


def scaling_relation_curve_fit(model_func: FunctionType, y_values: Quantity, y_errs: Quantity, x_values: Quantity,
                               x_errs: Quantity = None, y_norm: Quantity = None, x_norm: Quantity = None,
                               x_lims: Quantity = None, start_pars: list = None, y_name: str = 'Y',
                               x_name: str = 'X') -> ScalingRelation:
    """
    A function to fit a scaling relation with the scipy non-linear least squares implementation (curve fit), generate
    an XGA ScalingRelation product, and return it.

    :param FunctionType model_func: The function object of the model you wish to fit. PLEASE NOTE, the function must
        be defined in the style used in xga.models.misc; i.e. powerlaw(x: np.ndarray, k: float, a: float), where
        the first argument is for x values, and the following arguments are all fit parameters.
    :param Quantity y_values: The y data values to fit to.
    :param Quantity y_errs: The y error values of the data. These should be supplied as either a 1D Quantity with
        length N (where N is the length of y_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity x_values: The x data values to fit to.
    :param Quantity x_errs: The x error values of the data. These should be supplied as either a 1D Quantity with
        length N (where N is the length of x_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity y_norm: Quantity to normalise the y data by.
    :param Quantity x_norm: Quantity to normalise the x data by.
    :param Quantity x_lims: The range of x values in which this relation is valid, default is None. If this
        information is supplied, please pass it as a Quantity array, with the first element being the lower
        bound and the second element being the upper bound.
    :param list start_pars: The start parameters for the curve_fit run, default is None, which means curve_fit
        will use all ones.
    :param str y_name: The name to be used for the y-axis of the scaling relation (DON'T include the unit, that
        will be inferred from the astropy Quantity.
    :param str x_name: The name to be used for the x-axis of the scaling relation (DON'T include the unit, that
        will be inferred from the astropy Quantity.
    :return: An XGA ScalingRelation object with all the information about the data and fit, a view method, and a
        predict method.
    :rtype: ScalingRelation
    """
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm = _fit_initialise(y_values, y_errs, x_values,
                                                                                     x_errs, y_norm, x_norm)

    fit_par, fit_cov = curve_fit(model_func, x_fit_data.value, y_fit_data.value, sigma=y_fit_errs.value,
                                 absolute_sigma=True, p0=start_pars)
    fit_par_err = np.sqrt(np.diagonal(fit_cov))

    sr = ScalingRelation(fit_par, fit_par_err, model_func, x_norm, y_norm, x_name, y_name, 'Curve Fit',
                         x_fit_data * x_norm, y_fit_data * y_norm, x_fit_errs * x_norm, y_fit_errs * y_norm,
                         x_lims=x_lims)

    return sr


def scaling_relation_odr(model_func: FunctionType, y_values: Quantity, y_errs: Quantity, x_values: Quantity,
                         x_errs: Quantity = None, y_norm: Quantity = None, x_norm: Quantity = None,
                         x_lims: Quantity = None, start_pars: list = None, y_name: str = 'Y',
                         x_name: str = 'X') -> ScalingRelation:
    """
    A function to fit a scaling relation with the scipy orthogonal distance regression implementation, generate
    an XGA ScalingRelation product, and return it.

    :param FunctionType model_func: The function object of the model you wish to fit. PLEASE NOTE, the function must
        be defined in the style used in xga.models.misc; i.e. powerlaw(x: np.ndarray, k: float, a: float), where
        the first argument is for x values, and the following arguments are all fit parameters. The scipy ODR
        implementation requires functions of a different style, and I try to automatically convert the input function
        to that style, but to help that please avoid using one letter parameter names in any custom function you might
        want to use.
    :param Quantity y_values: The y data values to fit to.
    :param Quantity y_errs: The y error values of the data. These should be supplied as either a 1D Quantity with
        length N (where N is the length of y_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity x_values: The x data values to fit to.
    :param Quantity x_errs: The x error values of the data. These should be supplied as either a 1D Quantity with
        length N (where N is the length of x_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity y_norm: Quantity to normalise the y data by.
    :param Quantity x_norm: Quantity to normalise the x data by.
    :param Quantity x_lims: The range of x values in which this relation is valid, default is None. If this
        information is supplied, please pass it as a Quantity array, with the first element being the lower
        bound and the second element being the upper bound.
    :param list start_pars: The start parameters for the ODR run, default is all ones.
    :param str y_name: The name to be used for the y-axis of the scaling relation (DON'T include the unit, that
        will be inferred from the astropy Quantity.
    :param str x_name: The name to be used for the x-axis of the scaling relation (DON'T include the unit, that
        will be inferred from the astropy Quantity.
    :return: An XGA ScalingRelation object with all the information about the data and fit, a view method, and a
        predict method.
    :rtype: ScalingRelation
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

    # The first thing I want to do is check that the newly converted function gives the same result for a
    #  simple test as the original model
    if model_func(5, *start_pars) != converted_model_func(start_pars, 5):
        raise XGAFunctionConversionError('I attempted to convert the input model function to the standard'
                                         ' required by ODR, but it is not returning the same value as the '
                                         'original for this test case.')

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

    sr = ScalingRelation(fit_par, fit_par_err, model_func, x_norm, y_norm, x_name, y_name, 'ODR', x_fit_data*x_norm,
                         y_fit_data*y_norm, x_fit_errs*x_norm, y_fit_errs*y_norm, odr_output=fit_results, x_lims=x_lims)

    return sr


def scaling_relation_lira(y_values: Quantity, y_errs: Quantity, x_values: Quantity, x_errs: Quantity = None,
                          y_norm: Quantity = None, x_norm: Quantity = None, x_lims: Quantity = None, y_name: str = 'Y',
                          x_name: str = 'X', num_steps: int = 100000, num_chains: int = 4, num_burn_in: int = 10000) \
        -> ScalingRelation:
    """
    A function to fit a power law scaling relation with the excellent R fitting package LIRA
    (https://doi.org/10.1093/mnras/stv2374), this function requires a valid R installation, along with LIRA (and its
    dependencies such as JAGS), as well as the Python module rpy2.

    :param Quantity y_values: The y data values to fit to.
    :param Quantity y_errs: The y error values of the data. These should be supplied as either a 1D Quantity with
        length N (where N is the length of y_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity x_values: The x data values to fit to.
    :param Quantity x_errs: The x error values of the data. These should be supplied as either a 1D Quantity with
        length N (where N is the length of x_values), or an Nx2 Quantity with lower and upper errors.
    :param Quantity y_norm: Quantity to normalise the y data by.
    :param Quantity x_norm: Quantity to normalise the x data by.
    :param Quantity x_lims: The range of x values in which this relation is valid, default is None. If this
        information is supplied, please pass it as a Quantity array, with the first element being the lower
        bound and the second element being the upper bound.
    :param str y_name: The name to be used for the y-axis of the scaling relation (DON'T include the unit, that
        will be inferred from the astropy Quantity.
    :param str x_name: The name to be used for the x-axis of the scaling relation (DON'T include the unit, that
        will be inferred from the astropy Quantity.
    :param int num_steps: The number of steps to take in each chain.
    :param int num_chains: The number of chains to run.
    :param int num_burn_in: The number of steps to discard as a burn in period. This is also used as the adapt
        parameter of the LIRA fit.
    :return: An XGA ScalingRelation object with all the information about the data and fit, a view method, and a
        predict method.
    :rtype: ScalingRelation
    """
    # Due to the nature of this function, a wrapper for the LIRA R fitting package, I'm going to try the
    #  necessary imports here, because the external dependencies are likely to only be used in this function
    #  and as such I don't want to force the user to have them to use XGA.
    try:
        from rpy2.robjects.packages import importr
        from rpy2 import robjects
        robjects.r['options'](warn=-1)
    except ImportError:
        raise XGAOptionalDependencyError('LIRA is an R fitting package, and as such you need to have installed '
                                         'rpy2 to use this function')

    # We use the rpy2 module to interface with an underlying R installation, and import the basic R components
    base_pack = importr('base')
    utils_pack = importr('utils')

    # Now we import the thing we're actually interested in, the LIRA package
    try:
        lira_pack = importr('lira')
    except robjects.packages.PackageNotInstalledError:
        raise XGAOptionalDependencyError('While the rpy2 module is installed, you do not appear to have installed '
                                         'the LIRA fitting package to your R environment')

    # Slightly different data preparation to the other fitting methods, this one returns logged data and errors
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm = _fit_initialise(y_values, y_errs, x_values,
                                                                                     x_errs, y_norm, x_norm, True)

    # And now we have to make some R objects so that we can pass it through our R interface to the LIRA package
    x_fit_data = robjects.FloatVector(x_fit_data.value)
    y_fit_data = robjects.FloatVector(y_fit_data.value)
    x_fit_errs = robjects.FloatVector(x_fit_errs.value)
    y_fit_errs = robjects.FloatVector(y_fit_errs.value)

    # This runs the LIRA fit and grabs the output data frame, from that I can read the chains for the different
    #  parameters
    chains = lira_pack.lira(x_fit_data, y_fit_data, delta_x=x_fit_errs, delta_y=y_fit_errs, n_iter=num_steps,
                            n_chains=num_chains, n_adapt=num_burn_in, export=False, print_summary=False,
                            print_diagnostic=False)[0][0]

    # Read out the alpha parameter chain and convert to a numpy array
    alpha_par_chain = np.power(10, np.array(chains.rx2['alpha.YIZ']))
    alpha_par_val = np.mean(alpha_par_chain)
    alpha_par_err = np.std(alpha_par_chain)

    # Read out the beta parameter chain and convert to a numpy array
    beta_par_chain = np.array(chains.rx2['beta.YIZ'])
    beta_par_val = np.mean(beta_par_chain)
    beta_par_err = np.std(beta_par_chain)

    # Read out the intrinsic scatter chain and convert to a numpy array
    sigma_par_chain = np.array(chains.rx2['sigma.YIZ.0'])
    sigma_par_val = np.mean(sigma_par_chain)
    sigma_par_err = np.std(sigma_par_chain)

    fit_par = np.array([beta_par_val, alpha_par_val])
    fit_par_err = np.array([beta_par_err, alpha_par_err])

    # This call to the fit initialisation function DOESN'T produce logged data, do this so the plot works
    #  properly - it expects non logged data
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm = _fit_initialise(y_values, y_errs, x_values,
                                                                                     x_errs, y_norm, x_norm)

    # I'm re-formatting the chains into a shape that the ScalingRelation class will understand.
    xga_chains = np.concatenate([beta_par_chain.reshape(len(beta_par_chain), 1),
                                 alpha_par_chain.reshape(len(alpha_par_chain), 1)], axis=1)

    sr = ScalingRelation(fit_par, fit_par_err, power_law, x_norm, y_norm, x_name, y_name, 'LIRA', x_fit_data * x_norm,
                         y_fit_data * y_norm, x_fit_errs * x_norm, y_fit_errs * y_norm, chains=xga_chains,
                         x_lims=x_lims, scatter_par=np.array([sigma_par_val, sigma_par_err]),
                         scatter_chain=sigma_par_chain)

    return sr


def scaling_relation_emcee():
    raise NotImplementedError("This fitting method has not yet been implemented, please consider LIRA "
                              "as an alternative.")
