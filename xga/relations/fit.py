#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/03/2025, 13:56. Copyright (c) The Contributors
import inspect
from types import FunctionType
from typing import Tuple, Union
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
                    y_norm: Quantity = None, x_norm: Quantity = None, log_data: bool = False,
                    point_names: Union[np.ndarray, list] = None, third_dim: Union[np.ndarray, Quantity, list] = None) \
        -> Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, np.ndarray, Quantity]:
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
    :param np.ndarray/list point_names: A possible set of source names associated with all the data points
    :param np.ndarray/Quantity/list third_dim: A possible set of extra data used to colour data points.
    :return: The x data, x errors, y data, and y errors. Also the x_norm, y_norm, and the names of non-NaN points.
    :rtype: Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, np.ndarray, Quantity]
    """
    # Check the lengths of the value and uncertainty quantities, as well as the extra information that can also
    #  flow through this function
    if len(x_values) != len(y_values):
        raise ValueError("The x and y quantities must have the same number of entries!")
    elif len(y_errs) != len(y_values):
        raise ValueError("Uncertainty quantities must have the same number of entries as the value quantities.")
    elif x_errs is not None and len(x_errs) != len(x_values):
        raise ValueError("Uncertainty quantities must have the same number of entries as the value quantities.")
    # Not involved in the fitting process, but comes through here so that the sources dropped due to NaN values
    #  also have the values dropped in these variables
    elif point_names is not None and len(point_names) != len(x_values):
        ValueError("The 'point_names' argument is a different length ({p}) to the input data "
                   "({d}).".format(p=len(point_names), d=len(x_values)))
    elif third_dim is not None and len(third_dim) != len(x_values):
        ValueError("The 'third_dim' argument is a different length ({p}) to the input data "
                   "({d}).".format(p=len(third_dim), d=len(x_values)))
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
    #  First have to identify which entries in both the x and y arrays are NaN
    x_not_nans = np.where(~np.isnan(x_values))[0]
    y_not_nans = np.where(~np.isnan(y_values))[0]
    all_not_nans = np.intersect1d(x_not_nans, y_not_nans)

    # We also check for negative uncertainties, which are obviously bogus and cause plotting/fit issues later on - we
    #  take the opposite approach (in terms of boolean logic) to identifying the bad entries, because its easier
    x_err_are_neg = np.where(x_errs < 0)[0]
    y_err_are_neg = np.where(y_errs < 0)[0]

    all_err_not_neg = np.intersect1d(np.setdiff1d(np.arange(0, len(x_errs)), x_err_are_neg),
                                     np.setdiff1d(np.arange(0, len(y_errs)), y_err_are_neg))

    # Intersect the two selection criteria
    all_acc = np.intersect1d(all_not_nans, all_err_not_neg)

    # And we'll repeat the warning exercise if any were excluded because they have negative uncertainties
    thrown_away = len(x_values) - len(all_acc)
    if thrown_away != 0:
        warn("{} sources have NaN values or negative uncertainties and have been excluded".format(thrown_away),
             stacklevel=2)

    # Only values that aren't NaN and don't have negative errors will be permitted
    x_values = x_values[all_acc]
    y_values = y_values[all_acc]
    # We're not changing the error arrays here because I'll do that in the place where I ensure the error arrays
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

    # Doing what we did to the value arrays further up, removing any entries that haven't passed our criteria of not
    #  having a NaN and not having negative errors
    y_errs = y_errs[all_acc]
    x_errs = x_errs[all_acc]

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

    # Make sure point_names actually is an array (if supplied) and remove the NaN entry equivalents
    if point_names is not None:
        if isinstance(point_names, list):
            point_names = np.array(point_names)
        point_names = point_names[all_acc]
    elif point_names is None:
        point_names = None

    # Same deal with the third dimension data that can optionally be supplied to the scaling relations (though
    #  isn't used in the fit process, it's just for colouring data points in a view method).
    if third_dim is not None:
        if isinstance(third_dim, list):
            third_dim = Quantity(third_dim)
        third_dim = third_dim[all_acc]
    elif third_dim is None:
        third_dim = None

    return x_fit_data, x_fit_err, y_fit_data, y_fit_err, x_norm, y_norm, point_names, third_dim


def scaling_relation_curve_fit(model_func: FunctionType, y_values: Quantity, y_errs: Quantity, x_values: Quantity,
                               x_errs: Quantity = None, y_norm: Quantity = None, x_norm: Quantity = None,
                               x_lims: Quantity = None, start_pars: list = None, y_name: str = 'Y',
                               x_name: str = 'X', dim_hubb_ind: Union[float, int] = None,
                               point_names: Union[np.ndarray, list] = None,
                               third_dim_info: Union[np.ndarray, Quantity] = None, third_dim_name: str = None,
                               x_en_bounds: Quantity = None, y_en_bounds: Quantity = None) -> ScalingRelation:
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
    :param float/int dim_hubb_ind: This is used to tell the ScalingRelation which power of E(z) has been applied
        to the y-axis data, this can then be used by the predict method to remove the E(z) contribution from
        predictions. The default is None.
    :param np.ndarray/list point_names: The source names associated with the data points passed in to this scaling
        relation, can be used for diagnostic purposes (i.e. identifying which source an outlier belongs to).
    :param np.ndarray/Quantity third_dim_info: A set of data points which represent a faux third dimension. They should
        not have been involved in the fitting process, and the relation should not be in three dimensions, but these
        can be used to colour the data points in a view method.
    :param str third_dim_name: The name of the third dimension data.
    :param Tuple[Quantity] x_en_bounds: If the value on the x-axis of this relation is 'energy bound', those bounds
        can be specified here (e.g. if the value is 0.5-2.0 keV luminosity you would pass a non-scalar quantity with
        the first entry being 0.5 and the second 2.0; Quantity([0.5, 2.0], 'keV'). The default is None.
    :param Tuple[Quantity] y_en_bounds: If the value on the y-axis of this relation is 'energy bound', those bounds
        can be specified here (e.g. if the value is 0.5-2.0 keV luminosity you would pass a non-scalar quantity with
        the first entry being 0.5 and the second 2.0; Quantity([0.5, 2.0], 'keV'). The default is None.
    :return: An XGA ScalingRelation object with all the information about the data and fit, a view method, and a
        predict method.
    :rtype: ScalingRelation
    """
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm, point_names, \
        third_dim_info = _fit_initialise(y_values, y_errs, x_values, x_errs, y_norm, x_norm, point_names=point_names,
                                         third_dim=third_dim_info)

    fit_par, fit_cov = curve_fit(model_func, x_fit_data.value, y_fit_data.value, sigma=y_fit_errs.value,
                                 absolute_sigma=True, p0=start_pars)
    fit_par_err = np.sqrt(np.diagonal(fit_cov))

    sr = ScalingRelation(fit_par, fit_par_err, model_func, x_norm, y_norm, x_name, y_name, fit_method='Curve Fit',
                         x_data=x_fit_data * x_norm, y_data=y_fit_data * y_norm, x_err=x_fit_errs * x_norm,
                         y_err=y_fit_errs * y_norm, x_lims=x_lims, dim_hubb_ind=dim_hubb_ind, point_names=point_names,
                         third_dim_info=third_dim_info, third_dim_name=third_dim_name, x_en_bounds=x_en_bounds,
                         y_en_bounds=y_en_bounds)

    return sr


def scaling_relation_odr(model_func: FunctionType, y_values: Quantity, y_errs: Quantity, x_values: Quantity,
                         x_errs: Quantity = None, y_norm: Quantity = None, x_norm: Quantity = None,
                         x_lims: Quantity = None, start_pars: list = None, y_name: str = 'Y',
                         x_name: str = 'X', dim_hubb_ind: Union[float, int] = None,
                         point_names: Union[np.ndarray, list] = None,
                         third_dim_info: Union[np.ndarray, Quantity] = None, third_dim_name: str = None,
                         x_en_bounds: Quantity = None, y_en_bounds: Quantity = None) -> ScalingRelation:
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
    :param float/int dim_hubb_ind: This is used to tell the ScalingRelation which power of E(z) has been applied
        to the y-axis data, this can then be used by the predict method to remove the E(z) contribution from
        predictions. The default is None.
    :param np.ndarray/list point_names: The source names associated with the data points passed in to this scaling
        relation, can be used for diagnostic purposes (i.e. identifying which source an outlier belongs to).
    :param np.ndarray/Quantity third_dim_info: A set of data points which represent a faux third dimension. They should
        not have been involved in the fitting process, and the relation should not be in three dimensions, but these
        can be used to colour the data points in a view method.
    :param str third_dim_name: The name of the third dimension data.
    :param Tuple[Quantity] x_en_bounds: If the value on the x-axis of this relation is 'energy bound', those bounds
        can be specified here (e.g. if the value is 0.5-2.0 keV luminosity you would pass a non-scalar quantity with
        the first entry being 0.5 and the second 2.0; Quantity([0.5, 2.0], 'keV'). The default is None.
    :param Tuple[Quantity] y_en_bounds: If the value on the y-axis of this relation is 'energy bound', those bounds
        can be specified here (e.g. if the value is 0.5-2.0 keV luminosity you would pass a non-scalar quantity with
        the first entry being 0.5 and the second 2.0; Quantity([0.5, 2.0], 'keV'). The default is None.
    :return: An XGA ScalingRelation object with all the information about the data and fit, a view method, and a
        predict method.
    :rtype: ScalingRelation
    """
    if start_pars is None:
        # Setting up the start_pars, as ODR doesn't have a default value like curve_fit
        num_par = len(list(inspect.signature(model_func).parameters.keys())) - 1
        start_pars = np.ones(num_par)

    # Standard data preparation
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm, point_names, \
        third_dim_info = _fit_initialise(y_values, y_errs, x_values, x_errs, y_norm, x_norm, point_names=point_names,
                                         third_dim=third_dim_info)

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

    sr = ScalingRelation(fit_par, fit_par_err, model_func, x_norm, y_norm, x_name, y_name, fit_method='ODR',
                         x_data=x_fit_data * x_norm, y_data=y_fit_data * y_norm, x_err=x_fit_errs * x_norm,
                         y_err=y_fit_errs * y_norm, x_lims=x_lims, odr_output=fit_results, dim_hubb_ind=dim_hubb_ind,
                         point_names=point_names, third_dim_info=third_dim_info, third_dim_name=third_dim_name,
                         x_en_bounds=x_en_bounds, y_en_bounds=y_en_bounds)

    return sr


def scaling_relation_lira(y_values: Quantity, y_errs: Quantity, x_values: Quantity, x_errs: Quantity = None,
                          y_norm: Quantity = None, x_norm: Quantity = None, x_lims: Quantity = None, y_name: str = 'Y',
                          x_name: str = 'X', num_steps: int = 100000, num_chains: int = 4, num_burn_in: int = 10000,
                          dim_hubb_ind: Union[float, int] = None, point_names: Union[np.ndarray, list] = None,
                          third_dim_info: Union[np.ndarray, Quantity] = None, third_dim_name: str = None,
                          x_en_bounds: Quantity = None, y_en_bounds: Quantity = None) -> ScalingRelation:
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
    :param float/int dim_hubb_ind: This is used to tell the ScalingRelation which power of E(z) has been applied
        to the y-axis data, this can then be used by the predict method to remove the E(z) contribution from
        predictions. The default is None.
    :param np.ndarray/list point_names: The source names associated with the data points passed in to this scaling
        relation, can be used for diagnostic purposes (i.e. identifying which source an outlier belongs to).
    :param np.ndarray/Quantity third_dim_info: A set of data points which represent a faux third dimension. They should
        not have been involved in the fitting process, and the relation should not be in three dimensions, but these
        can be used to colour the data points in a view method.
    :param str third_dim_name: The name of the third dimension data.
    :param Tuple[Quantity] x_en_bounds: If the value on the x-axis of this relation is 'energy bound', those bounds
        can be specified here (e.g. if the value is 0.5-2.0 keV luminosity you would pass a non-scalar quantity with
        the first entry being 0.5 and the second 2.0; Quantity([0.5, 2.0], 'keV'). The default is None.
    :param Tuple[Quantity] y_en_bounds: If the value on the y-axis of this relation is 'energy bound', those bounds
        can be specified here (e.g. if the value is 0.5-2.0 keV luminosity you would pass a non-scalar quantity with
        the first entry being 0.5 and the second 2.0; Quantity([0.5, 2.0], 'keV'). The default is None.
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
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm, point_names, \
        third_dim_info = _fit_initialise(y_values, y_errs, x_values, x_errs, y_norm, x_norm, True, point_names,
                                         third_dim_info)

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
    x_fit_data, x_fit_errs, y_fit_data, y_fit_errs, x_norm, y_norm, \
        throw_away, sec_throw_away = _fit_initialise(y_values, y_errs, x_values, x_errs, y_norm, x_norm)

    # I'm re-formatting the chains into a shape that the ScalingRelation class will understand.
    xga_chains = np.concatenate([beta_par_chain.reshape(len(beta_par_chain), 1),
                                 alpha_par_chain.reshape(len(alpha_par_chain), 1)], axis=1)

    sr = ScalingRelation(fit_par, fit_par_err, power_law, x_norm, y_norm, x_name, y_name, fit_method='LIRA',
                         x_data=x_fit_data * x_norm, y_data=y_fit_data * y_norm, x_err=x_fit_errs * x_norm,
                         y_err=y_fit_errs * y_norm, x_lims=x_lims, chains=xga_chains,
                         scatter_par=np.array([sigma_par_val, sigma_par_err]), scatter_chain=sigma_par_chain,
                         dim_hubb_ind=dim_hubb_ind, point_names=point_names, third_dim_info=third_dim_info,
                         third_dim_name=third_dim_name, x_en_bounds=x_en_bounds, y_en_bounds=y_en_bounds)

    return sr


def scaling_relation_emcee():
    raise NotImplementedError("This fitting method has not yet been implemented, please consider LIRA "
                              "as an alternative.")
