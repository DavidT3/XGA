#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 16/08/2021, 16:20. Copyright (c) David J Turner

import inspect
from datetime import date
from typing import List
from warnings import warn

import numpy as np
import scipy.odr as odr
from astropy.units import Quantity, Unit, UnitConversionError
from cycler import cycler
from getdist import plots, MCSamples
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from ..models import MODEL_PUBLICATION_NAMES

# This is the default colour cycle for the AggregateScalingRelation view method
PRETTY_COLOUR_CYCLE = ['tab:gray', 'tab:blue', 'darkgreen', 'firebrick', 'slateblue', 'goldenrod']

# Given the existing structure of this part of XGA, I would have written a BaseRelation class in xga.products.base,
#  but I can't think of a reason why individual scaling relations should have their own classes. One general class
#  should be enough. Also I don't want relations to be able to be stored in source objects - these are for samples
#  only


class ScalingRelation:
    """
    This class is designed to store all information pertaining to a scaling relation fit, either performed by XGA
    or from literature. It also aims to make creating publication quality plots simple and easy.
    """
    def __init__(self, fit_pars: np.ndarray, fit_par_errs: np.ndarray, model_func, x_norm: Quantity, y_norm: Quantity,
                 x_name: str, y_name: str, fit_method: str = 'unknown', x_data: Quantity = None,
                 y_data: Quantity = None, x_err: Quantity = None, y_err: Quantity = None, x_lims: Quantity = None,
                 odr_output: odr.Output = None, chains: np.ndarray = None, relation_name: str = None,
                 relation_author: str = 'XGA', relation_year: str = str(date.today().year), relation_doi: str = '',
                 scatter_par: np.ndarray = None, scatter_chain: np.ndarray = None):
        """
        The init for the ScalingRelation class, all information necessary to enable the different functions of
        this class will be supplied by the user here.

        :param np.ndarray fit_pars: The results of the fit to a model that describes this scaling relation.
        :param np.ndarray fit_par_errs: The uncertainties on the fit results for this scalin relation.
        :param model_func: A Python function of the model which this scaling relation is described by.
            PLEASE NOTE, the function must be defined in the style used in xga.models.misc;
            i.e. powerlaw(x: np.ndarray, k: float, a: float), where the first argument is for x values, and the
            following arguments are all fit parameters.
        :param Quantity x_norm: Quantity to normalise the x data by.
        :param Quantity y_norm: Quantity to normalise the y data by.
        :param str x_name: The name to be used for the x-axis of the plot (DON'T include the unit, that will be
            inferred from an astropy Quantity.
        :param str y_name: The name to be used for the y-axis of the plot (DON'T include the unit, that will be
            inferred from an astropy Quantity.
        :param str fit_method: The method used to fit this data, if known.
        :param Quantity x_data: The x-data used to fit this scaling relation, if available. This should be
            the raw, un-normalised data.
        :param Quantity y_data: The y-data used to fit this scaling relation, if available. This should be
            the raw, un-normalised data.
        :param Quantity x_err: The x-errors used to fit this scaling relation, if available. This should be
            the raw, un-normalised data.
        :param Quantity y_err: The y-errors used to fit this scaling relation, if available. This should be
            the raw, un-normalised data.
        :param Quantity x_lims: The range of x values in which this relation is valid, default is None. If this
            information is supplied, please pass it as a Quantity array, with the first element being the lower
            bound and the second element being the upper bound.
        :param odr.Output odr_output: The orthogonal distance regression output object associated with this
            relation's fit, if available and applicable.
        :param np.ndarray chains: The parameter chains associated with this relation's fit, if available and
            applicable. They should be of shape N_stepxN_par, where N_steps is the number of steps (after burn-in
            is removed), and N_par is the number of parameters in the fit.
        :param str relation_name: A suitable name for this relation.
        :param str relation_author: The author who deserves credit for this relation.
        :param str relation_year: The year this relation was produced, default is the current year.
        :param str relation_doi: The DOI of the original paper this relation appeared in.
        :param np.ndarray scatter_par: A parameter describing the intrinsic scatter of y|x. Optional as many fits don't
            include this.
        :param np.ndarray scatter_chain: A corresponding MCMC chain for the scatter parameter. Optional.
        """
        # These will always be passed in, and are assumed to be in the order required by the model_func that is also
        #  passed in by the user.
        self._fit_pars = fit_pars
        self._fit_par_errs = fit_par_errs

        # This should be a Python function of the model which was fit to create this relation, and which will take the
        #  passed fit parameters as arguments
        self._model_func = model_func

        # These are very important, as many people apply normalisation before fitting, and they give us information
        #  about the units of the x and y axis in absence of any data (data doesn't have to be passed)
        self._x_norm = x_norm
        self._y_norm = y_norm

        # These are also required, otherwise any plots we make are going to look a bit dumb with no x or y axis labels
        self._x_name = x_name
        self._y_name = y_name

        # The default fit method is 'unknown', as we may not know the method of any relation from literature, but
        #  if the fit was performed by XGA then something more useful can be passed
        self._fit_method = fit_method

        # Again if this relation was generated by XGA then these things will be pased, but we might not have (or want)
        #  the data used to generate scaling relations from literature. If there is data we will check if it has the
        #  correct units
        if x_data is not None and x_data.unit == self.x_unit:
            self._x_data = x_data
        elif x_data is not None and x_data.unit != self.x_unit:
            raise UnitConversionError('Any x data ({d}) passed must have the same units as the x_norm ({n}) '
                                      'argument'.format(d=x_data.unit.to_string(), n=self.x_unit.to_string()))
        else:
            # An empty quantity is defined if there is no data, rather than leaving it as None
            self._x_data = Quantity([], self.x_unit)

        if x_err is not None and x_err.unit == self.x_unit:
            self._x_err = x_err
        elif x_err is not None and x_err.unit != self.x_unit:
            raise UnitConversionError('Any x error ({d}) passed must have the same units as the x_norm ({n}) '
                                      'argument'.format(d=x_err.unit.to_string(), n=self.x_unit.to_string()))
        else:
            # An empty quantity is defined if there is no data, rather than leaving it as None
            self._x_err = Quantity([], self.x_unit)

        if y_data is not None and y_data.unit == self.y_unit:
            self._y_data = y_data
        elif y_data is not None and y_data.unit != self.y_unit:
            raise UnitConversionError('Any y data ({d}) passed must have the same units as the y_norm ({n}) '
                                      'argument'.format(d=y_data.unit.to_string(), n=self.y_unit.to_string()))
        else:
            # An empty quantity is defined if there is no data, rather than leaving it as None
            self._y_data = Quantity([], self.y_unit)

        if y_err is not None and y_err.unit == self.y_unit:
            self._y_err = y_err
        elif y_err is not None and y_err.unit != self.y_unit:
            raise UnitConversionError('Any y error ({d}) passed must have the same units as the y_norm ({n}) '
                                      'argument'.format(d=y_err.unit.to_string(), n=self.y_unit.to_string()))
        else:
            # An empty quantity is defined if there is no data, rather than leaving it as None
            self._y_err = Quantity([], self.y_unit)

        # Need to do a unit check (and I'll allow conversions in this case) on the x limits that may or may not
        #  have been passed by the user.
        if x_lims is not None and not x_lims.unit.is_equivalent(self.x_unit):
            raise UnitConversionError("Limits on the valid x range must be in units ({lu}) that are compatible with "
                                      "this relation's x units ({ru})".format(lu=x_lims.unit.to_string(),
                                                                              ru=self.x_unit.to_string()))
        elif x_lims is not None and x_lims.unit.is_equivalent(self.x_unit):
            self._x_lims = x_lims.to(self.x_unit)
        else:
            # If nothing is passed then its just None
            self._x_lims = x_lims

        # If the profile was created by XGA and fitted with ODR, then the user can pass the output object
        #  from that fitting method - I likely won't do much with it but it will be accessible
        self._odr_output = odr_output

        # If the profile was created by XGA and fitted by LIRA or emcee, the user can pass chains. I'll include the
        #  ability to make chain plots and corner plots as well.
        if chains is not None and chains.shape[1] != len(self._fit_pars):
            raise ValueError("The passed chains don't have an 2nd dimension length ({nd}) equal to the number of fit"
                             " parameters ({np}).".format(nd=chains.shape[1], np=len(self._fit_pars)))
        else:
            self._chains = chains

        # If the user hasn't passed the name of the relation then I'll generate one from what I know so far
        if relation_name is None:
            self._name = self._y_name + '-' + self._x_name + ' ' + self.fit_method
        else:
            self._name = relation_name

        # For relations from literature especially I need to give credit the author, and the original paper
        self._author = relation_author
        self._year = str(relation_year)
        self._doi = relation_doi

        # Just grabbing the parameter names from the model function to plot on the y-axis
        self._par_names = list(inspect.signature(self._model_func).parameters)[1:]

        self._scatter = scatter_par

        if chains is not None and scatter_chain is not None and len(scatter_chain) != chains.shape[0]:
            raise ValueError("There must be the same number of steps in any scatter and parameter chains passed "
                             "to this relation.")
        self._scatter_chain = scatter_chain

    @property
    def pars(self) -> np.ndarray:
        """
        The parameters that describe this scaling relation, along with their uncertainties. They are in the order in
        which they are expected to be passed into the model function.

        :return: A numpy array of the fit parameters and their uncertainties, first column are parameters,
            second column are uncertainties.
        :rtype: np.ndarray
        """
        return np.concatenate([self._fit_pars.reshape((len(self._fit_pars), 1)),
                               self._fit_par_errs.reshape((len(self._fit_pars), 1))], axis=1)

    @property
    def model_func(self):
        """
        Provides the model function used to fit this relation.

        :return: The Python function of this relation's model.
        """
        return self._model_func

    @property
    def x_name(self) -> str:
        """
        A string containing the name of the x-axis of this relation.

        :return: A Python string containing the name.
        :rtype: str
        """
        return self._x_name

    @property
    def y_name(self) -> str:
        """
        A string containing the name of the x-axis of this relation.

        :return: A Python string containing the name.
        :rtype: str
        """
        return self._y_name

    @property
    def x_norm(self) -> Quantity:
        """
        The astropy quantity containing the x-axis normalisation used during fitting.

        :return: An astropy quantity object.
        :rtype: Quantity
        """
        return self._x_norm

    @property
    def y_norm(self) -> Quantity:
        """
        The astropy quantity containing the y-axis normalisation used during fitting.

        :return: An astropy quantity object.
        :rtype: Quantity
        """
        return self._y_norm

    @property
    def x_unit(self) -> Unit:
        """
        The astropy unit object relevant to the x-axis of this relation.

        :return: An Astropy Unit object.
        :rtype: Unit
        """
        return self._x_norm.unit

    @property
    def y_unit(self) -> Unit:
        """
        The astropy unit object relevant to the y-axis of this relation.

        :return: An Astropy Unit object.
        :rtype: Unit
        """
        return self._y_norm.unit

    @property
    def x_data(self) -> Quantity:
        """
        An astropy Quantity of the x-data used to fit this relation, or an empty quantity if that data
        is not available. The first column is the data, the second is the uncertainties.

        :return: An Astropy Quantity object, containing the data and uncertainties.
        :rtype: Quantity
        """
        num_points = len(self._x_data)
        return np.concatenate([self._x_data.reshape((num_points, 1)), self._x_err.reshape((num_points, 1))], axis=1)

    @property
    def y_data(self) -> Quantity:
        """
        An astropy Quantity of the y-data used to fit this relation, or an empty quantity if that data
        is not available. The first column is the data, the second is the uncertainties.

        :return: An Astropy Quantity object, containing the data and uncertainties.
        :rtype: Quantity
        """
        num_points = len(self._y_data)
        return np.concatenate([self._y_data.reshape((num_points, 1)), self._y_err.reshape((num_points, 1))], axis=1)

    @property
    def x_lims(self) -> Quantity:
        """
        If the user passed an x range in which the relation is valid on initialisation, then this will
        return those limits in the same units as the x-axis.

        :return: A quantity containing upper and lower x limits, or None.
        :rtype: Quantity
        """
        return self._x_lims

    @property
    def fit_method(self) -> str:
        """
        A descriptor for the fit method used to generate this scaling relation.

        :return: A string containing the name of the fit method.
        :rtype: str
        """
        return self._fit_method

    @property
    def name(self) -> str:
        """
        A property getter for the name of the relation, this may not be unique in cases where no name was
        passed on declaration.

        :return: String containing the name of the relation.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, new_val: str):
        """
        A property setter for the name of the relation, it isn't a crucial quantity and the user may want to change
        it after declaration has already happened.

        :param str new_val:
        """
        self._name = new_val

    @property
    def author(self) -> str:
        """
        A property getter for the author of the relation, if not from literature it will be XGA.

        :return: String containing the name of the author.
        :rtype: str
        """
        return self._author

    @property
    def year(self) -> str:
        """
        A property getter for the year that the relation was created/published, if not from literature it will be the
        current year.

        :return: String containing the year of publication/creation.
        :rtype: str
        """
        return self._year

    @property
    def doi(self) -> str:
        """
        A property getter for the doi of the original paper of the relation, if not from literature it will an
        empty string.

        :return: String containing the doi.
        :rtype: str
        """
        return self._doi

    @property
    def scatter_par(self) -> np.ndarray:
        """
        A getter for the scatter information.

        :return: The scatter parameter and its uncertainty. If no scatter information was passed on definition
            then this will return None.
        :rtype: np.ndarray
        """
        return self._scatter

    @property
    def scatter_chain(self) -> np.ndarray:
        """
        A getter for the scatter information chain.

        :return: The scatter chain. If no scatter information was passed on definition then this will return None.
        :rtype: np.ndarray
        """
        return self._scatter_chain

    @property
    def chains(self) -> np.ndarray:
        """
        Property getter for the parameter chains.

        :return: The MCMC chains of the fit for this scaling relation, if they were passed. Otherwise None.
        :rtype: np.ndarray
        """
        return self._chains

    @property
    def par_names(self) -> List:
        """
        Getter for the parameter names.

        :return: The names of the model parameters.
        :rtype: List
        """
        return self._par_names

    def view_chains(self, figsize: tuple = None):
        """
        Simple view method to quickly look at the MCMC chains for a scaling relation fit.

        :param tuple figsize: Desired size of the figure, if None will be set automatically.
        """
        if self._chains is None:
            raise ValueError('No chains are available for this scaling relation')

        num_ch = len(self._fit_pars)
        if self._scatter_chain is not None:
            num_ch += 1

        if figsize is None:
            fig, axes = plt.subplots(nrows=num_ch, figsize=(12, 2 * num_ch), sharex='col')
        else:
            fig, axes = plt.subplots(num_ch, figsize=figsize, sharex='col')

        # Now we iterate through the parameters and plot their chains
        for i in range(len(self._fit_pars)):
            ax = axes[i]
            ax.plot(self._chains[:, i], "k", alpha=0.5)
            ax.set_xlim(0, self._chains.shape[0])
            ax.set_ylabel(self._par_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        if num_ch > len(self._fit_pars):
            ax = axes[-1]
            ax.plot(self._scatter_chain, "k", alpha=0.5)
            ax.set_xlim(0, len(self._scatter_chain))
            ax.set_ylabel(r'$\sigma$')
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step Number")
        plt.show()

    def view_corner(self, figsize: tuple = (10, 10), cust_par_names: List[str] = None,
                    colour: str = 'tab:gray', save_path: str = None):
        """
        A convenient view method to examine the corner plot of the parameter posterior distributions.

        :param tuple figsize: The size of the figure.
        :param List[str] cust_par_names: A list of custom parameter names. If the names include LaTeX code do not
            include $$ math environment symbols - you may also need to pass a string literal (e.g. r"\sigma"). Do
            not include an entry for a scatter parameter.
        :param List[str] colour: Colour for the contours, the default is tab:gray.
        :param str save_path: The path where the figure produced by this method should be saved. Default is None, in
            which case the figure will not be saved.
        """

        # Checks whether custom parameter names were passed, and if they were it checks whether there are the right
        #  number
        if cust_par_names is not None and len(cust_par_names) == len(self._par_names):
            par_names = cust_par_names
        elif cust_par_names is not None and len(cust_par_names) != len(self._par_names):
            raise ValueError("cust_par_names must have one entry per parameter of the scaling relation model.")
        else:
            par_names = self._par_names

        if self._chains is None:
            raise ValueError('No chains are available for this scaling relation')

        if self._scatter_chain is None:
            samp_obj = MCSamples(samples=self._chains, label=self.name, names=par_names, labels=par_names)
        else:
            par_names += [r'\sigma']
            all_ch = np.hstack([self._chains, self._scatter_chain[..., None]])
            samp_obj = MCSamples(samples=all_ch, label=self.name, names=par_names, labels=par_names)

        g = plots.get_subplot_plotter(width_inch=figsize[0])
        if colour is not None:
            g.triangle_plot(samp_obj, filled=True, contour_colors=[colour])
        else:
            g.triangle_plot(samp_obj, filled=True)

        # If the user passed a save_path value, then we assume they want to save the figure
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def predict(self, x_values: Quantity) -> Quantity:
        """
        This method allows for the prediction of y values from this scaling relation, you just need to pass in an
        appropriate set of x values.

        :param Quantity x_values: The x values to predict y values for.
        :return: The predicted y values
        :rtype: Quantity
        """
        # Got to check that people aren't passing any nonsense x quantities in
        if not x_values.unit.is_equivalent(self.x_unit):
            raise UnitConversionError('Values of x passed to the predict method ({xp}) must be convertible '
                                      'to the x-axis units of this scaling relation '
                                      '({xr}).'.format(xp=x_values.unit.to_string(), xr=self.x_unit.to_string()))

        # This is a check that all passed x values are within the validity limits of this relation (if the
        #  user passed those on init) - if they aren't a warning will be issued
        if self.x_lims is not None and len(x_values[(x_values < self.x_lims[0]) | (x_values > self.x_lims[1])]) != 0:
            warn("Some of the x values you have passed are outside the validity range of this relation "
                 "({l}-{h}{u}).".format(l=self.x_lims[0].value, h=self.x_lims[1].value, u=self.x_unit.to_string()))

        # Units that are convertible to the x-units of this relation are allowed, so we make sure we convert
        #  to the exact units the fit was done in. This includes dividing by the x_norm value
        x_values = x_values.to(self.x_unit) / self.x_norm
        # Then we just pass the x_values into the model, along with fit parameters. Then multiply by
        #  the y normalisation
        predicted_y = self._model_func(x_values.value, *self.pars[:, 0]) * self.y_norm

        return predicted_y

    def view(self, x_lims: Quantity = None, log_scale: bool = True, plot_title: str = None, figsize: tuple = (10, 8),
             data_colour: str = 'black', model_colour: str = 'grey', grid_on: bool = False, conf_level: int = 90,
             custom_x_label: str = None, custom_y_label: str = None, fontsize: float = 15, legend_fontsize: float = 13,
             x_ticks: list = None, x_minor_ticks: list = None, y_ticks: list = None, y_minor_ticks: list = None,
             save_path: str = None):
        """
        A method that produces a high quality plot of this scaling relation (including the data it is based upon,
        if available).

        :param Quantity x_lims: If not set, this method will attempt to take appropriate limits from the x-data
            this relation is based upon, if that data is not available an error will be thrown.
        :param bool log_scale: If true then the x and y axes of the plot will be log-scaled.
        :param str plot_title: A custom title to be used for the plot, otherwise one will be generated automatically.
        :param tuple figsize: A custom figure size for the plot, default is (8, 8).
        :param str data_colour: The colour to use for the data points in the plot, default is black.
        :param str model_colour: The colour to use for the model in the plot, default is grey.
        :param bool grid_on: If True then a grid will be included on the plot. Default is True.
        :param int conf_level: The confidence level to use when plotting the model.
        :param str custom_x_label: Passing a string to this variable will override the x axis label
            of this plot, including the unit string.
        :param str custom_y_label: Passing a string to this variable will override the y axis label
            of this plot, including the unit string.
        :param float fontsize: The fontsize for axis labels.
        :param float legend_fontsize: The fontsize for text in the legend.
        :param list x_ticks: Customise which major x-axis ticks and labels are on the figure, default is None in which
            case they are determined automatically.
        :param list x_minor_ticks: Customise which minor x-axis ticks and labels are on the figure, default is
            None in which case they are determined automatically.
        :param list y_ticks: Customise which major y-axis ticks and labels are on the figure, default is None in which
            case they are determined automatically.
        :param list y_minor_ticks: Customise which minor y-axis ticks and labels are on the figure, default is
            None in which case they are determined automatically.
        :param str save_path: The path where the figure produced by this method should be saved. Default is None, in
            which case the figure will not be saved.
        """
        # First we check that the passed axis limits are in appropriate units, if they weren't supplied then we check
        #  if any were supplied at initialisation, if that isn't the case then we make our own from the data, and
        #  if there's no data then we get stroppy
        if x_lims is not None and x_lims.unit.is_equivalent(self.x_unit):
            x_lims = x_lims.to(self.x_unit).value
        elif self.x_lims is not None:
            x_lims = self.x_lims.value
        elif x_lims is not None and not x_lims.unit.is_equivalent(self.x_unit):
            raise UnitConversionError('Limits on the x-axis ({xl}) must be convertible to the x-axis units of this '
                                      'scaling relation ({xr}).'.format(xl=x_lims.unit.to_string(),
                                                                        xr=self.x_unit.to_string()))
        elif x_lims is None and len(self._x_data) != 0:
            max_x_ind = np.argmax(self._x_data)
            min_x_ind = np.argmin(self._x_data)
            x_lims = [0.9*(self._x_data[min_x_ind].value - self._x_err[min_x_ind].value),
                      1.1*(self._x_data[max_x_ind].value + self._x_err[max_x_ind].value)]
        elif x_lims is None and len(self._x_data) == 0:
            raise ValueError('There is no data available to infer suitable axis limits from, please pass x limits.')

        # Setting up the matplotlib figure
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        ax = plt.gca()

        # Setting the axis limits
        ax.set_xlim(x_lims)

        # Making the scale log if requested
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        # Setup the aesthetics of the axis
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        # Plot the data with uncertainties, if any data is present in this scaling relation.
        if len(self.x_data) != 0:
            ax.errorbar(self._x_data.value, self._y_data.value, xerr=self._x_err.value, yerr=self._y_err.value,
                        fmt="x", color=data_colour, capsize=2)

        # Need to randomly sample from the fitted model
        num_rand = 10000
        model_pars = np.repeat(self._fit_pars[..., None], num_rand, axis=1).T
        model_par_errs = np.repeat(self._fit_par_errs[..., None], num_rand, axis=1).T

        model_par_dists = np.random.normal(model_pars, model_par_errs)

        model_x = np.linspace(*(x_lims / self.x_norm.value), 100)
        model_xs = np.repeat(model_x[..., None], num_rand, axis=1)

        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        model_realisations = self._model_func(model_xs, *model_par_dists.T) * self._y_norm
        model_mean = np.mean(model_realisations, axis=1)
        model_lower = np.percentile(model_realisations, lower, axis=1)
        model_upper = np.percentile(model_realisations, upper, axis=1)

        # I want the name of the function to include in labels and titles, but if its one defined in XGA then
        #  I can grab the publication version of the name - it'll be prettier
        mod_name = self._model_func.__name__
        for m_name in MODEL_PUBLICATION_NAMES:
            mod_name = mod_name.replace(m_name, MODEL_PUBLICATION_NAMES[m_name])

        relation_label = " ".join([self._author, self._year, '-', mod_name,
                                   "- {cf}% Confidence".format(cf=conf_level)])
        plt.plot(model_x * self._x_norm.value, self._model_func(model_x, *model_pars[0, :]) * self._y_norm.value,
                 color=model_colour, label=relation_label)

        plt.plot(model_x * self._x_norm.value, model_upper, color=model_colour, linestyle="--")
        plt.plot(model_x * self._x_norm.value, model_lower, color=model_colour, linestyle="--")
        ax.fill_between(model_x * self._x_norm.value, model_lower, model_upper, where=model_upper >= model_lower,
                        facecolor=model_colour, alpha=0.6, interpolate=True)

        # I can dynamically grab the units in LaTeX formatting from the Quantity objects (thank you astropy)
        #  However I've noticed specific instances where the units can be made prettier
        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + self.x_unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + self.y_unit.to_string("latex").strip("$") + r"\right]$"

        # Dimensionless quantities can be fitted too, and this make the axis label look nicer by not having empty
        #  square brackets
        if x_unit == r"$\left[\\mathrm{}\right]$":
            x_unit = ''
        if y_unit == r"$\left[\\mathrm{}\right]$":
            y_unit = ''

        # The scaling relation object knows what its x and y axes are called, though the user may pass
        #  their own if they wish
        if custom_x_label is None:
            plt.xlabel("{xn} {un}".format(xn=self._x_name, un=x_unit), fontsize=fontsize)
        else:
            plt.xlabel(custom_x_label, fontsize=fontsize)

        if custom_y_label is None:
            plt.ylabel("{yn} {un}".format(yn=self._y_name, un=y_unit), fontsize=fontsize)
        else:
            plt.ylabel(custom_y_label, fontsize=fontsize)

        # The user can also pass a plot title, but if they don't then I construct one automatically
        if plot_title is None and self._fit_method != 'unknown':
            plot_title = 'Scaling Relation - {mod} fitted with {fm}'.format(mod=mod_name, fm=self._fit_method)
        elif plot_title is None and self._fit_method == 'unknown':
            plot_title = '{n} Scaling Relation'.format(n=self._name)

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
        #  Though as I don't much like the look of the grid it is off by default, and users can enable it if they
        #  want to.
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

        # Here we check whether the user has manually set any of the ticks to be displayed (major or minor, either axis)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
        if x_minor_ticks is not None:
            ax.set_xticks(x_minor_ticks, minor=True)
            ax.set_xticklabels(x_minor_ticks, minor=True)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks)
        if y_minor_ticks is not None:
            ax.set_xticks(y_minor_ticks, minor=True)
            ax.set_xticklabels(y_minor_ticks, minor=True)

        plt.legend(loc="best", fontsize=legend_fontsize)
        plt.tight_layout()

        # If the user passed a save_path value, then we assume they want to save the figure
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def __add__(self, other):
        to_combine = [self]
        if type(other) == list:
            to_combine += other
        elif isinstance(other, ScalingRelation):
            to_combine.append(other)
        elif isinstance(other, AggregateScalingRelation):
            to_combine += other.relations
        else:
            raise TypeError("You may only add ScalingRelations, or a list of ScalingRelations, to this object.")
        return AggregateScalingRelation(to_combine)


class AggregateScalingRelation:
    """
    This class is akin to the BaseAggregateProfile class, in that it is the result of a sum of ScalingRelation
    objects. References to the component objects will be stored within the structure of this class, and it primarily
    exists to allow plots with multiple relations to be generated.
    """
    def __init__(self, relations: List[ScalingRelation]):
        # There aren't specific classes for different types of relations, but I do need to check that whatever
        #  relations are being added together have the same x and y units
        x_units = [sr.x_unit for sr in relations]
        if len(set(x_units)) != 1:
            raise UnitConversionError("All component scaling relations must have the same x-axis units.")
        y_units = [sr.y_unit for sr in relations]
        if len(set(y_units)) != 1:
            raise UnitConversionError("All component scaling relations must have the same y-axis units.")

        # Set some unit attributes for this class just so it takes one call to retrieve them
        self._x_unit = relations[0].x_unit
        self._y_unit = relations[0].y_unit

        # Making sure that the axis units match is the key check before allowing this class to be instantiated, but
        #  I'm also going to go through and see if the names of the x and y axes are the same and issue warnings if
        #  not
        x_names = [sr.x_name for sr in relations]
        if len(set(x_names)) != 1:
            self._x_name = " or ".join(list(set(x_names)))
            warn('Not all of these ScalingRelations have the same x-axis names.')
        else:
            self._x_name = relations[0].x_name

        y_names = [sr.y_name for sr in relations]
        if len(set(y_names)) != 1:
            self._y_name = " or ".join(list(set(y_names)))
            warn('Not all of these ScalingRelations have the same y-axis names.')
        else:
            self._y_name = relations[0].y_name

        # This stores the relations as an attribute
        self._relations = relations

    # The relations are the key attribute of this class, and the mechanism I've set up is that these objects
    #  are created by adding ScalingRelations together, as such there will be no setter to alter this after
    #  declaration
    @property
    def relations(self) -> List[ScalingRelation]:
        """
        This returns the list of ScalingRelation instances that make up this aggregate scaling relation.

        :return: A list of ScalingRelation instances.
        :rtype: List[ScalingRelation]
        """
        return self._relations

    @property
    def x_unit(self) -> Unit:
        """
        The astropy unit object relevant to the x-axis of this relation.

        :return: An Astropy Unit object.
        :rtype: Unit
        """
        return self._x_unit

    @property
    def y_unit(self) -> Unit:
        """
        The astropy unit object relevant to the y-axis of this relation.

        :return: An Astropy Unit object.
        :rtype: Unit
        """
        return self._y_unit

    def view_corner(self, figsize: tuple = (10, 10), cust_par_names: List[str] = None,
                    contour_colours: List[str] = None, save_path: str = None):
        """
        A corner plot viewing method that will combine chains from all the relations that make up this
        aggregate scaling relation and display them using getdist.

        :param tuple figsize: The size of the figure.
        :param List[str] cust_par_names: A list of custom parameter names. If the names include LaTeX code do not
            include $$ math environment symbols - you may also need to pass a string literal (e.g. r"\sigma"). Do
            not include an entry for a scatter parameter.
        :param List[str] contour_colours: Custom colours for the contours, there should be one colour
            per scaling relation.
        :param str save_path: The path where the figure produced by this method should be saved. Default is None, in
            which case the figure will not be saved.
        """
        # First off checking that every relation has chains, otherwise we can't do this
        not_chains = [r.chains is None for r in self._relations]
        # Which parameter names are the same, they should all be the same
        par_names = list(set([",".join(r.par_names) for r in self._relations]))
        # Checking which relations also have a scatter chain
        not_scatter_chains = [r.scatter_chain is None for r in self._relations]

        # Stopping this method if anything is amiss
        if any(not_chains):
            raise ValueError('Not all scaling relations have parameter chains, cannot view aggregate corner plot.')
        elif len(par_names) != 1:
            raise ValueError('Not all scaling relations have the same model parameter names, cannot view aggregate'
                             ' corner plot.')
        elif len(contour_colours) != len(self._relations):
            raise ValueError("If you pass a list of contour colours, there must be one entry per scaling relation.")

        # The number of non-scatter parameters in the scaling relation models
        num_pars = len(self._relations[0].par_names)

        samples = []
        # Need to remove $ from the labels because getdist adds them itself
        par_names = [n.replace('$', '') for n in self._relations[0].par_names]
        # Setup the getdist sample objects
        if not any(not_scatter_chains):
            # For if there ARE scatter chains
            if cust_par_names is not None and len(cust_par_names) == num_pars:
                par_names = cust_par_names

            par_names += [r'\sigma']
            for rel in self._relations:
                all_ch = np.hstack([rel.chains, rel.scatter_chain[..., None]])
                samp_obj = MCSamples(samples=all_ch, label=rel.name, names=par_names, labels=par_names)
                samples.append(samp_obj)
        else:
            if cust_par_names is not None and len(cust_par_names) == num_pars:
                par_names = cust_par_names

            for rel in self._relations:
                # For if there aren't scatter chains
                samp_obj = MCSamples(samples=rel.chains, label=rel.name, names=par_names, labels=par_names)
                samples.append(samp_obj)

        # And generate the triangle plot
        g = plots.get_subplot_plotter(width_inch=figsize[0])

        if contour_colours is not None:
            g.triangle_plot(samples, filled=True, contour_colors=contour_colours)
        else:
            g.triangle_plot(samples, filled=True)

        # If the user passed a save_path value, then we assume they want to save the figure
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def view(self, x_lims: Quantity = None, log_scale: bool = True, plot_title: str = None, figsize: tuple = (10, 8),
             colour_list: list = None, grid_on: bool = False, conf_level: int = 90, show_data: bool = True,
             fontsize: float = 15, legend_fontsize: float = 13, x_ticks: list = None, x_minor_ticks: list = None,
             y_ticks: list = None, y_minor_ticks: list = None, save_path: str = None):
        """
        A method that produces a high quality plot of the component scaling relations in this
        AggregateScalingRelation.

        :param Quantity x_lims: If not set, this method will attempt to take appropriate limits from the x-data
            this relation is based upon, if that data is not available an error will be thrown.
        :param bool log_scale: If true then the x and y axes of the plot will be log-scaled.
        :param str plot_title: A custom title to be used for the plot, otherwise one will be generated automatically.
        :param tuple figsize: A custom figure size for the plot, default is (8, 8).
        :param list colour_list: A list of matplotlib colours to use as a custom colour cycle.
        :param bool grid_on: If True then a grid will be included on the plot. Default is True.
        :param int conf_level: The confidence level to use when plotting the model.
        :param bool show_data: Controls whether data points are shown on the view, as it can quickly become
            confusing with multiple relations on one axis.
        :param float fontsize: The fontsize for axis labels.
        :param float legend_fontsize: The fontsize for text in the legend.
        :param list x_ticks: Customise which major x-axis ticks and labels are on the figure, default is None in which
            case they are determined automatically.
        :param list x_minor_ticks: Customise which minor x-axis ticks and labels are on the figure, default is
            None in which case they are determined automatically.
        :param list y_ticks: Customise which major y-axis ticks and labels are on the figure, default is None in which
            case they are determined automatically.
        :param list y_minor_ticks: Customise which minor y-axis ticks and labels are on the figure, default is
            None in which case they are determined automatically.
        :param str save_path: The path where the figure produced by this method should be saved. Default is None, in
            which case the figure will not be saved.
        """
        # Very large chunks of this are almost direct copies of the view method of ScalingRelation, but this
        #  was the easiest way of setting this up so I think the duplication is justified.

        # Set up the colour cycle
        if colour_list is None:
            colour_list = PRETTY_COLOUR_CYCLE
        new_col_cycle = cycler(color=colour_list)

        # This part decides the x_lims of the plot, much the same as in the ScalingRelation view but it works
        #  on a combined sets of x-data or combined built in validity ranges, though user limits passed to view
        #  will still override everything else
        comb_x_data = np.concatenate([sr.x_data for sr in self._relations])

        # Combining any x_lims defined at init for these relations is slightly more complicated, as if they weren't
        #  defined then they will be None, so I have different behaviours dependent on how many sets of built in
        #  x_lims there are
        existing_x_lims = [sr.x_lims for sr in self._relations if sr.x_lims is not None]
        if len(existing_x_lims) == 0:
            comb_x_lims = None
        elif len(existing_x_lims) == 1:
            comb_x_lims = existing_x_lims[0]
        else:
            comb_x_lims = np.concatenate(existing_x_lims)

        if x_lims is not None and not x_lims.unit.is_equivalent(self.x_unit):
            raise UnitConversionError('Limits on the x-axis ({xl}) must be convertible to the x-axis units of this '
                                      'scaling relation ({xr}).'.format(xl=x_lims.unit.to_string(),
                                                                        xr=self.x_unit.to_string()))
        elif x_lims is not None and x_lims.unit.is_equivalent(self.x_unit):
            x_lims = x_lims.to(self.x_unit).value
        elif comb_x_lims is not None:
            x_lims = np.array([comb_x_lims.value.min(), comb_x_lims.value.max()])
        elif x_lims is None and len(comb_x_data) != 0:
            max_x_ind = np.argmax(comb_x_data[:, 0])
            min_x_ind = np.argmin(comb_x_data[:, 0])
            x_lims = [0.9 * (comb_x_data[min_x_ind, 0].value - comb_x_data[min_x_ind, 1].value),
                      1.1 * (comb_x_data[max_x_ind, 0].value + comb_x_data[max_x_ind, 1].value)]
        elif x_lims is None and len(comb_x_data) == 0:
            raise ValueError('There is no data available to infer suitable axis limits from, please pass x limits.')

        # Setting up the matplotlib figure
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        ax = plt.gca()
        ax.set_prop_cycle(new_col_cycle)

        # Setting the axis limits
        ax.set_xlim(x_lims)

        # Making the scale log if requested
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        # Setup the aesthetics of the axis
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        for rel in self._relations:
            # This is a horrifying bodge, but I do just want the colour out and I can't be bothered to figure out
            #  how to use the colour cycle object properly
            if len(rel.x_data.value[:, 0]) == 0 or not show_data:
                # Sets up a null error bar instance for the colour basically
                d_out = ax.errorbar(None, None, xerr=None, yerr=None, fmt="x", capsize=2, label='')
            else:
                d_out = ax.errorbar(rel.x_data.value[:, 0], rel.y_data.value[:, 0], xerr=rel.x_data.value[:, 1],
                                    yerr=rel.y_data.value[:, 1], fmt="x", capsize=2)

            d_colour = d_out[0].get_color()

            # Need to randomly sample from the fitted model
            num_rand = 10000
            model_pars = np.repeat(rel.pars[:, 0, None], num_rand, axis=1).T
            model_par_errs = np.repeat(rel.pars[:, 1, None], num_rand, axis=1).T

            model_par_dists = np.random.normal(model_pars, model_par_errs)

            model_x = np.linspace(*(x_lims / rel.x_norm.value), 100)
            model_xs = np.repeat(model_x[..., None], num_rand, axis=1)

            upper = 50 + (conf_level / 2)
            lower = 50 - (conf_level / 2)

            model_realisations = rel.model_func(model_xs, *model_par_dists.T) * rel._y_norm
            model_mean = np.mean(model_realisations, axis=1)
            model_lower = np.percentile(model_realisations, lower, axis=1)
            model_upper = np.percentile(model_realisations, upper, axis=1)

            # I want the name of the function to include in labels and titles, but if its one defined in XGA then
            #  I can grab the publication version of the name - it'll be prettier
            mod_name = rel.model_func.__name__
            for m_name in MODEL_PUBLICATION_NAMES:
                mod_name = mod_name.replace(m_name, MODEL_PUBLICATION_NAMES[m_name])

            if rel.author != 'XGA':
                relation_label = " ".join([rel.author, rel.year])
            else:
                relation_label = rel.name + ' Scaling Relation'
            plt.plot(model_x * rel.x_norm.value, rel.model_func(model_x, *model_pars[0, :]) * rel.y_norm.value,
                     color=d_colour, label=relation_label)

            plt.plot(model_x * rel.x_norm.value, model_upper, color=d_colour, linestyle="--")
            plt.plot(model_x * rel.x_norm.value, model_lower, color=d_colour, linestyle="--")
            ax.fill_between(model_x * rel.x_norm.value, model_lower, model_upper, where=model_upper >= model_lower,
                            facecolor=d_colour, alpha=0.6, interpolate=True)

        # I can dynamically grab the units in LaTeX formatting from the Quantity objects (thank you astropy)
        #  However I've noticed specific instances where the units can be made prettier
        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + self.x_unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + self.y_unit.to_string("latex").strip("$") + r"\right]$"

        # Dimensionless quantities can be fitted too, and this make the axis label look nicer by not having empty
        #  square brackets
        if x_unit == r"$\left[\\mathrm{}\right]$":
            x_unit = ''
        if y_unit == r"$\left[\\mathrm{}\right]$":
            y_unit = ''

        # The scaling relation object knows what its x and y axes are called
        plt.xlabel("{xn} {un}".format(xn=self._x_name, un=x_unit), fontsize=fontsize)
        plt.ylabel("{yn} {un}".format(yn=self._y_name, un=y_unit), fontsize=fontsize)

        # The user can also pass a plot title, but if they don't then I construct one automatically
        if plot_title is None:
            plot_title = 'Scaling Relation Comparison - {c}% Confidence Limits'.format(c=conf_level)

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
        #  Though as I don't much like the look of the grid it is off by default, and users can enable it if they
        #  want to.
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

        # Here we check whether the user has manually set any of the ticks to be displayed (major or minor, either axis)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
        if x_minor_ticks is not None:
            ax.set_xticks(x_minor_ticks, minor=True)
            ax.set_xticklabels(x_minor_ticks, minor=True)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks)
        if y_minor_ticks is not None:
            ax.set_xticks(y_minor_ticks, minor=True)
            ax.set_xticklabels(y_minor_ticks, minor=True)

        plt.legend(loc="best", fontsize=legend_fontsize)
        plt.tight_layout()

        # If the user passed a save_path value, then we assume they want to save the figure
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def __len__(self) -> int:
        return len(self._relations)

    def __add__(self, other):
        to_combine = self.relations
        if type(other) == list:
            to_combine += other
        elif isinstance(other, ScalingRelation):
            to_combine.append(other)
        elif isinstance(other, AggregateScalingRelation):
            to_combine += other.relations
        else:
            raise TypeError("You may only add ScalingRelations, AggregateScalingRelations, or a "
                            "list of ScalingRelations.")
        return AggregateScalingRelation(to_combine)






















