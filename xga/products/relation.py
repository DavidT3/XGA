#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 09/12/2020, 17:26. Copyright (c) David J Turner

import inspect

import corner
import numpy as np
import scipy.odr as odr
from astropy.units import Quantity, Unit, UnitConversionError
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from ..models import MODEL_PUBLICATION_NAMES

# This is just to make some instances of astropy LaTeX units prettier for plotting
PRETTY_UNITS = {'solMass': r'M$_{\odot}$', 'erg / s': r"erg s$^{-1}$"}

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
                 y_data: Quantity = None, x_err: Quantity = None, y_err: Quantity = None,
                 odr_output: odr.Output = None, chains: np.ndarray = None, relation_name: str = None,
                 relation_author: str = 'Turner et al. with XGA', relation_doi: str = ''):
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
        :param odr.Output odr_output: The orthogonal distance regression output object associated with this
        relation's fit, if available and applicable.
        :param np.ndarray chains: The parameter chains associated with this relation's fit, if available and
        applicable. They should be of shape N_stepxN_par, where N_steps is the number of steps (after burn-in
        is removed), and N_par is the number of parameters in the fit.
        :param str relation_name: A suitable name for this relation.
        :param str relation_author: The author who deserves credit for this relation.
        :param str relation_doi: The DOI of the original paper this relation appeared in.
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
            self._name = self._y_name + '-' + self._x_name
        else:
            self._name = relation_name

        # For relations from literature especially I need to give credit the author, and the original paper
        self._author = relation_author
        self._doi = relation_doi

        # Just grabbing the parameter names from the model function to plot on the y-axis
        self._par_names = list(inspect.signature(self._model_func).parameters)[1:]

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
        A property getter for the author of the relation, if not from literature it will be the name of the author
        of XGA.
        :return: String containing the name of the author.
        :rtype: str
        """
        return self._author

    @property
    def doi(self) -> str:
        """
        A property getter for the doi of the original paper of the relation, if not from literature it will an
        empty string.
        :return: String containing the doi.
        :rtype: str
        """
        return self._doi

    def view_chains(self, figsize: tuple = None):
        """
        Simple view method to quickly look at the MCMC chains for a scaling relation fit.
        :param tuple figsize: Desired size of the figure, if None will be set automatically.
        """
        if self._chains is None:
            raise ValueError('No chains are available for this scaling relation')

        if figsize is None:
            fig, axes = plt.subplots(nrows=len(self._fit_pars), figsize=(12, 2 * len(self._fit_pars)), sharex='col')
        else:
            fig, axes = plt.subplots(len(self._fit_pars), figsize=figsize, sharex='col')

        # Now we iterate through the parameters and plot their chains
        for i in range(len(self._fit_pars)):
            ax = axes[i]
            ax.plot(self._chains[:, i], "k", alpha=0.5)
            ax.set_xlim(0, self._chains.shape[0])
            ax.set_ylabel(self._par_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step Number")
        plt.show()

    def view_corner(self, figsize: tuple = (10, 10), conf_level: int = 90):
        """
        A convenient view method to examine the corner plot of the parameter posterior distributions.
        :param Tuple figsize: The desired figure size.
        :param int conf_level: The confidence level to use when indicating confidence limits on the distributions.
        """
        if self._chains is None:
            raise ValueError('No chains are available for this scaling relation')

        frac_conf_lev = [(50 - (conf_level / 2)) / 100, 0.5, (50 + (conf_level / 2)) / 100]
        fig = corner.corner(self._chains, labels=self._par_names, figsize=figsize, quantiles=frac_conf_lev,
                            show_titles=True)
        plt.suptitle("{n} Scaling Relation - {c}% Confidence".format(n=self._name, c=conf_level), fontsize=14, y=1.02)
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

        # Units that are convertible to the x-units of this relation are allowed, so we make sure we convert
        #  to the exact units the fit was done in. This includes dividing by the x_norm value
        x_values = x_values.to(self.x_unit) / self.x_norm
        # Then we just pass the x_values into the model, along with fit parameters. Then multiply by
        #  the y normalisation
        predicted_y = self._model_func(x_values.value, *self.pars[:, 0]) * self.y_norm

        return predicted_y

    def view(self, x_lims: Quantity = None, log_scale: bool = True, plot_title: str = None, figsize: tuple = (8, 8),
             data_colour: str = 'black', model_colour: str = 'grey', grid_on: bool = False, conf_level: int = 90):
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
        """
        # First we check that the passed axis limits are in appropriate units, if they weren't supplied
        #  then we make our own from the data, if there's no data then we get stroppy
        if x_lims is not None and x_lims.unit.is_equivalent(self.x_unit):
            x_lims = x_lims.to(self.x_unit).value
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

        # Plot the data with uncertainties, if any data is present in this scaling relation. If not then
        #  even though this command is called no data will appear because the x_data and y_data variables
        #  are empty quantities
        ax.errorbar(self._x_data.value, self._y_data.value, xerr=self._x_err.value, yerr=self._y_err.value,
                    fmt="x", color=data_colour, capsize=2, label=self._name + " Data")

        # Need to randomly sample from the fitted model
        num_rand = 300
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

        relation_label = " ".join([self._author, '-', mod_name,  "- {cf}% Confidence".format(cf=conf_level)])
        plt.plot(model_x * self._x_norm.value, self._model_func(model_x, *model_pars[0, :]) * self._y_norm.value,
                 color=model_colour, label=relation_label)

        plt.plot(model_x * self._x_norm.value, model_upper, color=model_colour, linestyle="--")
        plt.plot(model_x * self._x_norm.value, model_lower, color=model_colour, linestyle="--")
        ax.fill_between(model_x * self._x_norm.value, model_lower, model_upper, where=model_upper >= model_lower,
                        facecolor=model_colour, alpha=0.6, interpolate=True)

        # I can dynamically grab the units in LaTeX formatting from the Quantity objects (thank you astropy)
        #  However I've noticed specific instances where the units can be made prettier
        x_unit = '[' + self.x_unit.to_string() + ']'
        y_unit = '[' + self.y_unit.to_string() + ']'
        for og_unit in PRETTY_UNITS:
            x_unit = x_unit.replace(og_unit, PRETTY_UNITS[og_unit])
            y_unit = y_unit.replace(og_unit, PRETTY_UNITS[og_unit])

        # Dimensionless quantities can be fitted too, and this make the axis label look nicer by not having empty
        #  square brackets
        if x_unit == '[]':
            x_unit = ''
        if y_unit == '[]':
            y_unit = ''

        # The scaling relation object knows what its x and y axes are called
        plt.xlabel("{xn} {un}".format(xn=self._x_name, un=x_unit), fontsize=12)
        plt.ylabel("{yn} {un}".format(yn=self._y_name, un=y_unit), fontsize=12)

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

        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


class AggregateScalingRelation:
    def __init__(self):
        raise NotImplementedError("I'll get to this very soon")























