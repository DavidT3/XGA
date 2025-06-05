#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 05/06/2025, 11:36. Copyright (c) The Contributors

import inspect
import pickle
from copy import deepcopy
from datetime import date
from typing import List, Union, Tuple
from warnings import warn

import numpy as np
import scipy.odr as odr
from astropy.cosmology import Cosmology
from astropy.units import Quantity, Unit, UnitConversionError
from cycler import cycler
from getdist import plots, MCSamples
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TABLEAU_COLORS, BASE_COLORS, Colormap, CSS4_COLORS, Normalize
from matplotlib.ticker import FuncFormatter

from ..models import MODEL_PUBLICATION_NAMES, power_law

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
    :param float/int dim_hubb_ind: This is used to tell the ScalingRelation which power of E(z) has been applied
        to the y-axis data, this can then be used by the predict method to remove the E(z) contribution from
        predictions. The default is None.
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
    :param str model_colour: This variable can be used to set the colour that the fit should be displayed in
        when plotting. Setting it at definition or setting the property means that the colour doesn't have
        to be set for every view method, and it will be remembered when multiple relations are viewed together.
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
    """
    def __init__(self, fit_pars: np.ndarray, fit_par_errs: np.ndarray, model_func, x_norm: Quantity, y_norm: Quantity,
                 x_name: str, y_name: str, dim_hubb_ind=None, fit_method: str = 'unknown', x_data: Quantity = None,
                 y_data: Quantity = None, x_err: Quantity = None, y_err: Quantity = None, x_lims: Quantity = None,
                 odr_output: odr.Output = None, chains: np.ndarray = None, relation_name: str = None,
                 relation_author: str = 'XGA', relation_year: str = str(date.today().year), relation_doi: str = '',
                 scatter_par: np.ndarray = None, scatter_chain: np.ndarray = None, model_colour: str = None,
                 point_names: Union[np.ndarray, list] = None, third_dim_info: Union[np.ndarray, Quantity] = None,
                 third_dim_name: str = None, x_en_bounds: Quantity = None, y_en_bounds: Quantity = None):
        """
        The init for the ScalingRelation class, all information necessary to enable the different functions of
        this class will be supplied by the user here.
        """
        # These will always be passed in, and are assumed to be in the order required by the model_func that is also
        #  passed in by the user.
        self._fit_pars = np.array(fit_pars)
        self._fit_par_errs = np.array(fit_par_errs)

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

        # Wanted the relation to know if it had some power of E(z) applied to the y-axis data - this is quite common
        #  in galaxy cluster scaling relations to account for cosmological evolution of certain parameters
        self._ez_power = dim_hubb_ind

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

        # We are double-checking that the values for the x and y energy bounds for the measured quantities are legal
        if x_en_bounds is not None and (not isinstance(x_en_bounds, Quantity)
                                        or (isinstance(x_en_bounds, Quantity) and x_en_bounds.isscalar)
                                        or (isinstance(x_en_bounds, Quantity) and not x_en_bounds.isscalar and
                                            len(x_en_bounds) != 2)):
            raise TypeError("The 'x_en_bounds' argument must be either None, or a non-scalar Astropy Quantity with "
                            "two entries, the lower energy bound and the upper energy bound.")
        elif x_en_bounds is not None and x_en_bounds[0] >= x_en_bounds[1]:
            raise ValueError("The first entry in 'x_en_bounds' is larger than or equal to the second entry, as the "
                             "first entry is meant to be the lower energy bound this is not permitted.")

        if y_en_bounds is not None and (not isinstance(y_en_bounds, Quantity)
                                        or (isinstance(y_en_bounds, Quantity) and y_en_bounds.isscalar)
                                        or (isinstance(y_en_bounds, Quantity) and not y_en_bounds.isscalar and
                                            len(y_en_bounds) != 2)):
            raise TypeError("The 'y_en_bounds' argument must be either None, or a non-scalar Astropy Quantity with "
                            "two entries, the lower energy bound and the upper energy bound.")
        elif y_en_bounds is not None and y_en_bounds[0] >= y_en_bounds[1]:
            raise ValueError("The first entry in 'y_en_bounds' is larger than or equal to the second entry, as the "
                             "first entry is meant to be the lower energy bound this is not permitted.")

        # If we get this far then the energy bounds are fine, so we store them
        self._x_quantity_en_bounds = x_en_bounds
        self._y_quantity_en_bounds = y_en_bounds

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

        # If the user hasn't passed the name of the relation then the view method will generate one for itself, which
        #  is the main place that it is used
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

        # This sets an internal colour attribute so the default plotting colour is always the one that the
        #  user defined
        self._model_colour = model_colour

        # This checks the input for 'point_names', which can be used to associate each data point in this scaling
        #  relation with a source name so that outliers can be properly investigated.
        if (x_data is None or y_data is None) and point_names is not None:
            raise ValueError("You cannot set the 'point_names' argument if you have not passed data to "
                             "x_data and y_data.")
        elif point_names is not None and len(point_names) != len(x_data):
            raise ValueError("You have passed a 'point_names' argument that has a different number of entries ({dn}) "
                             "than the data given to this scaling relation ({d}).".format(dn=len(point_names),
                                                                                          d=len(x_data)))
        else:
            self._point_names = point_names

        # The user is allowed to pass information that can be used to colour the data points of a scaling relation
        #  when it is viewed. Here we check that, if present, the extra data are the right shape
        if (x_data is None or y_data is None) and third_dim_info is not None:
            raise ValueError("You cannot set the 'third_dim_info' argument if you have not passed data to "
                             "x_data and y_data.")
        elif third_dim_info is not None and len(third_dim_info) != len(x_data):
            raise ValueError("You have passed a 'third_dim_info' argument that has a different number of "
                             "entries ({dn}) than the data given to this scaling relation "
                             "({d}).".format(dn=len(third_dim_info), d=len(x_data)))
        elif third_dim_info is not None and third_dim_info.ndim != 1:
            raise ValueError("Only single-dimension Quantities are accepted by 'third_dim_info'.")
        elif third_dim_info is not None and third_dim_name is None:
            raise ValueError("If 'third_dim_info' is set, then the 'third_dim_name' argument must be as well.")
        elif third_dim_info is None and third_dim_name is not None:
            # If the user accidentally passed a name but no data then I will just null the name and let them carry on
            #  with a warning
            third_dim_name = None
            warn("A value was passed to 'third_dim_name' without a corresponding 'third_dim_info' "
                 "value, 'third_dim_name' has been set to None.", stacklevel=2)
        # Setting the attributes, if we've gotten this far then there are no problems
        self._third_dim_info = third_dim_info
        self._third_dim_name = third_dim_name

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
    def dimensionless_hubble_parameter(self) -> Union[float, int]:
        """
        This property should be set on the declaration of a scaling relation, and exists to tell the relation what
        power of E(z) has been applied to the y-axis data before fitting. This also helps the predict method remove
        the E(z) contribution (if any) from predictions.

        :return: The power of E(z) applied to the y-axis data before fitting. Default is None.
        :rtype: float/int
        """
        return self._ez_power

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

    @author.setter
    def author(self, new_val: str):
        """
        Property setter for the author of the relation.

        :param str new_val: The new author string.
        """
        if not isinstance(new_val, str):
            raise TypeError('You must set the author property with a string.')
        self._author = new_val

    @property
    def year(self) -> str:
        """
        A property getter for the year that the relation was created/published, if not from literature it will be the
        current year.

        :return: String containing the year of publication/creation.
        :rtype: str
        """
        return self._year

    @year.setter
    def year(self, new_val: Union[int, str]):
        """
        The property setter for the year related with a particular scaling relation.

        :param int/str new_val: The new value for the year of the relation, either an integer year that can be
            converted to a string, or a string representing a year.
        """
        if type(new_val) != int and type(new_val) != str:
            raise TypeError('You must set the year property with an integer or string.')
        elif type(new_val) == int:
            new_val = str(new_val)
        self._year = new_val

    @property
    def doi(self) -> str:
        """
        A property getter for the doi of the original paper of the relation, if not from literature it will an
        empty string.

        :return: String containing the doi.
        :rtype: str
        """
        return self._doi

    @doi.setter
    def doi(self, new_val: str):
        """
        The property setter for the DOI of the work related with the relation.

        :param str new_val: The new value of the doi.
        """
        if not isinstance(new_val, str):
            raise TypeError("You must set the doi property with a string.")

        self._doi = new_val

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

    @property
    def model_colour(self) -> str:
        """
        Property getter for the model colour assigned to this relation. If it wasn't set at definition or set
        via the property setter then it defaults to 'tab:gray'.

        :return: The currently set model colour. If one wasn't set on definition then we default to tab:gray.
        :rtype: str
        """
        if self._model_colour is not None:
            return self._model_colour
        else:
            return 'tab:gray'

    @model_colour.setter
    def model_colour(self, new_colour: str):
        """
        Property setter for the model colour attribute, which controls the colour used in plots for the fit
         of this relation. New colours are checked against matplotlibs list of named colours.

        :param str new_colour: The new matplotlib colour.
        """
        all_col = list(TABLEAU_COLORS.keys()) + list(CSS4_COLORS.keys()) + list(BASE_COLORS.keys())
        if new_colour not in all_col:
            all_names = ', '.join(all_col)
            raise ValueError("{c} is not a named matplotlib colour, please use one of the "
                             "following; {cn}".format(c=new_colour, cn=all_names))
        else:
            self._model_colour = new_colour

    @property
    def point_names(self) -> Union[np.ndarray, None]:
        """
        Returns an array of point names, with one entry per data point, and in the same order (unless the user passes
        a differently ordered name array than data array, there is no way we can detect that).

        :return: The names associated with the data points, if supplied on initialization. The default is None.
        :rtype: np.ndarray/None
        """
        if isinstance(self._point_names, list):
            return np.ndarray(self._point_names)
        else:
            return self._point_names

    @property
    def third_dimension_data(self) -> Union[Quantity, None]:
        """
        Returns a Quantity containing a third dimension of data associated with the data points (this can be used to
        colour the points in the view method), with one entry per data point, and in the same order (unless the
        user passes a differently ordered name array than data array, there is no way we can detect that).

        :return: The third dimension data associated with the data points, if supplied on initialization. The
            default is None.
        :rtype: Quantity/None
        """
        if isinstance(self._third_dim_info, (list, np.ndarray)):
            return Quantity(self._third_dim_info)
        else:
            return self._third_dim_info

    @property
    def third_dimension_name(self) -> Union[str, None]:
        """
        Returns the name of the third data dimension passed to this relation on initialization.

        :return: The name of the third dimension, if supplied on initialization. The default is None.
        :rtype: Quantity/None
        """
        return self._third_dim_name

    @property
    def x_energy_bounds(self) -> Quantity:
        """
        The energy bounds within which the x-axis data have been measured (e.g. a 0.5-2.0 keV luminosity).

        :return: A non-scalar Astropy Quantity with two entries if the x-data energy bounds have been set, and None if
            they have not.
        :rtype: Quantity
        """
        return self._x_quantity_en_bounds

    @x_energy_bounds.setter
    def x_energy_bounds(self, new_val: Quantity):
        """
        Set the energy bounds within which the x-axis data have been measured (e.g. a 0.5-2.0 keV luminosity).

        :param Quantity new_val:
        """
        if not isinstance(new_val, Quantity) or new_val.isscalar or len(new_val) != 2:
            raise TypeError("The new value of 'x_energy_bounds' must be a non-scalar Astropy Quantity with "
                            "two elements.")
        else:
            self._x_quantity_en_bounds = new_val

    @property
    def y_energy_bounds(self) -> Quantity:
        """
        The energy bounds within which the y-axis data have been measured (e.g. a 0.5-2.0 keV luminosity).

        :return: A non-scalar Astropy Quantity with two entries if the y-data energy bounds have been set, and None if
            they have not.
        :rtype: Quantity
        """
        return self._y_quantity_en_bounds

    @y_energy_bounds.setter
    def y_energy_bounds(self, new_val: Quantity):
        """
        Set the energy bounds within which the y-axis data have been measured (e.g. a 0.5-2.0 keV luminosity).

        :param Quantity new_val:
        """
        if not isinstance(new_val, Quantity) or new_val.isscalar or len(new_val) != 2:
            raise TypeError("The new value of 'y_energy_bounds' must be a non-scalar Astropy Quantity with "
                            "two elements.")
        else:
            self._y_quantity_en_bounds = new_val

    def view_chains(self, figsize: tuple = None, colour: str = None):
        """
        Simple view method to quickly look at the MCMC chains for a scaling relation fit.

        :param tuple figsize: Desired size of the figure, if None will be set automatically.
        :param str colour: The colour that the chains should be in the plot. Default is None in which case
            the value of the model_colour property of the relation is used.
        """
        # If the colour is None then we fetch the model colour property
        if colour is None:
            colour = self.model_colour

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
            ax.plot(self._chains[:, i], colour, alpha=0.7)
            ax.set_xlim(0, self._chains.shape[0])
            ax.set_ylabel(self._par_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        if num_ch > len(self._fit_pars):
            ax = axes[-1]
            ax.plot(self._scatter_chain, colour, alpha=0.7)
            ax.set_xlim(0, len(self._scatter_chain))
            ax.set_ylabel(r'$\sigma$')
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step Number")
        plt.show()

    def view_corner(self, figsize: tuple = (10, 10), cust_par_names: List[str] = None,
                    colour: str = None, save_path: str = None):
        r"""
        A convenient view method to examine the corner plot of the parameter posterior distributions.

        :param tuple figsize: The size of the figure.
        :param List[str] cust_par_names: A list of custom parameter names. If the names include LaTeX code do not
            include $$ math environment symbols - you may also need to pass a string literal (e.g. r"\sigma"). Do
            not include an entry for a scatter parameter.
        :param List[str] colour: Colour for the contours. Default is None in which case the value of the
            model_colour property of the relation is used.
        :param str save_path: The path where the figure produced by this method should be saved. Default is None, in
            which case the figure will not be saved.
        """
        # If the colour is None then we fetch the model colour property
        if colour is None:
            colour = self.model_colour

        # Checks whether custom parameter names were passed, and if they were it checks whether there are the right
        #  number
        if cust_par_names is not None and len(cust_par_names) == len(self._par_names):
            par_names = cust_par_names
        elif cust_par_names is not None and len(cust_par_names) != len(self._par_names):
            raise ValueError("cust_par_names must have one entry per parameter of the scaling relation model.")
        else:
            par_names = deepcopy(self._par_names)

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

    def predict(self, x_values: Quantity, redshift: Union[float, np.ndarray] = None,
                cosmo: Cosmology = None, x_errors: Quantity = None) -> Quantity:
        """
        This method allows for the prediction of y values from this scaling relation, you just need to pass in an
        appropriate set of x value(s). If a power of E(z) was applied to the y-axis data before fitting, and that
        information was passed on declaration (using 'dim_hubb_ind'), then a redshift and cosmology are required
        to remove out the E(z) contribution.

        :param Quantity x_values: The x value(s) to predict y value(s) for.
        :param float/np.ndarray redshift: The redshift(s) of the objects for which we wish to predict values. This is
            only necessary if the 'dim_hubb_ind' argument was set on declaration. Default is None.
        :param Cosmology cosmo: The cosmology in which we wish to predict values. This is only necessary if the
            'dim_hubb_ind' argument was set on declaration. Default is None.
        :param Quantity x_errors: The uncertainties for passed x-values. Default is None. If this argument is not None
            then uncertainties in x-value and the model fit will be propagated to a final prediction uncertainty. If
            minus and plus uncertainties are passed then they will be averaged before propagation.
        :return: The predicted y values (and predicted uncertainties if x-errors were passed).
        :rtype: Quantity
        """
        # Ensure no floats are being passed in, as we need units!
        if type(x_values) is not Quantity:
            raise TypeError("The 'x_values' argument must be an astropy quantity.")
        # Got to check that people aren't passing any nonsense x quantities in
        elif not x_values.unit.is_equivalent(self.x_unit):
            raise UnitConversionError('Values of x passed to the predict method ({xp}) must be convertible '
                                      'to the x-axis units of this scaling relation '
                                      '({xr}).'.format(xp=x_values.unit.to_string(), xr=self.x_unit.to_string()))

        # Check that if x errors have been passed, they're in the right units
        if x_errors is not None and x_errors.unit != x_values.unit:
            raise UnitConversionError("The x errors are not in the same units as 'x_values'.")
        elif (x_errors is not None and not x_errors.isscalar and not x_values.isscalar and
              len(x_errors) != len(x_values)):
            raise ValueError("The length of the 'x_errors' argument ({xe}) should be the same as the 'x_values' "
                             "argument({xv}).".format(xe=len(x_errors), xv=len(x_values)))
        elif x_errors is not None and x_errors.isscalar and not x_values.isscalar:
            raise ValueError("Pass either a non-scalar set of 'x_values' and 'x_errors', a scalar value for both, or a "
                             "scalar value for 'x_values' and a two-entry value for 'x_errors' (for plus and minus).")

        # We average the uncertainties if there are minus and plus values (bad I know)
        if x_errors is not None and x_errors.ndim == 2:
            x_errors = x_errors.mean(axis=1)
        elif x_errors is not None and x_values.isscalar and not x_errors.isscalar and len(x_errors) == 2:
            x_errors = x_errors.mean()

        # This is a check that all passed x values are within the validity limits of this relation (if the
        #  user passed those on init) - if they aren't a warning will be issued
        if self.x_lims is not None and len(x_values[(x_values < self.x_lims[0]) | (x_values > self.x_lims[1])]) != 0:
            warn("Some of the x values you have passed are outside the validity range of this relation "
                 "({l}-{h}{u}).".format(l=self.x_lims[0].value, h=self.x_lims[1].value, u=self.x_unit.to_string()),
                 stacklevel=2)

        # Need to check if any power of E(z) was applied to the y-axis data before fitting, if so (and no
        #  cosmo/redshift was passed) then it's time to throw an error.
        if (redshift is None or cosmo is None) and self._ez_power is not None:
            raise ValueError("A power of E(z) was applied to the y-axis data before fitting, as such you must pass"
                             " redshift and cosmology information to this predict method.")
        elif self._ez_power is not None and isinstance(redshift, float) and not x_values.isscalar:
            raise ValueError("You must supply one redshift for every entry in x_values.")
        elif self._ez_power is not None and isinstance(redshift, np.ndarray) and len(x_values) != len(redshift):
            raise ValueError("The x_values argument has {x} entries, and the redshift argument has {z} entries; "
                             "please supply one redshift per x_value.".format(x=len(x_values), z=len(redshift)))

        # Units that are convertible to the x-units of this relation are allowed, so we make sure we convert
        #  to the exact units the fit was done in. This includes dividing by the x_norm value
        x_values = x_values.to(self.x_unit)
        # Then we just pass the x_values into the model, along with fit parameters. Then multiply by
        #  the y normalisation
        predicted_y = self._model_func((x_values / self.x_norm).value, *self.pars[:, 0]) * self.y_norm

        # If there was a power of E(z) applied to the data, we undo it for the prediction.
        if self._ez_power is not None:
            # We store this so that error propogation can use it later
            ez = (cosmo.efunc(redshift)**self._ez_power)
            predicted_y /= ez
        elif not x_values.isscalar:
            # This means that error propagation doesn't need to keep checking whether there is an ez power stored
            ez = np.ones(len(predicted_y))
        elif x_values.isscalar:
            # And handles the case where the input x_values are scalar, in which case the 'len' call above would
            #  cause them to get stroppy and error
            ez = 1.

        # Now we propagate the uncertainties on the input parameters, if they have them (and if the model is a
        #  power law) - would be nice to generalise this somehow
        if x_errors is not None and self.model_func == power_law:
            # This is just the error propagation for a powerlaw - the standard form of a scaling relation
            term_one = ((self.y_norm.value * (1/ez) * (x_values.value/self.x_norm.value)**self.pars[0, 0]) *
                        self.pars[1, 1])**2

            term_two = (((self.y_norm.value * (1/ez) * self.pars[1, 0] * self.pars[0, 0] *
                          ((1/self.x_norm.value)**self.pars[0, 0]) *
                          x_values.value**(self.pars[0, 0] - 1)))*x_errors.value)**2

            term_three = ((self.y_norm.value*(1/ez)*self.pars[1, 0] *
                           ((x_values.value/self.x_norm.value)**self.pars[0, 0]) *
                           np.log(x_values.value/self.x_norm.value))*self.pars[0, 1])**2

            predicted_y_errs = Quantity(np.sqrt(term_one + term_two + term_three), self.y_unit)

            # We use a slightly different method of combining the predicted value and uncertainty depending on whether
            #  a single x-value was passed, or a set of them.
            if x_values.isscalar:
                predicted_y = Quantity([predicted_y, predicted_y_errs])
            else:
                predicted_y = np.vstack([predicted_y, predicted_y_errs]).T

        elif x_errors is not None and self.model_func != power_law:
            raise NotImplementedError("Error propagation for scaling relation models other than 'power_law' is not "
                                      "implemented yet.")

        return predicted_y

    def get_view(self, ax: Axes, x_lims: Quantity = None, log_scale: bool = True, plot_title: str = None,
                 data_colour: str = 'black', model_colour: str = None, grid_on: bool = False, conf_level: int = 90,
                 custom_x_label: str = None, custom_y_label: str = None, fontsize: float = 15, x_ticks: list = None,
                 x_minor_ticks: list = None, y_ticks: list = None, y_minor_ticks: list = None,
                 label_points: bool = False, point_label_colour: str = 'black', point_label_size: int = 10,
                 point_label_offset: tuple = (0.01, 0.01), show_third_dim: bool = None,
                 third_dim_cmap: Union[str, Colormap] = 'plasma', third_dim_norm_cmap: Normalize = Normalize,
                 third_dim_axis_formatters: dict = None, y_lims: Quantity = None, one_to_one: bool = False):
        """
        A get method that will populate a matplotlib axes with a high quality plot of this scaling relation (including
        the data it is based upon, if available), and then return it.

        :param Axes ax: The axes on which to draw the plot.
        :param Quantity x_lims: If not set, this method will attempt to take appropriate limits from the x-data
            this relation is based upon, if that data is not available an error will be thrown.
        :param bool log_scale: If true then the x and y axes of the plot will be log-scaled.
        :param str plot_title: A custom title to be used for the plot, otherwise one will be generated automatically.
        :param str data_colour: The colour to use for the data points in the plot, default is black.
        :param str model_colour: The colour to use for the model in the plot. Default is None in which case
            the value of the model_colour property of the relation is used.
        :param bool grid_on: If True then a grid will be included on the plot. Default is True.
        :param int conf_level: The confidence level to use when plotting the model.
        :param str custom_x_label: Passing a string to this variable will override the x axis label
            of this plot, including the unit string.
        :param str custom_y_label: Passing a string to this variable will override the y axis label
            of this plot, including the unit string.
        :param float fontsize: The fontsize for axis labels.
        :param list x_ticks: Customise which major x-axis ticks and labels are on the figure, default is None in which
            case they are determined automatically.
        :param list x_minor_ticks: Customise which minor x-axis ticks and labels are on the figure, default is
            None in which case they are determined automatically.
        :param list y_ticks: Customise which major y-axis ticks and labels are on the figure, default is None in which
            case they are determined automatically.
        :param list y_minor_ticks: Customise which minor y-axis ticks and labels are on the figure, default is
            None in which case they are determined automatically.
        :param bool label_points: If True, and source name information for each point was passed on the declaration of
            this scaling relation, then points will be accompanied by an index that can be used with the 'point_names'
            property to retrieve the source name for a point. Default is False.
        :param str point_label_colour: The colour of the label text.
        :param int point_label_size: The fontsize of the label text.
        :param bool show_third_dim: Colour the data points by the third dimension data passed in on creation of this
            scaling relation, with a colour bar to communicate values. Only possible if data were passed to
            'third_dim_info' on initialization. Default is None, which automatically gets converted to True if there
            is a third data dimension, and converted to False if there is not.
        :param str/Colormap third_dim_cmap: The colour map which should be used for the third dimension data points.
            A matplotlib colour map name or a colour map object may be passed. Default is 'plasma'. This essentially
            overwrites the 'data_colour' argument if show_third_dim is True.
        :param Normalize third_dim_norm_cmap: A matplotlib 'Normalize' class/subclass (e.g. LogNorm, SymLogNorm, etc.)
            that will be used to scale the colouring of the data points by the third data dimension. Note that
            a class, NOT A CLASS INSTANCE (e.g. LogNorm()) must be passed, as the normalisation will be set up in
            this method. Default is Normalization (linear scaling).
        :param dict third_dim_axis_formatters: A dictionary of formatters that can be applied to the colorbar
            axis. Allowed keys are; 'major' and 'minor'. The values associated with the keys should be
            instantiated matplotlib formatters.
        :param Tuple[float, float] point_label_offset: A fractional offset (in display coordinates) applied to the
            data point coordinates to determine the location a label should be added. You can use this to fine-tune
            the label positions relative to their data point.
        :param Quantity y_lims: If not set, this method will attempt to take appropriate limits from the y-data and/or
            relation line - setting any value other than None will override that.
        :param bool one_to_one: If True, a one-to-one line will be plotted on the scaling relation view. Default is
            False.
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
            max_x_ind = np.nanargmax(self._x_data)
            min_x_ind = np.nanargmin(self._x_data)
            x_lims = [0.9 * (self._x_data[min_x_ind].value - self._x_err[min_x_ind].value),
                      1.1 * (self._x_data[max_x_ind].value + self._x_err[max_x_ind].value)]
        elif x_lims is None and len(self._x_data) == 0:
            raise ValueError('There are no data available to infer suitable axis limits from, please pass x limits.')

        # Just grabs the model colour from the property if the user doesn't set a value for model_colour
        if model_colour is None:
            model_colour = self.model_colour

        # Setting the axis limits
        ax.set_xlim(x_lims)

        # Setup the aesthetics of the axis
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        # I wanted this to react to whether there is a third dimension of data or not, so that the warning below
        #  is still shown if the user sets show_third_dim=True when there is no third dimension, but otherwise they
        #  don't have to worry about/see that warning.
        if show_third_dim is None and self.third_dimension_data is None:
            show_third_dim = False
        elif show_third_dim is None and self.third_dimension_data is not None:
            show_third_dim = True

        # We check to see a) whether the user wants a third dimension of data communicated via the colour of the
        #  data, and b) if they actually passed the data necessary to make that happen. If there is no data but they
        #  have set show_third_dim=True, we set it back to False
        if show_third_dim and self.third_dimension_data is None:
            warn("The 'show_third_dim' argument should only be set to True if 'third_dim_info' was set on "
                 "the creation of this scaling relation. Setting 'show_third_dim' to False.", stacklevel=2)
            show_third_dim = False

        # Check that the passed normalization class is actually valid
        if ((show_third_dim and third_dim_norm_cmap != Normalize) and
                not Normalize.__subclasscheck__(third_dim_norm_cmap)):
            raise TypeError("The 'third_dim_norm_cmap' argument must be a subclass of matplotlib.color.Normalize")

        # Plot the data with uncertainties if any data is present in this scaling relation.
        if len(self.x_data) != 0 and not show_third_dim:
            ax.errorbar(self._x_data.value, self._y_data.value, xerr=self._x_err.value, yerr=self._y_err.value,
                        fmt="x", color=data_colour, capsize=2)
        elif len(self.x_data) != 0 and show_third_dim:
            # The user can either set the cmap with a string name, or actually pass a colormap object
            if isinstance(third_dim_cmap, str):
                cmap = cm.get_cmap(third_dim_cmap)
            else:
                cmap = third_dim_cmap
            # We want to normalise this colourmap to our specific data range
            norm = third_dim_norm_cmap(vmin=self.third_dimension_data.value.min(),
                                       vmax=self.third_dimension_data.value.max())
            # Now this mapper can be constructed so that we can take that information about the cmap and normalisation
            #  and use it with our data to calculate colours
            cmap_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            # This calculates the colours
            colours = cmap_mapper.to_rgba(self.third_dimension_data.value)
            # I didn't really want to do this but errorbar calls plot (rather than scatter) so it will only do one
            #  colour at a time.
            for c_ind, col in enumerate(colours):
                ax.errorbar(self._x_data[c_ind].value, self._y_data[c_ind].value, xerr=self._x_err[c_ind].value,
                            yerr=self._y_err[c_ind].value, fmt="x", c=colours[c_ind, :], capsize=2)

        # This will check a) if the scaling relation knows the source names associated with the points, and b) if the
        #  user wants us to label them
        if self.point_names is not None and label_points:
            # If both those conditions are satisfied then we will start to iterate through the data points
            for ind in range(len(self.point_names)):
                # These are the current points being read out to help position the overlaid text
                cur_x = self._x_data[ind].value
                cur_y = self._y_data[ind].value
                # Then we check to make sure neither coord is None, and add the index (which is used as the short
                #  ID for these points) to the axes - the user can then look at the number and use that to retrieve
                #  the name.
                if not np.isnan(cur_x) and not np.isnan(cur_y):
                    # This measures the x_size of the plot in display coordinates (TRANSFORMED from data, thus avoiding
                    #  any issues with the scaling of the axis)
                    x_size = ax.transData.transform((x_lims[1], 0))[0] - ax.transData.transform((x_lims[0], 0))[0]
                    # This does the same thing with the y-data
                    y_dat_lims = ax.get_ylim()
                    y_size = ax.transData.transform((0, y_dat_lims[1]))[1] - \
                             ax.transData.transform((0, y_dat_lims[0]))[1]
                    # Then we convert the current data coordinate into display coordinate system
                    cur_fig_coord = ax.transData.transform((cur_x, cur_y))
                    # And make a label coordinate by offsetting the x and y data coordinate by some fraction of the
                    #  overall size of the axis, in display coordinates, with the final coordinate transformed back
                    #  to data coordinates.
                    inv_tran = ax.transData.inverted()
                    lab_data_coord = inv_tran.transform((cur_fig_coord[0] + (point_label_offset[0] * x_size),
                                                         cur_fig_coord[1] + (point_label_offset[1] * y_size)))
                    plt.text(lab_data_coord[0], lab_data_coord[1], str(ind), fontsize=point_label_size,
                             color=point_label_colour)

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
        model_median = np.nanmedian(model_realisations, axis=1)
        model_lower = np.nanpercentile(model_realisations, lower, axis=1)
        model_upper = np.nanpercentile(model_realisations, upper, axis=1)

        # I want the name of the function to include in labels and titles, but if its one defined in XGA then
        #  I can grab the publication version of the name - it'll be prettier
        mod_name = self._model_func.__name__
        for m_name in MODEL_PUBLICATION_NAMES:
            mod_name = mod_name.replace(m_name, MODEL_PUBLICATION_NAMES[m_name])

        relation_label = " ".join([self._author, self._year, '-', mod_name,
                                   "- {cf}% Confidence".format(cf=conf_level)])
        plt.plot(model_x * self._x_norm.value, model_median, color=model_colour, label=relation_label)

        plt.plot(model_x * self._x_norm.value, model_upper, color=model_colour, linestyle="--")
        plt.plot(model_x * self._x_norm.value, model_lower, color=model_colour, linestyle="--")
        ax.fill_between(model_x * self._x_norm.value, model_lower, model_upper, where=model_upper >= model_lower,
                        facecolor=model_colour, alpha=0.6, interpolate=True)

        # Now the relation/data have been plotted, we'll see if the user wanted any custom y-axis limits. If not then
        #  nothing will happen and we'll go with whatever matplotlib decided. Also check that the input was
        #  appropriate, if there was one
        if y_lims is not None and not y_lims.unit.is_equivalent(self.y_unit):
            raise UnitConversionError('Limits on the y-axis ({yl}) must be convertible to the y-axis units of this '
                                      'scaling relation ({yr}).'.format(yl=y_lims.unit.to_string(),
                                                                        yr=self.y_unit.to_string()))
        elif y_lims is not None:
            # Setting the axis limits
            ax.set_ylim(y_lims.value)

        # Making the scale log if requested
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        # I can dynamically grab the units in LaTeX formatting from the Quantity objects (thank you astropy)
        #  However I've noticed specific instances where the units can be made prettier
        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + self.x_unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + self.y_unit.to_string("latex").strip("$") + r"\right]$"

        # Dimensionless quantities can be fitted too, and this make the axis label look nicer by not having empty
        #  square brackets
        if x_unit == r"$\left[\\mathrm{}\right]$" or x_unit == r'$\left[\mathrm{}\right]$':
            x_unit = ''
        if y_unit == r"$\left[\\mathrm{}\right]$" or y_unit == r'$\left[\mathrm{}\right]$':
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

        # The user can request a one-to-one line be overplotted on the view (obviously not relevant to general
        #  scaling relations, but great for comparisons)
        if one_to_one:
            min_from = min(min(x_axis_lims), min(y_axis_lims))
            max_to = max(max(x_axis_lims), max(y_axis_lims))
            plt.plot([min_from, max_to], [min_from, max_to], color='red', linestyle='dashed', label='1:1')

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
            ax.grid(False, which='both', axis='both')

        if grid_on and (max(y_axis_lims) / min(y_axis_lims)) < 10:
            ax.grid(which='minor', axis='y', linestyle='dotted', color='grey')
        elif grid_on:
            ax.grid(which='major', axis='y', linestyle='dotted', color='grey')
        else:
            ax.grid(False, which='both', axis='both')

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

        # If we did colour the data by a third dimension, then we should add a colour-bar to the relation
        if show_third_dim:
            # Setting up the colorbar axis
            cbar = plt.colorbar(cmap_mapper, ax=plt.gca())
            # And making sure we include units
            if self.third_dimension_data.unit.is_equivalent(''):
                cbar_lab = self.third_dimension_name
            else:
                cbar_lab = self.third_dimension_name + ' [' + self.third_dimension_data.unit.to_string('latex') + ']'

            # Set the axis label for the colobar
            cbar.ax.set_ylabel(cbar_lab, fontsize=fontsize)

            # Now we check to see if the user passed custom axis formatters for the colorbar axis, and if so then
            #  we apply them
            if third_dim_axis_formatters is not None:
                # Checks for and uses formatters that the user may have specified for the plot
                if 'minor' in third_dim_axis_formatters:
                    cbar.ax.yaxis.set_minor_formatter(third_dim_axis_formatters['minor'])
                if 'major' in third_dim_axis_formatters:
                    cbar.ax.yaxis.set_major_formatter(third_dim_axis_formatters['major'])

        return ax


    def view(self, x_lims: Quantity = None, log_scale: bool = True, plot_title: str = None, figsize: tuple = (10, 8),
             data_colour: str = 'black', model_colour: str = None, grid_on: bool = False, conf_level: int = 90,
             custom_x_label: str = None, custom_y_label: str = None, fontsize: float = 15, legend_fontsize: float = 13,
             x_ticks: list = None, x_minor_ticks: list = None, y_ticks: list = None, y_minor_ticks: list = None,
             save_path: str = None, label_points: bool = False, point_label_colour: str = 'black',
             point_label_size: int = 10, point_label_offset: tuple = (0.01, 0.01), show_third_dim: bool = None,
             third_dim_cmap: Union[str, Colormap] = 'plasma', third_dim_norm_cmap: Normalize = Normalize,
             third_dim_axis_formatters: dict = None, y_lims: Quantity = None, one_to_one: bool = False):
        """
        A method that produces a high quality plot of this scaling relation (including the data it is based upon,
        if available).

        :param Quantity x_lims: If not set, this method will attempt to take appropriate limits from the x-data
            this relation is based upon, if that data is not available an error will be thrown.
        :param bool log_scale: If True, then the x and y axes of the plot will be log-scaled.
        :param str plot_title: A custom title to be used for the plot, otherwise one will be generated automatically.
        :param tuple figsize: A custom figure size for the plot, default is (8, 8).
        :param str data_colour: The colour to use for the data points in the plot, default is black.
        :param str model_colour: The colour to use for the model in the plot. Default is None in which case
            the value of the model_colour property of the relation is used.
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
        :param bool label_points: If True, and source name information for each point was passed on the declaration of
            this scaling relation, then points will be accompanied by an index that can be used with the 'point_names'
            property to retrieve the source name for a point. Default is False.
        :param str point_label_colour: The colour of the label text.
        :param int point_label_size: The fontsize of the label text.
        :param bool show_third_dim: Colour the data points by the third dimension data passed in on creation of this
            scaling relation, with a colour bar to communicate values. Only possible if data were passed to
            'third_dim_info' on initialization. Default is None, which automatically gets converted to True if there
            is a third data dimension, and converted to False if there is not.
        :param str/Colormap third_dim_cmap: The colour map which should be used for the third dimension data points.
            A matplotlib colour map name or a colour map object may be passed. Default is 'plasma'. This essentially
            overwrites the 'data_colour' argument if show_third_dim is True.
        :param Normalize third_dim_norm_cmap: A matplotlib 'Normalize' class/subclass (e.g. LogNorm, SymLogNorm, etc.)
            that will be used to scale the colouring of the data points by the third data dimension. Note that
            a class, NOT A CLASS INSTANCE (e.g. LogNorm()) must be passed, as the normalisation will be set up in
            this method. Default is Normalization (linear scaling).
        :param dict third_dim_axis_formatters: A dictionary of formatters that can be applied to the colorbar
            axis. Allowed keys are; 'xmajor', 'xminor', 'ymajor', and 'yminor'. The values associated with the
            keys should be instantiated matplotlib formatters.
        :param Tuple[float, float] point_label_offset: A fractional offset (in display coordinates) applied to the
            data point coordinates to determine the location a label should be added. You can use this to fine-tune
            the label positions relative to their data point.
        :param Quantity y_lims: If not set, this method will attempt to take appropriate limits from the y-data and/or
            relation line - setting any value other than None will override that.
        :param bool one_to_one: If True, a one-to-one line will be plotted on the scaling relation view. Default is
            False.
        """
        # Setting up the matplotlib figure
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        ax = plt.gca()

        ax = self.get_view(ax, x_lims, log_scale, plot_title, data_colour, model_colour, grid_on, conf_level,
                           custom_x_label, custom_y_label, fontsize, x_ticks, x_minor_ticks, y_ticks, y_minor_ticks,
                           label_points, point_label_colour, point_label_size, point_label_offset, show_third_dim,
                           third_dim_cmap, third_dim_norm_cmap, third_dim_axis_formatters, y_lims, one_to_one)

        plt.legend(loc="best", fontsize=legend_fontsize)
        plt.tight_layout()

        # If the user passed a save_path value, then we assume they want to save the figure
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def save(self, save_path: str):
        """
        This method pickles and saves the scaling relation object. The save file is a pickled version of this object.

        :param str save_path: The path where this relation should be saved.
        """
        # if '/' in save_path and not os.path.exists('/'.join(save_path.split('/')[:-1])):
        #     raise FileNotFoundError('The path before your file name does not seem to exist.')

        # Pickles and saves this ScalingRelation instance.
        with open(save_path, 'wb') as picklo:
            pickle.dump(self, picklo)

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

    :param List[ScalingRelation] relations: A list of scaling relations objects to be combined in this object.
    """
    def __init__(self, relations: List[ScalingRelation]):
        """
        Init method for the AggregateScalingRelation, that allows for the joint viewing of sets of scaling relations.
        """
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
        r"""
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
        elif contour_colours is not None and len(contour_colours) != len(self._relations):
            raise ValueError("If you pass a list of contour colours, there must be one entry per scaling relation.")

        # This draws the colours from the model_colour parameters of the various relations, but only if each
        #  has a unique colour
        if contour_colours is None:
            # Use a set to check for duplicate colours, they are not allowed. This is primarily to catch
            #  instances where the model_colour property has not been set for all the relations, as then all
            #  the colours would be grey
            all_rel_cols = list(set([r.model_colour for r in self._relations]))
            # If there are N unique colours for N relations, then we'll use those colours, otherwise matplotlib
            #  can choose whatever colours it likes
            if len(all_rel_cols) == len(self._relations):
                # Don't use the all_rel_cols variable here is it is inherently unordered as a set() was used
                #  in its construction.
                contour_colours = [r.model_colour for r in self._relations]

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

    def get_view(self, ax: Axes, x_lims: Quantity = None, log_scale: bool = True, plot_title: str = None,
                 colour_list: list = None, grid_on: bool = False, conf_level: int = 90, show_data: bool = True,
                 fontsize: float = 15, x_ticks: list = None, x_minor_ticks: list = None, y_ticks: list = None,
                 y_minor_ticks: list = None, data_colour_list: list = None, data_shape_list: list = None,
                 custom_x_label: str = None, custom_y_label: str = None, y_lims: Quantity = None,
                 one_to_one: bool = False):
        """
        A method that populates a passed matplotlib axis with a high quality plot of the component scaling
        relations in this AggregateScalingRelation, and then returns it.

        :param Axes ax: The matplotlib Axes object to draw on.
        :param Quantity x_lims: If not set, this method will attempt to take appropriate limits from the x-data
            this relation is based upon, if that data is not available an error will be thrown.
        :param bool log_scale: If true then the x and y axes of the plot will be log-scaled.
        :param str plot_title: A custom title to be used for the plot, otherwise one will be generated automatically.
        :param list colour_list: A list of matplotlib colours to use as a custom colour cycle.
        :param bool grid_on: If True then a grid will be included on the plot. Default is True.
        :param int conf_level: The confidence level to use when plotting the model.
        :param bool show_data: Controls whether data points are shown on the view, as it can quickly become
            confusing with multiple relations on one axis.
        :param float fontsize: The fontsize for axis labels.
        :param list x_ticks: Customise which major x-axis ticks and labels are on the figure, default is None in which
            case they are determined automatically.
        :param list x_minor_ticks: Customise which minor x-axis ticks and labels are on the figure, default is
            None in which case they are determined automatically.
        :param list y_ticks: Customise which major y-axis ticks and labels are on the figure, default is None in which
            case they are determined automatically.
        :param list y_minor_ticks: Customise which minor y-axis ticks and labels are on the figure, default is
            None in which case they are determined automatically.
        :param list data_colour_list: A list of matplotlib colours to use as a colour cycle specifically for
            data points. This should be used when you want data points to be a different colour to their model.
        :param list data_shape_list: A list of matplotlib format shapes, to manually set the shapes of plotted
            data points.
        :param str custom_x_label: Passing a string to this variable will override the x-axis label of this
            plot, including the unit string.
        :param str custom_y_label: Passing a string to this variable will override the y-axis label of this
            plot, including the unit string.
        :param Quantity y_lims: If not set, this method will attempt to take appropriate limits from the y-data and/or
            relation line - setting any value other than None will override that.
        :param bool one_to_one: If True, a one-to-one line will be plotted on the scaling relation view. Default is
            False.
        """
        # Very large chunks of this are almost direct copies of the view method of ScalingRelation, but this
        #  was the easiest way of setting this up, so I think the duplication is justified.

        # Grabs the colours that may have been set for each relation, uses a set to check that there are
        #  no duplicates
        set_mod_cols = list(set([r.model_colour for r in self._relations]))
        # Set up the colour cycle, if the user hasn't passed a colour list we'll try to use colours set for the
        #  individual relations, but if they haven't all been set then we'll use the predefined colour cycle
        if colour_list is None and len(set_mod_cols) == len(self._relations):
            # Don't use the set_mod_cols variable as its unordered due to the use of set
            colour_list = [r.model_colour for r in self._relations]
        elif colour_list is None and len(set_mod_cols) != len(self._relations):
            colour_list = PRETTY_COLOUR_CYCLE
        new_col_cycle = cycler(color=colour_list)

        # If the user didn't pass their own list of DATA colours to use, then they will be the same as the
        #  model colours.
        if data_colour_list is None:
            data_colour_list = deepcopy(colour_list)
        elif data_colour_list is not None and len(data_colour_list) != len(self._relations):
            raise ValueError('If a data_colour_list is passed, then it must have the same number of entries as there'
                             ' are relations.')

        if data_shape_list is None:
            data_shape_list = ['x' for i in range(0, len(self._relations))]
        elif data_shape_list is not None and len(data_shape_list) != len(self._relations):
            raise ValueError('If a data_shape_list is passed, then it must have the same number of entries as there'
                             ' are relations.')

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
            max_x_ind = np.nanargmax(comb_x_data[:, 0])
            min_x_ind = np.nanargmin(comb_x_data[:, 0])
            x_lims = [0.9 * (comb_x_data[min_x_ind, 0].value - comb_x_data[min_x_ind, 1].value),
                      1.1 * (comb_x_data[max_x_ind, 0].value + comb_x_data[max_x_ind, 1].value)]
        elif x_lims is None and len(comb_x_data) == 0:
            raise ValueError('There is no data available to infer suitable axis limits from, please pass x limits.')

        ax.set_prop_cycle(new_col_cycle)

        # Setting the axis limits
        ax.set_xlim(x_lims)

        # Setup the aesthetics of the axis
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        for rel_ind, rel in enumerate(self._relations):
            # This is a horrifying bodge, but I do just want the colour out and I can't be bothered to figure out
            #  how to use the colour cycle object properly
            if len(rel.x_data.value[:, 0]) == 0 or not show_data:
                # Sets up a null error bar instance for the colour basically
                d_out = ax.errorbar(None, None, xerr=None, yerr=None, fmt=data_shape_list[rel_ind], capsize=2, label='',
                                    color=data_colour_list[rel_ind])
            else:
                d_out = ax.errorbar(rel.x_data.value[:, 0], rel.y_data.value[:, 0], xerr=rel.x_data.value[:, 1],
                                    yerr=rel.y_data.value[:, 1], fmt=data_shape_list[rel_ind], capsize=2,
                                    color=data_colour_list[rel_ind], alpha=0.7)

            m_colour = colour_list[rel_ind]

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
            model_median = np.nanmedian(model_realisations, axis=1)
            model_lower = np.nanpercentile(model_realisations, lower, axis=1)
            model_upper = np.nanpercentile(model_realisations, upper, axis=1)

            # I want the name of the function to include in labels and titles, but if its one defined in XGA then
            #  I can grab the publication version of the name - it'll be prettier
            mod_name = rel.model_func.__name__
            for m_name in MODEL_PUBLICATION_NAMES:
                mod_name = mod_name.replace(m_name, MODEL_PUBLICATION_NAMES[m_name])

            if rel.name is None:
                relation_label = " ".join([rel.author, rel.year])
            else:
                relation_label = rel.name + ' Scaling Relation'

            plt.plot(model_x * rel.x_norm.value, model_median, color=m_colour, label=relation_label)

            plt.plot(model_x * rel.x_norm.value, model_upper, color=m_colour, linestyle="--")
            plt.plot(model_x * rel.x_norm.value, model_lower, color=m_colour, linestyle="--")
            ax.fill_between(model_x * rel.x_norm.value, model_lower, model_upper, where=model_upper >= model_lower,
                            facecolor=m_colour, alpha=0.6, interpolate=True)

        # Now the relation/data have been plotted, we'll see if the user wanted any custom y-axis limits. If not then
        #  nothing will happen and we'll go with whatever matplotlib decided. Also check that the input was
        #  appropriate, if there was one
        if y_lims is not None and not y_lims.unit.is_equivalent(self.y_unit):
            raise UnitConversionError('Limits on the y-axis ({yl}) must be convertible to the y-axis units of this '
                                      'scaling relation ({yr}).'.format(yl=y_lims.unit.to_string(),
                                                                        yr=self.y_unit.to_string()))
        elif y_lims is not None:
            # Setting the axis limits
            ax.set_ylim(y_lims.value)

        # Making the scale log if requested
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        # I can dynamically grab the units in LaTeX formatting from the Quantity objects (thank you astropy)
        #  However I've noticed specific instances where the units can be made prettier
        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + self.x_unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + self.y_unit.to_string("latex").strip("$") + r"\right]$"

        # Dimensionless quantities can be fitted too, and this make the axis label look nicer by not having empty
        #  square brackets
        if x_unit == r"$\left[\\mathrm{}\right]$" or x_unit == r'$\left[\mathrm{}\right]$':
            x_unit = ''
        if y_unit == r"$\left[\\mathrm{}\right]$" or y_unit == r'$\left[\mathrm{}\right]$':
            y_unit = ''

        # The user is allowed to define their own x and y axis labels if they want, otherwise we construct it
        #  from the relations in this aggregate scaling relation.
        if custom_x_label is None:
            # The scaling relation object knows what its x-axis is called
            plt.xlabel("{xn} {un}".format(xn=self._x_name, un=x_unit), fontsize=fontsize)
        else:
            plt.xlabel(custom_x_label, fontsize=fontsize)

        if custom_y_label is None:
            # The scaling relation object knows what its y-axis is called
            plt.ylabel("{yn} {un}".format(yn=self._y_name, un=y_unit), fontsize=fontsize)
        else:
            plt.ylabel(custom_y_label, fontsize=fontsize)

        # The user can also pass a plot title, but if they don't then I construct one automatically
        if plot_title is None:
            plot_title = 'Scaling Relation Comparison - {c}% Confidence Limits'.format(c=conf_level)

        plt.title(plot_title, fontsize=13)

        # Use the axis limits quite a lot in this next bit, so read them out into variables
        x_axis_lims = ax.get_xlim()
        y_axis_lims = ax.get_ylim()

        # The user can request a one-to-one line be overplotted on the view (obviously not relevant to general
        #  scaling relations, but great for comparisons)
        if one_to_one:
            min_from = min(min(x_axis_lims), min(y_axis_lims))
            max_to = max(max(x_axis_lims), max(y_axis_lims))
            plt.plot([min_from, max_to], [min_from, max_to], color='red', linestyle='dashed', label='1:1')

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
            ax.grid(False, which='both', axis='both')

        if grid_on and (max(y_axis_lims) / min(y_axis_lims)) < 10:
            ax.grid(which='minor', axis='y', linestyle='dotted', color='grey')
        elif grid_on:
            ax.grid(which='major', axis='y', linestyle='dotted', color='grey')
        else:
            ax.grid(False, which='both', axis='both')

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

        return ax

    def view(self, x_lims: Quantity = None, log_scale: bool = True, plot_title: str = None, figsize: tuple = (10, 8),
             colour_list: list = None, grid_on: bool = False, conf_level: int = 90, show_data: bool = True,
             fontsize: float = 15, legend_fontsize: float = 13, x_ticks: list = None, x_minor_ticks: list = None,
             y_ticks: list = None, y_minor_ticks: list = None, save_path: str = None, data_colour_list: list = None,
             data_shape_list: list = None, custom_x_label: str = None, custom_y_label: str = None,
             y_lims: Quantity = None, one_to_one: bool = False):
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
        :param list data_colour_list: A list of matplotlib colours to use as a colour cycle specifically for
            data points. This should be used when you want data points to be a different colour to their model.
        :param list data_shape_list: A list of matplotlib format shapes, to manually set the shapes of plotted
            data points.
        :param str custom_x_label: Passing a string to this variable will override the x-axis label of this
            plot, including the unit string.
        :param str custom_y_label: Passing a string to this variable will override the y-axis label of this
            plot, including the unit string.
        :param Quantity y_lims: If not set, this method will attempt to take appropriate limits from the y-data and/or
            relation line - setting any value other than None will override that.
        :param bool one_to_one: If True, a one-to-one line will be plotted on the scaling relation view. Default is
            False.
        """
        # Setting up the matplotlib figure
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        ax = plt.gca()

        ax = self.get_view(ax, x_lims, log_scale, plot_title, colour_list, grid_on, conf_level, show_data, fontsize,
                           x_ticks, x_minor_ticks, y_ticks, y_minor_ticks, data_colour_list, data_shape_list,
                           custom_x_label, custom_y_label, y_lims, one_to_one)

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






















