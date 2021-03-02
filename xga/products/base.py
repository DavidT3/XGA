#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 02/03/2021, 10:39. Copyright (c) David J Turner

import inspect
import os
from typing import Tuple, List, Dict, Union
from warnings import warn

import corner
import emcee as em
import numpy as np
from astropy.units import Quantity, UnitConversionError, Unit, deg
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit, minimize

from ..exceptions import SASGenerationError, UnknownCommandlineError, XGAFitError, XGAInvalidModelError, \
    ModelNotAssociatedError
from ..models import MODEL_PUBLICATION_NAMES, PROF_TYPE_MODELS, PROF_TYPE_MODELS_STARTS, PROF_TYPE_MODELS_PRIORS, \
    MODEL_PUBLICATION_PAR_NAMES
from ..models.fitting import log_likelihood, log_prob
from ..utils import SASERROR_LIST, SASWARNING_LIST


class BaseProduct:
    """
    The super class for all SAS generated products in XGA. Stores relevant file path information, parses the std_err
    output of the generation process, and stores the instrument and ObsID that the product was generated for.
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str):
        """
        The initialisation method for the BaseProduct class.

        :param str path: The path to where the product file SHOULD be located.
        :param str stdout_str: The stdout from calling the terminal command.
        :param str stderr_str: The stderr from calling the terminal command.
        :param str gen_cmd: The command used to generate the product.
        """
        # This attribute stores strings that indicate why a product object has been deemed as unusable
        self._why_unusable = []

        # So this flag indicates whether we think this data product can be used for analysis
        self._usable = True
        if os.path.exists(path):
            self._path = path
        else:
            self._path = None
            self._usable = False
            self._why_unusable.append("ProductPathDoesNotExist")
        # Saving this in attributes for future reference
        self.unprocessed_stdout = stdout_str
        self.unprocessed_stderr = stderr_str
        self._sas_error, self._sas_warn, self._other_error = self.parse_stderr()
        self._obs_id = obs_id
        self._inst = instrument
        self._og_cmd = gen_cmd
        self._energy_bounds = (None, None)
        self._prod_type = None
        self._src_name = None

    # Users are not allowed to change this, so just a getter.
    @property
    def usable(self) -> bool:
        """
        Returns whether this product instance should be considered usable for an analysis.

        :return: A boolean flag describing whether this product should be used.
        :rtype: bool
        """
        return self._usable

    @property
    def path(self) -> str:
        """
        Property getter for the attribute containing the path to the product.

        :return: The product path.
        :rtype: str
        """
        return self._path

    @path.setter
    def path(self, prod_path: str):
        """
        Property setter for the attribute containing the path to the product.

        :param str prod_path: The product path.
        """
        if not os.path.exists(prod_path):
            prod_path = None
            # We won't be able to make use of this product if it isn't where we think it is
            self._usable = False
            self._why_unusable.append("ProductPathDoesNotExist")
        self._path = prod_path

    def parse_stderr(self) -> Tuple[List[str], List[Dict], List]:
        """
        This method parses the stderr associated with the generation of a product into errors confirmed to have
        come from SAS, and other unidentifiable errors. The SAS errors are returned with the actual error
        name, the error message, and the SAS routine that caused the error.

        :return: A list of dictionaries containing parsed, confirmed SAS errors, another containing SAS warnings,
            and another list of unidentifiable errors that occured in the stderr.
        :rtype: Tuple[List[Dict], List[Dict], List]
        """
        def find_sas(split_stderr: list, err_type: str) -> Tuple[List[dict], List[str]]:
            """
            Function to search for and parse SAS errors and warnings.

            :param list split_stderr: The stderr string split on line endings.
            :param str err_type: Should this look for errors or warnings?
            :return: Returns the dictionary of parsed errors/warnings, as well as all lines
                with SAS errors/warnings in.
            :rtype: Tuple[List[dict], List[str]]
            """
            parsed_sas = []
            # This is a crude way of looking for SAS error/warning strings ONLY
            sas_lines = [line for line in split_stderr if "** " in line and ": {}".format(err_type) in line]
            for err in sas_lines:
                try:
                    # This tries to split out the SAS task that produced the error
                    originator = err.split("** ")[-1].split(":")[0]
                    # And this should split out the actual error name
                    err_ident = err.split(": {} (".format(err_type))[-1].split(")")[0]
                    # Actual error message
                    err_body = err.split("({})".format(err_ident))[-1].strip("\n").strip(", ").strip(" ")

                    if err_type == "error":
                        # Checking to see if the error identity is in the list of SAS errors
                        sas_err_match = [sas_err for sas_err in SASERROR_LIST if err_ident.lower()
                                         in sas_err.lower()]
                    elif err_type == "warning":
                        # Checking to see if the error identity is in the list of SAS warnings
                        sas_err_match = [sas_err for sas_err in SASWARNING_LIST if err_ident.lower()
                                         in sas_err.lower()]

                    if len(sas_err_match) != 1:
                        originator = ""
                        err_ident = ""
                        err_body = ""
                except IndexError:
                    originator = ""
                    err_ident = ""
                    err_body = ""

                parsed_sas.append({"originator": originator, "name": err_ident, "message": err_body})
            return parsed_sas, sas_lines

        # Defined as empty as they are returned by this method
        sas_errs_msgs = []
        parsed_sas_warns = []
        other_err_lines = []
        # err_str being "" is ideal, hopefully means that nothing has gone wrong
        if self.unprocessed_stderr != "":
            # Errors will be added to the error summary, then raised later
            # That way if people try except the error away the object will have been constructed properly
            err_lines = [e for e in self.unprocessed_stderr.split('\n') if e != '']
            # Fingers crossed each line is a separate error
            parsed_sas_errs, sas_err_lines = find_sas(err_lines, "error")
            parsed_sas_warns, sas_warn_lines = find_sas(err_lines, "warning")

            sas_errs_msgs = ["{e} raised by {t} - {b}".format(e=e["name"], t=e["originator"], b=e["message"])
                             for e in parsed_sas_errs]

            # These are impossible to predict the form of, so they won't be parsed
            other_err_lines = [line for line in err_lines if line not in sas_err_lines
                               and line not in sas_warn_lines and line != "" and "warn" not in line]
            # Adding some advice
            for e_ind, e in enumerate(other_err_lines):
                if 'seg' in e.lower() and 'fault' in e.lower():
                    other_err_lines[e_ind] += ' - Try examining an image of the cluster with regions subtracted, ' \
                                              'and have a look at where your coordinate lies.'

        if len(sas_errs_msgs) > 0:
            self._usable = False
            self._why_unusable.append("SASErrorPresent")
        if len(other_err_lines) > 0:
            self._usable = False
            self._why_unusable.append("OtherErrorPresent")

        return sas_errs_msgs, parsed_sas_warns, other_err_lines

    @property
    def sas_errors(self) -> List[str]:
        """
        Property getter for the confirmed SAS errors associated with a product.

        :return: The list of confirmed SAS errors.
        :rtype: List[Dict]
        """
        return self._sas_error

    @property
    def sas_warnings(self) -> List[Dict]:
        """
        Property getter for the confirmed SAS warnings associated with a product.

        :return: The list of confirmed SAS warnings.
        :rtype: List[Dict]
        """
        return self._sas_warn

    def raise_errors(self):
        """
        Method to raise the errors parsed from std_err string.
        """
        for error in self._sas_error:
            raise SASGenerationError(error)

        # This is for any unresolved errors.
        for error in self._other_error:
            if "warning" not in error:
                raise UnknownCommandlineError("{}".format(error))

    @property
    def obs_id(self) -> str:
        """
        Property getter for the ObsID of this image. Admittedly this information is implicit in the location
        this object is stored in a source object, but I think it worth storing directly as a property as well.

        :return: The XMM ObsID of this image.
        :rtype: str
        """
        return self._obs_id

    @property
    def instrument(self) -> str:
        """
        Property getter for the instrument used to take this image. Admittedly this information is implicit
        in the location this object is stored in a source object, but I think it worth storing
        directly as a property as well.

        :return: The XMM instrument used to take this image.
        :rtype: str
        """
        return self._inst

    @property
    def type(self) -> str:
        """
        Property getter for the string identifier for the type of product this object is, mostly useful for
        internal methods of source objects.

        :return: The string identifier for this type of object.
        :rtype: str
        """
        return self._prod_type

    @property
    def errors(self) -> List[str]:
        """
        Property getter for non-SAS errors detected during the generation of a product.

        :return: A list of errors that aren't related to SAS.
        :rtype: List[str]
        """
        return self._other_error

    # This is a fundamental property of the generated product, so I won't allow it be changed.
    @property
    def energy_bounds(self) -> Tuple[Quantity, Quantity]:
        """
        Getter method for the energy_bounds property, which returns the rest frame energy band that this
        product was generated in.

        :return: Tuple containing the lower and upper energy limits as Astropy quantities.
        :rtype: Tuple[Quantity, Quantity]
        """
        return self._energy_bounds

    @property
    def src_name(self) -> str:
        """
        Method to return the name of the object a product is associated with. The product becomes
        aware of this once it is added to a source object.

        :return: The name of the source object this product is associated with.
        :rtype: str
        """
        return self._src_name

    # This needs a setter, as this property only becomes not-None when the product is added to a source object.
    @src_name.setter
    def src_name(self, name: str):
        """
        Property setter for the src_name attribute of a product, should only really be called by a source object,
        not by a user.

        :param str name: The name of the source object associated with this product.
        """
        self._src_name = name

    @property
    def not_usable_reasons(self) -> List:
        """
        Whenever the usable flag of a product is set to False (indicating you shouldn't use the product), a string
        indicating the reason is added to a list, which this property returns.

        :return: A list of reasons why this product is unusable.
        :rtype: List
        """
        return self._why_unusable

    @property
    def sas_command(self) -> str:
        """
        A property that returns the original SAS command used to generate this object.

        :return: String containing the command.
        :rtype: str
        """
        return self._og_cmd


class BaseAggregateProduct:
    """
    A base class for any XGA products that are an aggregate of an XGA SAS product.
    """
    def __init__(self, file_paths: list, prod_type: str, obs_id: str, instrument: str):
        self._all_paths = file_paths
        self._all_usable = True
        self._obs_id = obs_id
        self._inst = instrument
        self._prod_type = prod_type
        self._src_name = None

        # This was originally going to create the individual products here, but realised it was
        # easier to do in subclasses
        self._component_products = {}

        # Setting up energy limits, if they're ever required
        self._energy_bounds = (None, None)

    @property
    def src_name(self) -> str:
        """
        Method to return the name of the object a product is associated with. The product becomes
        aware of this once it is added to a source object.

        :return: The name of the source object this product is associated with.
        :rtype: str
        """
        return self._src_name

    # This needs a setter, as this property only becomes not-None when the product is added to a source object.
    @src_name.setter
    def src_name(self, name: str):
        """
        Property setter for the src_name attribute of a product, should only really be called by a source object,
        not by a user.

        :param str name: The name of the source object associated with this product.
        """
        self._src_name = name

    @property
    def obs_id(self) -> str:
        """
        Property getter for the ObsID of this image. Admittedly this information is implicit in the location
        this object is stored in a source object, but I think it worth storing directly as a property as well.

        :return: The XMM ObsID of this image.
        :rtype: str
        """
        return self._obs_id

    @property
    def instrument(self) -> str:
        """
        Property getter for the instrument used to take this image. Admittedly this information is implicit
        in the location this object is stored in a source object, but I think it worth storing
        directly as a property as well.

        :return: The XMM instrument used to take this image.
        :rtype: str
        """
        return self._inst

    @property
    def type(self) -> str:
        """
        Property getter for the string identifier for the type of product this object is, mostly useful for
        internal methods of source objects.

        :return: The string identifier for this type of object.
        :rtype: str
        """
        return self._prod_type

    @property
    def usable(self) -> bool:
        """
        Property getter for the boolean variable that tells you whether all component products have been
        found to be usable.

        :return: Boolean variable, are all component products usable?
        :rtype: bool
        """
        return self._all_usable

    # This is a fundamental property of the generated product, so I won't allow it be changed.
    @property
    def energy_bounds(self) -> Tuple[Quantity, Quantity]:
        """
        Getter method for the energy_bounds property, which returns the rest frame energy band that this
        product was generated in, if relevant.

        :return: Tuple containing the lower and upper energy limits as Astropy quantities.
        :rtype: Tuple[Quantity, Quantity]
        """
        return self._energy_bounds

    @property
    def sas_errors(self) -> List:
        """
        Equivelant to the BaseProduct sas_errors property, but reports any SAS errors stored in the component products.

        :return: A list of SAS errors related to component products.
        :rtype: List
        """
        sas_err_list = []
        for p in self._component_products:
            prod = self._component_products[p]
            sas_err_list += prod.sas_errors
        return sas_err_list

    @property
    def errors(self) -> List:
        """
        Equivelant to the BaseProduct errors property, but reports any non-SAS errors stored in the
        component products.

        :return: A list of non-SAS errors related to component products.
        :rtype: List
        """
        err_list = []
        for p in self._component_products:
            prod = self._component_products[p]
            err_list += prod.errors
        return err_list

    @property
    def unprocessed_stderr(self) -> List:
        """
        Equivelant to the BaseProduct sas_errors unprocessed_stderr, but returns a list of all the unprocessed
        standard error outputs.

        :return: List of stderr outputs.
        :rtype: List
        """
        unprocessed_err_list = []
        for p in self._component_products:
            prod = self._component_products[p]
            unprocessed_err_list.append(prod.unprocessed_stderr)
        return unprocessed_err_list

    def __len__(self) -> int:
        """
        The length of an AggregateProduct is the number of component products that makes it up.

        :return:
        :rtype: int
        """
        return len(self._component_products)

    def __iter__(self):
        """
        Called when initiating iterating through an AggregateProduct based object. Resets the counter _n.
        """
        self._n = 0
        return self

    def __next__(self):
        """
        Iterates the counter _n and returns the next entry in the the component_products dictionary.
        """
        if self._n < self.__len__():
            result = self.__getitem__(self._n)
            self._n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, ind):
        return list(self._component_products.values())[ind]


class BaseProfile1D:
    """
    The superclass for all 1D radial profile products, with built in fitting, viewing, and result retrieval
    functionality. Classes derived from BaseProfile1D can be added together to create Aggregate Profiles.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str,
                 inst: str, radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None, x_norm: Quantity = Quantity(1, ''),
                 y_norm: Quantity = Quantity(1, '')):
        """
        The init of the superclass 1D profile product. Unlikely to ever be declared by a user, but the base
        of all other 1D profiles in XGA - contains many useful functions.

        :param Quantity radii: The radii at which the y values of this profile have been measured.
        :param Quantity values: The y values of this profile.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable. If this
            value is supplied a set_storage_key value must also be supplied.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        :param Quantity x_norm: An astropy quantity to use to normalise the x-axis values, this is only used when
            plotting if the user tells the view method that they wish for the plot to use normalised x-axis data.
        :param Quantity y_norm: An astropy quantity to use to normalise the y-axis values, this is only used when
            plotting if the user tells the view method that they wish for the plot to use normalised y-axis data.
        """
        if type(radii) != Quantity or type(values) != Quantity:
            raise TypeError("Both the radii and values passed into this object definition must "
                            "be astropy quantities.")
        elif radii_err is not None and type(radii_err) != Quantity:
            raise TypeError("The radii_err variable must be an astropy Quantity, or None.")
        elif radii_err is not None and radii_err.unit != radii.unit:
            raise UnitConversionError("The radii_err unit must be the same as the radii unit.")
        elif values_err is not None and type(values_err) != Quantity:
            raise TypeError("The values_err variable must be an astropy Quantity, or None.")
        elif values_err is not None and values_err.unit != values.unit:
            raise UnitConversionError("The values_err unit must be the same as the values unit.")

        # Check for one dimensionality
        if radii.ndim != 1 or values.ndim != 1:
            raise ValueError("The radii and values arrays must be one-dimensional. The shape of radii is {0} "
                             "and the shape of values is {1}".format(radii.shape, values.shape))
        elif (radii_err is not None and radii_err.ndim != 1) or (values_err is not None and values_err.ndim != 1):
            raise ValueError("The radii_err and values_err arrays must be one-dimensional. The shape of "
                             "radii_err is {0} and the shape of values_err is "
                             "{1}".format(radii_err.shape, values_err.shape))
        # Making sure the arrays have the same number of entries
        elif radii.shape != values.shape:
            raise ValueError("The radii and values arrays must have the same shape. The shape of radii is {0} "
                             "and the shape of values is {1}".format(radii.shape, values.shape))
        elif (radii_err is not None and radii_err.shape != radii.shape) or \
                (values_err is not None and values_err.shape != values.shape):
            raise ValueError("radii_err must be the same shape as radii, and values_err must be the same shape "
                             "as values. The shape of radii_err is {0} where radii is {1}, and the shape of "
                             "values_err is {2} where values is {3}".format(radii_err.shape, radii.shape,
                                                                            values_err.shape, values.shape))

        # I'm actually going to enforce that the central coordinates passed when declaring a profile must
        #  be RA and Dec, I need the storage keys to be predictable to make everything neater
        if centre.unit != deg:
            raise UnitConversionError("The central coordinate value passed into a profile on declaration must be"
                                      " in RA and Dec coordinates.")

        # I'm also going to require that the profiles have knowledge of radii in degree units, also so I can make
        #  predictable storage strings. I don't really like to do this as it feels bodgy, but oh well
        if not radii.unit.is_equivalent('deg') and deg_radii is None and set_storage_key is None:
            raise ValueError("If the radii variable is not in units that are convertible to degrees, please pass "
                             "radii in degrees to deg_radii, this profile needs knowledge of the radii in degrees"
                             " to construct a storage key.")
        elif not radii.unit.is_equivalent('deg') and set_storage_key is None and len(deg_radii) != len(radii):
            raise ValueError("deg_radii is a different length to radii, they should be equivelant quantities, simply"
                             " in different units.")
        elif radii.unit.is_equivalent('deg') and set_storage_key is None:
            deg_radii = radii.to('deg')

        if deg_radii is not None:
            deg_radii = deg_radii.to("deg")
            self._deg_radii = deg_radii

        # Storing the key values in attributes
        self._radii = radii
        self._values = values
        self._radii_err = radii_err
        self._values_err = values_err
        self._centre = centre

        # This generates an array containing (hopefully) the original annular boundaries of the profile
        if self._radii_err is not None:
            upper_bounds = self._radii + self._radii_err
            bounds = np.insert(upper_bounds, 0, self._radii[0]-self._radii_err[0])

            if self._radii[0].value == 0:
                bounds[0] = self._radii[0]
                bounds[1] = bounds[1] + self._radii_err[0]
            self._rad_ann_bounds = bounds
        else:
            self._rad_ann_bounds = None

        # Just checking that if one of these values is combined, then both are. Doesn't make sense otherwise.
        if (obs_id == "combined" and inst != "combined") or (inst == "combined" and obs_id != "combined"):
            raise ValueError("If ObsID or inst is set to combined, then both must be set to combined.")

        # Storing the passed source name in an attribute, as well as the ObsID and instrument
        self._src_name = source_name
        self._obs_id = obs_id
        self._inst = inst

        # Going to have this convenient attribute for profile classes, I could just use the type() command
        #  when I wanted to know but this is easier.
        self._prof_type = "base"

        # Here is where information about fitted models is stored (and any failed fit attempts)
        self._good_model_fits = {}
        self._bad_model_fits = {}
        # Previously I stored model realisations in self._good_model_fits, but I'm splitting out into its
        #  own attribute. Primarily because I want to be able to add realisations from non-model sources in
        #  the Density1D profile product.
        self._realisations = {}

        # Some types of profiles will support a background value (like surface brightness), which will
        #  need to be incorporated into the fit and plotting.
        self._background = Quantity(0, self._values.unit)

        # Need to be able to store upper and lower energy bounds for those profiles that
        #  have them (like brightness profiles for instance)
        self._energy_bounds = (None, None)

        # This is where allowed realisation types are stored, but there are none for the base profile
        self._allowed_real_types = []

        # Checking the if associated_set_id is supplied, so is set_storage_key, and vica versa
        if not all([associated_set_id is None, set_storage_key is None]) and \
                not all([associated_set_id is not None, set_storage_key is not None]):
            raise ValueError("Both associated_set_id and set_storage_key must be None, or both must be not None.")

        # Putting the associated set ID into an attribute, if this profile wasn't generated by an AnnularSpectra
        #  then this will just be None. Same for the set_storage_key
        self._set_id = associated_set_id
        # Don't think this one will get a property, I can't see why the user would need it.
        self._set_storage_key = set_storage_key

        # Here we generate a storage key for the profile to use to place itself in XGA's storage structure
        if self._set_storage_key is not None:
            # If there is a storage key for a spectrum which generated this available, then our life becomes
            #  quite simple, as it has a lot of information in it.

            # In fact as the profile will also be indexed under the profile type name, we can just use this as
            #  our storage key
            self._storage_key = self._set_storage_key
        else:
            # Default storage key for profiles that don't implement their own storage key will include their radii
            #  and the central coordinate
            # Just being doubly sure its in degrees
            cent_chunk = "ra{r}_dec{d}_r".format(r=centre.value[0], d=centre.value[1])
            rad_chunk = "_".join(self._deg_radii.value.astype(str))
            self._storage_key = cent_chunk + rad_chunk

        # The y-axis label used to be stored in a dictionary in the init of models, but it makes more sense
        #  just declaring it in the init I think - it should be over-ridden in every subclass
        self._y_axis_name = "Unknown"

        # Akin to the usable attribute of other product classes, will be set for different reasons by different
        #  profile subclasses
        self._usable = True

        # These are the normalisation values for plotting (and possibly fitting at some point). I don't check their
        #  units at any point because the user is welcome to normalise their plots by whatever value they wish
        self._x_norm = x_norm
        self._y_norm = y_norm

    def fit(self, model: str, method: str = "mcmc", priors: list = None, start_pars: list = None,
            model_real: int = 1000, model_rad_steps: int = 300, conf_level: int = 90, num_walkers: int = 20,
            num_steps: int = 20000, progress_bar: bool = True, show_errors: bool = True):
        """
        The fitting method for 1D XGA profile products, it uses the type of profile which is calling the method
        to determine what models are allowed to be fit to the data. It also allowed for different fit methods,
        including scipy's curve_fit implementation of non-linear least squares, and emcee's ensemble MCMC fitter.

        :param str model: The name of the model to fit to the profile, the allowed_models method can be used to
            find a list of legal models for a specific type of profile.
        :param str method: Which fitting method should be used, default is "mcmc".
        :param list priors: The priors to use for the given model (IF USING MCMC FITTING), though default values
            will be used if no priors are passed. This should be a nested list, with one entry per parameter, and each
            entry being a list with two values, a lower limit and an upper limit (in that order).
        :param list start_pars: The start parameters to use for the given model (IF USING CURVE FIT), though default
            values will be used if no start parameters are passed. This should be a list, with one entry per model
            parameter.
        :param int model_real: The number of realisations of the fitted model to generate for uncertainty measurement.
        :param int model_rad_steps: The number of radial steps to use to generate realisations of the model.
        :param int conf_level: The confidence level to calculate uncertainties at.
        :param int num_walkers: The number of walkers for the emcee ensemble MCMC fitter to use, if using MCMC fitting.
        :param int num_steps: The number of steps for each walker to take, if using MCMC fitting.
        :param bool progress_bar: Controls whether a progress bar should be shown for MCMC fitting.
        :param bool show_errors: If there are errors during MCMC fitting, should they be displayed.
        """
        # These are the currently allowed fitting methods
        method = method.lower()
        fit_methods = ["curve_fit", "mcmc"]
        # Checking that the user hasn't chosen a method that isn't allowed
        if method not in fit_methods:
            raise ValueError("{0} is not an accepted fitting method, please choose one of these; "
                             "{1}".format(method, ", ".join(fit_methods)))
        elif method == "curve_fit" and priors is not None:
            warn("You have chosen curve_fit, and also provided priors, these will not be used.")

        # Stopping the user from making stupid model choices
        if self._prof_type == "base":
            raise XGAFitError("A BaseProfile1D object currently cannot have a model fitted to it, as there"
                              " is no physical context.")
        elif model not in PROF_TYPE_MODELS[self._prof_type]:
            allowed = list(PROF_TYPE_MODELS[self._prof_type].keys())
            prof_name = self._y_axis_name.lower()
            raise XGAInvalidModelError("{m} is not a valid model for a {p} profile, please choose from "
                                       "one of these; {a}".format(m=model, a=", ".join(allowed), p=prof_name))
        else:
            model_func = PROF_TYPE_MODELS[self._prof_type][model]

        # Changes confidence level to expected input for numpy percentile function
        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        # I used to grab the variable names from the function signature, but now I've implemented dictionaries
        #  that contain 'nice' parameter names for all the models (nice as in LaTeX formatted etc.)
        model_par_names = MODEL_PUBLICATION_PAR_NAMES[model]
        if start_pars is not None and len(start_pars) != len(model_par_names):
            raise ValueError("start_pars must either be None, or have an entry for each parameter expected by"
                             " the chosen model; {0} expects {1}".format(model, ", ".join(model_par_names)))
        elif start_pars is None:
            # If the user doesn't supply any starting parameters then we just have to use the default ones
            start_pars = PROF_TYPE_MODELS_STARTS[self._prof_type][model]

        # Even though we won't always need priors I'm just grab them anyway
        if priors is not None and len(priors) != len(model_par_names):
            raise ValueError("priors must either be None, or have an entry for each parameter expected by"
                             " the chosen model; {0} expects {1}".format(model, ", ".join(model_par_names)))
        elif priors is None:
            # If the user doesn't supply any priors then we use the default ones
            priors = PROF_TYPE_MODELS_PRIORS[self._prof_type][model]

        # I don't think I'm going to allow any fits without value uncertainties - just seems daft
        if self._values_err is None:
            raise XGAFitError("You cannot fit to a profile that doesn't have value uncertainties.")

        # Check whether a good fit result already exists for this model
        if model in self._good_model_fits:
            warn("{} already has a successful fit result for this profile".format(model))
            already_done = True
        else:
            already_done = False

        # Check whether this fit is in the bad fit dictionary
        if model in self._bad_model_fits:
            warn("{} already has a failed fit result for this profile".format(model))

        # Now we do the actual fitting part
        if method == "curve_fit" and not already_done:
            success = True
            # Curve fit is a simple non-linear least squares implementation, its alright but fragile
            try:
                fit_par, fit_cov = curve_fit(model_func, self._radii.value, self.values.value
                                             - self._background.value, p0=start_pars, sigma=self._values_err.value,
                                             absolute_sigma=True)
                # Grab the diagonal of the covariance matrix, then sqrt to get sigma values for each parameter
                fit_par_err = np.sqrt(np.diagonal(fit_cov))
                frac_err = np.divide(fit_par_err, fit_par, where=fit_par != 0)
                if frac_err.max() > 10:
                    warn("A parameter uncertainty is more than 10 times larger than the parameter, curve_fit "
                         "has failed.")
                    success = False
                # If there is an infinite value in the covariance matrix, it means curve_fit was
                #  unable to estimate it properly
                if np.inf in fit_cov:
                    success = False
            except RuntimeError:
                warn("RuntimeError was raised, curve_fit has failed.")
                success = False
                fit_par = np.full(len(start_pars), np.nan)
                fit_par_err = np.full(len(start_pars), np.nan)

        elif method == "mcmc" and not already_done:
            # I'm just defining these here so that the lines don't get too long for PEP standards
            r_dat = self.radii.value
            v_dat = self.values.value - self.background.value
            v_err = self.values_err.value
            n_par = len(priors)
            prior_arr = np.array(priors)

            for_max_like = lambda *args: -log_likelihood(*args, model_func)
            # This finds maximum likelihood parameter values for the model+data
            max_like_res = minimize(for_max_like, start_pars, args=(r_dat, v_dat, v_err))

            # This basically finds the order of magnitude (+1) of each parameter
            ml_rand_dev = np.power(10, np.floor(np.log10(np.abs(max_like_res.x))))

            # Then that order of magnitude is multiplied by a value drawn from a standard gaussian, and this is what
            #  we perturb the maximum likelihood values with - so we get random start parameters for all
            #  of our walkers
            pos = max_like_res.x + ml_rand_dev*np.random.randn(num_walkers, n_par)

            # It is possible that some of the start parameters we've generated are outside the prior, in which
            #  case emcee gets quite angry. Just in case I draw random values from the priors of all parameters,
            #  ready to be substituted in if a start par is outside the allowed range
            rand_uniform_pos = np.random.uniform(prior_arr[:, 0], prior_arr[:, 1], size=(num_walkers, n_par))

            # It is possible that the start parameters can be outside of the range allowed by the the priors. In which
            #  case the MCMC fit will get super upset but not actually throw an error.
            start_check_greater = np.greater_equal(pos, prior_arr[:, 0])
            start_check_lower = np.less_equal(pos, prior_arr[:, 1])
            # Any true value in this array is a parameter that isn't in the allowed prior range
            to_replace = ~(start_check_greater & start_check_lower)

            # So any start values that fall outside the allowed range will be swapped out with a value randomly drawn
            #  from the prior
            pos[to_replace] = rand_uniform_pos[to_replace]

            # This instantiates an Ensemble sampler with the number of walkers specified by the user,
            #  with the log probability as defined in the functions above
            sampler = em.EnsembleSampler(num_walkers, n_par, log_prob, args=(r_dat, v_dat, v_err,
                                                                             model_func, priors))
            try:
                # So now we start the sampler, running for the number of steps specified on function call, with
                #  the starting parameters defined in the if statement above this.
                sampler.run_mcmc(pos, num_steps, progress=progress_bar)
                success = True
            except ValueError as bugger:
                if show_errors:
                    print("SAMPLER ERROR", bugger)
                success = False

            if success:
                # The auto-correlation can produce an error that basically says not to trust the chains
                try:
                    # The sampler has a convenient auto-correlation time derivation, which returns the
                    #  auto-correlation time for each parameter - with this I simply choose the highest one and
                    #  round up to the nearest 100 to use as the burn-in
                    auto_corr = sampler.get_autocorr_time()
                    cut_off = int(np.ceil(auto_corr.max() / 100) * 100)
                    success = True
                except ValueError as bugger:
                    if show_errors:
                        print("AUTOCORRELATION VALUE ERROR", bugger)
                    success = False
                except em.autocorr.AutocorrError as bugger:
                    warn("AUTOCORRELATION ERROR " + str(bugger))
                    cut_off = int(0.1 * num_steps)
        # Now do some checks after the fit has run, primarily for any infinite values
        if not already_done and method == "curve_fit" and ((np.inf in fit_par or np.inf in fit_par_err)
                                 or (True in np.isnan(fit_par) or True in np.isnan(fit_par_err))):
            # This is obviously bad, and enough of a reason to call a fit bad as an outright failure to fit
            success = False

        # If the fit succeeded to our satisfaction then it gets stored in the good dictionary, otherwise we record
        #  it in the bad dictionary.
        if not already_done and success and method == "curve_fit":
            ext_model_par = np.repeat(fit_par[..., None], model_real, axis=1).T
            ext_model_par_err = np.repeat(fit_par_err[..., None], model_real, axis=1).T

            # This generates model_real random samples from the passed model parameters, assuming they are Gaussian
            model_par_dists = np.random.normal(ext_model_par, ext_model_par_err)

            # No longer need these now we've drawn the random samples
            del ext_model_par
            del ext_model_par_err

            # Setting up some radii between 0 and the maximum radius to sample the model at
            if self._radii_err is None:
                model_radii = np.linspace(0, self._radii[-1].value, model_rad_steps)
            else:
                model_radii = np.linspace(0, self._radii[-1].value + self._radii_err[-1].value, model_rad_steps)

            # Copies the chosen radii model_real times, much as with the ext_model_par definition
            ext_model_radii = np.repeat(model_radii[..., None], model_real, axis=1)

            # Generates model_real realisations of the model at the model_radii
            model_realisations = model_func(ext_model_radii, *model_par_dists.T)

            # Calculates the mean model value at each radius step
            model_mean = np.mean(model_realisations, axis=1)
            # Then calculates the values for the upper and lower limits (defined by the
            #  confidence level) for each radii
            model_lower = np.percentile(model_realisations, lower, axis=1)
            model_upper = np.percentile(model_realisations, upper, axis=1)

            # Store these realisations for statistics later on
            self._good_model_fits[model] = {"par": fit_par, "par_err": fit_par_err, "start_pars": start_pars,
                                            "model_func": model_func, "par_names": model_par_names,
                                            "conf_level": conf_level, "par_err_1sig": fit_par_err}
            self._realisations[model] = {"mod_real": model_realisations, "mod_radii": model_radii,
                                         "conf_level": conf_level, "mod_real_mean": model_mean,
                                         "mod_real_lower": model_lower, "mod_real_upper": model_upper}

        elif not already_done and success and method == "mcmc":
            thinning = int(num_steps / model_real)
            flat_samp = sampler.get_chain(discard=cut_off, thin=thinning, flat=True)
            pars_lower = np.percentile(flat_samp, lower, axis=0)
            pars_upper = np.percentile(flat_samp, upper, axis=0)
            fit_par = np.mean(flat_samp, axis=0)
            fit_par_mi = fit_par - pars_lower
            fit_par_pl = pars_upper - fit_par
            fit_par_1sig = np.std(flat_samp, axis=0)

            # Setting up some radii between 0 and the maximum radius to sample the model at
            if self._radii_err is None:
                model_radii = np.linspace(0, self._radii[-1].value, model_rad_steps)
            else:
                model_radii = np.linspace(0, self._radii[-1].value + self._radii_err[-1].value, model_rad_steps)

            # Copies the chosen radii model_real times, much as with the ext_model_par definition
            ext_model_radii = np.repeat(model_radii[..., None], flat_samp.shape[0], axis=1)

            # Generates model_real realisations of the model at the model_radii
            model_realisations = model_func(ext_model_radii, *flat_samp.T)
            model_mean = np.mean(model_realisations, axis=1)
            model_lower = np.percentile(model_realisations, lower, axis=1)
            model_upper = np.percentile(model_realisations, upper, axis=1)

            self._good_model_fits[model] = {"par": fit_par, "par_err_mi": fit_par_mi, "par_err_pl": fit_par_pl,
                                            "model_func": model_func, "sampler": sampler, "thinning": thinning,
                                            "cut_off": cut_off, "par_names": model_par_names,
                                            "conf_level": conf_level, "par_err_1sig": fit_par_1sig}
            self._realisations[model] = {"mod_real": model_realisations, "mod_radii": model_radii,
                                         "conf_level": conf_level, "mod_real_mean": model_mean,
                                         "mod_real_lower": model_lower, "mod_real_upper": model_upper}

        elif not already_done and not success and method == "mcmc":
            self._bad_model_fits[model] = {"start_pars": start_pars}

        elif not already_done and not success and method == "curve_fit":
            self._bad_model_fits[model] = {"priors": priors}

    def get_realisation(self, real_type: str) -> Dict:
        """
        Get method for model realisation data, this includes the array of realisations, the radii at which
        the realisations are generated, the upper and lower bounds, the mean, and the confidence level.

        :param str real_type: The type of realisation to be retrieved, most often a model name, or the key
            associated with a particular function that generated realisations (such as inv_abel_model).
        :return: The realisation dictionary with relevant information in it, or None if no matching
            realisation exists.
        :rtype: Dict
        """
        if real_type in self._allowed_real_types or real_type in self._good_model_fits:
            return self._realisations[real_type]
        else:
            return None

    def get_model_fit(self, model) -> Dict:
        """
        Get method for parameters of fitted models.

        :param model: The name of the model for which to retrieve parameters.
        :return: A dictionary containing the fit parameters, their uncertainties, an instance of the model
            function, and the initial parameters.
        :rtype: Dict
        """
        if model not in PROF_TYPE_MODELS[self._prof_type]:
            allowed = list(PROF_TYPE_MODELS[self._prof_type].keys())
            prof_name = self._y_axis_name.lower()
            raise XGAInvalidModelError("{m} is not a valid model for a {p} profile, please choose from "
                                       "one of these; {a}".format(m=model, a=", ".join(allowed), p=prof_name))
        elif model in self._bad_model_fits:
            raise XGAFitError("An attempt was made to fit {}, but it failed, no fit data can be "
                              "retrieved.".format(model))
        elif model not in self._good_model_fits:
            raise ModelNotAssociatedError("{} is valid for this profile, but hasn't been fit yet".format(model))

        return self._good_model_fits[model]

    def allowed_models(self):
        """
        This is a convenience function to tell the user what models can be used to fit a profile
        of the current type, what parameters are expected, and what the defaults are.
        """
        # Base profile don't have any type of model associated with them, so just making an empty list
        if self._prof_type == "base":
            allowed = []
        else:
            allowed = list(PROF_TYPE_MODELS[self._prof_type].keys())

        # These set up the dictionary of printables, and variables that store the longest entry for each column
        to_print = {}
        # Initial values are the column sizes of the headers
        longest_name = 12
        longest_pars = 21
        longest_defaults = 26
        for model in allowed:
            # Function object grabbed
            model_func = PROF_TYPE_MODELS[self._prof_type][model]
            # Looking for the variables in the function signature
            model_sig = inspect.signature(model_func)
            # Ignore the first argument, as it will be radius
            model_par_names = ", ".join([p.name for p in list(model_sig.parameters.values())[1:]])
            # The default start parameters of the fit
            start_pars = ", ".join([str(p) for p in PROF_TYPE_MODELS_STARTS[self._prof_type][model]])
            to_print[model] = [model, model_par_names, start_pars]
            if len(model) > longest_name:
                longest_name = len(model)
            if len(model_par_names) > longest_pars:
                longest_pars = len(model_par_names)
            if len(start_pars) > longest_defaults:
                longest_defaults = len(start_pars)

        if longest_name % 2 != 0:
            longest_name += 3
        else:
            longest_name += 2

        if longest_pars % 2 != 0:
            longest_pars += 3
        else:
            longest_pars += 2

        if longest_defaults % 2 != 0:
            longest_defaults += 3
        else:
            longest_defaults += 2

        # This next lot is just boring string formatting and printing, I'm sure you can figure it out.
        first_col = "|" + " " * np.ceil((longest_name - 12) / 2).astype(int) + " MODEL NAME " + " " * np.ceil(
            (longest_name - 12) / 2).astype(int) + "|"

        second_col = " " * np.ceil((longest_pars - 21) / 2).astype(int) + " EXPECTED PARAMETERS " + " " * np.ceil(
            (longest_pars - 21) / 2).astype(int) + "|"

        third_col = " "*np.ceil((longest_defaults-26) / 2).astype(int) + " DEFAULT START PARAMETERS " + \
                    " " * np.ceil((longest_defaults-26) / 2).astype(int) + "|"
        comb = first_col + second_col + third_col
        print("\n" + "-"*len(comb))
        print(first_col + second_col + third_col)
        print("-"*len(comb))
        for model in to_print:
            # I know this code is disgustingly ugly, but its not really important that you know how it works
            # And perhaps I'll rewrite it at some point, who knows
            the_line = "|" + " " * np.ceil((len(first_col) - len(to_print[model][0])) / 2).astype(int) + \
                       to_print[model][0] + " " * np.ceil((len(first_col) -
                                                           len(to_print[model][0])) / 2).astype(int) \
                       + "|"

            the_line += " "*np.ceil((len(second_col) -
                                     len(to_print[model][1])) / 2).astype(int) + to_print[model][1] + \
                        " "*np.ceil((len(second_col)-len(to_print[model][1])) / 2).astype(int) + "|"

            the_line += " " * np.ceil((len(third_col) -
                                       len(to_print[model][2])) / 2).astype(int) + to_print[model][
                2] + " " * np.ceil((len(third_col) - len(to_print[model][2])) / 2).astype(int) + "|"
            print(the_line)
        print("-" * len(comb) + "\n")

    def get_sampler(self, model: str) -> em.EnsembleSampler:
        """
        A get method meant to retrieve the MCMC ensemble sampler used to fit a particular
        model (supplied by the user). Checks are applied to the supplied model, to make
        sure that it is valid for the type of profile, that a good fit has actually been
        performed, and that the fit was performed with Emcee and not another method.

        :param str model: The name of the model for which to retrieve the sampler.
        :return: The Emcee sampler used to fit the user supplied model - if applicable.
        :rtype: em.EnsembleSampler
        """
        if model not in PROF_TYPE_MODELS[self._prof_type]:
            allowed = list(PROF_TYPE_MODELS[self._prof_type].keys())
            prof_name = self._y_axis_name.lower()
            raise XGAInvalidModelError("{m} is not a valid model for a {p} profile, please choose from "
                                       "one of these; {a}".format(m=model, a=", ".join(allowed), p=prof_name))
        elif model in self._bad_model_fits:
            raise XGAFitError("An attempt was made to fit {}, but it failed, no fit data can be "
                              "retrieved.".format(model))
        elif model not in self._good_model_fits:
            raise XGAFitError("{} is valid for this profile, but hasn't been fit yet".format(model))
        elif model in self._good_model_fits and "sampler" not in self._good_model_fits[model]:
            raise XGAFitError("{} was not fit with MCMC, and as such the sampler object cannot be "
                              "retrieved.".format(model))

        return self._good_model_fits[model]["sampler"]

    def get_chains(self, model: str) -> np.ndarray:
        """
        Get method for the sampler chains of an MCMC fit to the user supplied model. get_sampler is
        called to retrieve the sampler object, as well as perform validity checks on the model name.
        :param str model: The name of the model for which to retrieve the chains.
        :return: The sampler chains, with burn-in discarded, and with thinning applied.
        :rtype: np.ndarray
        """
        sampler = self.get_sampler(model)
        m_info = self.get_model_fit(model)

        return sampler.get_chain(discard=m_info["cut_off"], thin=m_info["thinning"])

    def get_flat_samples(self, model: str) -> np.ndarray:
        """
        Get method for the flattened samples of an MCMC fit to the user supplied model. get_sampler is
        called to retrieve the sampler object, as well as perform validity checks on the model name.

        :param str model: The name of the model for which to retrieve the flat samples.
        :return: The flattened posterior samples, with burn-in discarded, and with thinning applied.
        :rtype: np.ndarray
        """
        sampler = self.get_sampler(model)
        m_info = self.get_model_fit(model)

        return sampler.get_chain(discard=m_info["cut_off"], thin=m_info["thinning"], flat=True)

    def view_chains(self, model: str, figsize: Tuple = None):
        """
        Simple view method to quickly look at the MCMC chains for a given model fit, though bear in the mind that
        these chains have already been thinned.

        :param str model: The name of the model for which to view the MCMC chains.
        :param Tuple figsize: Desired size of the figure, if None will be set automatically.
        """
        chains = self.get_chains(model)
        m_info = self.get_model_fit(model)

        if figsize is None:
            fig, axes = plt.subplots(nrows=len(m_info["par_names"]), figsize=(12, 2*len(m_info["par_names"])),
                                     sharex='col')
        else:
            fig, axes = plt.subplots(len(m_info["par_names"]), figsize=figsize, sharex='col')

        plt.suptitle("{m} Parameter Chains".format(m=MODEL_PUBLICATION_NAMES[model]), fontsize=14, y=1.02)

        for i in range(len(m_info["par_names"])):
            ax = axes[i]
            ax.plot(chains[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(chains))
            ax.set_ylabel(m_info["par_names"][i], fontsize=13)
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Thinned Step Number", fontsize=13)
        plt.tight_layout()
        plt.show()

    def view_corner(self, model: str, figsize: Tuple = (8, 8)):
        """
        A convenient view method to examine the corner plot of the parameter posterior distributions.

        :param str model: The name of the model for which to view the corner plot.
        :param Tuple figsize: The desired figure size.
        """
        m_info = self.get_model_fit(model)
        samples = self.get_flat_samples(model)

        frac_conf_lev = [(50 - (m_info["conf_level"] / 2))/100, 0.5, (50 + (m_info["conf_level"] / 2))/100]
        fig = corner.corner(samples, labels=m_info["par_names"], figsize=figsize, quantiles=frac_conf_lev,
                            show_titles=True)
        t = self._y_axis_name
        plt.suptitle("{m} - {s} {t} Profile - {c}% Confidence".format(m=MODEL_PUBLICATION_NAMES[model], s=self.src_name,
                                                                      t=t, c=m_info["conf_level"]), fontsize=14, y=1.02)
        plt.show()

    def add_realisation(self, real_type: str, radii: Quantity, realisation: Quantity, conf_level: int = 90):
        """
        A method to add a realisation generated by some external process (such as the density
        measurement functions).

        :param str real_type: The type of realisation being added.
        :param Quantity radii: The radii at which the realisation is generated.
        :param Quantity realisation: The values of the realisation.
        :param int conf_level: The confidence level.
        """
        if real_type not in self._allowed_real_types:
            raise ValueError("{r} is not an acceptable realisation type, this profile object currently supports"
                             " the following; {a}".format(r=real_type, a=", ".join(self._allowed_real_types)))
        elif real_type in self._realisations:
            warn("There was already a realisation of this type stored in this profile, it has been overwritten.")

        if radii.shape[0] != realisation.shape[0]:
            raise ValueError("First axis of radii and realisation arrays must be the same length.")

        # Check that the radii units are alright
        if not radii.unit.is_equivalent(self.radii_unit):
            raise UnitConversionError("The supplied radii cannot be converted to the radius unit"
                                      " of this profile ({u})".format(u=self.radii_unit.to_string()))
        else:
            radii = radii.to(self.radii_unit)

        # Check that the realisation unit are alright
        if not realisation.unit.is_equivalent(self.values_unit):
            raise UnitConversionError("The supplied realisation cannot be converted to the values unit"
                                      " of this profile ({u})".format(u=self.values_unit.to_string()))
        else:
            realisation = realisation.to(self.values_unit)

        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        # Calculates the mean model value at each radius step
        model_mean = np.mean(realisation, axis=1)
        # Then calculates the values for the upper and lower limits (defined by the
        #  confidence level) for each radii
        model_lower = np.percentile(realisation, lower, axis=1)
        model_upper = np.percentile(realisation, upper, axis=1)

        self._realisations[real_type] = {"mod_real": realisation, "mod_radii": radii, "conf_level": conf_level,
                                         "mod_real_mean": model_mean, "mod_real_lower": model_lower,
                                         "mod_real_upper": model_upper}

    def generate_data_realisations(self, num_real: int):
        """
        A method to generate random realisations of the data points in this profile, using their y-axis values
        and uncertainties. This can be useful for error propagation for instance, and does not require a model fit
        to work. This method assumes that the y-errors are 1-sigma, which isn't necessarily the case.

        :param int num_real: The number of random realisations to generate.
        :return: An N x R astropy quantity, where N is the number of realisations and R is the number of radii
            at which there are data points in this profile.
        :rtype: Quantity
        """
        if self.values_err is None:
            raise ValueError("This profile has no y-error information, and as such you cannot generate random"
                             " realisations of the data.")

        # Here I copy the values and value uncertainties N times, where N is the number of realisations
        #  the user wants
        ext_values = np.repeat(self.values[..., None], num_real, axis=1).T
        ext_value_errs = np.repeat(self.values_err[..., None], num_real, axis=1).T

        # Then I just generate N realisations of the profiles using a normal distribution, though this does assume
        #  that the errors are one sigma which isn't necessarily true
        realisations = np.random.normal(ext_values, ext_value_errs)
        realisations = Quantity(realisations, self.values_unit)

        return realisations

    def view(self, figsize=(10, 7), xscale="log", yscale="log", xlim=None, ylim=None, models=True,
             back_sub: bool = True, just_models: bool = False, custom_title: str = None, draw_rads: dict = {},
             normalise_x: bool = False, normalise_y: bool = False):
        """
        A method that allows us to view the current profile, as well as any models that have been fitted to it,
        and their residuals.

        :param Tuple figsize: The desired size of the figure, the default is (10, 7)
        :param str xscale: The scaling to be applied to the x axis, default is log.
        :param str yscale: The scaling to be applied to the y axis, default is log.
        :param Tuple xlim: The limits to be applied to the x axis, upper and lower, default is
            to let matplotlib decide by itself.
        :param Tuple ylim: The limits to be applied to the y axis, upper and lower, default is
            to let matplotlib decide by itself.
        :param str models: Should the fitted models to this profile be plotted, default is True
        :param bool back_sub: Should the plotted data be background subtracted, default is True.
        :param bool just_models: Should ONLY the fitted models be plotted? Default is False
        :param str custom_title: A plot title to replace the automatically generated title, default is None.
        :param dict draw_rads: A dictionary of extra radii (as astropy Quantities) to draw onto the plot, where
            the dictionary key they are stored under is what they will be labelled.
            e.g. ({'r500': Quantity(), 'r200': Quantity()}
        :param bool normalise_x: Should the x-axis be normalised with the x_norm value passed on the definition of
            the profile object.
        :param bool normalise_y: Should the y-axis be normalised with the x_norm value passed on the definition of
            the profile object.
        """
        # Checks that any extra radii that have been passed are the correct units (i.e. the same as the radius units
        #  used in this profile)
        if not all([r.unit == self.radii_unit for r in draw_rads.values()]):
            raise UnitConversionError("All radii in draw_rad have to be in the same units as this profile, "
                                      "{}".format(self.radii_unit.to_string()))

        # Default is to show models, but that flag is set to False here if there are none, otherwise we get
        #  extra plotted stuff that doesn't make sense
        if len(self._good_model_fits) == 0:
            models = False
            just_models = False

        # If the user wants the x-axis to be normalised then we grab the value from the profile (though of course
        #  if the user didn't set it initially then self.x_norm will also be 1
        if normalise_x:
            x_norm = self.x_norm
        else:
            # Otherwise we set x_norm to a harmless values with no units and unity value
            x_norm = Quantity(1, '')
        if normalise_y:
            y_norm = self.y_norm
        else:
            y_norm = Quantity(1, '')

        # Setting up figure for the plot
        fig = plt.figure(figsize=figsize)
        # Grabbing the axis object and making sure the ticks are set up how we want
        main_ax = plt.gca()
        main_ax.minorticks_on()
        if models:
            # This sets up an axis for the residuals to be plotted on, if model plotting is enabled
            res_ax = fig.add_axes((0.125, -0.075, 0.775, 0.2))
            res_ax.minorticks_on()
            res_ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
            # Adds a zero line for reference, as its ideally where residuals would be
            res_ax.axhline(0.0, color="black")
        # Setting some aesthetic parameters for the main plotting axis
        main_ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        if self.type == "brightness_profile" and self.psf_corrected:
            leg_label = self.src_name + " PSF Corrected"
        else:
            leg_label = self.src_name

        # This subtracts the background if the user wants a background subtracted plot
        plot_y_vals = self.values.copy()
        if back_sub:
            plot_y_vals -= self.background

        rad_vals = self.radii.copy()
        plot_y_vals /= y_norm
        rad_vals /= x_norm

        # Now the actual plotting of the data
        if self.radii_err is not None and self.values_err is None:
            x_errs = (self.radii_err.copy() / x_norm).value
            line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, xerr=x_errs, fmt="x", capsize=2,
                                    label=leg_label)
        elif self.radii_err is None and self.values_err is not None:
            y_errs = (self.values_err.copy() / y_norm).value
            line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, yerr=y_errs, fmt="x", capsize=2,
                                    label=leg_label)
        elif self.radii_err is not None and self.values_err is not None:
            x_errs = (self.radii_err.copy() / x_norm).value
            y_errs = (self.values_err.copy() / y_norm).value
            line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, xerr=x_errs, yerr=y_errs, fmt="x", capsize=2,
                                    label=leg_label)
        else:
            line = main_ax.plot(rad_vals.value, plot_y_vals.value, 'x', label=leg_label)

        if just_models and models:
            line[0].set_visible(False)
            if len(line) != 1:
                for coll in line[1:]:
                    for art_obj in coll:
                        art_obj.set_visible(False)

        if not back_sub and self.background.value != 0:
            main_ax.axhline(self.background.value, label=leg_label + ' Background', linestyle='dashed',
                            color=line[0].get_color())

        if models:
            for model in self._good_model_fits:
                model_func = PROF_TYPE_MODELS[self._prof_type][model]
                info = self.get_realisation(model)
                pars = self.get_model_fit(model)["par"]

                mod_rad = info["mod_radii"]/x_norm
                mod_val = Quantity(model_func(info["mod_radii"], *pars), self.values_unit)/y_norm
                mod_low = Quantity(info["mod_real_lower"], self.values_unit)/y_norm
                mod_upp = Quantity(info["mod_real_upper"], self.values_unit)/y_norm

                mod_line = main_ax.plot(mod_rad.value, mod_val.value, label=MODEL_PUBLICATION_NAMES[model])
                model_colour = mod_line[0].get_color()

                main_ax.fill_between(mod_rad.value, mod_low.value, mod_upp.value, alpha=0.7, interpolate=True,
                                     where=mod_upp.value >= mod_low.value, facecolor=model_colour)
                main_ax.plot(mod_rad.value, mod_low.value, color=model_colour, linestyle="dashed")
                main_ax.plot(mod_rad.value, mod_upp.value, color=model_colour, linestyle="dashed")

                # This calculates and plots the residuals between the model and the data on the extra
                #  axis we added near the beginning of this method
                res_ax.plot(rad_vals.value, model_func(self.radii.value, *pars) - (plot_y_vals*y_norm).value, 'D',
                            color=model_colour)

        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + rad_vals.unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + plot_y_vals.unit.to_string("latex").strip("$") + r"\right]$"

        # Setting the main plot's x label
        main_ax.set_xlabel("Radius {}".format(x_unit), fontsize=13)
        if self._background.value == 0 or not back_sub:
            main_ax.set_ylabel(r"{l} {u}".format(l=self._y_axis_name, u=y_unit), fontsize=13)
        else:
            # If background has been subtracted it will be mentioned in the y axis label
            main_ax.set_ylabel(r"Background Subtracted {l} {u}".format(l=self._y_axis_name, u=y_unit), fontsize=13)

        main_leg = main_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1, borderaxespad=0)
        # This makes sure legend keys are shown, even if the data is hidden
        for leg_key in main_leg.legendHandles:
            leg_key.set_visible(True)

        # If the user has manually set limits then we can use them, only on the main axis because
        #  we grab those limits from the axes object for the residual axis later
        if xlim is not None:
            main_ax.set_xlim(xlim)
        if ylim is not None:
            main_ax.set_ylim(ylim)

        # Setup the scale that the user wants to see, again on the main axis
        main_ax.set_xscale(xscale)
        main_ax.set_yscale(yscale)
        if models:
            # We want the residual x axis limits to be identical to the main axis, as the
            # points should line up
            res_ax.set_xlim(main_ax.get_xlim())
            res_ax.set_xlabel("Radius {}".format(x_unit), fontsize=13)
            res_ax.set_xscale(xscale)
            # Grabbing the automatically assigned y limits for the residual axis, then finding the maximum
            #  difference from zero, increasing it by 10%, then setting that value is the new -+ limits
            # That way its symmetrical
            outer_ylim = 1.1 * max([abs(lim) for lim in res_ax.get_ylim()])
            res_ax.set_ylim(-outer_ylim, outer_ylim)
            res_ax.set_ylabel("Model - Data", fontsize=13)

        # Adds a title to this figure, changes depending on whether model fits are plotted as well
        if models and custom_title is None and len(self._good_model_fits) == 1:
            title_str = "{l} Profile - with model".format(l=self._y_axis_name)
        elif models and custom_title is None and len(self._good_model_fits) > 1:
            title_str = "{l} Profile - with models".format(l=self._y_axis_name)
        elif not models and custom_title is None:
            title_str = "{l} Profile".format(l=self._y_axis_name)
        else:
            # If the user doesn't like my title, they can supply their own
            title_str = custom_title

        # If this particular profile is not considered usable, the user should be made aware in the plot
        if not self._usable:
            title_str += " [CONSIDERED UNUSABLE]"
        # Actually plots the title
        plt.suptitle(title_str, y=0.91)

        # Calculate the y midpoint of the main axis, which is where any extra radius labels will be placed
        main_ylims = main_ax.get_ylim()
        y_pos = main_ylims[1]*0.90
        # If the user has passed radii to plot, then we plot them
        for r_name in draw_rads:
            main_ax.axvline(draw_rads[r_name].value, linestyle='dashed', color='black')
            main_ax.text(draw_rads[r_name].value * 1.01, y_pos, r_name, rotation=90, verticalalignment='center',
                         color='black', fontsize=14)

        # Use the axis limits quite a lot in this next bit, so read them out into variables
        x_axis_lims = main_ax.get_xlim()
        y_axis_lims = main_ax.get_ylim()

        # This dynamically changes how tick labels are formatted depending on the values displayed
        if max(x_axis_lims) < 1000 and not models:
            main_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
            main_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
        elif max(x_axis_lims) < 1000 and models:
            res_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
            res_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

        if max(y_axis_lims) < 1000 and min(y_axis_lims) > 0.1:
            main_ax.yaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
            main_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
        elif max(y_axis_lims) < 1000 and min(y_axis_lims) <= 0.1:
            main_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

        # And of course actually showing it
        plt.show()

    @property
    def good_model_fits(self) -> List:
        """
        A list of the names of models that have been successfully fitted to the profile.

        :return: A list of model names.
        :rtype: Dict
        """
        return list(self._good_model_fits.keys())

    # None of these properties concerning the radii and values are going to have setters, if the user
    #  wants to modify it then they can define a new product.
    @property
    def radii(self) -> Quantity:
        """
        Getter for the radii passed in at init. These radii correspond to radii where the values were measured.

        :return: Astropy quantity array of radii.
        :rtype: Quantity
        """
        return self._radii

    @property
    def radii_err(self) -> Quantity:
        """
        Getter for the uncertainties on the profile radii.

        :return: Astropy quantity array of radii uncertainties, or a None value if no radii_err where passed.
        :rtype: Quantity
        """
        return self._radii_err

    @property
    def radii_unit(self) -> Unit:
        """
        Getter for the unit of the radii passed by the user at init.

        :return: An astropy unit object.
        :rtype: Unit
        """
        return self._radii.unit

    @property
    def annulus_bounds(self) -> Quantity:
        """
        Getter for the original boundary radii of the annuli this profile may have been generated from. Only
        available if radii errors were passed on init.

        :return: An astropy quantity containing the boundary radii of the annuli, or None if not available.
        :rtype: Quantity
        """
        return self._rad_ann_bounds

    @property
    def values(self) -> Quantity:
        """
        Getter for the values passed by user at init.

        :return: Astropy quantity array of values.
        :rtype: Quantity
        """
        return self._values

    @property
    def values_err(self) -> Quantity:
        """
        Getter for uncertainties on the profile values.

        :return: Astropy quantity array of values uncertainties, or a None value if no values_err where passed.
        :rtype: Quantity
        """
        return self._values_err

    @property
    def values_unit(self) -> Unit:
        """
        Getter for the unit of the values passed by the user at init.

        :return: An astropy unit object.
        :rtype: Unit
        """
        return self._values.unit

    @property
    def background(self) -> Quantity:
        """
        Getter for the background associated with the profile values. If no background is set this will
        be zero.

        :return: Astropy scalar quantity.
        :rtype: Quantity
        """
        return self._background

    @property
    def centre(self) -> Quantity:
        """
        Property that returns the central coordinate that the profile was generated from.

        :return: An astropy quantity of the central coordinate
        :rtype: Quantity
        """
        return self._centre

    # This definitely doesn't get a setter, as its basically a proxy for type() return, it will not change
    #  during the life of the object
    @property
    def type(self) -> str:
        """
        Getter for a string representing the type of profile stored in this object.

        :return: String description of profile.
        :rtype: str
        """
        return self._prof_type + "_profile"

    @property
    def src_name(self) -> str:
        """
        Getter for the name attribute of this profile, what source object it was derived from.

        :return:
        :rtype: object
        """
        return self._src_name

    @src_name.setter
    def src_name(self, new_name):
        """
        Setter for the name attribute of this profile, what source object it was derived from.
        """
        self._src_name = new_name

    @property
    def obs_id(self) -> str:
        """
        Property getter for the ObsID this profile was made from. Admittedly this information is implicit
        in the location this object is stored in a source object, but I think it worth storing directly
        as a property as well.

        :return: XMM ObsID string.
        :rtype: str
        """
        return self._obs_id

    @property
    def instrument(self) -> str:
        """
        Property getter for the instrument this profile was made from. Admittedly this information is implicit
        in the location this object is stored in a source object, but I think it worth storing directly
        as a property as well.

        directly as a property as well.
        :return: XMM instrument name string.
        :rtype: str
        """
        return self._inst

    @property
    def energy_bounds(self) -> Union[Tuple[Quantity, Quantity], Tuple[None, None]]:
        """
        Getter method for the energy_bounds property, which returns the rest frame energy band that this
        profile was generated from

        :return: Tuple containing the lower and upper energy limits as Astropy quantities.
        :rtype: Union[Tuple[Quantity, Quantity], Tuple[None, None]]
        """
        return self._energy_bounds

    @property
    def set_ident(self) -> int:
        """
        If this profile was generated from an annular spectrum, this will contain the set_id of that annular spectrum.

        :return: The integer set ID of the annular spectrum that generated this, or None if it wasn't generated
            from an AnnularSpectra object.
        :rtype: int
        """
        return self._set_id

    @property
    def y_axis_label(self) -> str:
        """
        Property to return the name used for labelling the y-axis in any plot generated by a profile object.

        :return: The y_axis label.
        :rtype: str
        """
        return self._y_axis_name

    @y_axis_label.setter
    def y_axis_label(self, new_name: str):
        """
        This allows the user to set a new y axis label for any plots generated by a profile object.

        :param str new_name: The new y axis label.
        """
        if not isinstance(new_name, str):
            raise TypeError("Axis labels must be strings!")
        self._y_axis_name = new_name

    @property
    def associated_set_storage_key(self) -> str:
        """
        This property provides the storage key of the associated AnnularSpectra object, if the profile was generated
        from an AnnularSpectra. If it was not then a None value is returned.

        :return: The storage key of the associated AnnularSpectra, or None if not applicable.
        :rtype: str
        """
        return self._set_storage_key

    @property
    def deg_radii(self) -> Quantity:
        """
        The radii in degrees if available.

        :return: An astropy quantity containing the radii in degrees, or None.
        :rtype: Quantity
        """
        return self._deg_radii

    @property
    def storage_key(self) -> str:
        """
        This property returns the storage key which this object assembles to place the profile in
        an XGA source's storage structure. If the profile was generated from an AnnularSpectra then the key is based
        on the properties of the AnnularSpectra, otherwise it is based upon the properties of the specific profile.

        :return: String storage key.
        :rtype: str
        """
        return self._storage_key

    @property
    def usable(self) -> bool:
        """
        Whether the profile object can be considered usable or not, reasons for this decision will vary for
        different profile types.

        :return: A boolean variable.
        :rtype: bool
        """
        return self._usable

    @property
    def x_norm(self) -> Quantity:
        """
        The normalisation value for x-axis data passed on the definition of the this profile object.

        :return: An astropy quantity containing the normalisation value.
        :rtype: Quantity
        """
        return self._x_norm

    @x_norm.setter
    def x_norm(self, new_val: Quantity):
        """
        The setter for the normalisation value for x-axis data passed on the definition of the this profile object.
        :param Quantity new_val: The new value for the normalisation of x-axis data.
        """
        self._x_norm = new_val

    @property
    def y_norm(self) -> Quantity:
        """
        The normalisation value for y-axis data passed on the definition of the this profile object.

        :return: An astropy quantity containing the normalisation value.
        :rtype: Quantity
        """
        return self._y_norm

    @y_norm.setter
    def y_norm(self, new_val: Quantity):
        """
        The setter for the normalisation value for y-axis data passed on the definition of the this profile object.
        :param Quantity new_val: The new value for the normalisation of y-axis data.
        """
        self._y_norm = new_val

    def __len__(self):
        """
        The length of a BaseProfile1D object is equal to the length of the radii and values arrays
        passed in on init.

        :return: The number of bins in this radial profile.
        """
        return len(self._radii)

    def __add__(self, other):
        to_combine = [self]
        if type(other) == list:
            to_combine += other
        elif isinstance(other, BaseProfile1D):
            to_combine.append(other)
        elif isinstance(other, BaseAggregateProfile1D):
            to_combine += other.profiles
        else:
            raise TypeError("You may only add 1D Profiles, 1D Aggregate Profiles, or a list of 1D profiles"
                            " to this object.")
        return BaseAggregateProfile1D(to_combine)


class BaseAggregateProfile1D:
    """
    Quite a simple class that is generated when multiple 1D radial profile objects are added together. The
    purpose of instances of this class is simply to make it easy to view 1D radial profiles on the same axes.
    """
    def __init__(self, profiles: List[BaseProfile1D]):
        # This checks that all types of profiles in the profiles list are the same
        types = [type(p) for p in profiles]
        if len(set(types)) != 1:
            raise TypeError("All component profiles must be of the same type")

        # This checks that all profiles have the same x units
        x_units = [p.radii_unit for p in profiles]
        if len(set(x_units)) != 1:
            raise TypeError("All component profiles must have the same radii units.")

        # THis checks that they all have the same y units. This is likely to be true if they are the same
        #  type, but you never know
        y_units = [p.values_unit for p in profiles]
        if len(set(y_units)) != 1:
            raise TypeError("All component profiles must have the same value units.")

        # We check to see if all profiles either have a background, or not
        backs = [p.background.value != 0 for p in profiles]
        if len(set(backs)) != 1:
            raise ValueError("All component profiles must have a background, or not have a "
                             "background. You cannot profiles that do to profiles that don't.")
        elif backs[0]:
            # An attribute to tell us whether backgrounds are present in the component profiles
            self._back_avail = True
        else:
            self._back_avail = False

        # Here we check that all energy bounds are the same
        bounds = [p.energy_bounds for p in profiles]
        if len(set(bounds)) != 1:
            raise ValueError("All component profiles must have been generate from the same energy range,"
                             " otherwise they aren't directly comparable.")

        self._profiles = profiles
        self._radii_unit = x_units[0]
        self._values_unit = y_units[0]
        # Not doing a check that all the prof types are the same, because that should be included in the
        #  type check on the first line of this init
        self._prof_type = profiles[0].type.split("_profile")[0]
        self._energy_bounds = bounds[0]

        # We set the y-axis name attribute which is now expected by the plotting function, just grab it from the
        #  first component because we've already checked that they're all the same type
        self._y_axis_name = self._profiles[0].y_axis_label

    @property
    def radii_unit(self) -> Unit:
        """
        Getter for the unit of the radii passed by the user at init.

        :return: An astropy unit object.
        :rtype: Unit
        """
        return self._radii_unit

    @property
    def values_unit(self) -> Unit:
        """
        Getter for the unit of the values passed by the user at init.

        :return: An astropy unit object.
        :rtype: Unit
        """
        return self._values_unit

    @property
    def type(self) -> str:
        """
        Getter for a string representing the type of profile stored in this object.

        :return: String description of profile.
        :rtype: str
        """
        return self._prof_type

    @property
    def profiles(self) -> List[BaseProfile1D]:
        """
        This property is for the constituent profiles that makes up this aggregate profile.

        :return: A list of the profiles that make up this object.
        :rtype: List[BaseProfile1D]
        """
        return self._profiles

    @property
    def energy_bounds(self) -> Union[Tuple[Quantity, Quantity], Tuple[None, None]]:
        """
        Getter method for the energy_bounds property, which returns the rest frame energy band that
        the component profiles of this object were generated from.

        :return: Tuple containing the lower and upper energy limits as Astropy quantities.
        :rtype: Union[Tuple[Quantity, Quantity], Tuple[None, None]]
        """
        return self._energy_bounds

    def view(self, figsize: Tuple = (10, 7), xscale: str = "log", yscale: str = "log", xlim: Tuple = None,
             ylim: Tuple = None, model: str = None, back_sub: bool = True, legend: bool = True,
             just_model: bool = False, custom_title: str = None, draw_rads: dict = {}):
        """
        A method that allows us to see all the profiles that make up this aggregate profile, plotted
        on the same figure.

        :param Tuple figsize: The desired size of the figure, the default is (10, 7)
        :param str xscale: The scaling to be applied to the x axis, default is log.
        :param str yscale: The scaling to be applied to the y axis, default is log.
        :param Tuple xlim: The limits to be applied to the x axis, upper and lower, default is
            to let matplotlib decide by itself.
        :param Tuple ylim: The limits to be applied to the y axis, upper and lower, default is
            to let matplotlib decide by itself.
        :param str model: The name of the model fit to display, default is None. If the model
            hasn't been fitted, or it failed, then it won't be displayed.
        :param bool back_sub: Should the plotted data be background subtracted, default is True.
        :param bool legend: Should a legend with source names be added to the figure, default is True.
        :param bool just_model: Should only the models, not the data, be plotted. Default is False.
        :param str custom_title: A plot title to replace the automatically generated title, default is None.
        :param dict draw_rads: A dictionary of extra radii (as astropy Quantities) to draw onto the plot, where
            the dictionary key they are stored under is what they will be labelled.
            e.g. ({'r500': Quantity(), 'r200': Quantity()}
        """

        # Checks that any extra radii that have been passed are the correct units (i.e. the same as the radius units
        #  used in this profile)
        if not all([r.unit == self.radii_unit for r in draw_rads.values()]):
            raise UnitConversionError("All radii in draw_rad have to be in the same units as this profile, "
                                      "{}".format(self.radii_unit.to_string()))

        # Setting up figure for the plot
        fig = plt.figure(figsize=figsize)
        # Grabbing the axis object and making sure the ticks are set up how we want
        main_ax = plt.gca()
        main_ax.minorticks_on()
        if model is not None:
            # This sets up an axis for the residuals to be plotted on, if model plotting is enabled
            res_ax = fig.add_axes((0.125, -0.075, 0.775, 0.2))
            res_ax.minorticks_on()
            res_ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
            # Adds a zero line for reference, as its ideally where residuals would be
            res_ax.axhline(0.0, color="black")
        # Setting some aesthetic parameters for the main plotting axis
        main_ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        # Cycles through the component profiles of this aggregate profile, plotting them all
        for p in self._profiles:
            if p.type == "brightness_profile" and p.psf_corrected:
                leg_label = p.src_name + " PSF Corrected"
            else:
                leg_label = p.src_name

            # This subtracts the background if the user wants a background subtracted plot
            sub_values = p.values.value.copy()
            if back_sub:
                sub_values -= p.background.value

            # Now the actual plotting of the data
            if p.radii_err is not None and p.values_err is None:
                line = main_ax.errorbar(p.radii.value, sub_values, xerr=p.radii_err.value, fmt="x", capsize=2,
                                        label=leg_label)
            elif p.radii_err is None and p.values_err is not None:
                line = main_ax.errorbar(p.radii.value, sub_values, yerr=p.values_err.value, fmt="x", capsize=2,
                                        label=leg_label)
            elif p.radii_err is not None and p.values_err is not None:
                line = main_ax.errorbar(p.radii.value, sub_values, xerr=p.radii_err.value, yerr=p.values_err.value,
                                        fmt="x", capsize=2, label=leg_label)
            else:
                line = main_ax.plot(p.radii.value, sub_values, 'x', label=leg_label)

            # If the user only wants the models to be plotted, then this goes through the matplotlib
            #  artist objects that make up the line plot and hides them.
            # Take this approach because I still want them on the legend, and I want the colour to use
            #  for the model plot
            if just_model and model is not None:
                line[0].set_visible(False)
                if len(line) != 1:
                    for coll in line[1:]:
                        for art_obj in coll:
                            art_obj.set_visible(False)

            if not back_sub and p.background.value != 0:
                main_ax.axhline(p.background.value, label=leg_label + ' Background', linestyle='dashed',
                                color=line[0].get_color())

            # If the user passes a model name, and that model has been fitted to the data, then that
            #  model will be plotted
            if model is not None and model in p.good_model_fits:
                model_func = PROF_TYPE_MODELS[self._prof_type][model]
                info = p.get_realisation(model)
                pars = p.get_model_fit(model)["par"]

                colour = line[0].get_color()
                main_ax.plot(info["mod_radii"], model_func(info["mod_radii"], *pars), color=colour)
                main_ax.fill_between(info["mod_radii"], info["mod_real_lower"], info["mod_real_upper"],
                                     where=info["mod_real_upper"] >= info["mod_real_lower"], facecolor=colour,
                                     alpha=0.7, interpolate=True)
                main_ax.plot(info["mod_radii"], info["mod_real_lower"], color=colour, linestyle="dashed")
                main_ax.plot(info["mod_radii"], info["mod_real_upper"], color=colour, linestyle="dashed")

                # This calculates and plots the residuals between the model and the data on the extra
                #  axis we added near the beginning of this method
                res_ax.plot(p.radii.value, model_func(p.radii.value, *pars)-sub_values, 'D', color=colour)

        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + self.radii_unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + self.values_unit.to_string("latex").strip("$") + r"\right]$"

        # Setting the main plot's x label
        main_ax.set_xlabel("Radius {}".format(x_unit), fontsize=13)
        if not self._back_avail or not back_sub:
            main_ax.set_ylabel(r"{l} {u}".format(l=self._y_axis_name, u=y_unit), fontsize=13)
        else:
            # If background has been subtracted it will be mentioned in the y axis label
            main_ax.set_ylabel(r"Background Subtracted {l} {u}".format(l=self._y_axis_name, u=y_unit), fontsize=13)

        # Adds a legend with source names to the side if the user requested it
        # I let the user decide because there could be quite a few names in it and it could get messy
        if legend:
            # TODO I'd like this to dynamically choose the number of columns depending on the number of
            #  profiles but I got bored figuring how to do it
            main_leg = main_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1, borderaxespad=0)
            # This makes sure legend keys are shown, even if the data is hidden
            for leg_key in main_leg.legendHandles:
                leg_key.set_visible(True)

        # If the user has manually set limits then we can use them, only on the main axis because
        #  we grab those limits from the axes object for the residual axis later
        if xlim is not None:
            main_ax.set_xlim(xlim)
        if ylim is not None:
            main_ax.set_ylim(ylim)

        # Setup the scale that the user wants to see, again on the main axis
        main_ax.set_xscale(xscale)
        main_ax.set_yscale(yscale)
        if model is not None:
            # We want the residual x axis limits to be identical to the main axis, as the
            # points should line up
            res_ax.set_xlim(main_ax.get_xlim())
            res_ax.set_xlabel("Radius {}".format(x_unit), fontsize=13)
            res_ax.set_xscale(xscale)
            # Grabbing the automatically assigned y limits for the residual axis, then finding the maximum
            #  difference from zero, increasing it by 10%, then setting that value is the new -+ limits
            # That way its symmetrical
            outer_ylim = 1.1*max([abs(lim) for lim in res_ax.get_ylim()])
            res_ax.set_ylim(-outer_ylim, outer_ylim)
            res_ax.set_ylabel("Model - Data")

        # Adds a title to this figure, changes depending on whether model fits are plotted as well
        if model is not None and custom_title is None:
            plt.suptitle("{l} Profiles - {m} fit".format(l=self._y_axis_name,
                                                         m=MODEL_PUBLICATION_NAMES[model]), y=0.91)
        elif model is None and custom_title is None:
            plt.suptitle("{l} Profiles".format(l=self._y_axis_name), y=0.91)
        else:
            # If the user doesn't like my title, they can supply their own
            plt.suptitle(custom_title, y=0.91)

        # Calculate the y midpoint of the main axis, which is where any extra radius labels will be placed
        main_ylims = main_ax.get_ylim()
        y_mid = (main_ylims[1] - main_ylims[0]) / 2
        # If the user has passed radii to plot, then we plot them
        for r_name in draw_rads:
            main_ax.axvline(draw_rads[r_name].value, linestyle='dashed', color='black')
            main_ax.text(draw_rads[r_name].value * 1.01, y_mid, r_name, rotation=90, verticalalignment='center',
                         color='black', fontsize=14)

        # And of course actually showing it
        plt.show()

    def __add__(self, other):
        to_combine = self.profiles
        if type(other) == list:
            to_combine += other
        elif isinstance(other, BaseProfile1D):
            to_combine.append(other)
        elif isinstance(other, BaseAggregateProfile1D):
            to_combine += other.profiles
        else:
            raise TypeError("You may only add 1D Profiles, 1D Aggregate Profiles, or a list of 1D profiles"
                            " to this object.")
        return BaseAggregateProfile1D(to_combine)














