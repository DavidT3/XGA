#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 07/10/2020, 10:57. Copyright (c) David J Turner


import inspect
import os
from typing import Tuple, List, Dict
from warnings import warn

import numpy as np
from astropy.units import Quantity, UnitConversionError, Unit
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from ..exceptions import SASGenerationError, UnknownCommandlineError, XGAFitError, XGAInvalidModelError
from ..models import SB_MODELS, SB_MODELS_STARTS, DENS_MODELS, DENS_MODELS_STARTS, TEMP_MODELS, TEMP_MODELS_STARTS
from ..utils import SASERROR_LIST, SASWARNING_LIST

PROF_TYPE_YAXIS = {"base": "Unknown", "brightness": "Surface Brightness", "density": "Density",
                   "2d_temperature": "Projected Temperature", "3d_temperature": "3D Temperature"}
PROF_TYPE_MODELS = {"brightness": SB_MODELS, "density": DENS_MODELS, "2d_temperature": TEMP_MODELS,
                    "3d_temperature": TEMP_MODELS}
PROF_TYPE_MODELS_STARTS = {"brightness": SB_MODELS_STARTS, "density": DENS_MODELS_STARTS,
                           "2d_temperature": TEMP_MODELS_STARTS, "3d_temperature": TEMP_MODELS_STARTS}


class BaseProduct:
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, raise_properly: bool = True):
        """
        The initialisation method for the BaseProduct class.
        :param str path: The path to where the product file SHOULD be located.
        :param str stdout_str: The stdout from calling the terminal command.
        :param str stderr_str: The stderr from calling the terminal command.
        :param str gen_cmd: The command used to generate the product.
        :param bool raise_properly: Shall we actually raise the errors as Python errors?
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
        self.og_cmd = gen_cmd
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
                        sas_err_match = [sas_err for sas_err in SASERROR_LIST if err_ident.lower() in sas_err.lower()]
                    elif err_type == "warning":
                        # Checking to see if the error identity is in the list of SAS warnings
                        sas_err_match = [sas_err for sas_err in SASWARNING_LIST if err_ident.lower() in sas_err.lower()]

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
                               and line not in sas_warn_lines and line != ""]

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
    def errors(self) -> List[dict]:
        """
        Property getter for non-SAS errors detected during the generation of a product.
        :return: A list of dictionaries of parsed errors.
        :rtype: List[dict]
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


# TODO Obviously finish this, but also comment and docstring
class BaseAggregateProduct:
    def __init__(self, file_paths: list, prod_type: str, obs_id: str, instrument: str):
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
    def all_usable(self) -> bool:
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
        Equivelant to the BaseProduct sas_errors property, but reports any errors stored in the component products.
        :return: A list of SAS errors related to component products.
        :rtype: List
        """
        sas_err_list = []
        for p in self._component_products:
            prod = self._component_products[p]
            sas_err_list += prod.sas_errors
        return sas_err_list

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


# TODO Sweep through and docstring up in here
class BaseProfile1D:
    def __init__(self, radii: Quantity, values: Quantity, source_name: str, obs_id: str,
                 inst: str, radii_err: Quantity = None, values_err: Quantity = None):
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

        # Storing the key values in attributes
        self._radii = radii
        self._values = values
        self._radii_err = radii_err
        self._values_err = values_err

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

        # Some types of profiles will support a background value (like surface brightness), which will
        #  need to be incorporated into the fit and plotting.
        self._background = Quantity(0, self._values.unit)

    def fit(self, model: str, method: str = "mcmc", start_pars=None, model_real=1000, model_rad_steps=300,
            conf_level=90):
        # These are the currently allowed fitting methods
        method = method.lower()
        fit_methods = ["curve_fit", "mcmc"]
        # Checking that the user hasn't chosen a method that isn't allowed
        if method not in fit_methods:
            raise ValueError("{0} is not an accepted fitting method, please choose one of these; "
                             "{1}".format(method, ", ".join(fit_methods)))

        # Stopping the user from making stupid model choices
        if self._prof_type == "base":
            raise XGAFitError("A BaseProfile1D object currently cannot have a model fitted to it, as there"
                              " is no physical context.")
        elif model not in PROF_TYPE_MODELS[self._prof_type]:
            allowed = list(PROF_TYPE_MODELS[self._prof_type].keys())
            prof_name = PROF_TYPE_YAXIS[self._prof_type].lower()
            raise XGAInvalidModelError("{m} is not a valid model for a {p} profile, please choose from "
                                       "one of these; {a}".format(m=model, a=", ".join(allowed), p=prof_name))
        else:
            model_func = PROF_TYPE_MODELS[self._prof_type][model]

        # This inspect module lets me grab the parameters expected by the model dynamically, and check
        #  what the user might have passed in the start_pars variable against it
        model_sig = inspect.signature(model_func)
        # Ignore the first argument, as it will be radius
        model_par_names = [p.name for p in list(model_sig.parameters.values())[1:]]
        if start_pars is not None and len(start_pars) != len(model_par_names):
            raise ValueError("start_pars must either be None, or have an entry for each parameter expected by"
                             " the chosen model; {0} expects {1}".format(model, ", ".join(model_par_names)))
        elif start_pars is None:
            # If the user doesn't supply and starting parameters then we just have to use the default ones
            start_pars = PROF_TYPE_MODELS_STARTS[self._prof_type][model]

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
            try:
                fit_par, fit_cov = curve_fit(model_func, self._radii.value, self.values.value - self._background.value,
                                             p0=start_pars, sigma=self._values_err.value)
                # Grab the diagonal of the covariance matrix, then sqrt to get sigma values for each parameter
                fit_par_err = np.sqrt(np.diagonal(fit_cov))
            except RuntimeError:
                success = False
                fit_par = np.full(len(start_pars), np.nan)
                fit_par_err = np.full(len(start_pars), np.nan)
        elif method == "mcmc" and not already_done:
            success = False
            raise NotImplementedError("Haven't added MCMC fitting yet sozzle")

        # Now do some checks after the fit has run, primarily for any infinite values
        # TODO Possibly change this depending on the implementation of the MCMC fit
        if not already_done and ((np.inf in fit_par or np.inf in fit_par_err)
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

            # Changes confidence level to expected input for numpy percentile function
            upper = 50 + (conf_level / 2)
            lower = 50 - (conf_level / 2)

            # Calculates the mean model value at each radius step
            model_mean = np.mean(model_realisations, axis=1)
            # Then calculates the values for the upper and lower limits (defined by the
            #  confidence level) for each radii
            model_lower = np.percentile(model_realisations, lower, axis=1)
            model_upper = np.percentile(model_realisations, upper, axis=1)

            # Store these realisations for statistics later on
            self._good_model_fits[model] = {"par": fit_par, "par_err": fit_par_err, "start_pars": start_pars,
                                            "mod_real": model_realisations, "mod_radii": model_radii,
                                            "conf_level": conf_level, "mod_real_mean": model_mean,
                                            "mod_real_lower": model_lower, "mod_real_upper": model_upper}

        elif not already_done and success and method == "mcmc":
            raise NotImplementedError('HOW DID YOU GET HERE')
        elif not already_done and not success:
            self._bad_model_fits[model] = {"start_pars": start_pars}

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
            the_line = "|" + " " * np.ceil((len(first_col) - len(to_print[model][0])) / 2).astype(int) + \
                       to_print[model][0] + " " * np.ceil((len(first_col) - len(to_print[model][0])) / 2).astype(int) \
                       + "|"

            the_line += " "*np.ceil((len(second_col)-len(to_print[model][1])) / 2).astype(int) + to_print[model][1] + \
                        " "*np.ceil((len(second_col)-len(to_print[model][1])) / 2).astype(int) + "|"

            the_line += " " * np.ceil((len(third_col) - len(to_print[model][2])) / 2).astype(int) + to_print[model][
                2] + " " * np.ceil((len(third_col) - len(to_print[model][2])) / 2).astype(int) + "|"
            print(the_line)
        print("-" * len(comb) + "\n")

    def view(self, figsize=(8, 5), xscale="log", yscale="log", xlim=None, ylim=None, models=True):
        # Setting up figure for the plot
        plt.figure(figsize=figsize)

        # Grabbing the axis object and making sure the ticks are set up how we want
        ax = plt.gca()
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        # Taking off any background
        sub_values = self.values.value - self.background.value
        if self._radii_err is not None and self._values_err is None:
            plt.errorbar(self.radii.value, sub_values, xerr=self.radii_err.value, label="Data", fmt="x",
                         capsize=2)
        elif self._radii_err is None and self._values_err is not None:
            plt.errorbar(self.radii.value, sub_values, yerr=self.values_err.value, label="Data", fmt="x",
                         capsize=2)
        elif self._radii_err is not None and self._values_err is not None:
            plt.errorbar(self.radii.value, sub_values, xerr=self.radii_err.value, yerr=self.values_err.value,
                         label="Data", fmt="x", capsize=2)
        else:
            plt.plot(self.radii.value, sub_values, 'x', label="Data")

        # Setup the scale that the user wants to see
        plt.xscale(xscale)
        plt.yscale(yscale)

        # If the user has manually set limits then we can use them
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # If models have been fitted to this profile (and the user wants them plotted), then this runs through
        #  and adds them to the figure
        if models:
            for model in self._good_model_fits:
                model_func = PROF_TYPE_MODELS[self._prof_type][model]
                info = self._good_model_fits[model]
                line = plt.plot(info["mod_radii"], model_func(info["mod_radii"], *info["par"]),
                                label=model + " {}% Conf".format(info["conf_level"]))
                colour = line[0].get_color()
                plt.fill_between(info["mod_radii"], info["mod_real_lower"], info["mod_real_upper"],
                                 where=info["mod_real_upper"] >= info["mod_real_lower"], facecolor=colour,
                                 alpha=0.7, interpolate=True)
                plt.plot(info["mod_radii"], info["mod_real_lower"], color=colour, linestyle="dashed")
                plt.plot(info["mod_radii"], info["mod_real_upper"], color=colour, linestyle="dashed")

        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + self.radii_unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + self.values_unit.to_string("latex").strip("$") + r"\right]$"

        # Adding them to the figure
        plt.xlabel("Radius {}".format(x_unit))
        plt.ylabel(r"{l} {u}".format(l=PROF_TYPE_YAXIS[self._prof_type], u=y_unit))

        if self._obs_id == "combined":
            plt.title("{s} {l} Profile".format(s=self._src_name, l=PROF_TYPE_YAXIS[self._prof_type]))
        else:
            plt.title("{s}-{o}-{i} {l} Profile".format(s=self._src_name, l=PROF_TYPE_YAXIS[self._prof_type],
                                                       o=self.obs_id, i=self.instrument))

        # Just going to leave matplotlib to decide where the legend should live
        plt.legend(loc="best")

        # And of course actually showing it
        plt.show()

    # None of these properties concerning the radii and values are going to have setters, if the user wants to modify
    #  it then they can define a new product.
    @property
    def radii(self) -> Quantity:
        """
        Getter for the radii passed in at init. These radii correspond to radii where the values were measured
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

    # This definitely doesn't get a setter, as its basically a proxy for type() return, it will not change
    #  during the life of the object
    @property
    def prof_type(self) -> str:
        """
        Getter for a string representing the type of profile stored in this object.
        :return: String description of profile.
        :rtype: str
        """
        return self._prof_type

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

    def __len__(self):
        """
        The length of a BaseProfile1D object is equal to the length of the radii and values arrays
        passed in on init.
        :return: The number of bins in this radial profile.
        """
        return len(self._radii)
















