#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 26/08/2025, 19:01. Copyright (c) The Contributors

import inspect
import os
import pickle
from copy import deepcopy
from random import randint
from typing import Tuple, List, Dict, Union
from warnings import warn

import corner
import emcee as em
import numpy as np
from astropy.units import Quantity, UnitConversionError, Unit, deg
from getdist import plots, MCSamples
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit, minimize
from scipy.stats import truncnorm
from tabulate import tabulate

from ..exceptions import SASGenerationError, UnknownCommandlineError, XGAFitError, XGAInvalidModelError, \
    ModelNotAssociatedError
from ..models import PROF_TYPE_MODELS, BaseModel1D, MODEL_PUBLICATION_NAMES
from ..models.fitting import log_likelihood, log_prob
from ..utils import SASERROR_LIST, SASWARNING_LIST, OUTPUT


class BaseProduct:
    """
    The super class for all X-ray products in XGA. Stores relevant file path information, parses the std_err
    output of the generation process, and stores the instrument and ObsID that the product was generated for.

    :param str path: The path to where the product file SHOULD be located.
    :param str obs_id: The ObsID related to the product being declared.
    :param str instrument: The instrument related to the product being declared.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The command used to generate the product.
    :param dict extra_info: This allows the XGA processing steps to store some temporary extra information in this
        object for later use by another processing step. It isn't intended for use by a user and will only be
        accessible when defining a BaseProduct.
    :param bool force_remote: Used to force the product instantiation to treat the passed path string as a url to
            a remote dataset, and to use fsspec to read/stream the data.
    :param dict fsspec_kwargs: Optional arguments that can be passed fsspec when reading or streaming remote
        datasets - e.g. to pass credentials to access an S3 bucket. Default value is None, which sets the
        argument to {"anon": True}, making it instantly compatible with NASA archive S3 buckets.
    """
    def __init__(self, path: str, obs_id: str = "", instrument: str = "", stdout_str: str = "", stderr_str: str = "",
                 gen_cmd: str = "", extra_info: dict = None, force_remote: bool = False, fsspec_kwargs: dict = None):
        """
        The initialisation method for the BaseProduct class, the super class for all SAS generated products in XGA.
        Stores relevant file path information, parses the std_err output of the generation process, and stores the
        instrument and ObsID that the product was generated for.

        :param str path: The path to where the product file SHOULD be located.
        :param str obs_id: The ObsID related to the product being declared.
        :param str instrument: The instrument related to the product being declared.
        :param str stdout_str: The stdout from calling the terminal command.
        :param str stderr_str: The stderr from calling the terminal command.
        :param str gen_cmd: The command used to generate the product.
        :param dict extra_info: This allows the XGA processing steps to store some temporary extra information in this
            object for later use by another processing step. It isn't intended for use by a user and will only be
            accessible when defining a BaseProduct.
        :param bool force_remote: Used to force the product instantiation to treat the passed path string as a url to
            a remote dataset, and to use fsspec to read/stream the data.
        :param dict fsspec_kwargs: Optional arguments that can be passed fsspec when reading or streaming remote
            datasets - e.g. to pass credentials to access an S3 bucket. Default value is None, which sets the
            argument to {"anon": True}, making it instantly compatible with NASA archive S3 buckets.
        """

        # Here we try to identify if the file path that has been passed is local or remote, as it will change how we
        #  interact with it in the various product sub-classes
        if force_remote:
            # Here the user has forced us to treat the path as remote
            self._local_file = False
        elif path[:5] == "s3://" or path[:5] == "gs://":
            # Here we assume that the file is remote because it starts with the s3/gs identifier - this is for
            #  use with resources like the HEASARC open S3 bucket
            self._local_file = False
        else:
            # Otherwise we decide that the file is local
            self._local_file = True

        # Keep track of whether the user forced the path to be considered as a remote url or not, that information
        #  may be required in some warning/error messages later on
        self._force_remote = force_remote

        # We replace the default fsspec_kwargs value (None) with a dictionary indicating that no credentials are
        #  required to access the remote URL, which makes it instantly compatible with NASA archive S3 buckets.
        if fsspec_kwargs is None:
            fsspec_kwargs = {"anon": True}
        # We store the optional keyword arguments that the user can pass to facilitate access to
        #  remote files in an attribute
        self._fsspec_kwargs = fsspec_kwargs

        # This attribute stores strings that indicate why a product object has been deemed as unusable
        self._why_unusable = []

        # This flag indicates whether we think this data product can be used for analysis - it can be set to False
        #  for different reasons, but the most important is that the file cannot be found
        self._usable = True

        # Try to determine if the file exists - this will not currently check remote files
        if self._local_file and os.path.exists(path):
            self._path = path
        elif self._local_file:
            self._path = None
            self._usable = False
            self._why_unusable.append("ProductPathDoesNotExist")
        else:
            self._path = path

        # Turning null stderr and stdouts (for instance, if the product is just being loaded in, as opposed to it
        #  being generated by XGA and needing to be checked for process success) into empty strings
        if stdout_str is None:
            stdout_str = ""
        if stderr_str is None:
            stderr_str = ""
        # Keeping them in class attributes
        self.unprocessed_stdout = stdout_str
        self.unprocessed_stderr = stderr_str
        # Running checks on the stdout/err strings, looking for any obvious errors
        self._sas_error, self._sas_warn, self._other_error = self.parse_stderr()

        # Storing more of the input information in attributes
        self._obs_id = obs_id
        self._inst = instrument

        # Replacing a null input for the generation command with an empty string
        if gen_cmd is None:
            gen_cmd = ""
        self._og_cmd = gen_cmd

        self._energy_bounds = (None, None)
        self._prod_type = None
        self._src_name = None

        # Any extra information which a processing step might want to store in this base product - generally only
        #  used when a product has been generated that doesn't need its only product class, but is being put in
        #  a product class to be returned from 'execute_cmd'. I'm not even going to give it a property to hopefully
        #  highlight to the user that it shouldn't be accessed by them.
        self._extra_info = extra_info

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

    @property
    def local_file(self) -> bool:
        """
         A file is deemed remote by the presence of certain strings at the beginning of the path, or the
         user passing 'force_remote=True' at product initialization, otherwise it is considered to be local.

        :return: Returns a boolean flag describing if we think this product is pointed at a local file (True) or
            a remote file (False).
        :rtype: bool
        """
        return self._local_file

    @property
    def force_remote(self) -> bool:
        """
        A property providing the value of the 'force_remote' argument passed to this product at instantiation - that
        value controls how the init treats the file path.

        :return: The value of 'force_remote' argument passed to this product at instantiation.
        :rtype: bool
        """
        return self._force_remote

    @property
    def fsspec_kwargs(self) -> Union[dict, None]:
        """
        Property getter for the attribute containing the fsspec keyword arguments passed to this
        product at instantiation. These are for passing configuration information such as credentials for
        the remote access of S3 buckets

        :return: The fsspec keyword arguments passed to this product at instantiation.
        :rtype: dict
        """
        return self._fsspec_kwargs

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
    def telescope(self) -> str:
        """
        Property getter for the name of the telescope that this product was derived from.

        :return: The telescope name.
        :rtype: str
        """
        return self._tele

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
    A base class for any XGA products that are an aggregate of an XGA SAS product, for instance this is sub-classed
    to make the AnnularSpectra class. Users really shouldn't be instantiating these for themselves.

    :param list file_paths: The file paths of the main files for a given aggregate product.
    :param str prod_type: The product type of the individual elements.
    :param str obs_id: The ObsID related to the product.
    :param str instrument: The instrument related to the product.
    """
    def __init__(self, file_paths: list, prod_type: str, obs_id: str, instrument: str):
        """
        The init method for the BaseAggregateProduct class
        """
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
        aware of this once it is added to a source object. This is overridden in the AnnularSpectra class.

        :return: The name of the source object this product is associated with.
        :rtype: str
        """
        return self._src_name

    # This needs a setter, as this property only becomes not-None when the product is added to a source object.
    @src_name.setter
    def src_name(self, name: str):
        """
        Property setter for the src_name attribute of a product, should only really be called by a source object,
        not by a user. This is overridden in the AnnularSpectra class.

        :param str name: The name of the source object associated with this product.
        """
        self._src_name = name
        for p in self._component_products.values():
            p.src_name = name

    @property
    def obs_id(self) -> str:
        """
        Property getter for the ObsID of this AggregateProduct. Admittedly this information is implicit in the location
        this object is stored in a source object, but I think it worth storing directly as a property as well.

        :return: The ObsID of this AggregateProduct.
        :rtype: str
        """
        return self._obs_id

    @property
    def instrument(self) -> str:
        """
        Property getter for the instrument of this AggregateProduct. Admittedly this information is implicit
        in the location this object is stored in a source object, but I think it worth storing
        directly as a property as well.

        :return: The ObsID of this AggregateProduct.
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
        associated AnnularSpectra generates to place itself in XGA's storage structure.
    :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
        units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
        values converted to degrees, and allows this object to construct a predictable storage key.
    :param Quantity x_norm: An astropy quantity to use to normalise the x-axis values, this is only used when
        plotting if the user tells the view method that they wish for the plot to use normalised x-axis data.
    :param Quantity y_norm: An astropy quantity to use to normalise the y-axis values, this is only used when
        plotting if the user tells the view method that they wish for the plot to use normalised y-axis data.
    :param bool auto_save: Whether the profile should automatically save itself to disk at any point. The default is
        False, but all profiles generated through XGA processes acting on XGA sources will auto-save.
    :param str spec_model: The spectral model that was fit to annular spectra to measure the results that were
        used to create this profile. Only relevant to profiles that are generated from annular spectra, default
        is None.
    :param str fit_conf: The key that describes the fit-configuration used when fitting models to annular
        spectra to measure the results that were then used to create this profile. Only relevant to profiles that
        are generated from annular spectra, default is None.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None, x_norm: Quantity = Quantity(1, ''),
                 y_norm: Quantity = Quantity(1, ''), auto_save: bool = False, spec_model: str = None,
                 fit_conf: str = None):
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
            associated AnnularSpectra generates to place itself in XGA's storage structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        :param Quantity x_norm: An astropy quantity to use to normalise the x-axis values, this is only used when
            plotting if the user tells the view method that they wish for the plot to use normalised x-axis data.
        :param Quantity y_norm: An astropy quantity to use to normalise the y-axis values, this is only used when
            plotting if the user tells the view method that they wish for the plot to use normalised y-axis data.
        :param bool auto_save: Whether the profile should automatically save itself to disk at any point. The default
            is False, but all profiles generated through XGA processes acting on XGA sources will auto-save.
        :param str spec_model: The spectral model that was fit to annular spectra to measure the results that were
            used to create this profile. Only relevant to profiles that are generated from annular spectra, default
            is None.
        :param str fit_conf: The key that describes the fit-configuration used when fitting models to annular
            spectra to measure the results that were then used to create this profile. Only relevant to profiles that
            are generated from annular spectra, default is None.
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
            raise ValueError("If the 'radii' variable is not in units that are convertible to degrees, please pass "
                             "radii in degrees to 'deg_radii', this profile needs knowledge of the radii in degrees"
                             " to construct a storage key.")
        elif not radii.unit.is_equivalent('deg') and set_storage_key is None and len(deg_radii) != len(radii):
            raise ValueError("'deg_radii' is a different length to 'radii', they should be equivalent "
                             "quantities, simply in different units.")
        elif radii.unit.is_equivalent('deg') and set_storage_key is None:
            deg_radii = radii.to('deg')

        if deg_radii is not None:
            deg_radii = deg_radii.to("deg")
            self._deg_radii = deg_radii

        # Finally I'm going to make a check for infinite or NaN values in the passed value,
        #  radius, and uncertainty quantities
        if np.isnan(radii).any() or np.isinf(radii).any():
            raise ValueError("The radii quantity has NaN or infinite values")
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValueError("The values quantity has NaN or infinite values")
        if radii_err is not None and (np.isnan(radii_err).any() or np.isinf(radii_err).any()):
            raise ValueError("The radii_err quantity has NaN or infinite values")
        if values_err is not None and (np.isnan(values_err).any() or np.isinf(values_err).any()):
            raise ValueError("The values_err quantity has NaN or infinite values")
        if deg_radii is not None and (np.isnan(deg_radii).any() or np.isinf(deg_radii).any()):
            raise ValueError("The deg_radii quantity has NaN or infinite values")

        # And now we will check that no uncertainty values are negative, as that does not make sense yet can
        #  happen sometimes when XSPEC cannot constrain a parameter (for instance).
        if radii_err is not None and (radii_err < 0).any():
            raise ValueError("The radii_err quantity has negative values, which does not make sense "
                             "for an uncertainty.")
        if values_err is not None and (values_err < 0).any():
            raise ValueError("The values_err quantity has negative values, which does not make sense "
                             "for an uncertainty.")

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

        # The currently implemented and allowed types of fitting for a profile
        self._fit_methods = ['curve_fit', 'mcmc', 'odr']
        self._nice_fit_methods = {'curve_fit': 'Curve Fit', 'mcmc': 'MCMC', 'odr': 'ODR'}

        # Here is where information about fitted models is stored (and any failed fit attempts)
        self._good_model_fits = {m: {} for m in self._fit_methods}
        self._bad_model_fits = {m: {} for m in self._fit_methods}

        # Some types of profiles will support a background value (like surface brightness), which will
        #  need to be incorporated into the fit and plotting.
        self._background = Quantity(0, self._values.unit)

        # Need to be able to store upper and lower energy bounds for those profiles that
        #  have them (like brightness profiles for instance)
        self._energy_bounds = (None, None)

        # Checking if associated_set_id is supplied, so is set_storage_key, and vice versa
        if not all([associated_set_id is None, set_storage_key is None]) and \
                not all([associated_set_id is not None, set_storage_key is not None]):
            raise ValueError("Both associated_set_id and set_storage_key must be None, or both must be not None.")

        # Putting the associated set ID into an attribute, if this profile wasn't generated by an AnnularSpectra
        #  then this will just be None. Same for the set_storage_key
        self._set_id = associated_set_id
        # Don't think this one will get a property, I can't see why the user would need it.
        self._set_storage_key = set_storage_key

        # Here we define attributes to store the fit_conf and spec_model parameters - which detail the exact
        #  spectral model and configuration that was used to produce the profile (as such only relevant to
        #  profiles that come from annular spectral properties
        # First, we make sure that we don't have one of these passed without the other - that wouldn't make sense
        #  There must be a more elegant way of doing checks like this?
        if any([fit_conf is None, spec_model is None]) and not all([fit_conf is None, spec_model is None]):
            raise ValueError("Both the 'fit_conf' and 'spec_model' arguments must be None, or both must be not None.")
        # Currently restrict the input of fit_conf - only the string version is allowed.
        elif fit_conf is not None and not isinstance(fit_conf, str):
            raise TypeError("The 'fit_conf' argument must be the string-form of the spectral fit "
                            "configuration, not the dictionary-form.")

        self._spec_fit_conf = fit_conf
        self._spec_model = spec_model

        # Here we generate a storage key for the profile to use to place itself in XGA's storage structure
        if self._set_storage_key is not None:
            # If there is a storage key for a spectrum which generated this available, then our life becomes
            #  quite simple, as it has a lot of information in it.

            # In fact as the profile will also be indexed under the profile type name, we can just use this as
            #  our storage key
            self._storage_key = self._set_storage_key

            # There will only be a set storage key if this profile came from an annular spectrum, so now we can
            #  check if we were given a fit configuration key as well - if we were, we'll include it in the
            #  storage key. We don't check if both self._spec_fit_conf and self._spec_model are None here because
            #  we've already ensured that both variables have been set, or not set.
            if self._spec_fit_conf is not None:
                # TODO I NEED TO ENSURE THAT THE SPEC FIT CONF PASSED TO THESE PROFILES IS THE STRING VERSION, NOT
                #  THE DICTIONARY VERSION. TROUBLE IS I WROTE ALL OF THIS STUFF DEALING WITH DIFFERENT CONFIGURATIONS
                #  OF THE SAME MODEL SO LONG AGO NOW THAT I HAVE FORGOTTEN HOW
                self._storage_key += ("_" + self._spec_model + "_" + self._spec_fit_conf)
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

        if radii_err is not None:
            self._outer_rad = radii[-1] + radii_err[-1]
        else:
            self._outer_rad = radii[-1]

        self._save_path = None
        # We store the input for the 'auto_save' argument - this exists so that the auto-saving after successful
        #  model fits etc. can be turned off, as it isn't necessarily going to work for profiles that were defined
        #  outside of an XGA source analysis. All profiles created by XGA processes running through XGA sources will
        #  autosave, but the default behaviour of the class will be not to autosave.
        self._auto_save = auto_save

        # This attribute is null by default, and can only be set through a property - if set then (when profiles
        #  are combined into an BaseAggregateProfile1D for the purposes of plotting) the value will be used as the
        #  label for the profile, rather than just the name
        self._custom_agg_label = None

    def _model_allegiance(self, model: BaseModel1D):
        """
        This internal method with a silly name just checks whether a model instance has already been associated
        with a profile other than this one, or if it has any association with a profile. If there is an association
        with another profile then it throws an error, as that that can cause serious problems that have caught me
        out before (see issue #742). If there is no association it sets the models profile attribute.

        :param BaseModel1D model: An instance of a BaseModel1D class (or subclass) to check.
        """
        if model.profile is not None and model.profile != self:
            raise ModelNotAssociatedError("The passed model instance is already associated with another profile, and"
                                          " as such cannot be fit to this one. Ensure that individual model instances"
                                          " are declared for each profile you are fitting.")
        elif model.profile is None:
            model.profile = self

    def emcee_fit(self, model: BaseModel1D, num_steps: int, num_walkers: int, progress_bar: bool, show_warn: bool,
                  num_samples: int) -> Tuple[BaseModel1D, bool]:
        """
        A fitting function to fit an XGA model instance to the data in this profile using the emcee
        affine-invariant MCMC sampler, this should be called through .fit() for full functionality. An initial
        run of curve_fit is used to find start parameters for the sampler, though if that fails a maximum
        likelihood estimate is run, and if that fails the method will revert to using the start parameters
        set in the model instance.

        :param BaseModel1D model: The model to be fit to the data, you cannot pass a model name for this argument.
        :param int num_steps: The number of steps each chain should take.
        :param int num_walkers: The number of walkers to be run for the ensemble sampler.
        :param bool progress_bar: Whether a progress bar should be displayed.
        :param bool show_warn: Should warnings be printed out, otherwise they are just stored in the model
            instance (this also happens if show_warn is True).
        :param int num_samples: The number of random samples to take from the posterior distributions of
            the model parameters.
        :return: The model instance, and a boolean flag as to whether this was a successful fit or not.
        :rtype: Tuple[BaseModel1D, bool]
        """
        def find_to_replace(start_pos: np.ndarray, par_lims: np.ndarray) -> np.ndarray:
            """
            Tiny function to generate an array of which start positions are currently invalid and should
            be replaced.

            :param np.ndarray start_pos: The current start positions of the walkers/parameters to be checked.
            :param np.ndarray par_lims: The limits imposed on the start parameters.
            :return: A numpy array of True and False values, True means the value should be replaced/recalculated.
            :rtype: np.ndarray
            """
            # It is possible that the start parameters can be outside of the range allowed by the the priors. In which
            #  case the MCMC fit will get super upset but not actually throw an error.
            start_check_greater = np.greater_equal(start_pos, par_lims[:, 0])
            start_check_lower = np.less_equal(start_pos, par_lims[:, 1])
            # Any true value in this array is a parameter that isn't in the allowed prior range
            to_replace_arr = ~(start_check_greater & start_check_lower)

            return to_replace_arr

        # The very first thing I do is to check whether the passed model is ACTUALLY a model or a model name - I
        #  expect this confusion could arise because the fit() method (which is what users should REALLY be using)
        #  allows either an instance or a model name.
        if not isinstance(model, BaseModel1D):
            raise TypeError("This fitting method requires that a model instance be passed for the model argument, "
                            "rather than a model name.")
        # Then I check that the model instance hasn't already been fit to another profile - I would do this in the
        #  fit() method (because then I wouldn't have to it in every separate fitting method), but I can't
        else:
            self._model_allegiance(model)

        # Trying to read out the raw output unit of the model with current start parameters, rather than the
        #  final unit set by each model - this is to make sure we're doing regression on data of the right unit
        raw_mod_unit = model.model(self.radii[0], *model.start_pars).unit

        # I'm just defining these here so that the lines don't get too long for PEP standards
        y_data = (self.values.copy() - self._background).to(raw_mod_unit).value
        y_errs = self.values_err.copy().to(raw_mod_unit).value
        rads = self.fit_radii.copy().value
        success = True
        warning_str = ""

        for prior in model.par_priors:
            if prior['type'] != 'uniform':
                raise NotImplementedError("Sorry but we don't yet support non-uniform priors for profile fitting!")

        prior_list = [p['prior'].to(model.par_units[p_ind]).value for p_ind, p in enumerate(model.par_priors)]
        prior_arr = np.array(prior_list)

        # We can run a curve_fit fit to try and get start values for the model parameters, and if that fails
        #  we try maximum likelihood, and if that fails then we fall back on the default start parameters in the
        #  model.
        # Making a copy of the model, and setting the profile to None otherwise the allegiance check gets very upset
        curve_fit_model = deepcopy(model)
        curve_fit_model.profile = None

        curve_fit_model, success = self.nlls_fit(curve_fit_model, 10, show_warn=False)
        if success or curve_fit_model.fit_warning == "Very large parameter uncertainties":
            base_start_pars = np.array([p.value for p in curve_fit_model.model_pars])
        else:
            # This finds maximum likelihood parameter values for the model+data
            max_like_res = minimize(lambda *args: -log_likelihood(*args, model.model), model.unitless_start_pars,
                                    args=(rads, y_data, y_errs))
            # I'm now adding this checking step, which will revert to the default start parameters of the model if the
            #  maximum likelihood estimate produced insane results.
            base_start_pars = max_like_res.x

        # So if any of the max likelihood pars are outside their prior, we just revert back to the original
        #  start parameters of the model. This step may make the checks performed later for instances where all
        #  start positions for a parameter are outside the prior a bit pointless, but I'm leaving them in for safety.
        if find_to_replace(base_start_pars, prior_arr).any():
            warn("Maximum likelihood estimator has produced at least one start parameter that is outside"
                 " the allowed values defined by the prior, reverting to default start parameters for this model.",
                 stacklevel=2)
            base_start_pars = model.unitless_start_pars

        # This basically finds the order of magnitude of each parameter, so we know the scale on which we should
        #  randomly perturb
        ml_rand_dev = np.power(10, np.floor(np.log10(np.abs(base_start_pars)))) / 10

        # Then that order of magnitude is multiplied by a value drawn from a standard gaussian, and this is what
        #  we perturb the maximum likelihood values with - so we get random start parameters for all
        #  of our walkers
        pos = base_start_pars + (ml_rand_dev * np.random.randn(num_walkers, model.num_pars))

        # It is possible that some of the start parameters we've generated are outside the prior, in which
        #  case emcee gets quite angry. Just in case I draw random values from the priors of all parameters,
        #  ready to be substituted in, but only if I definitely can't get the start parameters from perturbing
        #  the max likelihood 'fit'
        rand_uniform_pos = np.random.uniform(prior_arr[:, 0], prior_arr[:, 1], size=(num_walkers, model.num_pars))

        # Check which of the current start parameters are currently valid
        to_replace = find_to_replace(pos, prior_arr)

        # Setting up decent starting values is very important, so first I'm going to check if every single starting
        #  value of any parameter is outside the bounds of our priors, because if that's true then the maximum
        #  likelihood 'fit' to get the initial starting parameters is probably a bit crappy
        all_bad = np.all(to_replace, axis=0)
        if any(all_bad):
            warn("All walker starting parameters for one or more of the model parameters are outside the priors, which"
                 "probably indicates a bad initial fit (which is used to get initial start parameters). Values will be"
                 " drawn from the priors directly.", stacklevel=2)
            # This replacement only affects those parameters for which ALL start positions are outside the
            #  prior range
            all_bad_inds = np.argwhere(all_bad).T[0]
            pos[:, all_bad_inds] = rand_uniform_pos[:, all_bad_inds]
            # Now need to re-calculate the to_replace array, as it is no longer valid
            to_replace = find_to_replace(pos, prior_arr)

        # Now, if there are still problems with the starting positions, we know it is likely because some of
        #  the random perturbations of the 'best fit' start parameters are outside the priors, thus we will
        #  iteratively try and draw more acceptable perturbations, and if there are any 'bad' start positions
        #  left after 100 tries then they'll just get assigned randomly drawn values from the priors
        iter_cnt = 0
        while True in to_replace and iter_cnt < 100:
            new_pos = base_start_pars + ml_rand_dev * np.random.randn(num_walkers, model.num_pars)
            pos[to_replace] = new_pos[to_replace]
            to_replace = find_to_replace(pos, prior_arr)
            iter_cnt += 1

        # So any start values that fall outside the allowed range will be swapped out with a value randomly drawn
        #  from the prior
        pos[to_replace] = rand_uniform_pos[to_replace]

        # This instantiates an Ensemble sampler with the number of walkers specified by the user,
        #  with the log probability as defined in the functions above
        sampler = em.EnsembleSampler(num_walkers, model.num_pars, log_prob, args=(rads, y_data, y_errs, model.model,
                                                                                  prior_list))
        try:
            # So now we start the sampler, running for the number of steps specified on function call, with
            #  the starting parameters defined in the if statement above this.
            sampler.run_mcmc(pos, num_steps, progress=progress_bar)
            model.acceptance_fraction = np.mean(sampler.acceptance_fraction)
            success = True
        except ValueError as bugger:
            warning_str = str(bugger)
            model.fit_warning = warning_str
            success = False

        rng = np.random.default_rng()
        if success:
            # The auto-correlation can produce an error that basically says not to trust the chains
            try:
                # The sampler has a convenient auto-correlation time derivation, which returns the
                #  auto-correlation time for each parameter - with this I simply choose the highest one and
                #  round up to the nearest 100 to use as the burn-in
                auto_corr = np.mean(sampler.get_autocorr_time())
                # Find the nearest hundred above the mean auto-correlation time, then multiply by two for
                #  burn-in region
                cut_off = int(np.ceil(auto_corr / 100) * 100)*2
                success = True
            except ValueError as bugger:
                model.fit_warning = str(bugger)
                success = False
                cut_off = None
            except em.autocorr.AutocorrError as bugger:
                model.fit_warning = str(bugger)
                cut_off = int(0.3 * num_steps)

            # Store the chosen cut off in the model instance
            model.cut_off = cut_off

            # If the fit is considered to have not completely failed then we store distributions and parameters
            #  in the model instance
            if success:
                # I am not going to thin the chains, apparently that can actually increase variance?
                flat_samp = sampler.get_chain(discard=cut_off, flat=True)
                # Construct a numpy array representing the indices of the flattened chains
                all_inds = np.arange(flat_samp.shape[0])
                # Use the numpy random submodule to choose randomly sample the flattened chains
                chosen_inds = rng.choice(all_inds, num_samples)
                chosen = flat_samp[chosen_inds, :]
                # Then give those chains the correct units and store them in the model instance
                par_dists = [Quantity(chosen[:, p_ind], model.par_units[p_ind]) for p_ind in range(model.num_pars)]
                model.par_dists = par_dists

                # Start constructing the uncertainties on the parameters, though with a more complex model where the
                #  parameters distributions are not gaussian using these parameter values would not be appropriate
                model_par_errs = []
                for p_dist in par_dists:
                    # Store the current unit
                    u = p_dist.unit
                    # Measure the 50th percentile value of the current parameter distribution
                    fiftieth = np.nanpercentile(p_dist, 50).value
                    # Find the upper and lower bounds of the 1sigma region of the distribution
                    upper = np.nanpercentile(p_dist, 84.1).value
                    lower = np.nanpercentile(p_dist, 15.9).value
                    # Store the upper and lower uncertainties with the correct units
                    model_par_errs.append(Quantity([fiftieth-lower, upper-fiftieth], u))

                # Store the model parameter and uncertainties in the model instance
                model.model_pars = [p_dist.mean() for p_dist in par_dists]
                model.model_par_errs = model_par_errs

        # Store the sampler in the model instance, useful for getting raw chains etc again in the future
        model.emcee_sampler = sampler

        # I show all the warnings at once, if the user wants that
        if model.fit_warning != "" and show_warn:
            print(model.fit_warning)

        # Tell the model whether we think the fit was successful or not
        model.success = success

        # And finally storing the fit method used in the model itself
        model.fit_method = "mcmc"

        # Explicitly deleting the curve fit model, just to be safe
        del curve_fit_model

        return model, success

    def nlls_fit(self, model: BaseModel1D, num_samples: int, show_warn: bool) -> Tuple[BaseModel1D, bool]:
        """
        A function to fit an XGA model instance to the data in this profile using the non-linear least squares
        curve_fit routine from scipy, this should be called through .fit() for full functionality

        :param BaseModel1D model: An instance of the model to be fit to this profile.
        :param int num_samples: The number of random samples to be drawn and stored in the model
            parameter distribution property.
        :param bool show_warn: Should warnings be printed out, otherwise they are just stored in the model
            instance (this also happens if show_warn is True).
        :return: The model (with best fit parameters stored within it), and a boolean flag as to whether the
            fit was successful or not.
        :rtype: Tuple[BaseModel1D, bool]
        """
        # The very first thing I do is to check whether the passed model is ACTUALLY a model or a model name - I
        #  expect this confusion could arise because the fit() method (which is what users should REALLY be using)
        #  allows either an instance or a model name.
        if not isinstance(model, BaseModel1D):
            raise TypeError("This fitting method requires that a model instance be passed for the model argument, "
                            "rather than a model name.")
        # Then I check that the model instance hasn't already been fit to another profile - I would do this in the
        #  fit() method (because then I wouldn't have to it in every separate fitting method), but I can't
        else:
            self._model_allegiance(model)

        # Trying to read out the raw output unit of the model with current start parameters, rather than the
        #  final unit set by each model - this is to make sure we're doing regression on data of the right unit
        raw_mod_unit = model.model(self.radii[0], *model.start_pars).unit

        y_data = (self.values.copy() - self._background).to(raw_mod_unit).value
        y_errs = self.values_err.copy().to(raw_mod_unit).value
        rads = self.fit_radii.copy().value
        success = True
        warning_str = ""
        
        lower_bounds = []
        upper_bounds = []
        for prior_ind, prior in enumerate(model.par_priors):
            if prior['type'] == 'uniform':
                conv_prior = prior['prior'].to(model.par_units[prior_ind]).value
                lower_bounds.append(conv_prior[0])
                upper_bounds.append(conv_prior[1])
            else:
                lower_bounds.append(-np.inf)
                upper_bounds.append(np.inf)

        # Curve fit is a simple non-linear least squares implementation, its alright but fragile
        try:
            fit_par, fit_cov = curve_fit(model.model, rads, y_data, p0=model.unitless_start_pars, sigma=y_errs,
                                         absolute_sigma=True, bounds=(lower_bounds, upper_bounds))

            # If there is an infinite value in the covariance matrix, it means curve_fit was
            #  unable to estimate it properly
            if np.inf in fit_cov:
                warning_str = "Infinity in covariance matrix"
                success = False
            else:
                # Grab the diagonal of the covariance matrix, then sqrt to get sigma values for each parameter
                fit_par_err = np.sqrt(np.diagonal(fit_cov))
                frac_err = np.divide(fit_par_err, fit_par, where=fit_par != 0)
                if frac_err.max() > 10:
                    warning_str = "Very large parameter uncertainties"
                    success = False
        except RuntimeError as r_err:
            warn("{}, curve_fit has failed.".format(str(r_err)), stacklevel=2)
            warning_str = str(r_err)
            success = False
            fit_par = np.full(len(model.model_pars), np.nan)
            fit_par_err = np.full(len(model.model_pars), np.nan)
        except ValueError as v_err:
            warn("{}, curve_fit has failed.".format(str(v_err)), stacklevel=2)
            warning_str = str(v_err)
            success = False
            fit_par = np.full(len(model.model_pars), np.nan)
            fit_par_err = np.full(len(model.model_pars), np.nan)

        # Using the parameter values and the covariance matrix I generate parameter distributions to store in the
        #  model instance.
        # Apparently this is what we should be using for random numbers from numpy now!
        rng = np.random.default_rng()
        if success:
            # I generate num_samples random points from the distribution of each parameter
            ext_model_par = np.repeat(fit_par[..., None], num_samples, axis=1).T
            ext_model_par_err = np.repeat(fit_par_err[..., None], num_samples, axis=1).T
            # This generates model_real random samples from the passed model parameters, assuming they are Gaussian
            model_par_dists = rng.normal(ext_model_par, ext_model_par_err)
            par_dists = [Quantity(model_par_dists[:, p_ind], model.par_units[p_ind])
                         for p_ind in range(0, len(fit_par))]
            model.par_dists = par_dists

        # Now we put the values BACK into quantities
        fit_par = [Quantity(p, model.par_units[p_ind]) for p_ind, p in enumerate(fit_par)]
        fit_par_err = [Quantity(p, model.par_units[p_ind]) for p_ind, p in enumerate(fit_par_err)]

        # And pass them into the model
        model.model_pars = fit_par
        model.model_par_errs = fit_par_err

        if not success:
            model.fit_warning = warning_str

        if show_warn and warning_str != "":
            warn(warning_str)

        # Tell the model whether we think the fit was successful or not
        model.success = success

        # And finally storing the fit method used in the model itself
        model.fit_method = "curve_fit"

        # And then the model gets sent back
        return model, success

    def _odr_fit(self, model: BaseModel1D, show_warn: bool):
        # TODO MAKE THIS A NON-INTERNAL METHOD WHEN ITS WRITTEN
        # TODO REMEMBER TO USE THE FIT RADII PROPERTY
        # Tell the model whether we think the fit was successful or not
        # model.success = success

        # The very first thing I do is to check whether the passed model is ACTUALLY a model or a model name - I
        #  expect this confusion could arise because the fit() method (which is what users should REALLY be using)
        #  allows either an instance or a model name.
        if not isinstance(model, BaseModel1D):
            raise TypeError("This fitting method requires that a model instance be passed for the model argument, "
                            "rather than a model name.")
        # Then I check that the model instance hasn't already been fit to another profile - I would do this in the
        #  fit() method (because then I wouldn't have to it in every separate fitting method), but I can't
        else:
            self._model_allegiance(model)

        # And finally storing the fit method used in the model itself
        model.fit_method = "odr"
        raise NotImplementedError("This fitting method is still under construction!")

    def fit(self, model: Union[str, BaseModel1D], method: str = "mcmc", num_samples: int = 10000,
            num_steps: int = 30000, num_walkers: int = 20, progress_bar: bool = True,
            show_warn: bool = True, force_refit: bool = False) -> BaseModel1D:
        """
        Method to fit a model to this profile's data, then store the resulting model parameter results. Each
        profile can store one instance of a type of model per fit method. So for instance you could fit both
        a 'beta' and 'double_beta' model to a surface brightness profile with curve_fit, and then you could
        fit 'double_beta' again with MCMC.

        If any of the parameters of the passed model have a uniform prior associated, and the chosen method
        is curve_fit, then those priors will be used to place bounds on those parameters.

        :param str/BaseModel1D model: Either an instance of an XGA model to be fit to this profile, or the name
            of a profile (e.g. 'beta', or 'simple_vikhlinin_dens').
        :param str method: The fit method to use, either 'curve_fit', 'mcmc', or 'odr'.
        :param int num_samples: The number of random samples to draw to create the parameter distributions
            that are saved in the model.
        :param int num_steps: Only applicable if using MCMC fitting, the number of steps each walker should take.
        :param int num_walkers: Only applicable if using MCMC fitting, the number of walkers to initialise
            for the ensemble sampler.
        :param bool progress_bar: Only applicable if using MCMC fitting, should a progress bar be shown.
        :param bool show_warn: Should warnings be printed out, otherwise they are just stored in the model
            instance (this also happens if show_warn is True).
        :param bool force_refit: Controls whether the profile will re-run the fit of a model that already has a good
            model fit stored. The default is False.
        :return: The fitted model object. The fitted model is also stored within the profile object.
        :rtype: BaseModel1D
        """
        # Make sure the method is lower case
        method = method.lower()

        # This chunk is just checking inputs and making sure they're valid
        if self._prof_type in PROF_TYPE_MODELS:
            # Put the allowed models for this profile type into a string
            allowed = ", ".join(PROF_TYPE_MODELS[self._prof_type])
        else:
            allowed = ""

        if self._prof_type == "base":
            raise XGAFitError("A BaseProfile1D object currently cannot have a model fitted to it, as there"
                              " is no physical context.")
        elif isinstance(model, str) and model.lower() not in PROF_TYPE_MODELS[self._prof_type]:
            raise XGAInvalidModelError("{p} is not available for this type of profile, please use one of the "
                                       "following models {a}".format(p=model, a=allowed))
        elif isinstance(model, str):
            model = PROF_TYPE_MODELS[self._prof_type][model](self.radii_unit, self.values_unit)
        elif isinstance(model, BaseModel1D) and model.name not in PROF_TYPE_MODELS[self._prof_type]:
            raise XGAInvalidModelError("{p} is not available for this type of profile, please use one of the "
                                       "following models {a}".format(p=model, a=allowed))
        elif isinstance(model, BaseModel1D) and (model.x_unit != self.radii_unit or model.y_unit != self.values_unit):
            raise UnitConversionError("The model instance passed to the fit method has units that are incompatible, "
                                      "with the data. This profile has an radius unit of {r} and a value unit of "
                                      "{v}".format(r=self.radii_unit.to_string(), v=self.values_unit.to_string()))

        # I don't think I'm going to allow any fits without value uncertainties - just seems daft
        if self._values_err is None:
            raise XGAFitError("You cannot fit to a profile that doesn't have value uncertainties.")

        # Checking that the method passed is valid
        if method not in self._fit_methods:
            allowed = ", ".join(self._fit_methods)
            raise XGAFitError("{me} is not a valid fitting method, please use one of these; {a}".format(me=method,
                                                                                                        a=allowed))

        # Check whether a good fit result already exists for this model. We use the storage_key property that
        #  XGA model objects generate from their name and their start parameters
        if not force_refit and model.name in self._good_model_fits[method]:
            warn("{m} already has a successful fit result for this profile using {me}, with those start "
                 "parameters".format(m=model.name, me=method), stacklevel=2)
            already_done = True
        elif model.name in self._bad_model_fits[method]:
            warn("{m} already has a failed fit result for this profile using {me} with those start "
                 "parameters".format(m=model.name, me=method), stacklevel=2)
            already_done = False
        else:
            already_done = False

        # Running the requested fitting method
        if not already_done and method == 'mcmc':
            model, success = self.emcee_fit(model, num_steps, num_walkers, progress_bar, show_warn, num_samples)
        elif not already_done and method == 'curve_fit':
            model, success = self.nlls_fit(model, num_samples, show_warn)
        elif not already_done and method == 'odr':
            model, success = self._odr_fit(model, show_warn)
        else:
            model = self.get_model_fit(model.name, method)

        # Storing the model in the internal dictionaries depending on whether the fit was successful or not
        if not already_done and success:
            self._good_model_fits[method][model.name] = model
        elif not already_done and not success:
            self._bad_model_fits[method][model.name] = model

        if self.auto_save:
            # This method means that a change has happened to the model, so it should be re-saved
            self.save()
        return model

    def allowed_models(self, table_format: str = 'fancy_grid'):
        """
        This is a convenience function to tell the user what models can be used to fit a profile
        of the current type, what parameters are expected, and what the defaults are.

        :param str table_format: The desired format of the allowed models table. This is passed to the
            tabulate module (allowed formats can be found here - https://pypi.org/project/tabulate/), and
            alters the way the printed table looks.
        """
        # Base profile don't have any type of model associated with them, so just making an empty list
        if self._prof_type == "base":
            warn("There are no implemented models for this profile type")
        else:
            allowed = list(PROF_TYPE_MODELS[self._prof_type].keys())

            # These just roll through the available models for this type of profile and construct strings of
            #  parameter names and start parameters to put in the table
            model_par_names = []
            model_par_starts = []
            for m in allowed:
                exp_pars = ""
                par_len = 0
                def_starts = ""
                def_len = 0

                # This chunk of code tries to make sure that the strings aren't too long to display nicely
                #  in the table
                mod_inst = PROF_TYPE_MODELS[self._prof_type][m]()
                for p_ind, p in enumerate(list(inspect.signature(mod_inst.model).parameters.values())[1:]):
                    if par_len > 35:
                        exp_pars += ' \n'
                        par_len = 0
                    next_par = '{}, '.format(p.name)
                    par_len += len(next_par)
                    exp_pars += next_par

                    if def_len > 35:
                        def_starts += ' \n'
                        def_len = 0
                    next_def = '{}, '.format(str(mod_inst.start_pars[p_ind]))
                    def_len += len(next_def)
                    def_starts += next_def

                # We slice out the last character because we know its going to be a spurious comma, just
                #  because of the lazy way I wrote the loop above
                model_par_names.append(exp_pars[:-2])
                model_par_starts.append(def_starts[:-2])

            # Construct the table data and display it using tabulate module
            tab_dat = [[allowed[i], model_par_names[i], model_par_starts[i]] for i in range(0, len(allowed))]
            print(tabulate(tab_dat, ["MODEL NAME", "EXPECTED PARAMETERS", "DEFAULT START VALUES"],
                           tablefmt=table_format))

    def get_model_fit(self, model: str, method: str) -> BaseModel1D:
        """
        A get method for fitted model objects associated with this profile. Models for which the fit failed will
        also be returned, but a warning will be shown to inform the user that the fit failed.

        :param str model: The name of the model to retrieve.
        :param str method: The method which was used to fit the model.
        :return: An instance of an XGA model object that was fitted to this profile and updated with the
            parameter values.
        :rtype: BaseModel1D
        """
        if model not in PROF_TYPE_MODELS[self._prof_type]:
            allowed = list(PROF_TYPE_MODELS[self._prof_type].keys())
            prof_name = self._y_axis_name.lower()
            raise XGAInvalidModelError("{m} is not a valid model for a {p} profile, please choose from "
                                       "one of these; {a}".format(m=model, a=", ".join(allowed), p=prof_name))
        elif model in self._bad_model_fits[method]:
            warn("An attempt was made to fit {m} with {me} but it failed, so treat the model with "
                 "suspicion".format(m=model, me=method))
            ret_model = self._bad_model_fits[method][model]
        elif model not in self._good_model_fits[method]:
            raise ModelNotAssociatedError("{m} is valid for this profile, but hasn't been fit with {me} "
                                          "yet".format(m=model, me=method))
        else:
            ret_model = self._good_model_fits[method][model]

        return ret_model

    def add_model_fit(self, model: BaseModel1D, method: str):
        """
        There are rare circumstances where XGA processes might wish to add a model to a profile from the outside,
        which is what this method allows you to do.

        :param BaseModel1D model: The XGA model object to add to the profile.
        :param str method: The method used to fit the model.
        """

        # Checking that the method passed is valid
        if method not in self._fit_methods:
            allowed = ", ".join(self._fit_methods)
            raise XGAFitError("{me} is not a valid fitting method, please use one of these; {a}".format(me=method,
                                                                                                        a=allowed))
        # Checking that the model is valid for this particular profile
        allowed = ", ".join(PROF_TYPE_MODELS[self._prof_type])
        if model.name not in PROF_TYPE_MODELS[self._prof_type]:
            raise XGAInvalidModelError("{p} is not valid for this type of profile, please use one of the "
                                       "following models {a}".format(p=model.name, a=allowed))
        elif model.x_unit != self.radii_unit or model.y_unit != self.values_unit:
            raise UnitConversionError("The model instance passed to the fit method has units that are incompatible, "
                                      "with the data. This profile has an radius unit of {r} and a value unit of "
                                      "{v}".format(r=self.radii_unit.to_string(), v=self.values_unit.to_string()))
        elif not model.success:
            raise ValueError("Please only add successful models to this profile.")
        else:
            self._good_model_fits[method][model.name] = model

        if self.auto_save:
            # This method means that a change has happened to the model, so it should be re-saved
            self.save()

    def remove_model_fit(self, model: Union[str, BaseModel1D], method: str):
        """
        This will remove an existing model fit for a particular fit method.

        :param str/BaseModel1D model: The model fit to delete.
        :param str method: The method used to fit the model.
        """
        # Making sure we have a string model name
        if isinstance(model, BaseModel1D):
            model = model.name

        # Checking the input model is valid for this profile
        if model not in PROF_TYPE_MODELS[self._prof_type]:
            raise XGAInvalidModelError("{m} is not a valid model for a {p} "
                                       "profile.".format(m=model, p=self._y_axis_name.lower()))

        # Checking that the method passed is valid
        if method not in self._fit_methods:
            allowed = ", ".join(self._fit_methods)
            raise XGAFitError("{me} is not a valid fitting method, the following are allowed; "
                              "{a}".format(me=method, a=allowed))

        if model not in self._good_model_fits[method]:
            raise XGAInvalidModelError("{m} is valid for this profile, but cannot be removed as it has not been "
                                       "fit.".format(m=model))
        else:
            # Finally remove the model
            del self._good_model_fits[method][model]

    def get_sampler(self, model: str) -> em.EnsembleSampler:
        """
        A get method meant to retrieve the MCMC ensemble sampler used to fit a particular
        model (supplied by the user). Checks are applied to the supplied model, to make
        sure that it is valid for the type of profile, that a good fit has actually been
        performed, and that the fit was performed with Emcee and not another method.

        :param str model: The name of the model for which to retrieve the sampler.
        :return: The Emcee sampler used to fit the user supplied model.
        :rtype: em.EnsembleSampler
        """
        model = self.get_model_fit(model, 'mcmc')
        return model.emcee_sampler

    def get_chains(self, model: str, discard: Union[bool, int] = True, flatten: bool = True,
                   thin: int = 1) -> np.ndarray:
        """
        Get method for the sampler chains of an MCMC fit to the user supplied model. get_sampler is
        called to retrieve the sampler object, as well as perform validity checks on the model name.

        :param str model: The name of the model for which to retrieve the chains.
        :param bool/int discard: Whether steps should be discarded for burn-in. If True then the cut off decided
            using the auto-correlation time will be used. If an integer is passed then this will be used as the
            number of steps to discard, and if False then no steps will be discarded.
        :param bool flatten: Should the chains of the multiple walkers be flattened into one chain per parameter.
        :param int thin: The thinning that should be applied to the chains. The default is 1, which means no
            thinning is applied.
        :return: The requested chains.
        :rtype: np.ndarray
        """
        model = self.get_model_fit(model, 'mcmc')

        if isinstance(discard, bool) and discard:
            chains = model.emcee_sampler.get_chain(discard=model.cut_off, flat=flatten, thin=thin)
        elif isinstance(discard, int):
            chains = model.emcee_sampler.get_chain(discard=discard, flat=flatten, thin=thin)
        else:
            chains = model.emcee_sampler.get_chain(flat=flatten, thin=thin)

        return chains

    def view_chains(self, model: str, discard: Union[bool, int] = True, thin: int = 1, figsize: Tuple = None):
        """
        Simple view method to quickly look at the MCMC chains for a given model fit.

        :param str model: The name of the model for which to view the MCMC chains.
        :param bool/int discard: Whether steps should be discarded for burn-in. If True then the cut off decided
            using the auto-correlation time will be used. If an integer is passed then this will be used as the
            number of steps to discard, and if False then no steps will be discarded.
        :param int thin: The thinning that should be applied to the chains. The default is 1, which means no
            thinning is applied.
        :param Tuple figsize: Desired size of the figure, if None will be set automatically.
        """
        chains = self.get_chains(model, discard, thin=thin, flatten=False)
        model_obj = self.get_model_fit(model, 'mcmc')

        if figsize is None:
            fig, axes = plt.subplots(nrows=model_obj.num_pars, figsize=(12, 2*model_obj.num_pars), sharex='col')
        else:
            fig, axes = plt.subplots(model_obj.num_pars, figsize=figsize, sharex='col')

        plt.suptitle("{m} Parameter Chains".format(m=model_obj.publication_name), fontsize=14, y=1.02)

        for i in range(model_obj.num_pars):
            cur_unit = model_obj.par_units[i]
            if cur_unit == Unit(''):
                par_unit_name = ""
            else:
                par_unit_name = r" $\left[" + cur_unit.to_string("latex").strip("$") + r"\right]$"
            ax = axes[i]
            ax.plot(chains[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(chains))
            ax.set_ylabel(model_obj.par_publication_names[i] + par_unit_name, fontsize=13)
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step Number", fontsize=13)
        plt.tight_layout()
        plt.show()

    def view_corner(self, model: str, figsize: Tuple = (8, 8)):
        """
        A convenient view method to examine the corner plot of the parameter posterior distributions.

        :param str model: The name of the model for which to view the corner plot.
        :param Tuple figsize: The desired figure size.
        """
        flat_chains = self.get_chains(model, flatten=True)
        model_obj = self.get_model_fit(model, 'mcmc')

        frac_conf_lev = [(50 - 34.1)/100, 0.5, (50 + 34.1)/100]

        # If any of the median parameter values are above 1e+4 we get corner to format them in scientific
        #  notation, to avoid super long numbers spilling over the edge of the corner plot. I will say that
        #  scientific notation in titles in corner doesn't look that great either, but its better than the
        #  alternative
        if np.any(np.median(flat_chains, axis=0) > 1e+4):
            fig = corner.corner(flat_chains, labels=model_obj.par_publication_names, figsize=figsize,
                                quantiles=frac_conf_lev, show_titles=True, title_fmt=".2e")
        else:
            fig = corner.corner(flat_chains, labels=model_obj.par_publication_names, figsize=figsize,
                                quantiles=frac_conf_lev, show_titles=True)
        t = self._y_axis_name
        plt.suptitle("{m} - {s} {t} Profile".format(m=model_obj.publication_name, s=self.src_name, t=t),
                     fontsize=14, y=1.02)
        plt.show()

    def view_getdist_corner(self, model: str, settings: dict = {}, figsize: tuple = (10, 10)):
        """
        A view method to see a corner plot generated with the getdist module, using flattened chains with
        burn-in removed (whatever the getdist message might say).

        :param str model: The name of the model for which to view the corner plot.
        :param dict settings: The settings dictionary for a getdist MCSample.
        :param tuple figsize: A tuple to set the size of the figure.
        """
        # Grab the flattened chains
        flat_chains = self.get_chains(model, flatten=True)
        model_obj = self.get_model_fit(model, 'mcmc')

        # Setting up parameter label name and unit pairs - will strip them of '$' in the next line - didn't do it
        #  here to make it a little easier to read
        labels = [[par_name, model_obj.par_units[par_ind].to_string('latex')] for par_ind, par_name
                  in enumerate(model_obj.par_publication_names)]

        # Need to remove $ from the labels because getdist adds them itself
        stripped_labels = [(lab_pair[0] + ((r"\: \left[" + lab_pair[1] + r'\right]')
                            if lab_pair[1] != '$\\mathrm{}$' else '')).replace('$', '') for lab_pair in labels]
        # Setup the getdist sample object
        gd_samp = MCSamples(samples=flat_chains, names=model_obj.par_names, labels=stripped_labels,
                            settings=settings)

        # And generate the triangle plot
        g = plots.get_subplot_plotter(width_inch=figsize[0])
        g.triangle_plot([gd_samp], filled=True)
        plt.show()

    def generate_data_realisations(self, num_real: int, truncate_zero: bool = False):
        """
        A method to generate random realisations of the data points in this profile, using their y-axis values
        and uncertainties. This can be useful for error propagation for instance, and does not require a model fit
        to work. This method assumes that the y-errors are 1-sigma, which isn't necessarily the case.

        :param int num_real: The number of random realisations to generate.
        :param bool truncate_zero: Should the data realisations be truncated at zero, default is False. This could
            be used for generating realisations of profiles where negative values are not physical.
        :return: An N x R astropy quantity, where N is the number of realisations and R is the number of radii
            at which there are data points in this profile.
        :rtype: Quantity
        """
        # If we have no error information on the profile y values, then we can hardly generate distributions from them
        if self.values_err is None:
            raise ValueError("This profile has no y-error information, and as such you cannot generate random"
                             " realisations of the data.")

        # The user can choose to ensure that the realisation distributions are truncated at zero, in which case we
        #  use a truncated normal distribution.
        if truncate_zero:
            # The truncnorm setup wants the limits in units of scale essentially
            trunc_lims = ((0 - self.values) / self.values_err, (np.inf - self.values) / self.values_err)
            realisations = truncnorm(trunc_lims[0], trunc_lims[1], loc=self.values,
                                     scale=self.values_err).rvs([num_real, len(self.values)])
        # But the default behaviour is to just use a normal distribution
        else:
            # Here I copy the values and value uncertainties N times, where N is the number of realisations
            #  the user wants
            ext_values = np.repeat(self.values[..., None], num_real, axis=1).T
            ext_value_errs = np.repeat(self.values_err[..., None], num_real, axis=1).T
            # Here I just generate N realisations of the profiles using a normal distribution, though this does assume
            #  that the errors are one sigma which isn't necessarily true
            realisations = np.random.normal(ext_values, ext_value_errs)

        # Ensure that our value distributions are passed back as quantities with the correct units
        realisations = Quantity(realisations, self.values_unit)

        return realisations

    def get_view(self, fig: Figure, main_ax: Axes, xscale: str = "log", yscale: str = "log", xlim: tuple = None,
                 ylim: tuple = None, models: bool = True,  back_sub: bool = True, just_models: bool = False,
                 custom_title: str = None, draw_rads: dict = {}, x_norm: Union[bool, Quantity] = False,
                 y_norm: Union[bool, Quantity] = False, x_label: str = None, y_label: str = None,
                 data_colour: str = 'black', model_colour: Union[str, List[str]] = 'seagreen',
                 show_legend: bool = True, show_residual_ax: bool = True, draw_vals: dict = {},
                 auto_legend: bool = True, joined_points: bool = False, axis_formatters: dict = None):
        """
        A get method for an axes (or multiple axes) showing this profile and model fits. The idea of this get method
        is that, whilst it is used by the view() method, it can also be called by external methods that wish to use
        the profile plot in concert with other views.

        :param Figure fig: The figure which has been set up for this profile plot.
        :param Axes main_ax: The matplotlib axes on which to show the image.
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
            e.g. {'r500': Quantity(), 'r200': Quantity()}
        :param bool x_norm: Controls whether the x-axis of the profile is normalised by another value, the default is
            False, in which case no normalisation is applied. If it is set to True then it will attempt to use the
            internal normalisation value (which can be set with the x_norm property), and if a quantity is passed it
            will attempt to normalise using that.
        :param bool y_norm: Controls whether the y-axis of the profile is normalised by another value, the default is
            False, in which case no normalisation is applied. If it is set to True then it will attempt to use the
            internal normalisation value (which can be set with the y_norm property), and if a quantity is passed it
            will attempt to normalise using that.
        :param str x_label: Custom label for the x-axis (excluding units, which will be added automatically).
        :param str y_label: Custom label for the y-axis (excluding units, which will be added automatically).
        :param str data_colour: Used to set the colour of the data points.
        :param str/List[str] model_colour: The matplotlib colour(s) that should be used for plotted model fits (if
            applicable). Either a single colour name, or a list of colour names, may be passed depending on the number
            of models that are being plotted - if there are multiple models, and a single colour is passed, the plot
            will revert to the default matplotlib colour cycler. If a list is passed, those colours will be cycled
            through instead (if there are insufficient entries for the number of models an error will be raised). The
            default value is 'seagreen'.
        :param bool show_legend: Whether the legend should be displayed or not. Default is True.
        :param bool show_residual_ax: Controls whether a lower axis showing the residuals between data and
            model (if a model is fitted and being shown) is displayed. Default is True.
        :param dict draw_vals: A dictionary of extra y-values (as astropy quantities) to draw onto the plot, where the
            dictionary key they are stored under is what they will be labelled (keys can be LaTeX
            formatted); e.g. {r'$T_{\rm{X,500}}$': Quantity(6, 'keV')}. Quantities with uncertainties may also be
            passed, and the error regions will be shaded; e.g. {r'$T_{\rm{X,500}}$': Quantity([6, 0.2, 0.3], 'keV')},
            where 0.2 is the negative error, and 0.3 is the positive error.
        :param bool auto_legend: If True, and show_legend has also been set to True, then the 'best' legend location
            will be defined by matplotlib, otherwise, if False, the legend will be added to the right hand side of the
            plot outside the main axes.
        :param bool joined_points: If True, the data in the profile will be plotted as a line, rather than points, as
            will any uncertainty regions.
        :param dict axis_formatters: A dictionary of formatters that can be applied to the profile plot. The keys
            can have the following values; 'xmajor', 'xminor', 'ymajor', and 'yminor'. The values associated with the
            keys should be instantiated matplotlib formatters.
        """

        # Checks that any extra radii that have been passed are the correct units (i.e. the same as the radius units
        #  used in this profile)
        if not all([r.unit == self.radii_unit for r in draw_rads.values()]):
            raise UnitConversionError("All radii in draw_rad have to be in the same units as this profile, "
                                      "{}".format(self.radii_unit.to_string()))

        # Checks that any extra y-axis values that have been passed are the correct units (i.e. the same as the
        #  y-value units used in this profile)
        if not all([v.unit == self.values_unit for v in draw_vals.values()]):
            raise UnitConversionError("All values in draw_vals have to be in the same units as this profile, "
                                      "{}".format(self.values_unit.to_string()))

        if axis_formatters is not None and \
                not all([k in ['xmajor', 'xminor', 'ymajor', 'yminor'] for k in axis_formatters.keys()]):
            raise KeyError("The axis_formatters dictionary may only contain the following keys; xmajor, xminor, "
                           "ymajor, and yminor.")

        # Default is to show models, but that flag is set to False here if there are none, otherwise we get
        #  extra plotted stuff that doesn't make sense
        if len(self.good_model_fits) == 0:
            models = False
            just_models = False

        # If the user wants the x-axis to be normalised then we grab the value from the profile (though of course
        #  if the user didn't set it initially then self.x_norm will also be 1
        if isinstance(x_norm, bool) and x_norm:
            x_norm = self.x_norm
            if self.x_norm == Quantity(1, ''):
                warn("No normalisation value is stored for the x-axis", stacklevel=2)
        elif isinstance(x_norm, Quantity):
            x_norm = x_norm
        elif isinstance(x_norm, bool) and not x_norm:
            # Otherwise we set x_norm to a harmless values with no units and unity value
            x_norm = Quantity(1, '')

        if isinstance(y_norm, bool) and y_norm:
            y_norm = self.y_norm
            if self.y_norm == Quantity(1, ''):
                warn("No normalisation value is stored for the y-axis", stacklevel=2)
        elif isinstance(y_norm, Quantity):
            y_norm = y_norm
        elif isinstance(y_norm, bool) and not y_norm:
            y_norm = Quantity(1, '')

        main_ax.minorticks_on()
        if models and show_residual_ax:
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

        rad_vals = self.fit_radii.copy()
        plot_y_vals /= y_norm
        rad_vals /= x_norm

        # Now the actual plotting of the data
        if self.radii_err is not None and self.values_err is None and not joined_points:
            x_errs = (self.radii_err.copy() / x_norm).value
            line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, xerr=x_errs, fmt="x", capsize=2,
                                    label=leg_label, color=data_colour)
        elif self.radii_err is None and self.values_err is not None and not joined_points:
            y_errs = (self.values_err.copy() / y_norm).value
            line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, yerr=y_errs, fmt="x", capsize=2,
                                    label=leg_label, color=data_colour)
        elif self.radii_err is not None and self.values_err is not None and not joined_points:
            x_errs = (self.radii_err.copy() / x_norm).value
            y_errs = (self.values_err.copy() / y_norm).value
            line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, xerr=x_errs, yerr=y_errs, fmt="x", capsize=2,
                                    label=leg_label, color=data_colour)
        elif joined_points:
            line = main_ax.plot(rad_vals.value, plot_y_vals.value, label=leg_label, color=data_colour)
            if self.values_err is not None:
                y_errs = (self.values_err.copy() / y_norm).value
                main_ax.fill_between(rad_vals.value, plot_y_vals.value - y_errs, plot_y_vals.value + y_errs,
                                     color=data_colour,  linestyle='dashdot', alpha=0.7)
        else:
            line = main_ax.plot(rad_vals.value, plot_y_vals.value, 'x', label=leg_label, color=data_colour)

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

            # Runs through the model fit methods, and the models fit with each method, and counts them - makes
            #  it a little neater to check how many colours we need for our colour cycles down below
            num_to_plot = len([1 for method in self._good_model_fits for model in self._good_model_fits[method]])

            # Now we have make sure that the model colours are set up properly - the user can either pass a string
            #  name or a list of string names, so we will either stick with one colour for one model, revert to the
            #  standard colour cycle if they only gave one colour for multiple models, accept the list of colours for
            #  a set of models, or throw an error that they didn't pass a long enough list of colours
            if isinstance(model_colour, str) and num_to_plot == 1:
                model_colour = [model_colour]
            elif isinstance(model_colour, str) and num_to_plot != 1:
                model_colour = [None]*num_to_plot
            elif isinstance(model_colour, list) and len(model_colour) != num_to_plot:
                raise ValueError("If the 'model_colour' argument is a list, it must have one entry per model-method "
                                 "combination. The passed list has {p} entries, and there are {mm} model-method "
                                 "combinations.".format(p=len(model_colour), mm=num_to_plot))

            # We use the slightly-no-longer-useful fit_radii property (it is only useful if any of the radii values
            #  are at zero, which used to be the case for most of the profiles generated by XGA). In the case where
            #  no radii values are zero, then fit_radii will just be the radii. Then we subtract the errors and add
            #  the errors, if they are available - to find the minimum and maximum radii we should plot the model to
            if self.radii_err is not None:
                lo_rad = (self.fit_radii-self.radii_err).min()
                hi_rad = (self.fit_radii+self.radii_err).max()
            else:
                lo_rad = self.fit_radii.min()
                hi_rad = self.fit_radii.max()
            mod_rads = np.linspace(lo_rad, hi_rad, 500)

            mod_col_ind = 0
            for method in self._good_model_fits:
                for model in self._good_model_fits[method]:
                    model_obj = self._good_model_fits[method][model]
                    mod_reals = model_obj.get_realisations(mod_rads)
                    # mean_model = np.mean(mod_reals, axis=1)
                    median_model = np.nanpercentile(mod_reals, 50, axis=1)

                    upper_model = np.nanpercentile(mod_reals, 84.1, axis=1)
                    lower_model = np.nanpercentile(mod_reals, 15.9, axis=1)

                    mod_lab = model_obj.publication_name + " - {}".format(self._nice_fit_methods[method])
                    cur_line = main_ax.plot(mod_rads.value / x_norm.value, median_model.value / y_norm, label=mod_lab,
                                 color=model_colour[mod_col_ind])
                    cur_color = cur_line[0].get_color()

                    main_ax.fill_between(mod_rads.value / x_norm.value, lower_model.value / y_norm.value,
                                         upper_model.value / y_norm.value, alpha=0.7, interpolate=True,
                                         where=upper_model.value >= lower_model.value, facecolor=cur_color)
                    main_ax.plot(mod_rads.value / x_norm.value, lower_model.value / y_norm.value, color=cur_color,
                                 linestyle="dashed")
                    main_ax.plot(mod_rads.value / x_norm.value, upper_model.value / y_norm.value, color=cur_color,
                                 linestyle="dashed")

                    # I only want this to trigger if the user has decided they want a residual axis. I expect most
                    #  of the time that they will, but for things like the Hydrostatic mass diagnostic plots I want
                    #  to be able to turn the residual axis off.
                    if show_residual_ax:
                        # This calculates and plots the residuals between the model and the data on the extra
                        #  axis we added near the beginning of this method
                        res = np.nanpercentile(model_obj.get_realisations(self.fit_radii), 50, axis=1) \
                              - (plot_y_vals * y_norm)
                        res_ax.plot(rad_vals.value, res.value, 'D', color=cur_color)

                    # Move the colour on!
                    mod_col_ind += 1

        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + rad_vals.unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + plot_y_vals.unit.to_string("latex").strip("$") + r"\right]$"

        # If the quantity being plotted is unitless then we don't want there to be empty brackets
        if x_unit == r"$\left[\mathrm{}\right]$":
            x_unit = ""
        if y_unit == r"$\left[\mathrm{}\right]$":
            y_unit = ""

        if x_label is None:
            # Setting the main plot's x label
            main_ax.set_xlabel("Radius {}".format(x_unit), fontsize=13)
        else:
            main_ax.set_xlabel(x_label + " {}".format(x_unit), fontsize=13)

        if y_label is None and (self._background.value == 0 or not back_sub):
            main_ax.set_ylabel(r"{l} {u}".format(l=self._y_axis_name, u=y_unit), fontsize=13)
        elif y_label is None:
            # If background has been subtracted it will be mentioned in the y axis label
            main_ax.set_ylabel(r"Background Subtracted {l} {u}".format(l=self._y_axis_name, u=y_unit), fontsize=13)
        elif y_label is not None:
            main_ax.set_ylabel(y_label + ' {}'.format(y_unit), fontsize=13)

        # If the user has manually set limits then we can use them, only on the main axis because
        #  we grab those limits from the axes object for the residual axis later
        if xlim is not None:
            main_ax.set_xlim(xlim)
        if ylim is not None:
            main_ax.set_ylim(ylim)

        # Setup the scale that the user wants to see, again on the main axis
        main_ax.set_xscale(xscale)
        main_ax.set_yscale(yscale)
        if models and show_residual_ax:
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
        if models and custom_title is None and len(self.good_model_fits) == 1:
            title_str = "{l} Profile - with model".format(l=self._y_axis_name)
        elif models and custom_title is None and len(self.good_model_fits) > 1:
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

        # If the user has passed radii to plot, then we plot them
        for r_name in draw_rads:
            d_rad = (draw_rads[r_name] / x_norm).value
            main_ax.axvline(d_rad, linestyle='dashed', color='black')
            main_ax.annotate(r_name, (d_rad * 1.01, 0.9), rotation=90, verticalalignment='center',
                             color='black', fontsize=14, xycoords=('data', 'axes fraction'))

        # Use the axis limits quite a lot in these next bits, so read them out into variables
        x_axis_lims = main_ax.get_xlim()
        y_axis_lims = main_ax.get_ylim()

        # If the user has passed extra values to plot, then we plot them
        for v_name in draw_vals:
            d_val = (draw_vals[v_name] / y_norm).value
            if draw_vals[v_name].isscalar:
                main_ax.axhline(d_val, linestyle='dashed', color=data_colour, alpha=0.8,
                                label=v_name)
            elif len(d_val) == 2:
                main_ax.axhline(d_val[0], linestyle='dashed', color=data_colour, alpha=0.8,
                                label=v_name)
                main_ax.fill_between(x_axis_lims, d_val[0]-d_val[1], d_val[0]+d_val[1], color=data_colour, alpha=0.5)
            elif len(d_val) == 3:
                main_ax.axhline(d_val[0], linestyle='dashed', color=data_colour, alpha=0.8,
                                label=v_name)
                main_ax.fill_between(x_axis_lims, d_val[0]-d_val[1], d_val[0]+d_val[2], color=data_colour, alpha=0.5)

            main_ax.set_xlim(x_axis_lims)

        # If the user wants a legend to be shown, then we create one
        if show_legend:
            # In this case the user wants matplotlib to decide where best to put the legend
            if auto_legend:
                main_leg = main_ax.legend(loc="best", ncol=1)
            # Otherwise the legend is placed outside the main axis, on the right hand side.
            else:
                main_leg = main_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1, borderaxespad=0)
            # This makes sure legend keys are shown, even if the data is hidden
            for leg_line in main_leg.legend_handles:
                leg_line.set_visible(True)

        # If this variable is not None it means that the user has specified their own formatters, and these will
        #  now override the automatic formatting

        if axis_formatters is not None:
            # We specify which axes object needs formatters applied, depends on whether the residual ax is being
            #  shown or not - slightly dodgy way of checking for a local declaration of the residual axes
            if show_residual_ax and 'res_ax' in locals():
                form_ax = res_ax
            else:
                form_ax = main_ax
            # Checks for and uses formatters that the user may have specified for the plot
            if 'xminor' in axis_formatters:
                form_ax.xaxis.set_minor_formatter(axis_formatters['xminor'])
            if 'xmajor' in axis_formatters:
                form_ax.xaxis.set_major_formatter(axis_formatters['xmajor'])

            # The y-axis formatters are applied to the main axis
            if 'yminor' in axis_formatters:
                main_ax.yaxis.set_minor_formatter(axis_formatters['yminor'])
            if 'ymajor' in axis_formatters:
                main_ax.yaxis.set_major_formatter(axis_formatters['ymajor'])

        else:
            # This dynamically changes how tick labels are formatted depending on the values displayed
            if max(x_axis_lims) < 100 and not models and min(x_axis_lims) > 0.1:
                main_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
                main_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
            elif max(x_axis_lims) < 100 and models and show_residual_ax:
                res_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
                res_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

            if max(y_axis_lims) < 100 and min(y_axis_lims) > 0.1:
                main_ax.yaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
                main_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
            elif max(y_axis_lims) < 100 and min(y_axis_lims) <= 0.1:
                main_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

        if models and show_residual_ax:
            return main_ax, res_ax
        else:
            return main_ax, None

    def view(self, figsize=(10, 7), xscale: str = "log", yscale:str = "log", xlim: tuple = None, ylim: tuple = None,
             models: bool = True, back_sub: bool = True, just_models: bool = False, custom_title: str = None,
             draw_rads: dict = {}, x_norm: Union[bool, Quantity] = False, y_norm: Union[bool, Quantity] = False,
             x_label: str = None, y_label: str = None, data_colour: str = 'black',
             model_colour: Union[str, List[str]] = 'seagreen', show_legend: bool = True, show_residual_ax: bool = True,
             draw_vals: dict = {}, auto_legend: bool = True, joined_points: bool = False, axis_formatters: dict = None):
        """
        A method that allows us to view the current profile, as well as any models that have been fitted to it,
        and their residuals. The models are plotted by generating random model realisations from the parameter
        distributions, then plotting the median values, with 1sigma confidence limits.

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
        :param bool x_norm: Controls whether the x-axis of the profile is normalised by another value, the default is
            False, in which case no normalisation is applied. If it is set to True then it will attempt to use the
            internal normalisation value (which can be set with the x_norm property), and if a quantity is passed it
            will attempt to normalise using that.
        :param bool y_norm: Controls whether the y-axis of the profile is normalised by another value, the default is
            False, in which case no normalisation is applied. If it is set to True then it will attempt to use the
            internal normalisation value (which can be set with the y_norm property), and if a quantity is passed it
            will attempt to normalise using that.
        :param str x_label: Custom label for the x-axis (excluding units, which will be added automatically).
        :param str y_label: Custom label for the y-axis (excluding units, which will be added automatically).
        :param str data_colour: Used to set the colour of the data points.
        :param str/List[str] model_colour: The matplotlib colour(s) that should be used for plotted model fits (if
            applicable). Either a single colour name, or a list of colour names, may be passed depending on the number
            of models that are being plotted - if there are multiple models, and a single colour is passed, the plot
            will revert to the default matplotlib colour cycler. If a list is passed, those colours will be cycled
            through instead (if there are insufficient entries for the number of models an error will be raised). The
            default value is 'seagreen'.
        :param bool show_legend: Whether the legend should be displayed or not. Default is True.
        :param bool show_residual_ax: Controls whether a lower axis showing the residuals between data and
            model (if a model is fitted and being shown) is displayed. Default is True.
        :param dict draw_vals: A dictionary of extra y-values (as astropy quantities) to draw onto the plot, where the
            dictionary key they are stored under is what they will be labelled (keys can be LaTeX
            formatted); e.g. {r'$T_{\rm{X,500}}$': Quantity(6, 'keV')}. Quantities with uncertainties may also be
            passed, and the error regions will be shaded; e.g. {r'$T_{\rm{X,500}}$': Quantity([6, 0.2, 0.3], 'keV')},
            where 0.2 is the negative error, and 0.3 is the positive error.
        :param bool auto_legend: If True, and show_legend has also been set to True, then the 'best' legend location
            will be defined by matplotlib, otherwise, if False, the legend will be added to the right hand side of the
            plot outside the main axes.
        :param bool joined_points: If True, the data in the profile will be plotted as a line, rather than points, as
            will any uncertainty regions.
        :param dict axis_formatters: A dictionary of formatters that can be applied to the profile plot. The keys
            can have the following values; 'xmajor', 'xminor', 'ymajor', and 'yminor'. The values associated with the
            keys should be instantiated matplotlib formatters.
        """
        # Setting up figure for the plot
        fig = plt.figure(figsize=figsize)
        # Grabbing the axis object and making sure the ticks are set up how we want
        main_ax = plt.gca()

        main_ax, res_ax = self.get_view(fig, main_ax, xscale, yscale, xlim, ylim, models, back_sub, just_models,
                                        custom_title, draw_rads, x_norm, y_norm, x_label, y_label, data_colour,
                                        model_colour, show_legend, show_residual_ax, draw_vals, auto_legend,
                                        joined_points, axis_formatters)

        # plt.tight_layout()
        plt.show()

        # Wipe the figure
        plt.close("all")

    def save_view(self, save_path: str, figsize=(10, 7), xscale: str = "log", yscale:str = "log", xlim: tuple = None,
                  ylim: tuple = None, models: bool = True, back_sub: bool = True, just_models: bool = False,
                  custom_title: str = None, draw_rads: dict = {}, x_norm: Union[bool, Quantity] = False,
                  y_norm: Union[bool, Quantity] = False, x_label: str = None, y_label: str = None,
                  data_colour: str = 'black', model_colour: Union[str, List[str]] = 'seagreen',
                  show_legend: bool = True, show_residual_ax: bool = True, draw_vals: dict = {},
                  auto_legend: bool = True, joined_points: bool = False, axis_formatters: dict = None):
        """
        A method that allows us to save a view of the current profile, as well as any models that have been
        fitted to it, and their residuals. The models are plotted by generating random model realisations from
        the parameter distributions, then plotting the median values, with 1sigma confidence limits.

        This method will not display a figure, just save it at the supplied save_path.

        :param str save_path: The path (including file name) where you wish to save the profile view.
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
        :param bool x_norm: Controls whether the x-axis of the profile is normalised by another value, the default is
            False, in which case no normalisation is applied. If it is set to True then it will attempt to use the
            internal normalisation value (which can be set with the x_norm property), and if a quantity is passed it
            will attempt to normalise using that.
        :param bool y_norm: Controls whether the y-axis of the profile is normalised by another value, the default is
            False, in which case no normalisation is applied. If it is set to True then it will attempt to use the
            internal normalisation value (which can be set with the y_norm property), and if a quantity is passed it
            will attempt to normalise using that.
        :param str x_label: Custom label for the x-axis (excluding units, which will be added automatically).
        :param str y_label: Custom label for the y-axis (excluding units, which will be added automatically).
        :param str data_colour: Used to set the colour of the data points.
        :param str/List[str] model_colour: The matplotlib colour(s) that should be used for plotted model fits (if
            applicable). Either a single colour name, or a list of colour names, may be passed depending on the number
            of models that are being plotted - if there are multiple models, and a single colour is passed, the plot
            will revert to the default matplotlib colour cycler. If a list is passed, those colours will be cycled
            through instead (if there are insufficient entries for the number of models an error will be raised). The
            default value is 'seagreen'.
        :param bool show_legend: Whether the legend should be displayed or not. Default is True.
        :param bool show_residual_ax: Controls whether a lower axis showing the residuals between data and
            model (if a model is fitted and being shown) is displayed. Default is True.
        :param dict draw_vals: A dictionary of extra y-values (as astropy quantities) to draw onto the plot, where the
            dictionary key they are stored under is what they will be labelled (keys can be LaTeX
            formatted); e.g. {r'$T_{\rm{X,500}}$': Quantity(6, 'keV')}. Quantities with uncertainties may also be
            passed, and the error regions will be shaded; e.g. {r'$T_{\rm{X,500}}$': Quantity([6, 0.2, 0.3], 'keV')},
            where 0.2 is the negative error, and 0.3 is the positive error.
        :param bool auto_legend: If True, and show_legend has also been set to True, then the 'best' legend location
            will be defined by matplotlib, otherwise, if False, the legend will be added to the right hand side of the
            plot outside the main axes.
        :param bool joined_points: If True, the data in the profile will be plotted as a line, rather than points, as
            will any uncertainty regions.
        :param dict axis_formatters: A dictionary of formatters that can be applied to the profile plot. The keys
            can have the following values; 'xmajor', 'xminor', 'ymajor', and 'yminor'. The values associated with the
            keys should be instantiated matplotlib formatters.
        """
        # Setting up figure for the plot
        fig = plt.figure(figsize=figsize)
        # Grabbing the axis object and making sure the ticks are set up how we want
        main_ax = plt.gca()

        main_ax, res_ax = self.get_view(fig, main_ax, xscale, yscale, xlim, ylim, models, back_sub, just_models,
                                        custom_title, draw_rads, x_norm, y_norm, x_label, y_label, data_colour,
                                        model_colour, show_legend, show_residual_ax, draw_vals, auto_legend,
                                        joined_points, axis_formatters)

        fig.savefig(save_path, bbox_inches='tight')

        # Wipe the figure
        plt.close("all")

    def save(self, save_path: str = None):
        """
        This method pickles and saves the profile object. This will be called automatically when the profile
        is initialised, and when changes are made to the profile (such as when a model is fitted). The save
        file is a pickled version of this object.

        :param str save_path: The path where this profile should be saved. By default this is None, which means
            this method will use the save_path attribute of the profile.
        """
        #  Checks to see if the user has supplied their own custom save path.
        if save_path is None and self.save_path is not None:
            save_path = self.save_path
        elif save_path is None and self.save_path is None:
            raise TypeError("Base profiles cannot be saved")

        # Pickles and saves this profile instance.
        with open(save_path, 'wb') as picklo:
            pickle.dump(self, picklo)

    @property
    def save_path(self) -> str:
        """
        Property getter that assembles the default XGA save path of this profile. The file name contains
        limited information; the type of profile, the source name, and a random integer.

        :return: The default XGA save path for this profile.
        :rtype: str
        """
        if self._save_path is None and self._prof_type != "base":
            temp_path = OUTPUT + "profiles/{sn}/{pt}_{sn}_{id}.xga"
            rand_prof_id = randint(0, int(1e+8))
            while os.path.exists(temp_path.format(pt=self.type, sn=self.src_name, id=rand_prof_id)):
                rand_prof_id = randint(0, int(1e+8))
            self._save_path = temp_path.format(pt=self.type, sn=self.src_name, id=rand_prof_id)

        return self._save_path

    @property
    def auto_save(self) -> bool:
        """
        Whether the profile will automatically save itself at any point (such as after successful model fits).

        :return: Boolean flag describing whether auto-save is turned on.
        :rtype: bool
        """
        return self._auto_save

    @auto_save.setter
    def auto_save(self, new_val: bool):
        """
        Whether the profile will automatically save itself at any point (such as after successful model fits).

        :param bool new_val: Boolean flag describing whether auto-save is turned on.
        """
        if isinstance(new_val, bool):
            self._auto_save = new_val
        else:
            raise TypeError("The 'auto_save' property must be set with a boolean variable.")

    @property
    def good_model_fits(self) -> List:
        """
        A list of the names of models that have been successfully fitted to the profile.

        :return: A list of model names.
        :rtype: Dict
        """
        models = []
        for method in self._good_model_fits:
            for model in self._good_model_fits[method]:
                if model not in models:
                    models.append(model)

        return models

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
    def fit_radii(self) -> Quantity:
        """
        This property gives the user a sanitised set of radii that is safe to use for fitting to XGA models, by
        which I mean if the first element is zero, then it will be replaced by a value slightly above zero that
        won't cause divide by zeros in the fit process.

        If the radius units are convertible to kpc then the zero value will be set to the equivalent of 1kpc, if
        they have pixel units then it will be set to one pixel, and if they are equivalent to degrees then it will
        be set to 1e−5 degrees. The value for degrees is loosely based on the value of 1kpc at a redshift of 1.

        :return: A Quantity with a set of radii that are 'safe' for fitting
        :rtype: Quantity
        """
        safe_rads = self._radii.copy()
        if safe_rads[0] == 0 and self.radii_unit.is_equivalent('kpc'):
            safe_rads[0] = Quantity(1, 'kpc').to(self.radii_unit)
        elif safe_rads[0] == 0 and self.radii_unit.is_equivalent('pix'):
            safe_rads[0] = Quantity(1, 'pix')
        elif safe_rads[0] == 0 and self.radii_unit.is_equivalent('deg'):
            safe_rads[0] = Quantity(1e-5, 'deg')

        return safe_rads

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
        if self.auto_save:
            # This method means that a change has happened to the model, so it should be re-saved
            self.save()

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
    def spec_fit_conf(self) -> str:
        """
        If this profile was generated from an annular spectrum, this property provides the fit-configuration key of
        the spectral fits that provided the properties used to build it.

        :return: The spectral fit-configuration key. If the spectral fit configuration key was never set, the
            return will be None.
        :rtype: str
        """
        return self._spec_fit_conf

    @property
    def spec_model(self) -> str:
        """
        If this profile was generated from an annular spectrum, this property provides the name of the model
        that was fit to the spectra in order to measure the properties used to build it.

        :return: The spectral model name. If the spectral model name was never set, the return will be None.
        :rtype: str
        """
        return self._spec_model

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
        if self.auto_save:
            # This method means that a change has happened to the model, so it should be re-saved
            self.save()

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
        if self.auto_save:
            # This method means that a change has happened to the model, so it should be re-saved
            self.save()

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
        if self.auto_save:
            # This method means that a change has happened to the model, so it should be re-saved
            self.save()

    @property
    def fit_options(self) -> List[str]:
        """
        Returns the supported fit options for XGA profiles.

        :return: List of supported fit options.
        :rtype: List[str]
        """
        return self._fit_methods

    @property
    def nice_fit_names(self) -> List[str]:
        """
        Returns nicer looking names for the supported fit options of XGA profiles.

        :return: List of nice fit options.
        :rtype: List[str]
        """
        return self._nice_fit_methods

    @property
    def outer_radius(self) -> Quantity:
        """
        Property that returns the outer radius used for the generation of this profile.

        :return: The outer radius used in the generation of the profile.
        :rtype: Quantity
        """
        return self._outer_rad

    @property
    def custom_aggregate_label(self) -> str:
        """
        This property is a label that should be used in place of the source name associated with this profile when
        plotting multiple profiles on one axis through an aggregate profile instance.

        :return: The custom label, default is None.
        :rtype: str
        """
        return self._custom_agg_label

    @custom_aggregate_label.setter
    def custom_aggregate_label(self, new_val: str):
        """
        Setter for the custom_aggregate_label property.

        :param str new_val: The new label.
        """
        if isinstance(new_val, str) or new_val is None:
            self._custom_agg_label = new_val
        else:
            raise TypeError("'custom_aggregate_label' must be a string, or None.")

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

    :param list profiles: A list of profile objects (of the same type) to include in this aggregate profile.
    """
    def __init__(self, profiles: List[BaseProfile1D]):
        """
        The init for the BaseAggregateProfile1D class.
        """

        # This checks that all profiles have the same x units - we used to explicitly check for Python instance
        #  type, but actually we do want profiles to be plottable on the same axis if they have the same units
        x_units = [p.radii_unit.to_string() for p in profiles]
        if len(set(x_units)) != 1:
            raise TypeError("All component profiles must have the same radii units.")

        # THis checks that they all have the same y units.
        y_units = [p.values_unit.to_string() for p in profiles]
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
        lo_bounds = [p.energy_bounds[0] for p in profiles]
        hi_bounds = [p.energy_bounds[1] for p in profiles]

        if len(set(lo_bounds)) != 1 or len(set(hi_bounds)) != 1:
            raise ValueError("All component profiles must have been generate from the same energy range,"
                             " otherwise they aren't directly comparable.")

        self._profiles = profiles
        self._radii_unit = x_units[0]
        self._values_unit = y_units[0]
        # Not doing a check that all the prof types are the same, because that should be included in the
        #  type check on the first line of this init
        self._prof_type = profiles[0].type.split("_profile")[0]
        self._energy_bounds = profiles[0].energy_bounds

        # We set the y-axis name attribute which is now expected by the plotting function, just grab it from the
        #  first component because we've already checked that they're all the same type
        self._y_axis_name = self._profiles[0].y_axis_label

        # Here I grab all the x_norm and y_norms, so that the view method of this aggregate profile can also
        #  apply normalisation to the separate profiles if the user wants
        self._x_norms = [p.x_norm for p in self._profiles]
        self._y_norms = [p.y_norm for p in self._profiles]

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

    @property
    def x_norms(self) -> List[Quantity]:
        """
        The collated x normalisation values for the constituent profiles of this aggregate profile.

        :return: A list of astropy quantities which represent the x-normalisations of the different profiles.
        :rtype: List[Quantity]
        """
        return self._x_norms

    @x_norms.setter
    def x_norms(self, new_vals: List[Quantity]):
        """
        Setter for the collated x normalisation values for the constituent profiles of this aggregate profile.

        :param List[Quantity] new_vals: A list of astropy quantities that the profile's x-axis values are
            to be normalised by, there must be one entry for each profile.
        """
        if len(new_vals) == len(self._x_norms):
            self._x_norms = new_vals
        else:
            raise ValueError("The new list passed for x-axis normalisations must be the same length"
                             " as the original.")

    @property
    def y_norms(self) -> List[Quantity]:
        """
        The collated y normalisation values for the constituent profiles of this aggregate profile.

        :return: A list of astropy quantities which represent the y-normalisations of the different profiles.
        :rtype: List[Quantity]
        """
        return self._y_norms

    @y_norms.setter
    def y_norms(self, new_vals: List[Quantity]):
        """
        Setter for the collated y normalisation values for the constituent profiles of this aggregate profile.

        :param List[Quantity] new_vals: A list of astropy quantities that the profile's y-axis values are
            to be normalised by, there must be one entry for each profile.
        """
        if len(new_vals) == len(self._y_norms):
            self._y_norms = new_vals
        else:
            raise ValueError("The new list passed for y-axis normalisations must be the same length"
                             " as the original.")

    def view(self, figsize: Tuple = (10, 7), xscale: str = "log", yscale: str = "log", xlim: Tuple = None,
             ylim: Tuple = None, model: str = None, back_sub: bool = True, show_legend: bool = True,
             just_model: bool = False, custom_title: str = None, draw_rads: dict = {}, x_norm: bool = False,
             y_norm: bool = False, x_label: str = None, y_label: str = None, save_path: str = None,
             draw_vals: dict = {}, auto_legend: bool = True, axis_formatters: dict = None,
             show_residual_ax: bool = True, joined_points: bool = False):
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
        :param bool show_legend: Should a legend with source names be added to the figure, default is True.
        :param bool just_model: Should only the models, not the data, be plotted. Default is False.
        :param str custom_title: A plot title to replace the automatically generated title, default is None.
        :param dict draw_rads: A dictionary of extra radii (as astropy Quantities) to draw onto the plot, where
            the dictionary key they are stored under is what they will be labelled.
            e.g. ({'r500': Quantity(), 'r200': Quantity()}. If normalise_x option is also used, and the x-norm values
            are not the same for each profile, then draw_rads will be disabled.
        :param bool x_norm: Should the x-axis values be normalised with the x_norm value passed on the
            definition of the constituent profile objects.
        :param bool y_norm: Should the y-axis values be normalised with the y_norm value passed on the
            definition of the constituent profile objects.
        :param str x_label: Custom label for the x-axis (excluding units, which will be added automatically).
        :param str y_label: Custom label for the y-axis (excluding units, which will be added automatically).
        :param str save_path: The path where the figure produced by this method should be saved. Default is None, in
            which case the figure will not be saved.
        :param dict draw_vals: A dictionary of extra y-values (as astropy quantities) to draw onto the plot, where the
                    dictionary key they are stored under is what they will be labelled (keys can be LaTeX formatted);
                    e.g. {r'$T_{\rm{X,500}}$': Quantity(6, 'keV')}. Quantities with uncertainties may also be
                    passed, and the error regions will be shaded; e.g. {r'$T_{\rm{X,500}}$':
                    Quantity([6, 0.2, 0.3], 'keV')}, where 0.2 is the negative error, and 0.3 is the positive error.
                    Finally, plotting colour may be specified by setting the value to a list, with the first entry
                    being the quantity, and the second being a colour;
                    e.g. {r'$T_{\rm{X,500}}$': [Quantity([6, 0.2, 0.3], 'keV'), 'tab:blue']}.
        :param bool auto_legend: If True, and show_legend has also been set to True, then the 'best' legend location
            will be defined by matplotlib, otherwise, if False, the legend will be added to the right hand side of the
            plot outside the main axes.
        :param dict axis_formatters: A dictionary of formatters that can be applied to the profile plot. The keys
            can have the following values; 'xmajor', 'xminor', 'ymajor', and 'yminor'. The values associated with the
            keys should be instantiated matplotlib formatters.
        :param bool show_residual_ax: Controls whether a lower axis showing the residuals between data and
            model (if a model is fitted and being shown) is displayed. Default is True.
        :param bool joined_points: If True, the data in the profiles will be plotted as a line, rather than points, as
            will any uncertainty regions.
        """

        # Checks that any extra radii that have been passed are the correct units (i.e. the same as the radius units
        #  used in this profile)
        if not all([r.unit == self.radii_unit for r in draw_rads.values()]):
            raise UnitConversionError("All radii in draw_rad have to be in the same units as this profile, "
                                      "{}".format(self.radii_unit.to_string()))

        # Checks that any entries in draw_vals are either quantities or lists
        if not all([isinstance(v, (Quantity, list)) for v in draw_vals.values()]):
            raise TypeError("All values in draw_vals must either be an astropy quantity, or list with the first"
                            "element being the astropy quantity, and the second being a string matplotlib colour.")

        # Checks that any extra y-axis values that have been passed are the correct units (i.e. the same as the
        #  y-value units used in this profile)
        if not all([v.unit == self.values_unit if isinstance(v, Quantity) else v[0].unit == self.values_unit
                    for v in draw_vals.values()]):
            raise UnitConversionError("All values in draw_vals have to be in the same units as this profile, "
                                      "{}".format(self.values_unit.to_string()))

        if axis_formatters is not None and \
                not all([k in ['xmajor', 'xminor', 'ymajor', 'yminor'] for k in axis_formatters.keys()]):
            raise KeyError("The axis_formatters dictionary may only contain the following keys; xmajor, xminor, "
                           "ymajor, and yminor.")

        # Set up the x normalisation and y normalisation variables
        if x_norm:
            x_norms = self.x_norms
        else:
            x_norms = [Quantity(1, '') for n in self.x_norms]

        if y_norm:
            y_norms = self.y_norms
        else:
            y_norms = [Quantity(1, '') for n in self.y_norms]

        # Need to make sure that draw_rads, if set, is compatible with the normalisations. The problem is that
        #  if the profiles all have different normalisations then the draw_rads values can't be normalised
        if len(draw_rads) != 0 and len(set(x_norms)) != 1:
            draw_rads = {}
        # Same deal with draw_vals, different y-norms would screw up plotting draw_vals
        if len(draw_vals) != 0 and len(set(y_norms)) != 1:
            draw_vals = {}

        # Setting up figure for the plot
        fig = plt.figure(figsize=figsize)
        # Grabbing the axis object and making sure the ticks are set up how we want
        main_ax = plt.gca()
        main_ax.minorticks_on()
        if model is not None and show_residual_ax:
            # This sets up an axis for the residuals to be plotted on, if model plotting is enabled
            res_ax = fig.add_axes((0.125, -0.075, 0.775, 0.2))
            res_ax.minorticks_on()
            res_ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
            # Adds a zero line for reference, as its ideally where residuals would be
            res_ax.axhline(0.0, color="black")
        # Setting some aesthetic parameters for the main plotting axis
        main_ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        # Cycles through the component profiles of this aggregate profile, plotting them all
        for p_ind, p in enumerate(self._profiles):
            if p.obs_id != 'combined':
                p_name = p.src_name + " {o}-{i}".format(o=p.obs_id, i=p.instrument.upper())
            else:
                p_name = p.src_name

            if p.type == "brightness_profile" and p.psf_corrected and p.custom_aggregate_label is None:
                leg_label = p_name + " PSF Corrected"
            elif p.custom_aggregate_label is None:
                leg_label = p_name
            else:
                leg_label = p.custom_aggregate_label

            # This subtracts the background if the user wants a background subtracted plot
            plot_y_vals = p.values.copy()
            if back_sub:
                plot_y_vals -= p.background

            rad_vals = p.radii.copy()
            plot_y_vals /= y_norms[p_ind]
            rad_vals /= x_norms[p_ind]

            # Now the actual plotting of the data
            if p.radii_err is not None and p.values_err is None and not joined_points:
                x_errs = (p.radii_err.copy() / x_norms[p_ind]).value
                line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, xerr=x_errs, fmt="x", capsize=2,
                                        label=leg_label)
            elif p.radii_err is None and p.values_err is not None and not joined_points:
                y_errs = (p.values_err.copy() / y_norms[p_ind]).value
                line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, yerr=y_errs, fmt="x", capsize=2,
                                        label=leg_label)
            elif p.radii_err is not None and p.values_err is not None and not joined_points:
                x_errs = (p.radii_err.copy() / x_norms[p_ind]).value
                y_errs = (p.values_err.copy() / y_norms[p_ind]).value
                line = main_ax.errorbar(rad_vals.value, plot_y_vals.value, xerr=x_errs, yerr=y_errs, fmt="x",
                                        capsize=2, label=leg_label)
            elif joined_points:
                line = main_ax.plot(rad_vals.value, plot_y_vals.value, label=leg_label)
                if p.values_err is not None:
                    y_errs = (p.values_err.copy() / y_norms[p_ind]).value
                    main_ax.fill_between(rad_vals.value, plot_y_vals.value - y_errs, plot_y_vals.value + y_errs,
                                         linestyle='dashdot', alpha=0.7)
            else:
                line = main_ax.plot(rad_vals.value, plot_y_vals.value, 'x', label=leg_label)

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
            if model is not None:
                # I've put them in this order because I would prefer mcmc over odr, and odr over curve_fit
                for method in ['mcmc', 'odr', 'curve_fit']:
                    try:
                        model_obj = p.get_model_fit(model, method)

                        # This makes sure that, if there are radius 'uncertainties' - the models are created so they
                        #  plot all the way from the leftmost errorbar, to the end of the rightmost
                        if p.radii_err is not None:
                            lo_rad = (p.fit_radii - p.radii_err).min()
                            hi_rad = (p.fit_radii + p.radii_err).max()
                        else:
                            lo_rad = p.fit_radii.min()
                            hi_rad = p.fit_radii.max()
                        mod_rads = np.linspace(lo_rad, hi_rad, 500)
                        mod_reals = model_obj.get_realisations(mod_rads)
                        median_model = np.nanpercentile(mod_reals, 50, axis=1)

                        upper_model = np.nanpercentile(mod_reals, 84.1, axis=1)
                        lower_model = np.nanpercentile(mod_reals, 15.9, axis=1)

                        colour = line[0].get_color()

                        mod_lab = model_obj.publication_name + " - {}".format(p.nice_fit_names[method])
                        mod_line = main_ax.plot(mod_rads.value / x_norms[p_ind].value,
                                                median_model.value/y_norms[p_ind], color=colour)

                        main_ax.fill_between(mod_rads.value / x_norms[p_ind].value,
                                             lower_model.value / y_norms[p_ind].value,
                                             upper_model.value / y_norms[p_ind].value, alpha=0.7, interpolate=True,
                                             where=upper_model.value >= lower_model.value, facecolor=colour)
                        main_ax.plot(mod_rads.value / x_norms[p_ind].value, lower_model.value / y_norms[p_ind].value,
                                     color=colour, linestyle="dashed")
                        main_ax.plot(mod_rads.value / x_norms[p_ind].value, upper_model.value / y_norms[p_ind].value,
                                     color=colour, linestyle="dashed")

                        if show_residual_ax:
                            # This calculates and plots the residuals between the model and the data on the extra
                            #  axis we added near the beginning of this method
                            res = np.nanpercentile(model_obj.get_realisations(p.radii), 50, axis=1) - \
                                  (plot_y_vals * y_norms[p_ind])
                            res_ax.plot(rad_vals.value, res.value, 'D', color=colour)

                        break
                    except ModelNotAssociatedError:
                        pass

        # Parsing the astropy units so that if they are double height then the square brackets will adjust size
        x_unit = r"$\left[" + rad_vals.unit.to_string("latex").strip("$") + r"\right]$"
        y_unit = r"$\left[" + plot_y_vals.unit.to_string("latex").strip("$") + r"\right]$"

        if x_label is None:
            # Setting the main plot's x label
            main_ax.set_xlabel("Radius {}".format(x_unit), fontsize=13)
        else:
            main_ax.set_xlabel(x_label + " {}".format(x_unit), fontsize=13)

        if y_label is None and (not self._back_avail or not back_sub):
            main_ax.set_ylabel(r"{l} {u}".format(l=self._y_axis_name, u=y_unit), fontsize=13)
        elif y_label is None:
            # If background has been subtracted it will be mentioned in the y axis label
            main_ax.set_ylabel(r"Background Subtracted {l} {u}".format(l=self._y_axis_name, u=y_unit), fontsize=13)
        elif y_label is not None:
            main_ax.set_ylabel(y_label + ' {}'.format(y_unit), fontsize=13)

        # If the user has manually set limits then we can use them, only on the main axis because
        #  we grab those limits from the axes object for the residual axis later
        if xlim is not None:
            main_ax.set_xlim(xlim)
        if ylim is not None:
            main_ax.set_ylim(ylim)

        # Setup the scale that the user wants to see, again on the main axis
        main_ax.set_xscale(xscale)
        main_ax.set_yscale(yscale)
        if model is not None and show_residual_ax:
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

        # If the user has passed radii to plot, then we plot them
        for r_name in draw_rads:
            d_rad = (draw_rads[r_name] / x_norms[0]).value
            main_ax.axvline(d_rad, linestyle='dashed', color='black')
            main_ax.annotate(r_name, (d_rad * 1.01, 0.9), rotation=90, verticalalignment='center',
                             color='black', fontsize=14, xycoords=('data', 'axes fraction'))

        # Reads out the axis limits of the plot thus far
        x_axis_lims = main_ax.get_xlim()
        # If the user has passed extra values to plot, then we plot them
        for v_name in draw_vals:
            if isinstance(draw_vals[v_name], Quantity):
                d_val = draw_vals[v_name].value
                cur_col = None
                is_scalar = draw_vals[v_name].isscalar
            else:
                d_val = draw_vals[v_name][0].value
                cur_col = draw_vals[v_name][1]
                is_scalar = draw_vals[v_name][0].isscalar

            if is_scalar:
                main_ax.axhline(d_val, linestyle='dashed', color=cur_col, alpha=0.8,
                                label=v_name)
            elif len(d_val) == 2:
                main_ax.axhline(d_val[0], linestyle='dashed', color=cur_col, alpha=0.8,
                                label=v_name)
                main_ax.fill_between(x_axis_lims, d_val[0] - d_val[1], d_val[0] + d_val[1], color=cur_col,
                                     alpha=0.5)
            elif len(d_val) == 3:
                main_ax.axhline(d_val[0], linestyle='dashed', color=cur_col, alpha=0.8,
                                label=v_name)
                main_ax.fill_between(x_axis_lims, d_val[0] - d_val[1], d_val[0] + d_val[2], color=cur_col,
                                     alpha=0.5)

            main_ax.set_xlim(x_axis_lims)

        # Adds a legend with source names to the side if the user requested it
        # I let the user decide because there could be quite a few names in it and it could get messy
        if show_legend:
            # This allows the user to exercise some control over where the legend is placed.
            if auto_legend:
                main_leg = main_ax.legend(loc="best", ncol=1)
            else:
                main_leg = main_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1, borderaxespad=0)
            # This makes sure legend keys are shown, even if the data is hidden - not we use legend_handles here
            #  rather than get_lines() because errorbars are not Line2D instances, but collections
            for leg_line in main_leg.legend_handles:
                leg_line.set_visible(True)

        # We specify which axes object needs formatters applied, depends on whether the residual ax is being
        #  shown or not - slightly dodgy way of checking for a local declaration of the residual axes
        if show_residual_ax and 'res_ax' in locals():
            form_ax = res_ax
        else:
            form_ax = main_ax
        # Checks for and uses formatters that the user may have specified for the plot
        if axis_formatters is not None and 'xminor' in axis_formatters:
            form_ax.xaxis.set_minor_formatter(axis_formatters['xminor'])
        if axis_formatters is not None and 'xmajor' in axis_formatters:
            form_ax.xaxis.set_major_formatter(axis_formatters['xmajor'])

        # The y-axis formatters are applied to the main axis
        if axis_formatters is not None and 'yminor' in axis_formatters:
            main_ax.yaxis.set_minor_formatter(axis_formatters['yminor'])
        if axis_formatters is not None and 'ymajor' in axis_formatters:
            main_ax.yaxis.set_major_formatter(axis_formatters['ymajor'])

        # If the user passed a save_path value, then we assume they want to save the figure
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')

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
