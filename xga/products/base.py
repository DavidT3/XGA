#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 15/07/2020, 00:06. Copyright (c) David J Turner


import os
from typing import Tuple, List, Dict

from astropy.units import Quantity

from xga.exceptions import SASGenerationError, UnknownCommandlineError
from xga.utils import SASERROR_LIST, SASWARNING_LIST


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
        # So this flag indicates whether we think this data product can be used for analysis
        self._usable = True
        if os.path.exists(path):
            self._path = path
        else:
            self._path = None
            self._usable = False
        # Saving this in attributes for future reference
        self.unprocessed_stdout = stdout_str
        self.unprocessed_stderr = stderr_str
        self._sas_error, self._sas_warn, self._other_error = self.parse_stderr()
        self._obs_id = obs_id
        self._inst = instrument
        self.og_cmd = gen_cmd
        self._energy_bounds = (None, None)
        self._prod_type = None
        self._obj_name = None

        self.raise_errors(raise_properly)

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
        self._path = prod_path

    def parse_stderr(self) -> Tuple[List[Dict], List[Dict], List]:
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
                        sas_err_match = [sas_err for sas_err in SASERROR_LIST if err_ident in sas_err]
                    elif err_type == "warning":
                        # Checking to see if the error identity is in the list of SAS warnings
                        sas_err_match = [sas_err for sas_err in SASWARNING_LIST if err_ident in sas_err]

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
        parsed_sas_errs = []
        parsed_sas_warns = []
        other_err_lines = []
        # err_str being "" is ideal, hopefully means that nothing has gone wrong
        if self.unprocessed_stderr != "":
            # Errors will be added to the error summary, then raised later
            # That way if people try except the error away the object will have been constructed properly
            err_lines = self.unprocessed_stderr.split('\n')  # Fingers crossed each line is a separate error
            parsed_sas_errs, sas_err_lines = find_sas(err_lines, "error")
            parsed_sas_warns, sas_warn_lines = find_sas(err_lines, "warning")

            # These are impossible to predict the form of, so they won't be parsed
            other_err_lines = [line for line in err_lines if line not in sas_err_lines
                               and line not in sas_warn_lines and line != ""]
        return parsed_sas_errs, parsed_sas_warns, other_err_lines

    @property
    def sas_errors(self) -> List[Dict]:
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

    def raise_errors(self, raise_flag: bool):
        """
        Method to raise the errors parsed from std_err string.
        :param raise_flag: Should this function actually raise the error properly.
        """
        if raise_flag:
            # I know this won't ever get to the later errors, I might change how this works later
            for error in self._sas_error:
                self._usable = False  # Just to make sure this object isn't used if the user uses try, except
                raise SASGenerationError("{e} raised by {t} - {b}".format(e=error["name"], t=error["originator"],
                                                                          b=error["message"]))
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
        Property getter for SAS errors detected during the generation of a product.
        :return: A list of dictionaries of parsed errors.
        :rtype: List[dict]
        """
        return self._sas_error

    @property
    def warnings(self) -> List[dict]:
        """
        Property getter for SAS warnings detected during the generation of a product.
        :return: A list of dictionaries of parsed errors.
        :rtype: List[dict]
        """
        return self._sas_error

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
    def obj_name(self) -> str:
        """
        Method to return the name of the object a product is associated with. The product becomes
        aware of this once it is added to a source object.
        :return: The name of the source object this product is associated with.
        :rtype: str
        """
        return self._obj_name

    # This needs a setter, as this property only becomes not-None when the product is added to a source object.
    @obj_name.setter
    def obj_name(self, name: str):
        """
        Property setter for the obj_name attribute of a product, should only really be called by a source object,
        not by a user.
        :param str name: The name of the source object associated with this product.
        """
        self._obj_name = name


# TODO Obviously finish this, but also comment and docstring
class BaseAggregateProduct:
    def __init__(self, file_paths: list, prod_type: str, obs_id: str, instrument: str):
        files_exist = [os.path.exists(f) for f in file_paths]
        self._all_usable = True
        self._obs_id = obs_id
        self._inst = instrument
        self._prod_type = prod_type
        self._obj_name = None

        # This was originally going to create the individual products here, but realised it was
        # easier to do in subclasses
        self._component_products = {}

        # Setting up energy limits, if they're ever required
        self._energy_bounds = (None, None)

    @property
    def obj_name(self) -> str:
        """
        Method to return the name of the object a product is associated with. The product becomes
        aware of this once it is added to a source object.
        :return: The name of the source object this product is associated with.
        :rtype: str
        """
        return self._obj_name

    # This needs a setter, as this property only becomes not-None when the product is added to a source object.
    @obj_name.setter
    def obj_name(self, name: str):
        """
        Property setter for the obj_name attribute of a product, should only really be called by a source object,
        not by a user.
        :param str name: The name of the source object associated with this product.
        """
        self._obj_name = name

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
            result = list(self._component_products.values())[self._n]
            self._n += 1
            return result
        else:
            raise StopIteration






