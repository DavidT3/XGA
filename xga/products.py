#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 01/07/2020, 14:50. Copyright (c) David J Turner

import os
import warnings
from typing import Tuple, List, Dict

import numpy as np
from astropy import wcs
from astropy.units import Quantity, UnitBase, UnitsError, deg, pix
from astropy.visualization import LogStretch, MinMaxInterval, ImageNormalize
from fitsio import read, read_header, FITSHDR, FITS, hdu
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from scipy.cluster.hierarchy import fclusterdata
from scipy.signal import fftconvolve

from xga.exceptions import SASGenerationError, UnknownCommandlineError, FailedProductError, \
    ModelNotAssociatedError, ParameterNotAssociatedError, RateMapPairError
from xga.sourcetools import ang_to_rad
from xga.utils import SASERROR_LIST, SASWARNING_LIST, xmm_sky, find_all_wcs


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


class Image(BaseProduct):
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, lo_en: Quantity, hi_en: Quantity, raise_properly: bool = True):
        """
        The initialisation method for the Image class.
        :param str path: The path to where the product file SHOULD be located.
        :param str stdout_str: The stdout from calling the terminal command.
        :param str stderr_str: The stderr from calling the terminal command.
        :param str gen_cmd: The command used to generate the product.
        :param Quantity lo_en: The lower energy bound used to generate this product.
        :param Quantity hi_en: The upper energy bound used to generate this product.
        :param bool raise_properly: Shall we actually raise the errors as Python errors?
        """
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, raise_properly)
        self._shape = None
        self._wcs_radec = None
        self._wcs_xmmXY = None
        self._wcs_xmmdetXdetY = None
        self._energy_bounds = (lo_en, hi_en)
        self._prod_type = "image"
        self._data = None
        self._header = None

    def _read_on_demand(self):
        """
        Internal method to read the image associated with this Image object into memory when it is requested by
        another method. Doing it on-demand saves on wasting memory.
        """
        # Usable flag to check that nothing went wrong in the image generation
        if self.usable:
            # Not all images produced by SAS are going to be needed all the time, so they will only be read in if
            # asked for.
            self._data = read(self.path).astype("float64")
            if self._data.min() < 0:
                # This throws a non-fatal warning to let the user know there are negative pixel values,
                #  and that they're being 'corrected'
                warnings.warn("You are loading an {} with elements that are < 0, "
                              "they will be set to 0.".format(self._prod_type))
                self._data[self._data < 0] = 0
            self._header = read_header(self.path)

            # As the image must be loaded to know the shape, I've waited until here to set the _shape attribute
            self._shape = self._data.shape
            # Will actually construct an image WCS as well because why not?
            # XMM images typically have two, both useful, so we'll find all available and store them
            wcses = find_all_wcs(self._header)

            # Just iterating through and assigning to the relevant attributes
            for w in wcses:
                axes = [ax.lower() for ax in w.axis_type_names]
                if "ra" in axes and "dec" in axes:
                    if self._wcs_radec is None:
                        self._wcs_radec = w
                elif "x" in axes and "y" in axes:
                    if self._wcs_xmmXY is None:
                        self._wcs_xmmXY = w
                elif "detx" in axes and "dety" in axes:
                    if self._wcs_xmmdetXdetY is None:
                        self._wcs_xmmdetXdetY = w
                else:
                    raise ValueError("This type of WCS is not recognised!")

            # I'll only strongly require that the pixel-RADEC WCS is found
            if self._wcs_radec is None:
                raise FailedProductError("SAS has generated this image without a WCS capable of "
                                         "going from pixels to RA-DEC.")

        else:
            raise FailedProductError("SAS failed to generate this product successfully, so you cannot "
                                     "access data from it. Check the usable attribute next time")

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Property getter for the resolution of the image. Standard XGA settings will make this 512x512.
        :return: The shape of the numpy array describing the image.
        :rtype: Tuple[int, int]
        """
        # This has to be run first, to check the image is loaded, otherwise how can we know the shape?
        # This if is here rather than in the method as some other properties of this class don't need the
        # image object, just some products derived from it.
        if self._data is None:
            self._read_on_demand()
        # There will not be a setter for this property, no-one is allowed to change the shape of the image.
        return self._shape

    @property
    def data(self) -> np.ndarray:
        """
        Property getter for the actual image data, in the form of a numpy array. Doesn't include
        any of the other stuff you get in a fits image, thats found in the hdulist property.
        :return: A numpy array of shape self.shape containing the image data.
        :rtype: np.ndarray
        """
        # Calling this ensures the image object is read into memory
        if self._data is None:
            self._read_on_demand()
        return self._data

    @data.setter
    def data(self, new_im_arr: np.ndarray):
        """
        Property setter for the image data. As the fits image is loaded in read-only mode,
        this won't alter the actual file (which is what I was going for), but it does allow
        user alterations to the image data they are analysing.
        :param np.ndarray new_im_arr: The new image data.
        """
        # Calling this ensures the image object is read into memory
        if self._data is None:
            self._read_on_demand()

        # Have to make sure the input is of the right type, and the right shape
        if not isinstance(new_im_arr, np.ndarray):
            raise TypeError("You may only assign a numpy array to the data attribute.")
        elif new_im_arr.shape != self._shape:
            raise ValueError("You may only assign a numpy array to the data attribute if it "
                             "is the same shape as the original.")
        else:
            self._data = new_im_arr

    # This one doesn't get a setter, as I require this WCS to not be none in the _read_on_demand method
    @property
    def radec_wcs(self) -> wcs.WCS:
        """
        Property getter for the WCS that converts back and forth between pixel values
        and RA-DEC coordinates. This one is the only WCS guaranteed to not-None.
        :return: The WCS object for RA and DEC.
        :rtype: wcs.WCS
        """
        # If this WCS is None then _read_on_demand definitely hasn't run, this one MUST be set
        if self._wcs_radec is None:
            self._read_on_demand()
        return self._wcs_radec

    # These two however, can be none, so the user should be allowed to set add WCS-es to those
    # that don't have them. Will be good for the coordinate transform methods
    @property
    def skyxy_wcs(self):
        """
        Property getter for the WCS that converts back and forth between pixel values
        and XMM XY Sky coordinates.
        :return: The WCS object for XMM X and Y sky coordinates.
        :rtype: wcs.WCS
        """
        # Deliberately checking the radec WCS, as the skyXY WCS is allowed to be None after the
        # read_on_demand call
        if self._wcs_radec is None:
            self._read_on_demand()
        return self._wcs_xmmXY

    @skyxy_wcs.setter
    def skyxy_wcs(self, input_wcs: wcs.WCS):
        """
        Property setter for the WCS that converts back and forth between pixel values
        and XMM XY Sky coordinates. This WCS is not guaranteed to be set from the image,
        so it is possible to add your own.
        :param wcs.WCS input_wcs: The user supplied WCS object to assign to skyxy_wcs property.
        """
        if not isinstance(input_wcs, wcs.WCS):
            # Obviously don't want people assigning non-WCS objects as this will be used internally
            TypeError("Can't assign a non-WCS object to this WCS property.")
        else:
            # Fetching the WCS axis names and lowering them for comparison
            axes = [w.lower() for w in input_wcs.axis_type_names]
            # Checking if the right names are present
            if "x" not in axes or "y" not in axes:
                raise ValueError("This WCS does not have the XY axes expected for the skyxy_wcs property.")
            else:
                self._wcs_xmmXY = input_wcs

    @property
    def detxy_wcs(self):
        """
        Property getter for the WCS that converts back and forth between pixel values
        and XMM DETXY detector coordinates.
        :return: The WCS object for XMM DETX and DETY detector coordinates.
        :rtype: wcs.WCS
        """
        # Deliberately checking the radec WCS, as the DETXY WCS is allowed to be None after the
        # read_on_demand call
        if self._wcs_radec is None:
            self._read_on_demand()
        return self._wcs_xmmdetXdetY

    @detxy_wcs.setter
    def detxy_wcs(self, input_wcs: wcs.WCS):
        """
        Property setter for the WCS that converts back and forth between pixel values
        and XMM DETXY detector coordinates. This WCS is not guaranteed to be set from the image,
        so it is possible to add your own.
        :param wcs.WCS input_wcs: The user supplied WCS object to assign to detxy_wcs property.
        """
        if not isinstance(input_wcs, wcs.WCS):
            # Obviously don't want people assigning non-WCS objects as this will be used internally
            TypeError("Can't assign a non-WCS object to this WCS property.")
        else:
            # Fetching the WCS axis names and lowering them for comparison
            axes = [w.lower() for w in input_wcs.axis_type_names]
            # Checking if the right names are present
            if "detx" not in axes or "dety" not in axes:
                raise ValueError("This WCS does not have the DETX DETY axes expected for the detxy_wcs property.")
            else:
                self._wcs_xmmdetXdetY = input_wcs
    
    # This absolutely doesn't get a setter considering its the header object with all the information
    #  about the image in.
    @property
    def header(self) -> FITSHDR:
        """
        Property getter allowing access to the astropy fits header object created when the image was read in.
        :return: The header of the primary data table of the image that was read in.
        :rtype: FITSHDR
        """
        return self._header

    def coord_conv(self, coord_pair: Quantity, output_unit: UnitBase) -> Quantity:
        """
        This will use the loaded WCSes, and astropy coordinates (including custom ones defined for this module),
        to perform common coordinate conversions for this product object.
        :param coord_pair: The input coordinate quantity to convert, in units of either deg,
        pix, xmm_sky, or xmm_det (xmm_sky and xmm_det are defined for this module).
        :param output_unit: The astropy unit to convert to, can be either deg, pix, xmm_sky, or
        xmm_det (xmm_sky and xmm_det are defined for this module).
        :return: The converted coordinates.
        :rtype: Quantity
        """
        allowed_units = ["deg", "xmm_sky", "xmm_det", "pix"]
        if coord_pair.unit.is_equivalent("deg"):
            coord_pair = coord_pair.to("deg")
        input_unit = coord_pair.unit.name
        out_name = output_unit.name

        if input_unit != out_name:
            # First off do some type checking
            if not isinstance(coord_pair, Quantity):
                raise TypeError("Please pass an astropy Quantity for the coord_pair.")
            # The coordinate pair must have two elements, no more no less
            elif coord_pair.shape != (2,):
                raise ValueError("Please supply x and y coordinates in one object.")
            # I know the proper way with astropy units is to do .to() but its easier with WCS this way
            elif input_unit not in allowed_units:
                raise UnitsError("Those coordinate units are not supported by this method, "
                                 "please use one of these: {}".format(", ".join(allowed_units)))
            elif out_name not in allowed_units:
                raise UnitsError("That output unit is not supported by this method, "
                                 "please use one of these: {}".format(", ".join(allowed_units)))

            # Check for presence of the right WCS
            if (input_unit == "xmm_sky" or out_name == "xmm_sky") and self.skyxy_wcs is None:
                raise ValueError("There is no XMM Sky XY WCS associated with this product.")
            elif (input_unit == "xmm_det" or out_name == "xmm_det") and self.detxy_wcs is None:
                raise ValueError("There is no XMM Detector XY WCS associated with this product.")

            # Now to do the actual conversion, which will include checking that the correct WCS is loaded
            # These go between degrees and pixels
            if input_unit == "deg" and out_name == "pix":
                # The second argument all_world2pix defines the origin, for numpy coords it should be 0
                out_coord = Quantity(self.radec_wcs.all_world2pix(*coord_pair, 0), output_unit).astype(int)
            elif input_unit == "pix" and out_name == "deg":
                out_coord = Quantity(self.radec_wcs.all_pix2world(*coord_pair, 0), output_unit)

            # These go between degrees and XMM sky XY coordinates
            elif input_unit == "deg" and out_name == "xmm_sky":
                interim = self.radec_wcs.all_world2pix(*coord_pair, 0)
                out_coord = Quantity(self.skyxy_wcs.all_pix2world(*interim, 0), xmm_sky)
            elif input_unit == "xmm_sky" and out_name == "deg":
                interim = self.skyxy_wcs.all_world2pix(*coord_pair, 0)
                out_coord = Quantity(self.radec_wcs.all_pix2world(*interim, 0), deg)

            # These go between XMM sky XY and pixel coordinates
            elif input_unit == "xmm_sky" and out_name == "pix":
                out_coord = Quantity(self.skyxy_wcs.all_world2pix(*coord_pair, 0), output_unit).astype(int)
            elif input_unit == "pix" and out_name == "xmm_sky":
                out_coord = Quantity(self.skyxy_wcs.all_pix2world(*coord_pair, 0), output_unit)

            # These go between degrees and XMM Det XY coordinates
            elif input_unit == "deg" and out_name == "xmm_det":
                interim = self.radec_wcs.all_world2pix(*coord_pair, 0)
                out_coord = Quantity(self.detxy_wcs.all_pix2world(*interim, 0), xmm_sky)
            elif input_unit == "xmm_det" and out_name == "deg":
                interim = self.detxy_wcs.all_world2pix(*coord_pair, 0)
                out_coord = Quantity(self.radec_wcs.all_pix2world(*interim, 0), deg)

            # These go between XMM det XY and pixel coordinates
            elif input_unit == "xmm_det" and out_name == "pix":
                out_coord = Quantity(self.detxy_wcs.all_world2pix(*coord_pair, 0), output_unit).astype(int)
            elif input_unit == "pix" and out_name == "xmm_det":
                out_coord = Quantity(self.detxy_wcs.all_pix2world(*coord_pair, 0), output_unit)

            # It is possible to convert between XMM coordinates and pixel and supply coordinates
            # outside the range covered by an image, but we can at least catch the error
            if out_name == "pix" and any(coord < 0 for coord in out_coord):
                raise ValueError("Pixel coordinates cannot be less than 0.")
        elif input_unit == out_name and out_name == 'pix':
            out_coord = coord_pair.astype(int)
        else:
            out_coord = coord_pair

        return out_coord

    def view(self, cross_hair: Quantity = None, mask: np.ndarray = None):
        """
        Quick and dirty method to view this image. Absolutely no user configuration is allowed, that feature
        is for other parts of XGA. Produces an image with log-scaling, and using the colour map gnuplot2.
        :param Quantity cross_hair: An optional parameter that can be used to plot a cross hair at
        the coordinates.
        :param np.ndarray mask: Allows the user to pass a numpy mask and view the masked
        data if they so choose.
        """
        if mask is not None and mask.shape != self.data.shape:
            raise ValueError("The shape of the mask array ({0}) must be the same as that of the data array "
                             "({1}).".format(mask.shape, self.data.shape))
        elif mask is not None and mask.shape == self.data.shape:
            plot_data = self.data * mask
        else:
            plot_data = self.data

        # Create figure object
        plt.figure(figsize=(7, 6))

        # Turns off any ticks and tick labels, we don't want them in an image
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in', which='both', top=False, right=False)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # Check if this is a combined product, because if it is then ObsID and instrument are both 'combined'
        #  and it makes the title ugly
        if self.obs_id != "combined":
            # Set the title with all relevant information about the image object in it
            plt.title("Log Scaled {n} - {o}{i} {l}-{u}keV {t}".format(n=self.obj_name, o=self.obs_id,
                                                                      i=self.instrument.upper(),
                                                                      l=self._energy_bounds[0].to("keV").value,
                                                                      u=self._energy_bounds[1].to("keV").value,
                                                                      t=self.type))
        else:
            plt.title("Log Scaled {n} - Combined {l}-{u}keV {t}".format(n=self.obj_name,
                                                                   l=self._energy_bounds[0].to("keV").value,
                                                                   u=self._energy_bounds[1].to("keV").value,
                                                                   t=self.type))

        # As this is a very quick view method, users will not be offered a choice of scaling
        #  There will be a more in depth way of viewing cluster data eventually
        norm = ImageNormalize(data=plot_data, interval=MinMaxInterval(), stretch=LogStretch())
        # I normalize with a log stretch, and use gnuplot2 colormap (pretty decent for clusters imo)
        if cross_hair is not None:
            pix_coord = self.coord_conv(cross_hair, pix).value
            plt.axvline(pix_coord[0], color="white", linewidth=0.5)
            plt.axhline(pix_coord[1], color="white", linewidth=0.5)

        plt.imshow(plot_data, norm=norm, origin="lower", cmap="gnuplot2")
        plt.colorbar()
        plt.tight_layout()
        # Display the image
        plt.show()

        # Wipe the figure
        plt.close("all")


class ExpMap(Image):
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, lo_en: Quantity, hi_en: Quantity, raise_properly: bool = True):
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, lo_en, hi_en, raise_properly)
        self._prod_type = "expmap"

    def get_exp(self, at_coord: Quantity) -> float:
        """
        A simple method that converts the given coordinates to pixels, then finds the exposure time
        at those coordinates.
        :param Quantity at_coord: A pair of coordinates to find the exposure time for.
        :return: The exposure time at the supplied coordinates.
        :rtype: Quantity
        """
        pix_coord = self.coord_conv(at_coord, pix).value
        exp = self.data[pix_coord[1], pix_coord[0]]
        return Quantity(exp, "s")


class RateMap(Image):
    def __init__(self, xga_image: Image, xga_expmap: ExpMap):
        if type(xga_image) != Image or type(xga_expmap) != ExpMap:
            raise TypeError("xga_image must be an XGA Image object, and xga_expmap must be an "
                            "XGA ExpMap object.")

        if xga_image.obs_id != xga_expmap.obs_id:
            raise RateMapPairError("The ObsIDs of xga_image ({0}) and xga_expmap ({1}) "
                                   "do not match".format(xga_image.obs_id, xga_expmap.obs_id))
        elif xga_image.instrument != xga_expmap.instrument:
            raise RateMapPairError("The instruments of xga_image ({0}) and xga_expmap ({1}) "
                                   "do not match".format(xga_image.instrument, xga_expmap.instrument))
        elif xga_image.energy_bounds != xga_expmap.energy_bounds:
            raise RateMapPairError("The energy bounds of xga_image ({0}) and xga_expmap ({1}) "
                                   "do not match".format(xga_image.energy_bounds, xga_expmap.energy_bounds))

        super().__init__(xga_image.path, xga_image.obs_id, xga_image.instrument, xga_image.unprocessed_stdout,
                         xga_image.unprocessed_stderr, "", xga_image.energy_bounds[0], xga_image.energy_bounds[1])
        self._prod_type = "ratemap"

        # Runs read on demand to grab the data for the image, as this was the input path to the super init call
        self._read_on_demand()
        # That reads in the WCS information (important), and stores the im data in _data
        # That is read out into this variable
        self._im_data = self.data
        # Then the path is changed, so that the exposure map file becomes the focus
        self._path = xga_expmap.path
        # Read on demand runs again and grabs the exposure map data
        self._read_on_demand()
        # Again read out into variable
        self._ex_data = self.data

        # Then divide image by exposure map to get rate map data.
        # Numpy divide lets me specify where we wish to divide, so we don't get any NaN results and divide by
        #  zero warnings
        self._data = np.divide(self._im_data, self._ex_data, out=np.zeros_like(self._im_data),
                               where=self._ex_data != 0)

        # Use exposure maps and basic edge detection to find the edges of the CCDs
        #  The exposure map values calculated on the edge of a CCD can be much smaller than it should be,
        #  which in turn can boost the rate map value there - hence useful to know which elements of an array
        #  are on an edge.
        det_map = self._ex_data.copy()
        # Turn the exposure map into something simpler, either on a detector or not
        det_map[self._ex_data != 0] = 1

        # Do the diff from top to bottom of the image, the append option adds a line of zeros at the end
        #  otherwise the resulting array would only be N-1 elements 'high'.
        hori_edges = np.diff(det_map, axis=0, append=0)
        # A 1 in this array means you're going from no chip to on chip, which means the coordinate where 1
        # is recorded is offset by 1 from the actual edge of the chip elements of this array.
        need_corr_y, need_corr_x = np.where(hori_edges == 1)
        # So that's why we add one to those y coordinates (as this is the vertical pass of np.diff
        new_y = need_corr_y + 1
        # Then make sure chip edge = 1, and everything else = 0
        hori_edges[need_corr_y, need_corr_x] = 0
        hori_edges[new_y, need_corr_x] = 1
        # -1 in this means going from chip to not-chip
        hori_edges[hori_edges == -1] = 1

        # The same process is repeated here, but in the x direction, so you're finding vertical edges
        vert_edges = np.diff(det_map, axis=1, append=0)
        need_corr_y, need_corr_x = np.where(vert_edges == 1)
        new_x = need_corr_x + 1
        vert_edges[need_corr_y, need_corr_x] = 0
        vert_edges[need_corr_y, new_x] = 1
        vert_edges[vert_edges == -1] = 1

        # Both passes are combined into one, with possible values of 0 (no edge), 1 (edge detected in one pass),
        #  and 2 (edge detected in both pass). Then configure the array to act as a mask that removes the
        #  edge pixels
        comb = hori_edges + vert_edges
        comb[comb == 0] = -1
        comb[comb != -1] = False
        comb[comb == -1] = 1

        # Store that mask as an attribute.
        self._edge_mask = comb

    def get_rate(self, at_coord: Quantity) -> float:
        """
        A simple method that converts the given coordinates to pixels, then finds the rate (in photons
        per second) and returns it.
        :param Quantity at_coord: A pair of coordinates to find the photon rate for.
        :return: The photon rate at the supplied coordinates.
        :rtype: Quantity
        """
        pix_coord = self.coord_conv(at_coord, pix).value
        rate = self.data[pix_coord[1], pix_coord[0]]
        return Quantity(rate, "s^-1")

    def simple_peak(self, mask: np.ndarray, out_unit: UnitBase = deg) -> Tuple[Quantity, bool]:
        """
        Simplest possible way to find the position of the peak of X-ray emission in a ratemap. This method
        takes a mask in the form of a numpy array, which allows the user to mask out parts of the ratemap
        that shouldn't be searched (outside of a certain region, or within point sources for instance).
        :param np.ndarray mask: A numpy array used to weight the data. It should be 0 for pixels that
        aren't to be searched, and 1 for those that are.
        :param UnitBase out_unit: The desired output unit of the peak coordinates, the default is degrees.
        :return: An astropy quantity containing the coordinate of the X-ray peak of this ratemap (given
        the user's mask), in units of out_unit, as specified by the user.
        :rtype: Tuple[Quantity, bool]
        """
        if mask.shape != self.data.shape:
            raise ValueError("The shape of the mask array ({0}) must be the same as that of the data array "
                             "({1}).".format(mask.shape, self.data.shape))

        # Creates the data array that we'll be searching. Takes into account the passed mask, as well as
        #  the edge mask designed to remove pixels at the edges of detectors, where RateMap values can
        #  be artificially boosted.
        masked_data = self.data * mask * self._edge_mask

        # Uses argmax to find the flattened coordinate of the max value, then unravel_index to convert
        #  it back to a 2D coordinate
        max_coords = np.unravel_index(np.argmax(masked_data == masked_data.max()), masked_data.shape)
        # Defines an astropy pix quantity of the peak coordinates
        peak_pix = Quantity([max_coords[1], max_coords[0]], pix)
        # Don't bother converting if the desired output coordinates are already pix, but otherwise use this
        #  objects coord_conv function to move to desired coordinate units.
        if out_unit != pix:
            peak_conv = self.coord_conv(peak_pix, out_unit)
        else:
            peak_conv = peak_pix

        # Find if the peak coordinates sit near an edge/chip gap
        edge_flag = self.near_edge(peak_pix)

        return peak_conv, edge_flag

    def clustering_peak(self, mask: np.ndarray, out_unit: UnitBase = deg, top_frac: float = 0.05) \
            -> Tuple[Quantity, bool]:
        """
        An experimental peak finding function that cuts out the top 5% (by default) of array elements
        (by value), and runs a hierarchical clustering algorithm on their positions. The motivation
        for this is that the cluster peak will likely be contained in that top 5%, and the only other
        pixels that might be involved are remnants of poorly removed point sources. So when clusters have
        been formed, we can take the one with the most entries, and find the maximal pixel of that cluster.
        Will be consistent with simple_peak under ideal circumstances.
        :param np.ndarray mask: A numpy array used to weight the data. It should be 0 for pixels that
        aren't to be searched, and 1 for those that are.
        :param UnitBase out_unit: The desired output unit of the peak coordinates, the default is degrees.
        :param float top_frac: The fraction of the elements (ordered in descending value) that should be used
        to generate clusters, and thus be considered for the cluster centre.
        :return: An astropy quantity containing the coordinate of the X-ray peak of this ratemap (given
        the user's mask), in units of out_unit, as specified by the user.
        :rtype: Tuple[Quantity, bool]
        """
        if mask.shape != self.data.shape:
            raise ValueError("The shape of the mask array ({0}) must be the same as that of the data array "
                             "({1}).".format(mask.shape, self.data.shape))

        # Creates the data array that we'll be searching. Takes into account the passed mask, as well as
        #  the edge mask designed to remove pixels at the edges of detectors, where RateMap values can
        #  be artificially boosted.
        masked_data = self.data * mask * self._edge_mask
        # How many non-zero elements are there in the array
        num_value = len(masked_data[masked_data != 0])
        # Find the number that corresponds to the top 5% (by default)
        to_select = round(num_value * top_frac)
        # Grab the inds of the pixels that are in the top 5% of values (by default)
        inds = np.unravel_index(np.argpartition(masked_data.flatten(), -to_select)[-to_select:], masked_data.shape)
        # Just formatting quickly for input into the clustering algorithm
        pairs = [[inds[0][i], inds[1][i]] for i in range(len(inds[0]))]
        # Hierarchical clustering using the inconsistent criterion with threshold 1. 'If a cluster node and all its
        # descendants have an inconsistent value less than or equal to 1, then all its leaf descendants belong to
        # the same flat cluster. When no non-singleton cluster meets this criterion, every node is assigned to its
        # own cluster.'
        cluster_inds = fclusterdata(pairs, 1)

        # Finds how many clusters there are, and how many points belong to each cluster
        uniq_vals, uniq_cnts = np.unique(cluster_inds, return_counts=True)
        # Choose the cluster with the most points associated with it
        chosen_clust = uniq_vals[np.argmax(uniq_cnts)]
        # Retrieves the inds for the main merged_data in the chosen cluster
        chosen_inds = np.where(cluster_inds == chosen_clust)[0]
        cutout = np.zeros(masked_data.shape)
        # Make a masking array to select only the points in the cluster
        cutout[inds[0][chosen_inds], inds[1][chosen_inds]] = 1
        # Mask the data
        masked_data = masked_data * cutout

        # Uses argmax to find the flattened coordinate of the max value, then unravel_index to convert
        #  it back to a 2D coordinate
        max_coords = np.unravel_index(np.argmax(masked_data == masked_data.max()), masked_data.shape)
        # Defines an astropy pix quantity of the peak coordinates
        peak_pix = Quantity([max_coords[1], max_coords[0]], pix)
        # Don't bother converting if the desired output coordinates are already pix, but otherwise use this
        #  objects coord_conv function to move to desired coordinate units.
        if out_unit != pix:
            peak_conv = self.coord_conv(peak_pix, out_unit)
        else:
            peak_conv = peak_pix

        # Find if the peak coordinates sit near an edge/chip gap
        edge_flag = self.near_edge(peak_pix)

        return peak_conv, edge_flag

    def convolved_peak(self, mask: np.ndarray, redshift: float, cosmology, out_unit: UnitBase = deg) \
            -> Tuple[Quantity, bool]:
        """
        A very experimental peak finding algorithm, credit for the idea and a lot of the code in this function
        go to Lucas Porth. A radial profile (for instance a project king profile for clusters) is convolved
        with the ratemap, using a suitable radius for the object type (so for a cluster r might be ~1000kpc). As
        such objects that are similar to this profile will be boosted preferentially over objects that aren't,
        making it less likely that we accidentally select the peak brightness pixel from a point source remnant or
        something similar. The convolved image is then masked to only look at the area of interest, and the peak
        brightness pixel is found.
        :param np.ndarray mask: A numpy array used to weight the data. It should be 0 for pixels that
        aren't to be searched, and 1 for those that are.
        :param float redshift: The redshift of the source that we wish to find the X-ray centroid of.
        :param cosmology: An astropy cosmology object.
        :param UnitBase out_unit: The desired output unit of the peak coordinates, the default is degrees.
        :return: An astropy quantity containing the coordinate of the X-ray peak of this ratemap (given
        the user's mask), in units of out_unit, as specified by the user.
        :rtype: Tuple[Quantity, bool]
        """
        def cartesian(arrays):
            arrays = [np.asarray(a) for a in arrays]
            shape = (len(x) for x in arrays)
            ix = np.indices(shape, dtype=int)
            ix = ix.reshape(len(arrays), -1).T
            for n, arr in enumerate(arrays):
                ix[:, n] = arrays[n][ix[:, n]]
            return ix

        def projected_king(r, pix_size, beta):
            n_pix = int(r / pix_size)
            _ = np.arange(-n_pix, n_pix + 1)
            ds = cartesian([_, _])
            r_grid = np.hypot(ds[:, 0], ds[:, 1]).reshape((len(_), len(_))) * pix_size

            func = (1 + (r_grid / r)**2)**((-3*beta) + 0.5)
            res = (r_grid < r) * func
            return res / np.sum(r_grid < r)

        raise NotImplementedError("The convolved peak method sort of works, but needs to be much more general"
                                  " before its available for proper use.")

        if mask.shape != self.data.shape:
            raise ValueError("The shape of the mask array ({0}) must be the same as that of the data array "
                             "({1}).".format(mask.shape, self.data.shape))

        start_pos = self.coord_conv(Quantity([int(self.shape[1]/2), int(self.shape[0]/2)], pix), deg)
        end_pos = self.coord_conv(Quantity([int(self.shape[1]/2) + 10, int(self.shape[0]/2)], pix), deg)

        separation = Quantity(np.sqrt(abs(start_pos[0].value - end_pos[0].value) ** 2 +
                                      abs(start_pos[1].value - end_pos[1].value) ** 2), deg)

        resolution = ang_to_rad(separation, redshift, cosmology) / 10

        # TODO Need to make this more general, with different profiles, also need to figure out what
        #  Lucas's code does and comment it
        # TODO Should probably go through different projected king profile parameters
        # TODO Could totally make this into a basic cluster finder combined with clustering algorithm
        filt = projected_king(1000, resolution.value, 3)
        n_cut = int(filt.shape[0] / 2)
        conv_data = fftconvolve(self.data*self._edge_mask, filt)[n_cut:-n_cut, n_cut:-n_cut]
        mask_conv_data = conv_data * mask

        max_coords = np.unravel_index(np.argmax(mask_conv_data == mask_conv_data.max()), mask_conv_data.shape)
        # Defines an astropy pix quantity of the peak coordinates
        peak_pix = Quantity([max_coords[1], max_coords[0]], pix)

        if out_unit != pix:
            peak_conv = self.coord_conv(peak_pix, out_unit)
        else:
            peak_conv = peak_pix

        # Find if the peak coordinates sit near an edge/chip gap
        edge_flag = self.near_edge(peak_pix)

        return peak_conv, edge_flag

    def near_edge(self, coord: Quantity) -> bool:
        """
        Uses the edge mask generated for RateMap objects to determine if the passed coordinates are near
        an edge/chip gap. If the coordinates are within +- 2 pixels of an edge the result will be true.
        :param Quantity coord: The coordinates to check.
        :return: A boolean flag as to whether the coordinates are near an edge.
        :rtype: bool
        """
        # Convert to pixel coordinates
        pix_coord = self.coord_conv(coord, pix).value

        # Checks the edge mask within a 5 by 5 array centered on the peak coord, if there are no edges then
        #  all elements will be 1 and it will sum to 25.
        edge_sum = self._edge_mask[pix_coord[1] - 2:pix_coord[1] + 3,
                                   pix_coord[0] - 2:pix_coord[0] + 3].sum()
        # If it sums to less then we know that there is an edge near the peak.
        if edge_sum != 25:
            edge_flag = True
        else:
            edge_flag = False

        return edge_flag


class EventList(BaseProduct):
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, raise_properly: bool = True):
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, raise_properly)
        self._prod_type = "events"


class Spectrum(BaseProduct):
    def __init__(self, path: str, rmf_path: str, arf_path: str, b_path: str, b_rmf_path: str, b_arf_path: str,
                 reg_type: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 raise_properly: bool = True):

        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, raise_properly)
        self._prod_type = "spectrum"

        if os.path.exists(rmf_path):
            self._rmf = rmf_path
        else:
            self._rmf = None
            self._usable = False

        if os.path.exists(arf_path):
            self._arf = arf_path
        else:
            self._arf = None
            self._usable = False

        if os.path.exists(b_path):
            self._back_spec = b_path
        else:
            self._back_spec = None
            self._usable = False

        if os.path.exists(b_rmf_path):
            self._back_rmf = b_rmf_path
        else:
            self._back_rmf = None
            self._usable = False

        if os.path.exists(b_arf_path):
            self._back_arf = b_arf_path
        else:
            self._arf_rmf = None
            self._usable = False

        allowed_regs = ["region", "r2500", "r500", "r200", "custom"]
        if reg_type in allowed_regs:
            self._reg_type = reg_type
        else:
            self._usable = False
            self._reg_type = None
            raise ValueError("{0} is not a supported region type, please use one of these; "
                             "{1}".format(reg_type, ", ".join(allowed_regs)))

        self._update_spec_headers("main")
        self._update_spec_headers("back")

        self._exp = None
        self._plot_data = {}
        self._luminosities = {}
        self._count_rate = {}

    def _update_spec_headers(self, which_spec: str):
        """
        An internal method that will 'push' the current class attributes that hold the paths to data products
        (like ARF and RMF) to the relevant spectrum file.
        :param str which_spec: A flag that tells the method whether to update the header of
         the main or background spectrum.
        """
        # This function is meant for internal use only, so I won't check that the passed-in file paths
        #  actually exist. This will have been checked already
        if which_spec == "main":
            with FITS(self._path, 'rw') as spec_fits:
                spec_fits[1].write_key("RESPFILE", self._rmf)
                spec_fits[1].write_key("ANCRFILE", self._arf)
                spec_fits[1].write_key("BACKFILE", self._back_spec)
                spec_fits[0].write_key("RESPFILE", self._rmf)
                spec_fits[0].write_key("ANCRFILE", self._arf)
                spec_fits[0].write_key("BACKFILE", self._back_spec)
        elif which_spec == "back":
            with FITS(self._back_spec, 'rw') as spec_fits:
                spec_fits[1].write_key("RESPFILE", self._back_rmf)
                spec_fits[1].write_key("ANCRFILE", self._back_arf)
                spec_fits[0].write_key("RESPFILE", self._back_rmf)
                spec_fits[0].write_key("ANCRFILE", self._back_arf)
        else:
            raise ValueError("Illegal value for which_spec, you shouldn't be using this internal function!")

    @property
    def path(self) -> str:
        """
        This method returns the path to the spectrum file of this object.
        :return: The path to the spectrum file associated with this object.
        :rtype: str
        """
        return self._path

    @path.setter
    def path(self, new_path: str):
        """
        This setter updates the path to the spectrum file, and then updates that file with the current values of
        the RMF, ARF, and background spectrum paths. WARNING: This does permanently alter the file, so use your
        own spectrum file with caution.
        :param str new_path: The updated path to the spectrum file.
        """
        if os.path.exists(new_path):
            self._path = new_path
            # Call this here because it'll replace any existing arf and rmf file paths with the ones
            #  currently loaded in the instance of this object.
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new spectrum file does not exist")

    @property
    def rmf(self) -> str:
        """
        This method returns the path to the RMF file of the main spectrum of this object.
        :return: The path to the RMF file associated with the main spectrum of this object.
        :rtype: str
        """
        return self._rmf

    @rmf.setter
    def rmf(self, new_path: str):
        """
        This setter updates the path to the main RMF file, then writes that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!
        :param str new_path: The path to the new RMF file.
        """
        if os.path.exists(new_path):
            self._rmf = new_path
            # Push to the actual file
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new RMF file does not exist")

    @property
    def arf(self) -> str:
        """
        This method returns the path to the ARF file of the main spectrum of this object.
        :return: The path to the ARF file associated with the main spectrum of this object.
        :rtype: str
        """
        return self._arf

    @arf.setter
    def arf(self, new_path: str):
        """
        This setter updates the path to the main ARF file, then writes that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!
        :param str new_path: The path to the new ARF file.
        """
        if os.path.exists(new_path):
            self._arf = new_path
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new ARF file does not exist")

    @property
    def background(self) -> str:
        """
        This method returns the path to the background spectrum.
        :return: Path of the background spectrum.
        :rtype: str
        """
        return self._back_spec

    @background.setter
    def background(self, new_path: str):
        """
        This method is the setter for the background spectrum. It can be used to change the background
        spectrum file associated with this object, and will write that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!
        :param str new_path: The path to the new background spectrum.
        """
        if os.path.exists(new_path):
            self._back_spec = new_path
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new background spectrum file does not exist")

    @property
    def background_rmf(self) -> str:
        """
        This method returns the path to the background spectrum's RMF file.
        :return: The path the the background spectrum's RMF.
        :rtype: str
        """
        return self._back_rmf

    @background_rmf.setter
    def background_rmf(self, new_path: str):
        """
        This setter method will change the RMF associated with the background spectrum, then write
        that change to the background spectrum file.
        :param str new_path: The path to the background spectrum's new RMF.
        """
        if os.path.exists(new_path):
            self._back_rmf = new_path
            self._update_spec_headers("back")
        else:
            raise FileNotFoundError("That new background RMF file does not exist")

    @property
    def background_arf(self) -> str:
        """
        This method returns the path to the background spectrum's ARF file.
        :return: The path the the background spectrum's ARF.
        :rtype: str
        """
        return self._back_arf

    @background_arf.setter
    def background_arf(self, new_path: str):
        """
        This setter method will change the ARF associated with the background spectrum, then write
        that change to the background spectrum file.
        :param str new_path: The path to the background spectrum's new ARF.
        """
        if os.path.exists(new_path):
            self._back_arf = new_path
            self._update_spec_headers("back")
        else:
            raise FileNotFoundError("That new background ARF file does not exist")

    # This is an intrinsic property of the generated spectrum, so users will not be allowed to change this
    @property
    def reg_type(self) -> str:
        """
        Getter method for the type of region this spectrum was generated for. e.g. 'region' - which would
        mean it represents the spectrum inside a region specificied by region files, or 'r500' - which
        would mean the radius of a cluster where the mean density is 500 times critical density of the Universe.
        :return: The region type this spectrum was generated for
        :rtype: str
        """
        return self._reg_type

    @property
    def exposure(self) -> Quantity:
        """
        Property that returns the spectrum exposure time used by XSPEC.
        :return: Spectrum exposure time.
        :rtype: Quantity
        """
        if self._exp is None:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this Spectrum")
        else:
            exp = Quantity(self._exp, 's')

        return exp

    def add_fit_data(self, model: str, tab_line, plot_data: hdu.table.TableHDU):
        """
        Method that adds information specific to a spectrum from an XSPEC fit to this object. This includes
        individual spectrum exposure and count rate, as well as calculated luminosities, and plotting
        information for data and model.
        :param str model: String representation of the XSPEC model fitted to the data.
        :param tab_line: The line of the SPEC_INFO table produced by xga_extract.tcl that is relevant to this
        spectrum object.
        :param hdu.table.TableHDU plot_data: The PLOT{N} table in the file produced by xga_extract.tcl that is
        relevant to this spectrum object.
        """
        # This stores the exposure time that XSPEC uses for this specific spectrum.
        if self._exp is None:
            self._exp = float(tab_line["EXPOSURE"])

        # This is the count rate and error for this spectrum.
        self._count_rate[model] = [float(tab_line["COUNT_RATE"]), float(tab_line["COUNT_RATE_ERR"])]

        # Searches for column headers with 'Lx' in them (this has to be dynamic as the user can calculate
        #  luminosity in as many bands as they like)
        lx_inds = np.where(np.char.find(tab_line.dtype.names, "Lx") == 0)[0]
        lx_cols = np.array(tab_line.dtype.names)[lx_inds]

        # Constructs a dictionary of luminosities and their errors for the different energy bands
        #  in this XSPEC fit.
        lx_dict = {}
        for col in lx_cols:
            lx_info = col.split("_")
            if lx_info[2][-1] == "-" or lx_info[2][-1] == "+":
                en_band = "bound_{l}-{u}".format(l=lx_info[1], u=lx_info[2][:-1])
                err_type = lx_info[-1][-1]
            else:
                en_band = "bound_{l}-{u}".format(l=lx_info[1], u=lx_info[2])
                err_type = ""

            if en_band not in lx_dict:
                lx_dict[en_band] = [0, 0, 0]

            if err_type == "":
                lx_dict[en_band][0] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")
            elif err_type == "-":
                lx_dict[en_band][1] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")
            elif err_type == "+":
                lx_dict[en_band][2] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")

        self._luminosities[model] = lx_dict

        self._plot_data[model] = {"x": plot_data["X"][:], "x_err": plot_data["XERR"][:],
                                  "y": plot_data["Y"][:], "y_err": plot_data["YERR"][:],
                                  "model": plot_data["YMODEL"][:]}

    def get_luminosities(self, model: str, lo_en: Quantity = None, hi_en: Quantity = None):
        """
        Returns the luminosities measured for this spectrum from a given model.
        :param model: Name of model to fetch luminosities for.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :return: Luminosity measurement, either for all energy bands, or the one requested with the energy
        limit parameters. Luminosity measurements are presented as three column numpy arrays, with column 0
        being the value, column 1 being err-, and column 2 being err+.
        """
        # Checking the input energy limits are valid, and assembles the key to look for lums in those energy
        #  bounds. If the limits are none then so is the energy key
        if lo_en is not None and hi_en is not None and lo_en > hi_en:
            raise ValueError("The low energy limit cannot be greater than the high energy limit")
        elif lo_en is not None and hi_en is not None:
            en_key = "bound_{l}-{u}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value)
        else:
            en_key = None

        # Checks that the requested region, model and energy band actually exist
        if len(self._luminosities) == 0:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this source")
        elif model not in self._luminosities:
            av_mods = ", ".join(self._luminosities.keys())
            raise ModelNotAssociatedError("{0} has not been fitted to this spectrum; "
                                          "available models are {1}".format(model, av_mods))
        elif en_key is not None and en_key not in self._luminosities[model]:
            av_bands = ", ".join([en.split("_")[-1] + "keV" for en in self._luminosities[model].keys()])
            raise ParameterNotAssociatedError("{l}-{u}keV was not an energy band for the fit with {m}; available "
                                              "energy bands are {b}".format(l=lo_en.to("keV").value,
                                                                            u=hi_en.to("keV").value,
                                                                            m=model, b=av_bands))

        if en_key is None:
            return self._luminosities[model]
        else:
            return self._luminosities[model][en_key]

    def get_rate(self, model: str) -> Quantity:
        """
        Fetches the count rate for a particular model fitted to this spectrum.
        :param model: The model to fetch count rate for.
        :return: Count rate in counts per second.
        :rtype: Quantity
        """
        if model not in self._count_rate:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this Spectrum")
        else:
            rate = Quantity(self._count_rate[model], 's^-1')

        return rate

    def view(self, lo_en: Quantity = Quantity(0.0, "keV"), hi_en: Quantity = Quantity(30.0, "keV")):
        """
        Very simple method to plot the data/models associated with this Spectrum object,
        between certain energy limits.
        :param Quantity lo_en: The lower energy limit from which to plot the spectrum.
        :param Quantity hi_en: The upper energy limit to plot the spectrum to.
        """
        if lo_en > hi_en:
            raise ValueError("hi_en cannot be greater than lo_en")
        else:
            lo_en = lo_en.to("keV").value
            hi_en = hi_en.to("keV").value

        if len(self._plot_data.keys()) != 0:
            # Create figure object
            plt.figure(figsize=(8, 5))

            # Set the plot up to look nice and professional.
            ax = plt.gca()
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

            # Set the title with all relevant information about the spectrum object in it
            plt.title("{n} - {o}{i} {r} Spectrum".format(n=self.obj_name, o=self.obs_id, i=self.instrument.upper(),
                                                         r=self.reg_type))
            for mod_ind, mod in enumerate(self._plot_data):
                x = self._plot_data[mod]["x"]
                # If the defaults are left, just update them to the min and max of the dataset
                #  to avoid unsightly gaps at the sides of the plot
                if lo_en == 0.:
                    lo_en = x.min()
                if hi_en == 30.0:
                    hi_en = x.max()

                # Cut the x dataset to just the energy range we want
                plot_x = x[(x > lo_en) & (x < hi_en)]

                if mod_ind == 0:
                    # Read out the data just for line length reasons
                    # Make the cuts based on energy values supplied to the view method
                    plot_y = self._plot_data[mod]["y"][(x > lo_en) & (x < hi_en)]
                    plot_xerr = self._plot_data[mod]["x_err"][(x > lo_en) & (x < hi_en)]
                    plot_yerr = self._plot_data[mod]["y_err"][(x > lo_en) & (x < hi_en)]
                    plot_mod = self._plot_data[mod]["model"][(x > lo_en) & (x < hi_en)]

                    plt.errorbar(plot_x, plot_y, xerr=plot_xerr, yerr=plot_yerr, fmt="k+", label="data", zorder=1)
                else:
                    # Don't want to re-plot data points as they should be identical, so if there is another model
                    #  only it will be plotted
                    plot_mod = self._plot_data[mod]["model"][(x > lo_en) & (x < hi_en)]

                # The model line is put on
                plt.plot(plot_x, plot_mod, label=mod, linewidth=1.5)

            # Generate the legend for the data and model(s)
            plt.legend(loc="best")

            # Ensure axis is limited to the chosen energy range
            plt.xlim(lo_en, hi_en)

            plt.xlabel("Energy [keV]")
            plt.ylabel("Normalised Counts s$^{-1}$ keV$^{-1}$")

            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

            plt.tight_layout()
            # Display the spectrum
            plt.show()

            # Wipe the figure
            plt.close("all")

        else:
            warnings.warn("There are no XSPEC fits associated with this Spectrum, you can't view it.")


class AnnularSpectra(BaseProduct):
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, raise_properly: bool = True):
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, raise_properly)


# Defining a dictionary to map from string product names to their associated classes
PROD_MAP = {"image": Image, "expmap": ExpMap, "events": EventList, "spectrum": Spectrum}





