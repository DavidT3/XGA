#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 02/08/2021, 17:28. Copyright (c) David J Turner

import os
import warnings
from typing import Tuple, List, Union
from copy import deepcopy

import numpy as np
import pandas as pd
from astropy import wcs
from astropy.convolution import Kernel
from astropy.units import Quantity, UnitBase, UnitsError, deg, pix, UnitConversionError, Unit
from astropy.visualization import LogStretch, MinMaxInterval, ImageNormalize, BaseStretch
from fitsio import read, read_header, FITSHDR
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from regions import read_ds9, PixelRegion, SkyRegion
from scipy.cluster.hierarchy import fclusterdata
from scipy.signal import fftconvolve

from . import BaseProduct, BaseAggregateProduct
from ..exceptions import FailedProductError, RateMapPairError, NotPSFCorrectedError, IncompatibleProductError
from ..sourcetools import ang_to_rad
from ..utils import xmm_sky, xmm_det, find_all_wcs

EMOSAIC_INST = {"EPN": "pn", "EMOS1": "mos1", "EMOS2": "mos2"}


class Image(BaseProduct):
    """
    This class stores image data from X-ray observations. It also allows easy, direct, access to that data, and
    implements many helpful methods with extra functionality (including coordinate transforms, peak finders, and
    a powerful view method).
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 lo_en: Quantity, hi_en: Quantity, reg_file_path: str = '', smoothed: bool = False,
                 smoothed_info: Union[dict, Kernel] = None, obs_inst_combs: List[List] = None):
        """
        The initialisation method for the Image class.

        :param str path: The path to where the product file SHOULD be located.
        :param str stdout_str: The stdout from calling the terminal command.
        :param str stderr_str: The stderr from calling the terminal command.
        :param str gen_cmd: The command used to generate the product.
        :param Quantity lo_en: The lower energy bound used to generate this product.
        :param Quantity hi_en: The upper energy bound used to generate this product.
        :param str reg_file_path: Path to a region file for this image.
        :param bool smoothed: Has this image been smoothed, default is False. This information can also be
            set after the instantiation of an image.
        :param dict/Kernel smoothed_info: Information on how the image was smoothed, given either by the Astropy
            kernel used or a dictionary of information (required structure detailed in
            parse_smoothing). Default is None
        :param List[List] obs_inst_combs: Supply a list of lists of ObsID-Instrument combinations if the image
            is combined and wasn't made by emosaic (e.g. [['0404910601', 'pn'], ['0404910601', 'mos1'],
            ['0404910601', 'mos2'], ['0201901401', 'pn'], ['0201901401', 'mos1'], ['0201901401', 'mos2']].
        """
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd)
        self._shape = None
        self._wcs_radec = None
        self._wcs_xmmXY = None
        self._wcs_xmmdetXdetY = None
        self._energy_bounds = (lo_en, hi_en)
        self._prod_type = "image"
        self._data = None
        self._header = None

        # This is a flag to let XGA know that the Image object has been PSF corrected
        self._psf_corrected = False
        # These give extra information about the PSF correction, but can't be set unless PSF
        #  corrected is true
        self._psf_correction_algorithm = None
        self._psf_num_bins = None
        self._psf_num_iterations = None
        self._psf_model = None

        # This checks whether a region file has been passed, and if it has then processes it
        if reg_file_path != '' and os.path.exists(reg_file_path):
            self._regions = self._process_regions(reg_file_path)
            self._reg_file_path = reg_file_path
        elif reg_file_path != '' and not os.path.exists(reg_file_path):
            warnings.warn("That region file path does not exist")
            self._regions = []
            self._reg_file_path = reg_file_path
        else:
            self._regions = []
            self._reg_file_path = ''

        self._smoothed = smoothed
        # If the user says at this point that the image has been smoothed, then we try and parse the smoothing info
        if smoothed:
            self._smoothed_method, self._smoothed_info = self.parse_smoothing(smoothed_info)
        else:
            self._smoothed_info = None
            self._smoothed_method = None

        # I want combined images to be aware of the ObsIDs and Instruments that have gone into them
        if obs_id == 'combined' or instrument == 'combined':
            if "CREATOR" in self.header and "emosaic" in self.header['CREATOR']:
                # We search for the instrument names of the various components
                ind_inst_hdrs = [h for h in self.header if 'EMSCI' in h]
                # Then use the length of the list to find out how many components there are
                num_ims = len(ind_inst_hdrs)
                # If this image is the combined product of only one ObsID's instruments, then there will be no EMSCA
                #  headers detailing the different ObsIDs, so we just use the ObsID header
                if len([h for h in self.header if 'EMSCA' in h]) == 0:
                    oi_pairs = [[self.header["OBS_ID"], EMOSAIC_INST[self.header["EMSCI"+str(ind).zfill(3)]]] for
                                ind in range(1, num_ims+1)]
                else:
                    oi_pairs = [[self.header["EMSCA" + str(ind).zfill(3)],
                                 EMOSAIC_INST[self.header["EMSCI" + str(ind).zfill(3)]]]
                                for ind in range(1, num_ims + 1)]

                # So now we have a list of lists of ObsID-Instrument combinations, we shall store them
                self._comb_oi_pairs = oi_pairs

            # In the case of the combined image not being made by emosaic, we need to take the info from
            #  the obs_inst_combs parameter
            elif "CREATOR" not in self.header or "emosaic" not in self.header['CREATOR'] and obs_inst_combs is not None:
                # We check to make sure that each entry in obs_inst_combs is a two element list
                if any([len(e) != 2 for e in obs_inst_combs]):
                    raise ValueError("Entries in the obs_inst_combs list must be lists structured as [ObsID, Inst]")
                # And if it passes that we check that the instrument values are one of the allowed list
                elif any([e[1] not in EMOSAIC_INST.values() for e in obs_inst_combs]):
                    raise ValueError("Instruments are currently only allowed to be 'pn', 'mos1', or 'mos2'.")

                self._comb_oi_pairs = obs_inst_combs

            # And if the user hasn't passed the obs_inst_combs list then we kick off
            elif "CREATOR" not in self.header or "emosaic" not in self.header['CREATOR'] and obs_inst_combs is None:
                raise ValueError("If a combined image has not been made with emosaic, you have to "
                                 " pass ObsID and Instrument combinations using obs_inst_combs")

        else:
            self._comb_oi_pairs = None

    def _read_on_demand(self):
        """
        Internal method to read the image associated with this Image object into memory when it is requested by
        another method. Doing it on-demand saves on wasting memory.
        """
        # Usable flag to check that nothing went wrong in the image generation
        if self.usable:
            try:
                # Not all images produced by SAS are going to be needed all the time, so they will only be read in if
                # asked for.
                self._data = read(self.path).astype("float64")
            except OSError:
                raise FileNotFoundError("FITSIO read cannot open {f}, possibly because there is a problem with "
                                        "the file, it doesn't exist, or maybe an SFTP problem? This product is "
                                        "associated with {s}.".format(f=self.path, s=self.src_name))
            if self._data.min() < 0:
                # This throws a non-fatal warning to let the user know there are negative pixel values,
                #  and that they're being 'corrected'
                warnings.warn("You are loading an {} with elements that are < 0, "
                              "they will be set to 0.".format(self._prod_type))
                self._data[self._data < 0] = 0

            # As the image must be loaded to know the shape, I've waited until here to set the _shape attribute
            self._shape = self._data.shape
        else:
            reasons = ", ".join(self.not_usable_reasons)
            raise FailedProductError("SAS failed to generate this product successfully, so you cannot access "
                                     "data from it; reason give is {}. Check the usable attribute next "
                                     "time".format(reasons))

    def _read_wcs_on_demand(self):
        """
        The equivalent of _read_on_demand, but for the header and wcs information. These are often
        required more than the data for individual images (as the merged images are generally used
        for analysis), so this function is split out in the interests of efficiency.
        """
        try:
            # Reads only the header information
            self._header = read_header(self.path)
        except OSError:
            raise FileNotFoundError("FITSIO read_header cannot open {f}, possibly because there is a problem with "
                                    "the file, it doesn't exist, or maybe an SFTP problem? This product is associated "
                                    "with {s}.".format(f=self.path, s=self.src_name))

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

    def _process_regions(self, path: str = None, reg_list: List[Union[PixelRegion, SkyRegion]] = None) \
            -> List[PixelRegion]:
        """
        This internal function just takes the path to a region file and processes it into a form that
        this object requires for viewing.

        :param str path: The path of a region file to be processed, can be None but only if the
            other argument is given.
        :param List[PixelRegion/SkyRegion] reg_list: A list of region objects to be processed, default is None.
        :return: A list of pixel regions.
        :rtype: List[PixelRegion]
        """
        # This method can deal with either an input of a region file path or of a list of region objects, but
        #  firstly we need to check that at least one of the inputs isn't None
        if all([path is None, reg_list is None]):
            raise ValueError("Either a path or a list of region objects must be passed, you have passed neither")
        elif all([path is not None, reg_list is not None]):
            raise ValueError("You have passed both a path and a list of regions, pass one or the other.")

        # The behaviour here depends on whether regions or a path have been passed
        if path is not None:
            ds9_regs = read_ds9(path)
        else:
            ds9_regs = deepcopy(reg_list)

        # Checking what kind of regions there are, as that changes whether they need to be converted or not
        final_regs = []
        for reg in ds9_regs:
            if isinstance(reg, PixelRegion):
                final_regs.append(reg)
            else:
                # Regions in sky coordinates need to be in pixels for overlaying on the image
                final_regs.append(reg.to_pixel(self._wcs_radec))

        return final_regs

    @staticmethod
    def parse_smoothing(info: Union[dict, Kernel]) -> Tuple[str, dict]:
        """
        Parses smoothing information that has been passed into the Image object, either on init or afterwards. If an
        Astropy kernel is provided then the name, and kernel parameter, will be extracted. If a dictionary is provided
        then it must have a) 'method' key with the name of the smoothing method, and b) a 'pars' key with another
        dictionary of all the parameters involved in smoothing;
        e.g. {'method': 'Gaussian2DKernel', 'pars': {'amplitude': 0.63, 'x_mean': 0.0, 'y_mean': 0.0, 'x_stddev': 0.5,
                                                     'y_stddev': 0.5, 'theta': 0.0}}

        :param dict/Kernel info: The information dictionary or astropy kernel used for the smoothing.
        :return: The name of the kernel/method and parameter information
        :rtype: Tuple[str, dict]
        """

        if not isinstance(info, (Kernel, dict)):
            raise TypeError("You may only pass smoothing information in the form of an Astropy Kernel or "
                            "a dictionary")
        elif isinstance(info, dict) and ("method" not in info.keys() or "pars" not in info.keys()):
            raise KeyError("If an info dictionary is passed, it must contain a 'method' key (whose value is the name"
                           " of the smoothing method), and a 'pars' key (a dictionary of the parameters involved in "
                           "the smoothing).")

        if isinstance(info, Kernel):
            # Not super happy with this, but the Astropy kernel.model.name attribute I would have preferred to
            #  use doesn't appear to be set for model objects upon which Kernels are based
            method_name = str(info).split(".")[-1].split(" object")[0]
            method_pars = dict(zip(info.model.param_names, info.model.parameters.copy()))
        else:
            method_name = info['method']
            method_pars = info['pars']

        return method_name, method_pars

    @property
    def smoothing_info(self) -> dict:
        """
        If the image has been smoothed, then this property getter will return information on the smoothing done.

        :return: A dictionary of information on the smoothing applied (if any). Default is None if no
            smoothing applied.
        :rtype: dict
        """
        return self._smoothed_info

    @smoothing_info.setter
    def smoothing_info(self, new_info: Union[dict, Kernel]):
        """
        If the Image was not declared smoothed on initialisation, smoothing information can be added with this
        property setter. It will be parsed and the smoothed property will be set to True.

        :param dict/Kernel new_info: The new smoothing information to be added to the product.
        """
        self._smoothed_method, self._smoothed_info = self.parse_smoothing(new_info)
        self._smoothed = True

    @property
    def smoothed_method(self) -> str:
        """
        The name of the smoothing method (or kernel) that has been applied.

        :return: Name of method/kernel, default is None if no smoothing has been applied.
        :rtype: str
        """
        return self._smoothed_method

    @property
    def smoothed(self) -> bool:
        """
        Property describing whether an image product has been smoothed or not.

        :return: Has the product been smoothed.
        :rtype: bool
        """
        return self._smoothed

    @property
    def storage_key(self) -> str:
        """
        The key under which this object should be stored in a source's product structure. Contains information
        about various aspects of the Image/RateMap/ExpMap.

        :return: The storage key.
        :rtype: str
        """
        # As for some reason I've allowed certain important info about these products to be updated after init,
        #  this getter actually generates the storage key on demand, rather than returning a stored value

        # Start with the simple stuff - these products are energy bound, so that info will be in ALL keys
        key = "bound_{l}-{u}".format(l=float(self._energy_bounds[0].value), u=float(self._energy_bounds[1].value))

        # Then add PSF correction information
        if self._psf_corrected:
            key += "_" + self.psf_model + "_" + str(self.psf_bins) + "_" + self.psf_algorithm + \
                         str(self.psf_iterations)

        # And now smoothing information
        if self.smoothed:
            # Grab the smoothing parameter's names and values, then smoosh them into a string
            sp = "_".join([str(k)+str(v) for k, v in self._smoothed_info.items()])
            # Then add the parameters and method name to the storage key
            key += "_sm{sm}_sp{sp}".format(sm=self._smoothed_method, sp=sp)

        return key

    @property
    def obs_inst_combos(self) -> list:
        """
        This property getter will provide ObsID-Instrument information on the constituent images that make up
        this total image (if it is combined), otherwise it will just provide the single ObsID-Instrument combo.

        :return: A list of lists of ObsID-Instrument combinations, or a list containing one ObsID and one instrument.
        :rtype: list
        """
        if self._comb_oi_pairs is not None:
            return self._comb_oi_pairs
        else:
            return [self.obs_id, self.instrument]

    @property
    def obs_ids(self) -> list:
        """
        Property getter for the ObsIDs that are involved in this image, if combined. Otherwise will return a list
        with one element, the single relevant ObsID.

        :return: List of ObsIDs involved in this image.
        :rtype: list
        """
        if self._comb_oi_pairs is None:
            ret_list = [self.obs_id]
        else:
            # This is a really ugly way of doing what a set and a list() operator could do, but I wanted to make
            #  absolutely sure that the order was preserved
            ret_list = []
            for o in self._comb_oi_pairs:
                if o[0] not in ret_list:
                    ret_list.append(o[0])

        return ret_list

    @property
    def instruments(self) -> dict:
        """
        Equivelant to the BaseSource instruments property, this will return a dictionary of ObsIDs with lists of
        instruments that are associated with them in a combined image. If the image is not combined then an equivelant
        dictionary with one key (the ObsID), with the associated value being a list with one entry (the instrument).

        :return: A dictionary of ObsIDs and their associated instruments
        :rtype: dict
        """
        # If this attribute is None then this product isn't combined, so we do the fallback for a single
        #  ObsID-Instrument combination
        if self._comb_oi_pairs is None:
            ret_dict = {self.obs_id: [self.instrument]}
        # Otherwise we construct the promised dictionary
        else:
            ret_dict = {o: [i[1] for i in self._comb_oi_pairs if i[0] == o] for o in self.obs_ids}

        return ret_dict

    @property
    def inventory_entry(self) -> pd.Series:
        """
        This allows an Image product to generate its own entry for the XGA file generation inventory.

        :return: The new line entry for the inventory.
        :rtype: pd.Series
        """
        # The filename, devoid of the rest of the path
        f_name = self.path.split('/')[-1]

        if self._comb_oi_pairs is None:
            new_line = pd.Series([f_name, self.obs_id, self.instrument, self.storage_key, "", self.type],
                                 ['file_name', 'obs_id', 'inst', 'info_key', 'src_name', 'type'], dtype=str)
        else:
            o_str = "/".join(e[0] for e in self._comb_oi_pairs)
            i_str = "/".join(e[1] for e in self._comb_oi_pairs)
            new_line = pd.Series([f_name, o_str, i_str, self.storage_key, "", self.type],
                                 ['file_name', 'obs_ids', 'insts', 'info_key', 'src_name', 'type'], dtype=str)

        return new_line

    @property
    def regions(self) -> List[PixelRegion]:
        """
        Property getter for regions associated with this image.

        :return: Returns a list of regions, if they have been associated with this object.
        :rtype: List[PixelRegion]
        """
        return self._regions

    @regions.setter
    def regions(self, new_reg: Union[str, List[Union[SkyRegion, PixelRegion]]]):
        """
        A setter for regions associated with this object, a region file path is passed, then that file
        is processed into the required format.

        :param str/List[SkyRegion/PixelRegion] new_reg: A new region file path, or a list of region objects.
        """
        if not isinstance(new_reg, (str, list)):
            raise TypeError("Please pass either a path to a region file or a list of "
                            "SkyRegion/PixelRegion objects.")

        if isinstance(new_reg, str) and new_reg != '' and os.path.exists(new_reg):
            self._reg_file_path = new_reg
            self._regions = self._process_regions(new_reg)
        elif isinstance(new_reg, str) and new_reg == '':
            pass
        elif isinstance(new_reg, str):
            warnings.warn("That region file path does not exist")
        elif isinstance(new_reg, List) and all([isinstance(r, (SkyRegion, PixelRegion)) for r in new_reg]):
            self._reg_file_path = ""
            self._regions = self._process_regions(reg_list=new_reg)
        else:
            raise ValueError("That value of new_reg is not valid, please pass either a path to a region file or "
                             "a list of SkyRegion/PixelRegion objects")

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

    @data.deleter
    def data(self):
        """
        Property deleter for data contained in this Image instance, or whatever subclass of the Image class you
        may be using. The self._data array is removed from memory, and then self._data is explicitly set to None
        so that self._read_on_demand() will be triggered if you ever want the data from this object again.
        """
        del self._data
        self._data = None

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
            self._read_wcs_on_demand()
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
            self._read_wcs_on_demand()
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
            self._read_wcs_on_demand()
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
        if self._header is None:
            self._read_wcs_on_demand()
        return self._header

    def coord_conv(self, coords: Quantity, output_unit: Union[Unit, str]) -> Quantity:
        """
        This will use the loaded WCSes, and astropy coordinates (including custom ones defined for this module),
        to perform common coordinate conversions for this product object.

        :param Quantity coords: The input coordinates quantity to convert, in units of either deg,
            pix, xmm_sky, or xmm_det (xmm_sky and xmm_det are defined for this module).
        :param Unit/str output_unit: The astropy unit to convert to, can be either deg, pix, xmm_sky, or
            xmm_det (xmm_sky and xmm_det are defined for this module).
        :return: The converted coordinates.
        :rtype: Quantity
        """
        # If a string representation was passed, we make it an astropy unit
        if isinstance(output_unit, str) and output_unit not in ['xmm_sky', 'xmm_det']:
            output_unit = Unit(output_unit)
        elif isinstance(output_unit, str) and output_unit == 'xmm_sky':
            output_unit = xmm_sky
        elif isinstance(output_unit, str) and output_unit == 'xmm_det':
            output_unit = xmm_det

        allowed_units = ["deg", "xmm_sky", "xmm_det", "pix"]
        if coords.unit.is_equivalent("deg"):
            coords = coords.to("deg")
        input_unit = coords.unit.name
        out_name = output_unit.name

        if input_unit != out_name:
            # First off do some type checking
            if not isinstance(coords, Quantity):
                raise TypeError("Please pass an astropy Quantity for the coords.")
            # The coordinate pair must have two elements per row, no more no less
            elif len(coords.shape) == 1 and coords.shape != (2,):
                raise ValueError("You have supplied an array with {} values, coordinate pairs "
                                 "should have two.".format(coords.shape[0]))
            # This changes individual coordinate pairs into the form that this function expects
            elif len(coords.shape) == 1:
                coords = coords[:, None].T
            # Checks that multiple pairs of coordinates are in the right format
            elif len(coords.shape) != 1 and coords.shape[1] != 2:
                raise ValueError("You have supplied an array with {} columns, there can only be "
                                 "two.".format(coords.shape[1]))
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
                out_coord = Quantity(self.radec_wcs.all_world2pix(coords, 0), output_unit).round(0).astype(int)
            elif input_unit == "pix" and out_name == "deg":
                out_coord = Quantity(self.radec_wcs.all_pix2world(coords, 0), output_unit)

            # These go between degrees and XMM sky XY coordinates
            elif input_unit == "deg" and out_name == "xmm_sky":
                interim = self.radec_wcs.all_world2pix(coords, 0)
                out_coord = Quantity(self.skyxy_wcs.all_pix2world(interim, 0), xmm_sky)
            elif input_unit == "xmm_sky" and out_name == "deg":
                interim = self.skyxy_wcs.all_world2pix(coords, 0)
                out_coord = Quantity(self.radec_wcs.all_pix2world(interim, 0), deg)

            # These go between XMM sky XY and pixel coordinates
            elif input_unit == "xmm_sky" and out_name == "pix":
                out_coord = Quantity(self.skyxy_wcs.all_world2pix(coords, 0), output_unit).round(0).astype(int)
            elif input_unit == "pix" and out_name == "xmm_sky":
                out_coord = Quantity(self.skyxy_wcs.all_pix2world(coords, 0), output_unit)

            # These go between degrees and XMM Det XY coordinates
            elif input_unit == "deg" and out_name == "xmm_det":
                interim = self.radec_wcs.all_world2pix(coords, 0)
                out_coord = Quantity(self.detxy_wcs.all_pix2world(interim, 0), xmm_sky)
            elif input_unit == "xmm_det" and out_name == "deg":
                interim = self.detxy_wcs.all_world2pix(coords, 0)
                out_coord = Quantity(self.radec_wcs.all_pix2world(interim, 0), deg)

            # These go between XMM det XY and pixel coordinates
            elif input_unit == "xmm_det" and out_name == "pix":
                out_coord = Quantity(self.detxy_wcs.all_world2pix(coords, 0), output_unit).round(0).astype(int)
            elif input_unit == "pix" and out_name == "xmm_det":
                out_coord = Quantity(self.detxy_wcs.all_pix2world(coords, 0), output_unit)

            # It is possible to convert between XMM coordinates and pixel and supply coordinates
            # outside the range covered by an image, but we can at least catch the error
            if out_name == "pix" and np.any(out_coord < 0) and self._prod_type != "psf":
                raise ValueError("You've converted to pixel coordinates, and some elements are less than zero.")
            # Have to compare to the [1] element of shape because numpy arrays are flipped and we want
            #  to compare x to x
            elif out_name == "pix" and np.any(out_coord[:, 0].value> self.shape[1]) and self._prod_type != "psf":
                raise ValueError("You've converted to pixel coordinates, and some x coordinates are larger than the "
                                 "image x-shape.")
            # Have to compare to the [0] element of shape because numpy arrays are flipped and we want
            #  to compare y to y
            elif out_name == "pix" and np.any(out_coord[:, 1].value > self.shape[0]) and self._prod_type != "psf":
                raise ValueError("You've converted to pixel coordinates, and some y coordinates are larger than the "
                                 "image y-shape.")

            # If there was only pair passed in, we'll return a flat numpy array
            if out_coord.shape == (1, 2):
                out_coord = out_coord[0, :]

            # if out_coord.shape ==
        elif input_unit == out_name and out_name == 'pix':
            out_coord = coords.round(0).astype(int)
        else:
            out_coord = coords

        return out_coord

    @property
    def psf_corrected(self) -> bool:
        """
        Tells the user (and XGA), whether an Image based object has been PSF corrected or not.

        :return: Boolean flag, True means this object has been PSF corrected, False means it hasn't
        :rtype: bool
        """
        return self._psf_corrected

    @psf_corrected.setter
    def psf_corrected(self, new_val):
        """
        Allows the psf_corrected flag to be altered.
        """
        self._psf_corrected = new_val

    @property
    def psf_algorithm(self) -> Union[str, None]:
        """
        If this object has been PSF corrected, this property gives the name of the algorithm used.

        :return: The name of the algorithm used to correct for PSF effects, or None if the object
            hasn't been PSF corrected.
        :rtype: Union[str, None]
        """
        return self._psf_correction_algorithm

    @psf_algorithm.setter
    def psf_algorithm(self, new_val: str):
        """
        If this object has been PSF corrected, this property setter allows you to set the
        name of the algorithm used. If it hasn't been PSF corrected then an error will be triggered.
        """
        if self._psf_corrected:
            self._psf_correction_algorithm = new_val
        else:
            raise NotPSFCorrectedError("You are trying to set the PSF Correction algorithm for an Image"
                                       " that hasn't been PSF corrected.")

    @property
    def psf_bins(self) -> Union[int, None]:
        """
        If this object has been PSF corrected, this property gives the number of bins that the X and Y axes
        were divided into to generate the PSFGrid.

        :return: The number of bins in X and Y for which PSFs were generated, or None if the object
            hasn't been PSF corrected.
        :rtype: Union[int, None]
        """
        return self._psf_num_bins

    @psf_bins.setter
    def psf_bins(self, new_val: int):
        """
        If this object has been PSF corrected, this property setter allows you to store the
        number of bins in X and Y for which PSFs were generated. If it hasn't been PSF corrected
        then an error will be triggered.
        """
        if self._psf_corrected:
            self._psf_num_bins = new_val
        else:
            raise NotPSFCorrectedError("You are trying to set the number of PSF bins for an Image"
                                       " that hasn't been PSF corrected.")

    @property
    def psf_iterations(self) -> Union[int, None]:
        """
        If this object has been PSF corrected, this property gives the number of iterations that the
        algorithm went through to create this image.

        :return: The number of iterations the PSF correction algorithm went through, or None if the
            object hasn't been PSF corrected.
        :rtype: Union[int, None]
        """
        return self._psf_num_iterations

    @psf_iterations.setter
    def psf_iterations(self, new_val: int):
        """
        If this object has been PSF corrected, this property setter allows you to store the
        number of iterations that the algorithm went through to create this image. If it hasn't
        been PSF corrected then an error will be triggered.
        """
        if self._psf_corrected:
            self._psf_num_iterations = new_val
        else:
            raise NotPSFCorrectedError("You are trying to set the number of algorithm iterations for an Image"
                                       " that hasn't been PSF corrected.")

    @property
    def psf_model(self) -> Union[str, None]:
        """
        If this object has been PSF corrected, this property gives the name of the PSF model used.

        :return: The name of the PSF model used to correct for PSF effects, or None if the object
            hasn't been PSF corrected.
        :rtype: Union[str, None]
        """
        return self._psf_model

    @psf_model.setter
    def psf_model(self, new_val: str):
        """
        If this object has been PSF corrected, this property setter allows you to add the
        name of the PSF model used. If it hasn't been PSF corrected then an error will be triggered.
        """
        if self._psf_corrected:
            self._psf_model = new_val
        else:
            raise NotPSFCorrectedError("You are trying to set the PSF model for an Image that hasn't "
                                       "been PSF corrected.")

    def get_count(self, at_coord: Quantity) -> float:
        """
        A simple method that converts the given coordinates to pixels, then finds the number of counts
        at those coordinates.

        :param Quantity at_coord: Coordinate at which to find the number of counts.
        :return: The counts at the supplied coordinates.
        :rtype: Quantity
        """
        pix_coord = self.coord_conv(at_coord, pix).value
        cts = self.data[pix_coord[1], pix_coord[0]]
        return Quantity(cts, "ct")

    def simple_peak(self, mask: np.ndarray, out_unit: Union[UnitBase, str] = deg) -> Tuple[Quantity, bool]:
        """
        Simplest possible way to find the position of the peak of X-ray emission in an Image. This method
        takes a mask in the form of a numpy array, which allows the user to mask out parts of the ratemap
        that shouldn't be searched (outside of a certain region, or within point sources for instance).

        Results from this can be less valid than the RateMap implementation (especially if the object you care
        about is off-axis), as that takes into account vignetting corrected exposure times.

        :param np.ndarray mask: A numpy array used to weight the data. It should be 0 for pixels that
            aren't to be searched, and 1 for those that are.
        :param UnitBase/str out_unit: The desired output unit of the peak coordinates, the default is degrees.
        :return: An astropy quantity containing the coordinate of the X-ray peak of this ratemap (given
            the user's mask), in units of out_unit, as specified by the user. A null value is also returned in
            place of the boolean flag describing whether the coordinates are near an edge or not that RateMap returns.
        :rtype: Tuple[Quantity, None]
        """
        # The code is essentially identical to that in simple_peak in RateMap, but I'm tired and can't be bothered
        #  to do this properly so I'll just copy it over
        if mask.shape != self.data.shape:
            raise ValueError("The shape of the mask array ({0}) must be the same as that of the data array "
                             "({1}).".format(mask.shape, self.data.shape))

        # Creates the data array that we'll be searching. Takes into account the passed mask
        masked_data = self.data * mask

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
            peak_conv = peak_pix.astype(int)

        return peak_conv, None

    def get_view(self, ax: Axes, cross_hair: Quantity = None, mask: np.ndarray = None,
                 chosen_points: np.ndarray = None, other_points: List[np.ndarray] = None, zoom_in: bool = False,
                 manual_zoom_xlims: tuple = None, manual_zoom_ylims: tuple = None,
                 radial_bins_pix: np.ndarray = np.array([]), back_bin_pix: np.ndarray = None,
                 stretch: BaseStretch = LogStretch(), mask_edges: bool = True, view_regions: bool = False,
                 ch_thickness: float = 0.8) -> Axes:
        """
        The method that creates and populates the view axes, separate from actual view so outside methods
        can add a view to other matplotlib axes.

        :param Axes ax: The matplotlib axes on which to show the image.
        :param Quantity cross_hair: An optional parameter that can be used to plot a cross hair at
            the coordinates. Up to two cross-hairs can be plotted, as any more can be visually confusing. If
            passing two, each row of a quantity is considered to be a separate coordinate pair.
        :param np.ndarray mask: Allows the user to pass a numpy mask and view the masked
            data if they so choose.
        :param np.ndarray chosen_points: A numpy array of a chosen point cluster from a hierarchical peak finder.
        :param list other_points: A list of numpy arrays of point clusters that weren't chosen by the
            hierarchical peak finder.
        :param bool zoom_in: Sets whether the figure limits should be set automatically so that borders with no
            data are reduced.
        :param tuple manual_zoom_xlims: If set, this will override the automatic zoom in and manually set a part
            of the x-axis to limit the image to, default is None. Pass a tuple with two elements, first being the
            lower limit, second the upper limit. Variable zoom_in must still be true for these limits
            to be applied.
        :param tuple manual_zoom_ylims: If set, this will override the automatic zoom in and manually set a part
            of the y-axis to limit the image to, default is None. Pass a tuple with two elements, first being the
            lower limit, second the upper limit. Variable zoom_in must still be true for these limits
            to be applied.
        :param np.ndarray radial_bins_pix: Radii (in units of pixels) of annuli to plot on top of the image, will
            only be triggered if a cross_hair coordinate is also specified and contains only one coordinate.
        :param np.ndarray back_bin_pix: The inner and outer radii (in pixel units) of the annulus used to measure
            the background value for a given profile, will only be triggered if a cross_hair coordinate is
            also specified and contains only one coordinate.
        :param BaseStretch stretch: The astropy scaling to use for the image data, default is log.
        :param bool mask_edges: If viewing a RateMap, this variable will control whether the chip edges are masked
            to remove artificially bright pixels, default is True.
        :param bool view_regions: If regions have been associated with this object (either on init or using
            the 'regions' property setter, should they be displayed. Default is False.
        :param float ch_thickness: The desired linewidth of the crosshair(s), can be useful to increase this in
            certain circumstances. Default is 0.8.
        :return: A populated figure displaying the view of the data.
        :rtype: Axes
        """

        if mask is not None and mask.shape != self.data.shape:
            raise ValueError("The shape of the mask array ({0}) must be the same as that of the data array "
                             "({1}).".format(mask.shape, self.data.shape))
        elif mask is not None and mask.shape == self.data.shape:
            plot_data = self.data * mask
        else:
            plot_data = self.data

        # If we're showing a RateMap, then we're gonna apply an edge mask to remove all the artificially brightened
        #  pixels that we can - it makes the view look better
        if type(self) == RateMap and mask_edges:
            plot_data *= self.edge_mask

        ax.tick_params(axis='both', direction='in', which='both', top=False, right=False)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # Check if this is a combined product, because if it is then ObsID and instrument are both 'combined'
        #  and it makes the title ugly
        if self.obs_id == "combined":
            ident = 'Combined'
        else:
            ident = "{o} {i}".format(o=self.obs_id, i=self.instrument.upper())

        if self.src_name is not None:
            title = "{n} - {i} {l}-{u}keV {t}".format(n=self.src_name, i=ident,
                                                      l=self._energy_bounds[0].to("keV").value,
                                                      u=self._energy_bounds[1].to("keV").value, t=self.type)
        else:
            title = "{i} {l}-{u}keV {t}".format(i=ident, l=self._energy_bounds[0].to("keV").value,
                                                u=self._energy_bounds[1].to("keV").value, t=self.type)

        # Its helpful to be able to distinguish PSF corrected image/ratemaps from the title
        if self.psf_corrected:
            title += ' - PSF Corrected'

        # And smoothed as well
        if self.smoothed:
            title += ' - Smoothed'

        ax.set_title(title)

        # As this is a very quick view method, users will not be offered a choice of scaling
        #  There will be a more in depth way of viewing cluster data eventually
        norm = ImageNormalize(data=plot_data, interval=MinMaxInterval(), stretch=stretch)
        # I normalize with a log stretch, and use gnuplot2 colormap (pretty decent for clusters imo)

        # If we want to plot point clusters on the image, then we go here
        if chosen_points is not None:
            # Add the point cluster points
            ax.plot(chosen_points[:, 0], chosen_points[:, 1], '+', color='black', label="Chosen Point Cluster")
            ax.legend(loc="best")

        if other_points is not None:
            for cl in other_points:
                ax.plot(cl[:, 0], cl[:, 1], 'D')

        # If we want a cross hair, then we put one on here
        if cross_hair is not None:
            # For the case of a single coordinate
            if cross_hair.shape == (2,):
                # Converts from whatever input coordinate to pixels
                pix_coord = self.coord_conv(cross_hair, pix).value
                # Drawing the horizontal and vertical lines
                ax.axvline(pix_coord[0], color="white", linewidth=ch_thickness)
                ax.axhline(pix_coord[1], color="white", linewidth=ch_thickness)

                # Drawing annular radii on the image, if they are enabled and passed. Only works with a
                #  single coordinate, otherwise we wouldn't know which to centre on
                for ann_rad in radial_bins_pix:
                    artist = Circle(pix_coord, ann_rad, fill=False, ec='white', linewidth=1.5)
                    ax.add_artist(artist)

                # This draws the background region on as well, if present
                if back_bin_pix is not None:
                    inn_artist = Circle(pix_coord, back_bin_pix[0], fill=False, ec='white', linewidth=1.6,
                                        linestyle='dashed')
                    out_artist = Circle(pix_coord, back_bin_pix[1], fill=False, ec='white', linewidth=1.6,
                                        linestyle='dashed')
                    ax.add_artist(inn_artist)
                    ax.add_artist(out_artist)

            # For the case of two coordinate pairs
            elif cross_hair.shape == (2, 2):
                # Converts from whatever input coordinate to pixels
                pix_coord = self.coord_conv(cross_hair, pix).value

                # This draws the first crosshair
                ax.axvline(pix_coord[0, 0], color="white", linewidth=ch_thickness)
                ax.axhline(pix_coord[0, 1], color="white", linewidth=ch_thickness)

                # And this the second
                ax.axvline(pix_coord[1, 0], color="white", linewidth=ch_thickness, linestyle='dashed')
                ax.axhline(pix_coord[1, 1], color="white", linewidth=ch_thickness, linestyle='dashed')

            else:
                # I don't want to bring someone's code grinding to a halt just because they passed crosshair wrong,
                #  it isn't essential so I'll just display a warning
                warnings.warn("You have passed a cross_hair quantity that has more than two coordinate "
                              "pairs in it, or is otherwise the wrong shape.")

        # Adds the actual image to the axis.
        ax.imshow(plot_data, norm=norm, origin="lower", cmap="gnuplot2")

        # If the user wants regions on the image, this is where they get added
        if view_regions:
            # We can just loop through the _regions attribute because its default is an empty
            #  list, so no need to check
            for reg in self._regions:
                # Use the regions module conversion method to go to a matplotlib artist
                reg_art = reg.as_artist()
                # Set line thickness and add to the axes
                reg_art.set_linewidth(1.4)
                ax.add_artist(reg_art)

        # This sets the limits of the figure depending on the options that have been passed in
        if zoom_in and manual_zoom_xlims is None and manual_zoom_ylims is None:
            # I don't like doing local imports, but this is the easiest way
            from xga.imagetools import data_limits
            x_lims, y_lims = data_limits(plot_data)
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
        elif zoom_in and manual_zoom_xlims is not None and manual_zoom_ylims is not None:
            ax.set_xlim(manual_zoom_xlims)
            ax.set_ylim(manual_zoom_ylims)
        elif zoom_in and manual_zoom_xlims is not None and manual_zoom_ylims is None:
            ax.set_xlim(manual_zoom_xlims)
        elif zoom_in and manual_zoom_xlims is None and manual_zoom_ylims is not None:
            ax.set_ylim(manual_zoom_ylims)

        return ax

    def view(self, cross_hair: Quantity = None, mask: np.ndarray = None, chosen_points: np.ndarray = None,
             other_points: List[np.ndarray] = None, figsize: Tuple = (10, 8), zoom_in: bool = False,
             manual_zoom_xlims: tuple = None, manual_zoom_ylims: tuple = None,
             radial_bins_pix: np.ndarray = np.array([]), back_bin_pix: np.ndarray = None,
             stretch: BaseStretch = LogStretch(), mask_edges: bool = True, view_regions: bool = False,
             ch_thickness: float = 0.8):
        """
        Powerful method to view this Image/RateMap/Expmap, with different options that can be used for eyeballing
        and producing figures for publication.

        :param Quantity cross_hair: An optional parameter that can be used to plot a cross hair at
            the coordinates. Up to two cross-hairs can be plotted, as any more can be visually confusing. If
            passing two, each row of a quantity is considered to be a separate coordinate pair.
        :param np.ndarray mask: Allows the user to pass a numpy mask and view the masked
            data if they so choose.
        :param np.ndarray chosen_points: A numpy array of a chosen point cluster from a hierarchical peak finder.
        :param list other_points: A list of numpy arrays of point clusters that weren't chosen by the
            hierarchical peak finder.
        :param Tuple figsize: Allows the user to pass a custom size for the figure produced by this method.
        :param bool zoom_in: Sets whether the figure limits should be set automatically so that borders with no
            data are reduced.
        :param tuple manual_zoom_xlims: If set, this will override the automatic zoom in and manually set a part
            of the x-axis to limit the image to, default is None. Pass a tuple with two elements, first being the
            lower limit, second the upper limit. Variable zoom_in must still be true for these limits
            to be applied.
        :param tuple manual_zoom_ylims: If set, this will override the automatic zoom in and manually set a part
            of the y-axis to limit the image to, default is None. Pass a tuple with two elements, first being the
            lower limit, second the upper limit. Variable zoom_in must still be true for these limits
            to be applied.
        :param np.ndarray radial_bins_pix: Radii (in units of pixels) of annuli to plot on top of the image, will
            only be triggered if a cross_hair coordinate is also specified and contains only one coordinate.
        :param np.ndarray back_bin_pix: The inner and outer radii (in pixel units) of the annulus used to measure
            the background value for a given profile, will only be triggered if a cross_hair coordinate is
            also specified and contains only one coordinate.
        :param BaseStretch stretch: The astropy scaling to use for the image data, default is log.
        :param bool mask_edges: If viewing a RateMap, this variable will control whether the chip edges are masked
            to remove artificially bright pixels, default is True.
        :param bool view_regions: If regions have been associated with this object (either on init or using
            the 'regions' property setter, should they be displayed. Default is False.
        :param float ch_thickness: The desired linewidth of the crosshair(s), can be useful to increase this in
            certain circumstances. Default is 0.8.
        """

        # Create figure object
        fig = plt.figure(figsize=figsize)

        # Turns off any ticks and tick labels, we don't want them in an image
        ax = plt.gca()

        ax = self.get_view(ax, cross_hair, mask, chosen_points, other_points, zoom_in, manual_zoom_xlims,
                           manual_zoom_ylims, radial_bins_pix, back_bin_pix, stretch, mask_edges, view_regions,
                           ch_thickness)
        plt.colorbar(ax.images[0])
        plt.tight_layout()
        # Display the image
        plt.show()

        # Wipe the figure
        plt.close("all")


class ExpMap(Image):
    """
    A very simple subclass of the Image product class - designed to allow for easy interaction with exposure maps.
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, lo_en: Quantity, hi_en: Quantity):
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, lo_en, hi_en)
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

    def get_count(self, *args, **kwargs):
        """
        Inherited method from Image superclass, disabled as does not make sense in an exposure map.
        """
        pass

    @property
    def smoothing_info(self) -> None:
        """
        As exposure maps cannot be smoothed, this property overrides the Image smoothing_info getter, and
        returns None.

        :return: None, as exposure maps cannot be smoothed.
        :rtype: None
        """
        return None

    @smoothing_info.setter
    def smoothing_info(self, new_info):
        """
        As exposure maps cannot be smoothed, this property overrides the Image smoothing_info setter, and will
        raise a TypeError if you try to pass any smoothing information to this object.

        :param new_info: The new smoothing information to be added to the product.
        """
        raise TypeError("ExpMap products cannot be smoothed, and as such cannot have smoothing info added.")


class RateMap(Image):
    """
    A very powerful class which allows interactions with 'RateMaps', though these are not directly generated by
    SAS, they are images divided by matching exposure maps, to provide a count rate image.
    """
    def __init__(self, xga_image: Image, xga_expmap: ExpMap, reg_file_path: str = ''):
        """
        This initialises a RateMap instance, where a count-rate image is divided by an exposure map, to create a map
        of X-ray counts.

        :param Image xga_image: The image component of the RateMap.
        :param ExpMap xga_expmap: The exposure map component of the RateMap.
        :param str reg_file_path:
        """
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

        # Reading in the PSF status from the Image passed in
        self._psf_corrected = xga_image.psf_corrected
        self._psf_model = xga_image.psf_model
        self._psf_num_bins = xga_image.psf_bins
        self._psf_num_iterations = xga_image.psf_iterations
        self._psf_correction_algorithm = xga_image.psf_algorithm

        self._path = xga_image.path
        self._im_path = xga_image.path
        self._expmap_path = xga_expmap.path

        self._im_obj = xga_image
        self._ex_obj = xga_expmap

        self._data = None
        # TODO Could I combine these two, and just make edges = 2, on sensor = 1 etc?
        self._edge_mask = None
        self._on_sensor_mask = None

        # Don't have to do any checks, they'll be done for me in the image object.
        self._im_obj.regions = reg_file_path

    def _construct_on_demand(self):
        """
        This method is complimentary to the _read_on_demand method of the base Image class, and ensures that
        the ratemap array is only created if the user actually asks for it. Otherwise a lot of time is wasted
        reading in files for individual images and exposure maps that are rarely used.
        """
        # This helps avoid a circular import issue
        from ..imagetools.misc import edge_finder

        # Divide image by exposure map to get rate map data.
        # Numpy divide lets me specify where we wish to divide, so we don't get any NaN results and divide by
        #  zero warnings
        # Set up an array of zeros to receive the ratemap data when its calculated
        self._data = np.zeros(self.image.shape)
        # Need to set the shape attribute as I no longer call _read_on_demand directly for this object
        self._shape = self._data.shape
        np.divide(self.image.data, self.expmap.data, out=self._data, where=self.expmap.data != 0)

        # Use exposure maps and basic edge detection to find the edges of the CCDs
        #  The exposure map values calculated on the edge of a CCD can be much smaller than it should be,
        #  which in turn can boost the rate map value there - hence useful to know which elements of an array
        #  are on an edge.
        comb = edge_finder(self.expmap, keep_corners=True)

        # Possible values of 0 (no edge), 1 (edge detected in one pass), and 2 (edge detected in both pass). Then
        # configure the array to act as a mask that removes the edge pixels
        comb[comb == 0] = -1
        comb[comb != -1] = False
        comb[comb == -1] = 1

        # Store that edge mask as an attribute.
        self._edge_mask = comb

        # This sets every element of the exposure map that isn't zero to one, that way it becomes a simple
        #  mask as to whether you are on or off the sensor
        det_map = self.expmap.data.copy()
        det_map[det_map != 0] = 1
        self._on_sensor_mask = det_map

        # Re-setting some paths to make more sense
        self._path = self._im_path

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Property getter for the resolution of the ratemap. Standard XGA settings will make this 512x512.

        :return: The shape of the numpy array describing the ratemap.
        :rtype: Tuple[int, int]
        """
        if self._data is None:
            self._construct_on_demand()
        # There will not be a setter for this property, no-one is allowed to change the shape of the image.
        return self._shape

    @property
    def data(self) -> np.ndarray:
        """
        Property getter for ratemap data, overrides the method in the base Image class. This is because
        the ratemap class has a _construct_on_demand method that creates the ratemap data, which needs
        to be called instead of _read_on_demand.

        :return: A numpy array of shape self.shape containing the ratemap data.
        :rtype: np.ndarray
        """
        # Calling this ensures the image object is read into memory
        if self._data is None:
            self._construct_on_demand()
        return self._data.copy()

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
        return Quantity(rate, "ct/s^-1")

    def get_count(self, at_coord: Quantity) -> float:
        """
        A simple method that converts the given coordinates to pixels, then finds the number of counts
        at those coordinates.

        :param Quantity at_coord: Coordinate at which to find the number of counts.
        :return: The counts at the supplied coordinates.
        :rtype: Quantity
        """
        pix_coord = self.coord_conv(at_coord, pix).value
        cts = self.image.data[pix_coord[1], pix_coord[0]]
        return Quantity(cts, "ct")

    def get_exp(self, at_coord: Quantity) -> float:
        """
        A simple method that converts the given coordinates to pixels, then finds the exposure time
        at those coordinates.

        :param Quantity at_coord: A pair of coordinates to find the exposure time for.
        :return: The exposure time at the supplied coordinates.
        :rtype: Quantity
        """
        pix_coord = self.coord_conv(at_coord, pix).value
        exp = self.expmap.data[pix_coord[1], pix_coord[0]]
        return Quantity(exp, "s")

    def simple_peak(self, mask: np.ndarray, out_unit: Union[UnitBase, str] = deg) -> Tuple[Quantity, bool]:
        """
        Simplest possible way to find the position of the peak of X-ray emission in a ratemap. This method
        takes a mask in the form of a numpy array, which allows the user to mask out parts of the ratemap
        that shouldn't be searched (outside of a certain region, or within point sources for instance).

        :param np.ndarray mask: A numpy array used to weight the data. It should be 0 for pixels that
            aren't to be searched, and 1 for those that are.
        :param UnitBase/str out_unit: The desired output unit of the peak coordinates, the default is degrees.
        :return: An astropy quantity containing the coordinate of the X-ray peak of this ratemap (given
            the user's mask), in units of out_unit, as specified by the user. Also returned is a boolean flag
            that tells the caller if the peak is near a chip edge.
        :rtype: Tuple[Quantity, bool]
        """
        if mask.shape != self.data.shape:
            raise ValueError("The shape of the mask array ({0}) must be the same as that of the data array "
                             "({1}).".format(mask.shape, self.data.shape))

        # Creates the data array that we'll be searching. Takes into account the passed mask, as well as
        #  the edge mask designed to remove pixels at the edges of detectors, where RateMap values can
        #  be artificially boosted.
        masked_data = self.data * mask * self.edge_mask

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
            peak_conv = peak_pix.astype(int)

        # Find if the peak coordinates sit near an edge/chip gap
        edge_flag = self.near_edge(peak_pix)

        return peak_conv, edge_flag

    def clustering_peak(self, mask: np.ndarray, out_unit: Union[UnitBase, str] = deg, top_frac: float = 0.05,
                        max_dist: float = 5, clean_point_clusters: bool = False) \
            -> Tuple[Quantity, bool, np.ndarray, List[np.ndarray]]:
        """
        An experimental peak finding function that cuts out the top 5% (by default) of array elements
        (by value), and runs a hierarchical clustering algorithm on their positions. The motivation
        for this is that the cluster peak will likely be contained in that top 5%, and the only other
        pixels that might be involved are remnants of poorly removed point sources. So when clusters have
        been formed, we can take the one with the most entries, and find the maximal pixel of that cluster.
        Should be consistent with simple_peak under ideal circumstances.

        :param np.ndarray mask: A numpy array used to weight the data. It should be 0 for pixels that
            aren't to be searched, and 1 for those that are.
        :param UnitBase/str out_unit: The desired output unit of the peak coordinates, the default is degrees.
        :param float top_frac: The fraction of the elements (ordered in descending value) that should be used
            to generate clusters, and thus be considered for the cluster centre.
        :param float max_dist: The maximum distance criterion for the hierarchical clustering algorithm, in pixels.
        :param bool clean_point_clusters: If this is set to true then the point clusters which are not believed
            to host the peak pixel will be cleaned, meaning that if they have less than 4 pixels associated with
            them then they will be removed.
        :return: An astropy quantity containing the coordinate of the X-ray peak of this ratemap (given
            the user's mask), in units of out_unit, as specified by the user. Finally, the coordinates of the points
            in the chosen cluster are returned, as is a list of all the coordinates of all the other clusters.
        :rtype: Tuple[Quantity, bool]
        """
        if mask.shape != self.data.shape:
            raise ValueError("The shape of the mask array ({0}) must be the same as that of the data array "
                             "({1}).".format(mask.shape, self.data.shape))

        # Creates the data array that we'll be searching. Takes into account the passed mask, as well as
        #  the edge mask designed to remove pixels at the edges of detectors, where RateMap values can
        #  be artificially boosted.
        masked_data = self.data * mask * self.edge_mask
        # How many non-zero elements are there in the array
        num_value = len(masked_data[masked_data != 0])
        # Find the number that corresponds to the top 5% (by default)
        to_select = round(num_value * top_frac)
        # Grab the inds of the pixels that are in the top 5% of values (by default)
        inds = np.unravel_index(np.argpartition(masked_data.flatten(), -to_select)[-to_select:], masked_data.shape)
        # Just formatting quickly for input into the clustering algorithm
        pairs = [[inds[0][i], inds[1][i]] for i in range(len(inds[0]))]

        # USED TO USE Hierarchical clustering using the inconsistent criterion with threshold 1. 'If a
        # cluster node and all its descendants have an inconsistent value less than or equal to 1, then all its
        # leaf descendants belong to the same flat cluster. When no non-singleton cluster meets this criterion,
        # every node is assigned to its own cluster.'

        # Now use a default distance criterion of 5 pixels (maximum intra point cluster distance of 5 pixels),
        #  which works better for low surface brightness clusters. This may still change in the future as I refine
        #  it, but its working well for now!
        cluster_inds = fclusterdata(pairs, max_dist, criterion="distance")

        # Finds how many clusters there are, and how many points belong to each cluster
        uniq_vals, uniq_cnts = np.unique(cluster_inds, return_counts=True)
        # Choose the cluster with the most points associated with it
        chosen_clust = uniq_vals[np.argmax(uniq_cnts)]
        # Retrieves the inds for the main merged_data in the chosen cluster
        chosen_inds = np.where(cluster_inds == chosen_clust)[0]
        # X column, then Y column - these are pixel coordinates
        chosen_coord_pairs = np.stack([inds[1][chosen_inds], inds[0][chosen_inds]]).T

        # Grabbing the none chosen point cluster indexes
        other_clusts = [np.where(cluster_inds == cl)[0] for cl in uniq_vals if cl != chosen_clust]
        # And more importantly the coordinates of the none chosen point clusters
        other_coord_pairs = [np.stack([inds[1][cl_ind], inds[0][cl_ind]]).T for cl_ind in other_clusts]

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

        if clean_point_clusters:
            cleaned_clusters = []
            for pcl in other_coord_pairs:
                if len(pcl) > 4:
                    cleaned_clusters.append(pcl)

            other_coord_pairs = cleaned_clusters

        return peak_conv, edge_flag, chosen_coord_pairs, other_coord_pairs

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
        conv_data = fftconvolve(self.data*self.edge_mask, filt)[n_cut:-n_cut, n_cut:-n_cut]
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
        # The try except is to catch instances where the pix coord conversion fails because its off an image,
        #  in which case we classify it as near an edge
        try:
            # Convert to pixel coordinates
            pix_coord = self.coord_conv(coord, pix).value

            # Checks the edge mask within a 5 by 5 array centered on the peak coord, if there are no edges then
            #  all elements will be 1 and it will sum to 25.
            edge_sum = self.edge_mask[pix_coord[1] - 2:pix_coord[1] + 3,
                                      pix_coord[0] - 2:pix_coord[0] + 3].sum()
            # If it sums to less then we know that there is an edge near the peak.
            if edge_sum != 25:
                edge_flag = True
            else:
                edge_flag = False

        except ValueError:
            edge_flag = True

        return edge_flag

    def signal_to_noise(self, source_mask: np.ndarray, back_mask: np.ndarray, exp_corr: bool = True,
                        allow_negative: bool = False):
        """
        A signal to noise calculation method which takes information on source and background regions, then uses
        that to calculate a signal to noise for the source. This was primarily motivated by the desire to produce
        valid SNR values for combined data, where uneven exposure times across the combined field of view could
        cause issues with the usual approach of just summing the counts in the region images and scaling by area.
        This method can also measure signal to noises without exposure time correction.

        :param np.ndarray source_mask: The mask which defines the source region, ideally with interlopers removed.
        :param np.ndarray back_mask: The mask which defines the background region, ideally with interlopers removed.
        :param bool exp_corr: Should signal to noises be measured with exposure time correction, default is True. I
            recommend that this be true for combined observations, as exposure time could change quite dramatically
            across the combined product.
        :param bool allow_negative: Should pixels in the background subtracted count map be allowed to go below
            zero, which results in a lower signal to noise (and can result in a negative signal to noise).
        :return: A signal to noise value for the source region.
        :rtype: float
        """
        # Perform some quick checks on the masks to check they are broadly compatible with this ratemap
        if source_mask.shape != self.shape:
            raise ValueError("The source mask shape {sm} is not the same as the ratemap shape "
                             "{rt}!".format(sm=source_mask.shape, rt=self.shape))
        elif not (source_mask >= 0).all() or not (source_mask <= 1).all():
            raise ValueError("The source mask has illegal values in it, there should only be ones and zeros.")
        elif back_mask.shape != self.shape:
            raise ValueError("The background mask shape {bm} is not the same as the ratemap shape "
                             "{rt}!".format(bm=back_mask.shape, rt=self.shape))
        elif not (back_mask >= 0).all() or not (back_mask <= 1).all():
            raise ValueError("The background mask has illegal values in it, there should only be ones and zeros.")

        # Find the total mask areas. As the mask is just an array of ones and zeros we can just sum the
        #  whole thing to find the total pixel area covered.
        src_area = (source_mask*self.sensor_mask).sum()
        back_area = (back_mask*self.sensor_mask).sum()

        # Exposure correction takes into account the different exposure times of the individual pixels
        if exp_corr:
            # Find an average background per pixel COUNT RATE by dividing the total cr in the background region by the
            #  number of pixels in the mask
            av_back = (self.data * back_mask).sum() / back_area
            # Then we use the exposure map to create a background COUNT map for the observation, by multiplying the
            #  average background count rate by the exposure map
            scaled_source_back_counts = self.expmap.data * av_back * source_mask
            # Then we create a background subtracted map of the source by subtracting the background map
            source_map = (self.image.data * source_mask) - scaled_source_back_counts
            # Some pixels could be negative now, but if we're not allowing negative values then they get
            #  set to zero
            if not allow_negative:
                source_map[source_map < 0] = 0
            # Then we sum the source count map to find a total source count value, and divide that by the square root
            #  of the total number of counts (NON BACKGROUND SUBTRACTED) within the source mask
            sn = source_map.sum() / np.sqrt((self.image.data * source_mask).sum())
        else:
            # Calculate an area normalisation so the background counts can be scaled to the source counts properly
            area_norm = src_area / back_area
            # Find the total counts within the source area
            tot_cnt = (self.image.data * source_mask).sum()
            # Find the total counts within the background area
            bck_cnt = (self.image.data * back_mask).sum()

            # Signal to noise is then just finding the source counts by subtracting the area scaled background counts
            #  and dividing by the square root of the total counts within the source area
            sn = (tot_cnt - bck_cnt*area_norm) / np.sqrt(tot_cnt)

        return sn

    @property
    def edge_mask(self) -> np.ndarray:
        """
        Returns the edge mask calculated for this RateMap in the form of a numpy array

        :return: A boolean numpy array in the same shape as the RateMap.
        :rtype: ndarray
        """
        if self._edge_mask is None:
            self._construct_on_demand()
        return self._edge_mask

    @property
    def sensor_mask(self) -> np.ndarray:
        """
        Returns the detector map calculated for this RateMap. Values of 1 mean on chip,
        values of 0 mean off chip.

        :return: A boolean numpy array in the same shape as the RateMap.
        :rtype: ndarray
        """
        if self._on_sensor_mask is None:
            self._construct_on_demand()
        return self._on_sensor_mask

    @property
    def expmap_path(self) -> str:
        """
        Similar to the path property, but for the exposure map that went into this ratemap.

        :return: The exposure map path.
        :rtype: str
        """
        return self._expmap_path

    @property
    def image(self) -> Image:
        """
        This property allows the user to access the input Image object for this ratemap.

        :return: The input XGA Image object used to create this ratemap.
        :rtype: Image
        """
        return self._im_obj

    @property
    def expmap(self) -> ExpMap:
        """
        This property allows the user to access the input ExpMap object for this ratemap.

        :return: The input XGA ExpMap object used to create this ratemap.
        :rtype: ExpMap
        """
        return self._ex_obj


class PSF(Image):
    def __init__(self, path: str, psf_model: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str):
        lo_en = Quantity(0, 'keV')
        hi_en = Quantity(100, 'keV')
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, lo_en, hi_en)
        self._prod_type = "psf"
        self._psf_centre = None
        self._psf_model = psf_model

    def get_val(self, at_coord: Quantity) -> float:
        """
        A simple method that converts the given coordinates to pixels, then finds the exposure time
        at those coordinates.

        :param Quantity at_coord: A pair of coordinates to find the exposure time for.
        :return: The exposure time at the supplied coordinates.
        :rtype: Quantity
        """
        pix_coord = self.coord_conv(at_coord, pix).value
        neg_ind = np.where(pix_coord < 0)
        # This wouldn't deal with PSF images that weren't the same size in x and y, but XGA
        #  doesn't generate PSFs like that so its fine.
        too_big_ind = np.where(pix_coord >= self.shape[0])

        pix_coord[too_big_ind[0], :] = [0, 0]
        pix_coord[neg_ind[0], :] = [0, 0]

        val = self.data[pix_coord[:, 1], pix_coord[:, 0]]
        # This is a difficult decision, what to do about requested coordinates that are outside the PSF range.
        #  I think I'm going to set those coordinates to the minimum PSF value
        val[too_big_ind[0]] = 0
        val[neg_ind[0]] = 0
        return val

    def resample(self, im_prod: Image, half_side_length: Quantity) -> np.ndarray:
        """
        This method resamples a psfgen created PSF image to the same scale as the passed Image object. This
        is very important because psfgen makes these PSF images with a standard pixel size of 1 arcsec x 1 arcsec,
        and it can't be changed when calling the routine. Thankfully, due to the wonders of WCS, it is possible
        to construct a new array with the same pixel size as a given image. Very important for when we want
        to deconvolve with an image and correct for the PSF.

        :param Image im_prod:
        :param Quantity half_side_length:
        :return: The resampled PSF.
        :rtype: np.ndarray
        """
        if im_prod.obs_id != self.obs_id:
            raise IncompatibleProductError("Image ObsID ({o1}) is not the same as PSF ObsID "
                                           "({o2})".format(o1=im_prod.obs_id, o2=self.obs_id))
        elif im_prod.instrument != self.instrument:
            raise IncompatibleProductError("Image instrument ({i1}) is not the same as PSF instrument "
                                           "({i2})".format(i1=im_prod.instrument, i2=self.instrument))
        if half_side_length.unit != pix:
            raise UnitConversionError("side_length must be in pixels")

        # Location at which the PSF was generated, but in image pixel coordinates
        im_pix_psf_gen = im_prod.coord_conv(self.ra_dec, pix).value

        hs = half_side_length.value.astype(int)
        grid = np.meshgrid(np.arange(im_pix_psf_gen[0] - hs, im_pix_psf_gen[0] + hs),
                           np.arange(im_pix_psf_gen[1] - hs, im_pix_psf_gen[1] + hs))
        pix_coords = Quantity(np.stack([grid[0].ravel(), grid[1].ravel()]).T, 'pix')
        # Just tell the conversion method that I want degrees out, and the image's WCS is used to convert
        deg_coords = im_prod.coord_conv(pix_coords, deg)

        # Now that we're in degrees, it should be possible to look up the PSF values at these positions
        # using the PSF wcs headers
        psf_vals = self.get_val(deg_coords)

        # Now need to reshape the values into a 2D array
        new_psf = np.reshape(psf_vals, (hs*2, hs*2))

        # Renormalising this resampled PSF
        new_psf /= new_psf.sum()

        return new_psf

    @property
    def ra_dec(self) -> Quantity:
        """
        A property that fetches the RA-DEC that the PSF was generated at.

        :return: An astropy quantity of the ra and dec that the PSF was generated at.
        :rtype: Quantity
        """
        if self._psf_centre is None:
            # I've put this here because if there is an error generating the PSF, and this is in the __init__ as
            #  it was before, then the file not found error is triggered before the XGA SAS error and then the user
            #  has no idea whats wrong.
            self._psf_centre = Quantity([self.header.get("CRVAL1"), self.header.get("CRVAL2")], deg)
        return self._psf_centre

    @property
    def model(self) -> str:
        """
        This is the model that was used to generate this PSF.

        :return: XMM SAS psfgen model name.
        :rtype: str
        """
        return self._psf_model


class PSFGrid(BaseAggregateProduct):
    def __init__(self, file_paths: list, bins: int, psf_model: str, x_bounds: np.ndarray, y_bounds: np.ndarray,
                 obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str):
        super().__init__(file_paths, 'psf', obs_id, instrument)
        self._psf_model = psf_model
        # Set none here because if I want positions of PSFs and there has been an error during generation, the user
        #  will only see the FileNotFoundError not the SAS error
        self._grid_loc = None
        self._nbins = bins
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds

        for f_ind, f in enumerate(file_paths):
            # I pass the whole stdout and stderr for each PSF, even though they will include ALL the PSFs in this
            #  grid, its a bit of a bodge but life goes on eh?
            interim = PSF(f, psf_model, obs_id, instrument, stdout_str, stderr_str, gen_cmd)
            # The dictionary key the PSF will be stored under - the key corresponds to the numpy y-x
            #  index from which it was generated
            pos = np.unravel_index(f_ind, (bins, bins))
            pos_key = "_".join([str(p) for p in pos])
            self._component_products[pos_key] = interim

        # I set up the BaseAggregateProduct class to iterate across its dictionary of products,
        #  so thats why I can do for p in self
        # This tells the world whether every single product associated with this AggregateProduct is usable.
        self._all_usable = all(p.usable for p in self)

    # These next two are fundamental properties of the psf files generation process, and
    # can't be changed after the fact.
    @property
    def num_bins(self) -> int:
        """
        Getter for the number of bins in X and Y that this PSFGrid has PSF objects for.

        :return: The number of bins per side used to generate this PSFGrid
        :rtype: int
        """
        return self._nbins

    @property
    def model(self) -> str:
        """
        This is the model that was used to generate the component PSFs in this PSFGrid.

        :return: XMM SAS psfgen model name.
        :rtype: str
        """
        return self._psf_model

    @property
    def x_bounds(self) -> np.ndarray:
        """
        The x lower (column 0) and x upper (column 1) bounds of the PSFGrid bins.
        :return: N x 2 numpy array, where N is the total number of PSFGrid bins.
        :rtype: np.ndarray
        """
        return self._x_bounds

    @property
    def y_bounds(self) -> np.ndarray:
        """
        The y lower (column 0) and y upper (column 1) bounds of the PSFGrid bins.

        :return: N x 2 numpy array, where N is the total number of PSFGrid bins.
        :rtype: np.ndarray
        """
        return self._y_bounds

    @property
    def grid_locs(self) -> Quantity:
        """
        A 3D quantity containing the central position of each PSF in the grid.

        :return: A 3D Quantity
        :rtype:
        """
        if self._grid_loc is None:
            for pos in self._component_products:
                self._grid_loc[pos[0], pos[1], :] = self._component_products.ra_dec
        return self._grid_loc

    def unload_data(self):
        """
        A convenience method that will iterate through the component PSFs of this object and remove their data from
        memory using the data property deleter. This ensures that, if the data needs to be accessed again, the call
        to .data will read in the PSFs and all will be well, hopefully.
        """
        for p in self._component_products:
            del self._component_products[p].data


