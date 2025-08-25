#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 25/08/2025, 15:48. Copyright (c) The Contributors

import os
import warnings
from copy import deepcopy
from typing import Tuple, List, Union, Dict

import numpy as np
import pandas as pd
from astropy import wcs
from astropy.convolution import Kernel, Gaussian2DKernel, convolve_fft
from astropy.units import Quantity, UnitBase, UnitsError, deg, pix, UnitConversionError, Unit
from astropy.visualization import MinMaxInterval, ImageNormalize, BaseStretch, ManualInterval
from astropy.visualization.stretch import LogStretch, SinhStretch, AsinhStretch, SqrtStretch, SquaredStretch, \
    LinearStretch
from fitsio import read, read_header, FITSHDR
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse
from matplotlib.widgets import Button, RangeSlider, Slider
from regions import PixelRegion, SkyRegion, EllipsePixelRegion, CirclePixelRegion, PixCoord, Regions
from scipy.cluster.hierarchy import fclusterdata
from scipy.signal import fftconvolve

from . import BaseProduct, BaseAggregateProduct
from ..exceptions import FailedProductError, RateMapPairError, NotPSFCorrectedError, IncompatibleProductError
from ..sourcetools import ang_to_rad
from ..utils import xmm_sky, xmm_det

EMOSAIC_INST = {"EPN": "pn", "EMOS1": "mos1", "EMOS2": "mos2"}
plt.rcParams['keymap.save'] = ''
plt.rcParams['keymap.quit'] = ''
stretch_dict = {'LOG': LogStretch(), 'SINH': SinhStretch(), 'ASINH': AsinhStretch(), 'SQRT': SqrtStretch(),
                'SQRD': SquaredStretch(), 'LIN': LinearStretch()}


class Image(BaseProduct):
    """
    This class stores image data from X-ray observations. It also allows easy, direct, access to that data, and
    implements many helpful methods with extra functionality (including coordinate transforms, peak finders, and
    a powerful view method).

    :param str path: The path to where the product file SHOULD be located.
    :param str obs_id: The ObsID related to the Image being declared.
    :param str instrument: The instrument related to the Image being declared.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The command used to generate the product.
    :param Quantity lo_en: The lower energy bound used to generate this product.
    :param Quantity hi_en: The upper energy bound used to generate this product.
    :param str/List[SkyRegion/PixelRegion]/dict regs: A region list file path, a list of region objects, or a
        dictionary of region lists with ObsIDs as dictionary keys.
    :param dict/SkyRegion/PixelRegion matched_regs: Similar to the regs argument, but in this case for a region
        that has been designated as 'matched', i.e. is the subject of a current analysis. This should either be
        supplied as a single region object, or as a dictionary of region objects with ObsIDs as keys, or None values
        if there is no match. Such a dictionary can be retrieved from a source using the 'matched_regions'
        property. Default is None.
    :param bool smoothed: Has this image been smoothed, default is False. This information can also be
        set after the instantiation of an image.
    :param dict/Kernel smoothed_info: Information on how the image was smoothed, given either by the Astropy
        kernel used or a dictionary of information (required structure detailed in
        parse_smoothing). Default is None
    :param List[List] obs_inst_combs: Supply a list of lists of ObsID-Instrument combinations if the image
        is combined and wasn't made by emosaic (e.g. [['0404910601', 'pn'], ['0404910601', 'mos1'],
        ['0404910601', 'mos2'], ['0201901401', 'pn'], ['0201901401', 'mos1'], ['0201901401', 'mos2']].
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 lo_en: Quantity, hi_en: Quantity, regs: Union[str, List[Union[SkyRegion, PixelRegion]], dict] = '',
                 matched_regs: Union[SkyRegion, PixelRegion, dict] = None, smoothed: bool = False,
                 smoothed_info: Union[dict, Kernel] = None, obs_inst_combs: List[List] = None):
        """
        The initialisation method for the Image class.
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
        # Adding an attribute to tell the product what its data units are, as there are subclasses of Image
        self._data_unit = Unit("ct")

        # This is a flag to let XGA know that the Image object has been PSF corrected
        self._psf_corrected = False
        # These give extra information about the PSF correction, but can't be set unless PSF
        #  corrected is true
        self._psf_correction_algorithm = None
        self._psf_num_bins = None
        self._psf_num_iterations = None
        self._psf_model = None

        # This checks whether a region file has been passed, and if it has then processes it. If a list or dictionary
        #  of regions has been passed instead (as is allowed) then the behaviour is modified slightly.
        if isinstance(regs, str) and regs != '' and os.path.exists(regs):
            self._regions = self._process_regions(regs)
            self._reg_file_path = regs
        elif isinstance(regs, str) and regs != '' and not os.path.exists(regs):
            warnings.warn("That region file path does not exist", stacklevel=2)
            self._regions = {}
            self._reg_file_path = regs
        elif isinstance(regs, (list, dict)):
            self._regions = self._process_regions(reg_objs=regs)
            self._reg_file_path = ''
        else:
            self._regions = {}
            self._reg_file_path = ''

        # This uses an internal function to process and return matched regions in a standard form, which is what
        #  I really should have done for the chunk above but oh well!
        self._matched_regions = self._process_matched_regions(matched_regs)

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
        # Import here to avoid circular import woes
        from ..imagetools.misc import find_all_wcs

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

    def _process_regions(self, path: str = None, reg_objs: Union[List[Union[PixelRegion, SkyRegion]], dict] = None) \
            -> dict:
        """
        This internal function just takes the path to a region file and processes it into a form that
        this object requires for viewing.

        :param str path: The path of a region file to be processed, can be None but only if the
            other argument is given.
        :param Union[List[PixelRegion/SkyRegion]/dict] reg_objs: A list or dictionary of region objects to be
            processed, default is None.
        :return: A dictionary of lists of pixel regions, with dictionary keys being ObsIDs.
        :rtype: Dict
        """
        # This method can deal with either an input of a region file path or of a list of region objects, but
        #  firstly we need to check that at least one of the inputs isn't None
        if all([path is None, reg_objs is None]):
            raise ValueError("Either a path or a list of region objects must be passed, you have passed neither")
        elif all([path is not None, reg_objs is not None]):
            raise ValueError("You have passed both a path and a list of regions, pass one or the other.")

        # The behaviour here depends on whether regions or a path have been passed
        if path is not None:
            ds9_regs = Regions.read(path, format='ds9').regions
        else:
            ds9_regs = deepcopy(reg_objs)

        # As we now support passing either a dictionary or a list of region objects, some preprocessing is needed
        if type(ds9_regs) == list:
            ds9_regs = {self._obs_id: ds9_regs}
        elif type(ds9_regs) == dict:
            obs_keys = ds9_regs.keys()
            # Checks whether all the ObsIDs present in the XGA product are represented in the region dictionary
            check = [o in obs_keys for o in self.obs_ids]
            if not all(check):
                missing = np.array(self.obs_ids)[~np.array(check)]
                raise KeyError("The passed region dictionary does not have an ObsID entry for every ObsID "
                               "associated with this object, the following are "
                               "missing; {a}.".format(a=','.join(missing)))

        # Checking what kind of regions there are, as that changes whether they need to be converted or not
        final_regs = {}
        # Top level of iteration is through the ObsID keys
        for o in ds9_regs:
            # Setting up an entry in the output dictionary for that ObsID
            final_regs[o] = []
            for reg in ds9_regs[o]:
                if isinstance(reg, PixelRegion):
                    final_regs[o].append(reg)
                else:
                    # Regions in sky coordinates need to be in pixels for overlaying on the image
                    final_regs[o].append(reg.to_pixel(self._wcs_radec))

        return final_regs

    def _process_matched_regions(self, matched_reg_input: Union[SkyRegion, PixelRegion, dict]):
        """
        This processes input matched region information, making sure that it is in an acceptable format, and then
        returning a dictionary in the form expected by this class. Also makes sure that all matched regions are
        converted to pixel coordinates.

        :param SkyRegion/PixelRegion/dict matched_reg_input: A region that has been designated as 'matched', i.e.
            is the subject of a current analysis. This should either be supplied as a single region object, or as
            a dictionary of region objects with ObsIDs as keys, or None values if there is no match. Such a
            dictionary can be retrieved from a source using the 'matched_regions' property.
        :return: A dictionary with ObsIDs as keys, and matching regions as values. If a single region is passed then
            the ObsID key it is paired with is set to the current ObsID of this object.
        :rtype: dict
        """
        # It is possible to set this to None, in which case no information is recorded.
        if matched_reg_input is None:
            matched_reg_input = {}
        # This is triggered when a dictionary is passed, and all of its values are regions or None (indicating no match)
        elif isinstance(matched_reg_input, dict) and all([r is None or isinstance(r, (SkyRegion, PixelRegion))
                                                          for o, r in matched_reg_input.items()]):
            obs_keys = matched_reg_input.keys()
            # Checks whether all the ObsIDs present in the XGA product are represented in the region dictionary
            check = [o in obs_keys for o in self.obs_ids]
            if not all(check):
                missing = np.array(self.obs_ids)[~np.array(check)]
                raise KeyError("The passed matched region dictionary does not have an ObsID entry for every ObsID "
                               "associated with this object, the following are "
                               "missing; {a}.".format(a=','.join(missing)))
        # This is triggered when a dictionary is passed but not all of its values are regions
        elif isinstance(matched_reg_input, dict) and not all([r is None or isinstance(r, (SkyRegion, PixelRegion))
                                                              for o, r in matched_reg_input.items()]):
            raise TypeError('The input matched region dictionary has entries that are not a SkyRegion or PixelRegion.')
        # If one single region is passed, it's put in a dictionary with the current ObsID of the object as the key
        elif isinstance(matched_reg_input, (PixelRegion, SkyRegion)):
            matched_reg_input = {self._obs_id: matched_reg_input}
        else:
            raise TypeError("The input matched region is not a dictionary of regions, nor is it a single "
                            "PixelRegion or SkyRegion instance.")

        # Finally we run through any matched regions that made it this far, and make sure that they
        #  are all in pixel coordinates (it makes it easier for plotting etc. later)
        for obs_id, matched_reg in matched_reg_input.items():
            if matched_reg is not None and not isinstance(matched_reg, PixelRegion):
                matched_reg_input[obs_id] = matched_reg.to_pixel(self._wcs_radec)

        return matched_reg_input

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
    def regions(self) -> Dict:
        """
        Property getter for regions associated with this image.

        :return: Returns a dictionary of regions, if they have been associated with this object.
        :rtype: Dict[PixelRegion]
        """
        return self._regions

    @regions.setter
    def regions(self, new_reg: Union[str, List[Union[SkyRegion, PixelRegion]], dict]):
        """
        A setter for regions associated with this object, a region file path or a list/dict of regions is passed, then
        that file/set of regions is processed into the required format. If a list of regions is passed, it will
        be assumed that they are for the ObsID of the image. In the case of passing a dictionary of regions to a
        combined image we require that each ObsID that goes into the image has an entry in the dictionary.

        :param str/List[SkyRegion/PixelRegion]/dict new_reg: A new region file path, a list of region objects, or a
            dictionary of region lists with ObsIDs as dictionary keys.
        """
        if not isinstance(new_reg, (str, list, dict)):
            raise TypeError("Please pass either a path to a region file, a list of "
                            "SkyRegion/PixelRegion objects, or a dictionary of lists of SkyRegion/PixelRegion objects "
                            "with ObsIDs as keys.")

        # Checks to make sure that a region file path exists, if passed, then processes the file
        if isinstance(new_reg, str) and new_reg != '' and os.path.exists(new_reg):
            self._reg_file_path = new_reg
            self._regions = self._process_regions(new_reg)
        # Possible for an empty string to be passed in which case nothing happens
        elif isinstance(new_reg, str) and new_reg == '':
            pass
        elif isinstance(new_reg, str):
            warnings.warn("That region file path does not exist")
        # If an existing list of regions are passed then we just process them and assign them to regions attribute
        elif isinstance(new_reg, List) and all([isinstance(r, (SkyRegion, PixelRegion)) for r in new_reg]):
            self._reg_file_path = ""
            self._regions = self._process_regions(reg_objs=new_reg)
        elif isinstance(new_reg, dict) and all([all([isinstance(r, (SkyRegion, PixelRegion)) for r in rl])
                                                for o, rl in new_reg.items()]):
            self._reg_file_path = ""
            self._regions = self._process_regions(reg_objs=new_reg)
        else:
            raise ValueError("That value of new_reg is not valid, please pass either a path to a region file or "
                             "a list/dictionary of SkyRegion/PixelRegion objects")

    @property
    def matched_regions(self) -> Dict:
        """
        Property getter for any regions which have been designated a 'match' in the current analysis, if
        they have been set.

        :return: Returns a dictionary of matched regions, if they have been associated with this object.
        :rtype: Dict[PixelRegion]
        """
        return self._matched_regions

    @matched_regions.setter
    def matched_regions(self, new_reg: Union[str, List[Union[SkyRegion, PixelRegion]], dict]):
        """
        A setter for matched regions associated with this object, with a new single matched region or dictionary of
        matched regions (with keys being ObsIDs and one entry for each ObsID associated with this object) being passed.
        If a single region is passed then it will be assumed that it is associated with the current ObsID of this
        object.

        :param dict/SkyRegion/PixelRegion new_reg: A region that has been designated as 'matched', i.e. is the
            subject of a current analysis. This should either be supplied as a single region object, or as a
            dictionary of region objects with ObsIDs as keys.
        """
        if new_reg is not None and not isinstance(new_reg, (PixelRegion, SkyRegion, dict)):
            raise TypeError("Please pass either a dictionary of SkyRegion/PixelRegion objects with ObsIDs as "
                            "keys, or a single SkyRegion/PixelRegion object. Alternatively pass None for no match.")

        self._matched_regions = self._process_matched_regions(new_reg)

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

    @property
    def data_unit(self) -> Unit:
        """
        The unit of the data associated with this photometric product.

        :return: An astropy unit object describing the units of this objects' data.
        :rtype: Unit
        """
        return self._data_unit

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
                # We define an interim variable, in case the result is NaN - this now causes a warning that we
                #  wish to avoid, so we replace NaN with a negative number that will cause a failure further down
                inter_coord = Quantity(self.radec_wcs.all_world2pix(coords, 0), output_unit).round(0)
                out_coord = np.nan_to_num(inter_coord, nan=Quantity(-100, 'pix')).astype(int)
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
            # Have to compare to the [1] element of shape because numpy arrays are flipped, and we want
            #  to compare x to x
            elif out_name == "pix" and np.any(out_coord[:, 0].value >= self.shape[1]) and self._prod_type != "psf":
                raise ValueError("You've converted to pixel coordinates, and some x coordinates are larger than the "
                                 "image x-shape.")
            # Have to compare to the [0] element of shape because numpy arrays are flipped, and we want
            #  to compare y to y
            elif out_name == "pix" and np.any(out_coord[:, 1].value >= self.shape[0]) and self._prod_type != "psf":
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
                 ch_thickness: float = 0.8, low_val_lim: float = None, upp_val_lim: float = None,
                 custom_title: str = None) -> Axes:
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
            only be triggered if a cross_hair coordinate is also specified, as this acts as the central coordinate
            of the annuli. If two cross-hair coordinates are specified, the first will be used as the centre.
        :param np.ndarray back_bin_pix: The inner and outer radii (in pixel units) of the annulus used to measure
            the background value for a given profile, will only be triggered if a cross_hair coordinate is also
            specified, as this acts as the central coordinate of the annuli. If two cross-hair coordinates are
            specified, the first will be used as the centre.
        :param BaseStretch stretch: The astropy scaling to use for the image data, default is log.
        :param bool mask_edges: If viewing a RateMap, this variable will control whether the chip edges are masked
            to remove artificially bright pixels, default is True.
        :param bool view_regions: If regions have been associated with this object (either on init or using
            the 'regions' property setter, should they be displayed. Default is False.
        :param float ch_thickness: The desired linewidth of the crosshair(s), can be useful to increase this in
            certain circumstances. Default is 0.8.
        :param float low_val_lim: This can be used to set a lower limit for the value range across which an image
            is scaled and normalised (i.e. a ManualInterval from Astropy). The default is None, and if low_val_lim is
            not None, upp_val_lim must be as well.
        :param float upp_val_lim: This can be used to set an upper limit for the value range across which an image
            is scaled and normalised (i.e. a ManualInterval from Astropy). The default is None, and if upp_val_lim is
            not None, low_val_lim must be as well.
        :param str custom_title: If set, this will overwrite the automatically generated title for this
            visualisation. Default is None.
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

        # We check that the values that set manual value limits are legal, otherwise we throw an error
        check_lim_arg = [low_val_lim is None, upp_val_lim is None]
        if any(check_lim_arg) and not all(check_lim_arg):
            raise ValueError("Either 'low_val_lim' and 'upp_val_lim' are both None, or both have values.")
        elif not all(check_lim_arg) and low_val_lim >= upp_val_lim:
            raise ValueError("The 'low_val_lim' argument must be lower than 'upp_val_lim'.")
        elif not all(check_lim_arg):
            interval = ManualInterval(low_val_lim, upp_val_lim)
        else:
            interval = MinMaxInterval()

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

        # Ugly nested if statement but oh well I'm in a hurry - if the custom title is None then we auto generate a
        #  title - otherwise we use the custom title and don't add anything to it
        if custom_title is None:
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
        else:
            title = custom_title

        ax.set_title(title)

        # As this is a very quick view method, users will not be offered a choice of scaling
        #  There will be a more in depth way of viewing cluster data eventually
        norm = ImageNormalize(data=plot_data, interval=interval, stretch=stretch)
        # I normalize with a log stretch, and use gnuplot2 colormap (pretty decent for clusters imo)

        # If we want to plot point clusters on the image, then we go here
        if chosen_points is not None:
            # Add the point cluster points
            ax.plot(chosen_points[:, 0], chosen_points[:, 1], '+', color='black', label="Chosen Point Cluster")
            ax.legend(loc="best")

        if other_points is not None:
            for cl in other_points:
                ax.plot(cl[:, 0], cl[:, 1], 'D')

        # If we want a cross-hair, then we put one on here
        if cross_hair is not None:
            # For the case of a single coordinate
            if cross_hair.shape == (2,):
                # Converts from whatever input coordinate to pixels
                pix_coord = self.coord_conv(cross_hair, pix).value
                # Drawing the horizontal and vertical lines
                ax.axvline(pix_coord[0], color="white", linewidth=ch_thickness)
                ax.axhline(pix_coord[1], color="white", linewidth=ch_thickness)

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

                # Here I reset the pix_coord variable, so it ONLY contains the first entry. This is for the benefit
                #  of the annulus-drawing part of the code that comes after
                pix_coord = pix_coord[0, :]

            else:
                # I don't want to bring someone's code grinding to a halt just because they passed crosshair wrong,
                #  it isn't essential, so I'll just display a warning
                warnings.warn("You have passed a cross_hair quantity that has more than two coordinate "
                              "pairs in it, or is otherwise the wrong shape.")
                # Just in case annuli were also passed, I set the coordinate to None so that it knows something is wrong
                pix_coord = None

            if pix_coord is not None:
                # Drawing annular radii on the image, if they are enabled and passed. If multiple coordinates have been
                #  passed then I assume that they want to centre on the first entry
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

        # Adds the actual image to the axis.
        ax.imshow(plot_data, norm=norm, origin="lower", cmap="gnuplot2")

        # If the user wants regions on the image, this is where they get added
        if view_regions:
            # We can just loop through the _regions attribute because its default is an empty
            #  list, so no need to check
            flattened_reg = [r for o, rl in self._regions.items() for r in rl]
            for reg in flattened_reg:
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
             ch_thickness: float = 0.8, low_val_lim: float = None, upp_val_lim: float = None, custom_title: str = None):
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
        :param float low_val_lim: This can be used to set a lower limit for the value range across which an image
            is scaled and normalised (i.e. a ManualInterval from Astropy). The default is None, and if low_val_lim is
            not None, upp_val_lim must be as well.
        :param float upp_val_lim: This can be used to set an upper limit for the value range across which an image
            is scaled and normalised (i.e. a ManualInterval from Astropy). The default is None, and if upp_val_lim is
            not None, low_val_lim must be as well.
        :param str custom_title: If set, this will overwrite the automatically generated title for this
            visualisation. Default is None.
        """

        # Create figure object
        fig = plt.figure(figsize=figsize)

        ax = plt.gca()
        ax = self.get_view(ax, cross_hair, mask, chosen_points, other_points, zoom_in, manual_zoom_xlims,
                           manual_zoom_ylims, radial_bins_pix, back_bin_pix, stretch, mask_edges, view_regions,
                           ch_thickness, low_val_lim, upp_val_lim, custom_title)
        cbar = plt.colorbar(ax.images[0])
        cbar.ax.set_ylabel(self.data_unit.to_string('latex'), fontsize=15)
        plt.tight_layout()
        # Display the image
        plt.show()

        # Wipe the figure
        plt.close("all")

    def save_view(self, save_path: str, cross_hair: Quantity = None, mask: np.ndarray = None,
                  chosen_points: np.ndarray = None, other_points: List[np.ndarray] = None, figsize: Tuple = (10, 8),
                  zoom_in: bool = False, manual_zoom_xlims: tuple = None, manual_zoom_ylims: tuple = None,
                  radial_bins_pix: np.ndarray = np.array([]), back_bin_pix: np.ndarray = None,
                  stretch: BaseStretch = LogStretch(), mask_edges: bool = True, view_regions: bool = False,
                  ch_thickness: float = 0.8, low_val_lim: float = None, upp_val_lim: float = None,
                  custom_title: str = None):
        """
        This is entirely equivalent to the view() method, but instead of displaying the view it will save it to
        a path of your choosing.

        :param str save_path: The path (including file name) where you wish to save the view.
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
        :param float low_val_lim: This can be used to set a lower limit for the value range across which an image
            is scaled and normalised (i.e. a ManualInterval from Astropy). The default is None, and if low_val_lim is
            not None, upp_val_lim must be as well.
        :param float upp_val_lim: This can be used to set an upper limit for the value range across which an image
            is scaled and normalised (i.e. a ManualInterval from Astropy). The default is None, and if upp_val_lim is
            not None, low_val_lim must be as well.
        :param str custom_title: If set, this will overwrite the automatically generated title for this
            visualisation. Default is None.
        """

        # Create figure object
        fig = plt.figure(figsize=figsize)

        # Turns off any ticks and tick labels, we don't want them in an image
        ax = plt.gca()

        ax = self.get_view(ax, cross_hair, mask, chosen_points, other_points, zoom_in, manual_zoom_xlims,
                           manual_zoom_ylims, radial_bins_pix, back_bin_pix, stretch, mask_edges, view_regions,
                           ch_thickness, low_val_lim, upp_val_lim, custom_title)
        cbar = plt.colorbar(ax.images[0], label=self.data_unit.to_string('latex'))
        cbar.ax.set_ylabel(self.data_unit.to_string('latex'), fontsize=15)
        plt.tight_layout()

        # Save figure to disk
        plt.savefig(save_path)

        # Wipe the figure
        plt.close("all")

    def edit_regions(self, figsize: Tuple = (7, 7), cmap: str = 'gnuplot2', reg_save_path: str = None,
                     cross_hair: Quantity = None, radial_bins_pix: Quantity = Quantity(np.array([]), 'pix'),
                     back_bin_pix: Quantity = None):
        """
        This allows for displaying, interacting with, editing, and adding new regions to an image. These can
        then be saved as a new region file. It also allows for the dynamic adjustment of which regions
        are displayed, the scaling of the image, and smoothing, in order to make placing new regions as
        simple as possible. If a save path for region files is passed, then it will be possible to save
        new region files in RA-Dec coordinates.

        :param Tuple figsize: Allows the user to pass a custom size for the figure produced by this class.
        :param str cmap: The colour map to use for displaying the image. Default is gnuplot2.
        :param str reg_save_path: A string that will have ObsID values added before '.reg' to construct
            save paths for the output region lists (if that feature is activated by the user). Default is
            None, in which case saving will be disabled.
        :param Quantity cross_hair: An optional parameter that can be used to plot a cross hair at
            the coordinates. Up to two cross-hairs can be plotted, as any more can be visually confusing. If
            passing two, each row of a quantity is considered to be a separate coordinate pair.
        :param Quantity radial_bins_pix: Radii (in units of pixels) of annuli to plot on top of the image, will
            only be triggered if a cross_hair coordinate is also specified and contains only one coordinate.
        :param Quantity back_bin_pix: The inner and outer radii (in pixel units) of the annulus used to measure
            the background value for a given profile, will only be triggered if a cross_hair coordinate is
            also specified and contains only one coordinate.
        """
        # TODO UPDATE THE DOCSTRING WHEN I HAVE INTEGRATED THIS WITH THE REST OF XGA
        view_inst = self._InteractiveView(self, figsize, cmap, reg_save_path, cross_hair, radial_bins_pix,
                                          back_bin_pix)
        view_inst.edit_view()

    def dynamic_view(self, figsize: Tuple = (7, 7), cmap: str = 'gnuplot2', cross_hair: Quantity = None,
                     radial_bins_pix: Quantity = Quantity(np.array([]), 'pix'),
                     back_bin_pix: Quantity = None):
        """
        This allows for displaying regions on an image. It also allows for the dynamic adjustment of which regions
        are displayed, the scaling of the image, and smoothing.

        :param Tuple figsize: Allows the user to pass a custom size for the figure produced by this class.
        :param str cmap: The colour map to use for displaying the image. Default is gnuplot2.
        :param Quantity cross_hair: An optional parameter that can be used to plot a cross hair at
            the coordinates. Up to two cross-hairs can be plotted, as any more can be visually confusing. If
            passing two, each row of a quantity is considered to be a separate coordinate pair.
        :param Quantity radial_bins_pix: Radii (in units of pixels) of annuli to plot on top of the image, will
            only be triggered if a cross_hair coordinate is also specified and contains only one coordinate.
        :param Quantity back_bin_pix: The inner and outer radii (in pixel units) of the annulus used to measure
            the background value for a given profile, will only be triggered if a cross_hair coordinate is
            also specified and contains only one coordinate.
        """
        view_inst = self._InteractiveView(self, figsize, cmap, None, cross_hair, radial_bins_pix, back_bin_pix)
        view_inst.dynamic_view()

    class _InteractiveView:
        """
        An internal class of the Image class, designed to enable the interactive and dynamic editing of regions
        for an observation (with the capability of adding completely new regions as well). This is 'private' as
        I can't really see a use-case where the user would define an instance of this themselves.
        """
        def __init__(self, phot_prod, figsize: Tuple = (7, 7), cmap: str = "gnuplot2", reg_save_path: str = None,
                     cross_hair: Quantity = None, radial_bins_pix: Quantity = Quantity(np.array([]), 'pix'),
                     back_bin_pix: Quantity = None):
            """
            The init of the _InteractiveView class, which enables dynamic viewing of XGA photometric products.

            :param Image/RateMap/ExpMap phot_prod: The XGA photometric product which we want to interact with.
            :param Tuple figsize: Allows the user to pass a custom size for the figure produced by this class.
            :param str cmap: The colour map to use for displaying the image. Default is gnuplot2.
            :param str reg_save_path: A string that will have ObsID values added before '.reg' to construct
                save paths for the output region lists (if that feature is activated by the user). Default is
                None, in which case saving will be disabled.
            :param Quantity cross_hair: An optional parameter that can be used to plot a cross hair at
                the coordinates. Up to two cross-hairs can be plotted, as any more can be visually confusing. If
                passing two, each row of a quantity is considered to be a separate coordinate pair.
            :param Quantity radial_bins_pix: Radii (in units of pixels) of annuli to plot on top of the image, will
                only be triggered if a cross_hair coordinate is also specified and contains only one coordinate.
            :param Quantity back_bin_pix: The inner and outer radii (in pixel units) of the annulus used to measure
                the background value for a given profile, will only be triggered if a cross_hair coordinate is
                also specified and contains only one coordinate.
            """
            # Just saving a reference to the photometric object that declared this instance of this class, and
            #  then making a copy of whatever regions are associated with it
            self._parent_phot_obj = phot_prod
            self._regions = deepcopy(phot_prod.regions)

            # Store the passed-in save path for regions in an attribute for later
            if reg_save_path is not None and reg_save_path[-4:] == '.reg':
                self._reg_save_path = reg_save_path
            elif reg_save_path is not None and reg_save_path[-4:] != '.reg':
                raise ValueError("The last four characters of the save path must be '.reg', as extra strings "
                                 "will be inserted into the save path to account for different region files for "
                                 "different ObsIDs.")
            else:
                self._reg_save_path = None

            # This is for storing references to artists with an ObsID key, so we know which artist belongs
            #  to which ObsID. Populated in the first part of _draw_regions. We also construct the reverse so that
            #  an artist instance can be easily used to lookup the ObsID it belongs to
            self._obsid_artists = {o: [] for o in self._parent_phot_obj.obs_ids}
            self._artist_obsids = {}
            # In the same vein I setup a lookup dictionary for artist to region
            self._artist_region = {}

            # Setting up the figure within which all the axes (data, buttons, etc.) are placed
            in_fig = plt.figure(figsize=figsize)
            # Storing the figure in an attribute, as well as the image axis (i.e. the axis on which the data
            #  are displayed) in another attribute, for convenience.
            self._fig = in_fig
            self._im_ax = plt.gca()
            self._ax_loc = self._im_ax.get_position()

            # Setting up the look of the data axis, removing ticks and tick labels because it's an image
            self._im_ax.tick_params(axis='both', direction='in', which='both', top=False, right=False)
            self._im_ax.xaxis.set_ticklabels([])
            self._im_ax.yaxis.set_ticklabels([])

            # Setting up some visual stuff that is used in multiple places throughout the class
            # First the colours of buttons in an active and inactive state (the region toggles)
            self._but_act_col = "0.85"
            self._but_inact_col = "0.99"
            # Now the standard line widths used both for all regions, and for the region that is currently selected
            self._reg_line_width = 1.2
            self._sel_reg_line_width = 2.3
            # These are the increments when adjusting the regions by pressing wasd and qe. So for the size and
            #  angle of the selected region.
            self._size_step = 2
            self._rot_step = 10

            # Setting up and storing the connections to events on the matplotlib canvas. These are what
            #  allow specific methods to be triggered when things like button presses or clicking on the
            #  figure occur. They are stored in attributes, though I'm not honestly sure that's necessary
            # Not all uses of this class will make use of all of these connections, but I'm still defining them
            #  all here anyway
            self._pick_cid = self._fig.canvas.mpl_connect("pick_event", self._on_region_pick)
            self._move_cid = self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self._rel_cid = self._fig.canvas.mpl_connect("button_release_event", self._on_release)
            self._undo_cid = self._fig.canvas.mpl_connect("key_press_event", self._key_press)
            self._click_cid = self._fig.canvas.mpl_connect("button_press_event", self._click_event)

            # All uses of this class (both editing regions and just having a vaguely interactive view of the
            #  observation) will have these buttons that allow regions to be turned off and on, so they are
            #  defined here. All buttons are defined in separate axes.
            # These buttons act as toggles, they are all active by default and clicking one will turn off the source
            #  type its associated with. Clicking it again will turn it back on.
            # This button toggles extended (green) sources.
            top_pos = self._ax_loc.y1-0.0771
            ext_src_loc = plt.axes([0.045, top_pos, 0.075, 0.075])
            self._ext_src_button = Button(ext_src_loc, "EXT", color=self._but_act_col)
            self._ext_src_button.on_clicked(self._toggle_ext)

            # This button toggles point (red) sources.
            pnt_src_loc = plt.axes([0.045, top_pos-(0.075 + 0.005), 0.075, 0.075])
            self._pnt_src_button = Button(pnt_src_loc, "PNT", color=self._but_act_col)
            self._pnt_src_button.on_clicked(self._toggle_pnt)

            # This button toggles types of region other than green or red (mostly valid for XCS XAPA sources).
            oth_src_loc = plt.axes([0.045, top_pos-2*(0.075 + 0.005), 0.075, 0.075])
            self._oth_src_button = Button(oth_src_loc, "OTHER", color=self._but_act_col)
            self._oth_src_button.on_clicked(self._toggle_oth)

            # This button toggles custom source regions
            cust_src_loc = plt.axes([0.045, top_pos-3*(0.075 + 0.005), 0.075, 0.075])
            self._cust_src_button = Button(cust_src_loc, "CUST", color=self._but_act_col)
            self._cust_src_button.on_clicked(self._toggle_cust)

            # These are buttons that can be present depending on the usage of the class
            self._new_ell_button = None
            self._new_circ_button = None

            # A dictionary describing the current type of regions that are on display
            self._cur_act_reg_type = {"EXT": True, "PNT": True, "OTH": True, "CUST": True}

            # These set up the default colours, red for point, green for extended, and white for custom. I already
            #  know these colour codes because this is what the regions module colours translate into in matplotlib
            # Maybe I should automate this rather than hard coding
            self._colour_convert = {(1.0, 0.0, 0.0, 1.0): 'red', (0.0, 0.5019607843137255, 0.0, 1.0): 'green',
                                    (1.0, 1.0, 1.0, 1.0): 'white'}
            # There can be other coloured regions though, XAPA for instance has lots of subclasses of region. This
            #  loop goes through the regions and finds their colour name / matplotlib colour code and adds it to the
            #  dictionary for reference
            for region in [r for o, rl in self._regions.items() for r in rl]:
                art_reg = region.as_artist()
                self._colour_convert[art_reg.get_edgecolor()] = region.visual["edgecolor"]

            # This just provides a conversion between name and colour tuple, the inverse of colour_convert
            self._inv_colour_convert = {v: k for k, v in self._colour_convert.items()}

            # Unfortunately I cannot rely on regions being of an appropriate type (Ellipse/Circle) for what they
            #  are. For instance XAPA point source regions are still ellipses, just with the height and width
            #  set equal. So this dictionary is an independent reference point for the shape, with entries for the
            #  original regions made in the first part of _draw_regions, and otherwise set when a new region is added.
            self._shape_dict = {}

            # I also wish to keep track of whether a particular region has been edited or not, for reference when
            #  outputting the final edited region list (if it is requested). I plan to do this with a similar approach
            #  to the shape_dict, have a dictionary with artists as keys, but then have a boolean as a value. Will
            #  also be initially populated in the first part of _draw_regions.
            self._edited_dict = {}

            # This controls whether interacting with regions is allowed - turned off for the dynamic view method
            #  as that is not meant for editing regions
            self._interacting_on = False
            # The currently selected region is referenced in this attribute
            self._cur_pick = None
            # The last coordinate ON THE IMAGE that was clicked is stored here. Initial value is set to the centre
            self._last_click = (phot_prod.shape[0] / 2, phot_prod.shape[1] / 2)
            # This describes whether the artist stored in _cur_pick (if there is one) is right now being clicked
            #  and held - this is used for enabling clicking and dragging so the method knows when to stop.
            self._select = False
            self._history = []

            # These store the current settings for colour map, stretch, and scaling
            self._cmap = cmap
            self._interval = MinMaxInterval()
            self._stretch = stretch_dict['LOG']
            # This is just a convenient place to store the name that XGA uses for the current stretch - it lets us
            #  access the current stretch instance from stretch_dict more easily (and accompanying buttons etc.)
            self._active_stretch_name = 'LOG'
            # This is used to store all the button instances created for controlling stretch
            self._stretch_buttons = {}

            # Here we define attribute to store the data and normalisation in. I copy the data to make sure that
            #  the original information doesn't get changed when smoothing is applied.
            self._plot_data = self._parent_phot_obj.data.copy()
            # It's also possible to mask and display the data, and the current mask is stored in this attribute
            self._plot_mask = np.ones(self._plot_data.shape)
            self._norm = self._renorm()

            # The output of the imshow command lives in here
            self._im_plot = None
            # Adds the actual image to the axis.
            self._replot_data()

            # This bit is where all the stretch buttons are set up, as well as the slider. All methods should
            #  be able to use re-stretching so that's why this is all in the init
            ax_slid = plt.axes([self._ax_loc.x0, 0.885, 0.7771, 0.03], facecolor="white")
            # Hides the ticks to make it look nicer
            ax_slid.set_xticks([])
            ax_slid.set_yticks([])
            # Use the initial defined MinMaxInterval to get the initial range for the RangeSlider - used both
            #  as upper and lower boundaries and starting points for the sliders.
            init_range = self._interval.get_limits(self._plot_data)
            # Define the RangeSlider instance, set the value text to invisible, and connect to the method it activates
            self._vrange_slider = RangeSlider(ax_slid, 'DATA INTERVAL', *init_range, valinit=init_range)
            # We move the RangeSlider label so that is sits within the bar
            self._vrange_slider.label.set_x(0.6)
            self._vrange_slider.label.set_y(0.45)
            self._vrange_slider.valtext.set_visible(False)
            self._vrange_slider.on_changed(self._change_interval)

            # Sets up an initial location for the stretch buttons to iterate over, so I can make this
            #  as automated as possible. An advantage is that I can just add new stretches to the stretch_dict,
            #  and they should be automatically added here.
            loc = [self._ax_loc.x0 - (0.075 + 0.005), 0.92, 0.075, 0.075]
            # Iterate through the stretches that I chose to store in the stretch_dict
            for stretch_name, stretch in stretch_dict.items():
                # Increments the position of the button
                loc[0] += (0.075 + 0.005)
                # Sets up an axis for the button we're about to create
                stretch_loc = plt.axes(loc)

                # Sets the colour for this button. Sort of unnecessary to do it like this because LOG should always
                #  be the initially active stretch, but better to generalise
                if stretch_name == self._active_stretch_name:
                    col = self._but_act_col
                else:
                    col = self._but_inact_col
                # Creates the button for the current stretch
                self._stretch_buttons[stretch_name] = Button(stretch_loc, stretch_name, color=col)

                # Generates and adds the function for the current stretch button
                self._stretch_buttons[stretch_name].on_clicked(self._change_stretch(stretch_name))

            # This is the bit where we set up the buttons and slider for the smoothing function
            smooth_loc = plt.axes([self._ax_loc.x1 + 0.005, top_pos, 0.095, 0.075])
            self._smooth_button = Button(smooth_loc, "SMOOTH", color=self._but_inact_col)
            self._smooth_button.on_clicked(self._toggle_smooth)

            ax_smooth_slid = plt.axes([self._ax_loc.x1 + 0.03, self._ax_loc.y0+0.002, 0.05, 0.685], facecolor="white")
            # Hides the ticks to make it look nicer
            ax_smooth_slid.set_xticks([])
            # Define the Slider instance, add and position a label, and connect to the method it activates
            self._smooth_slider = Slider(ax_smooth_slid, 'KERNEL RADIUS', 0.5, 5, valinit=1, valstep=0.5,
                                         orientation='vertical')
            # Remove the annoying line representing initial value that is automatically added
            self._smooth_slider.hline.remove()
            # We move the Slider label so that is sits within the bar
            self._smooth_slider.label.set_rotation(270)
            self._smooth_slider.label.set_x(0.5)
            self._smooth_slider.label.set_y(0.45)
            self._smooth_slider.on_changed(self._change_smooth)

            # We also create an attribute to store the current value of the slider in. Not really necessary as we
            #  could always fetch the value out of the smooth slider attribute but its neater this way I think
            self._kernel_rad = self._smooth_slider.val

            # This is a definition for a save button that is used in edit_view
            self._save_button = None

            # Adding a button to apply a mask generated from the regions, largely to help see if any emission
            #  from an object isn't being properly removed.
            mask_loc = plt.axes([self._ax_loc.x0 + (0.075 + 0.005), self._ax_loc.y0 - 0.08, 0.075, 0.075])
            self._mask_button = Button(mask_loc, "MASK", color=self._but_inact_col)
            self._mask_button.on_clicked(self._toggle_mask)

            # This next part allows for the over-plotting of annuli to indicate analysis regions, this can be
            #  very useful to give context when manually editing regions. The only way I know of to do this is
            #  with artists, but unfortunately artists (and iterating through the artist property of the image axis)
            #  is the way a lot of stuff in this class works. So here I'm going to make a new class attribute
            #  that stores which artists are added to visualise analysis areas and therefore shouldn't be touched.
            self._ignore_arts = []
            # As this was largely copied from the get_view method of Image, I am just going to define this
            #  variable here for ease of testing
            ch_thickness = 0.8
            # If we want a cross-hair, then we put one on here
            if cross_hair is not None:
                # For the case of a single coordinate
                if cross_hair.shape == (2,):
                    # Converts from whatever input coordinate to pixels
                    pix_coord = self._parent_phot_obj.coord_conv(cross_hair, pix).value
                    # Drawing the horizontal and vertical lines
                    self._im_ax.axvline(pix_coord[0], color="white", linewidth=ch_thickness)
                    self._im_ax.axhline(pix_coord[1], color="white", linewidth=ch_thickness)

                # For the case of two coordinate pairs
                elif cross_hair.shape == (2, 2):
                    # Converts from whatever input coordinate to pixels
                    pix_coord = self._parent_phot_obj.coord_conv(cross_hair, pix).value

                    # This draws the first crosshair
                    self._im_ax.axvline(pix_coord[0, 0], color="white", linewidth=ch_thickness)
                    self._im_ax.axhline(pix_coord[0, 1], color="white", linewidth=ch_thickness)

                    # And this the second
                    self._im_ax.axvline(pix_coord[1, 0], color="white", linewidth=ch_thickness, linestyle='dashed')
                    self._im_ax.axhline(pix_coord[1, 1], color="white", linewidth=ch_thickness, linestyle='dashed')

                    # Here I reset the pix_coord variable, so it ONLY contains the first entry. This is for the benefit
                    #  of the annulus-drawing part of the code that comes after
                    pix_coord = pix_coord[0, :]

                else:
                    # I don't want to bring someone's code grinding to a halt just because they passed crosshair wrong,
                    #  it isn't essential, so I'll just display a warning
                    warnings.warn("You have passed a cross_hair quantity that has more than two coordinate "
                                  "pairs in it, or is otherwise the wrong shape.")
                    # Just in case annuli were also passed, I set the coordinate to None so that it knows something is
                    # wrong
                    pix_coord = None

                if pix_coord is not None:
                    # Drawing annular radii on the image, if they are enabled and passed. If multiple coordinates have
                    #  been passed then I assume that they want to centre on the first entry
                    for ann_rad in radial_bins_pix:
                        # Creates the artist for the current annular region
                        artist = Circle(pix_coord, ann_rad.value, fill=False, ec='white',
                                        linewidth=ch_thickness)
                        # Means it can't be interacted with
                        artist.set_picker(False)
                        # Adds it to the list that lets the class know it needs to not treat it as a region
                        #  found by a source detector
                        self._ignore_arts.append(artist)
                        # And adds the artist to axis
                        self._im_ax.add_artist(artist)

                    # This draws the background region on as well, if present
                    if back_bin_pix is not None:
                        # The background annulus is guaranteed to only have two entries, inner and outer
                        inn_artist = Circle(pix_coord, back_bin_pix[0].value, fill=False, ec='white',
                                            linewidth=ch_thickness, linestyle='dashed')
                        out_artist = Circle(pix_coord, back_bin_pix[1].value, fill=False, ec='white',
                                            linewidth=ch_thickness, linestyle='dashed')
                        # Make sure neither region can be interacted with
                        inn_artist.set_picker(False)
                        out_artist.set_picker(False)
                        # Add to the ignore list and to the axis
                        self._im_ax.add_artist(inn_artist)
                        self._ignore_arts.append(inn_artist)
                        self._im_ax.add_artist(out_artist)
                        self._ignore_arts.append(out_artist)

            # This chunk checks to see if there were any matched regions associated with the parent
            #  photometric object, and if so it adds them to the image and makes sure that they
            #  cannot be interacted with
            for obs_id, match_reg in self._parent_phot_obj.matched_regions.items():
                if match_reg is not None:
                    art_reg = match_reg.as_artist()
                    # Setting the style for these regions, to make it obvious that they are different from
                    #  any other regions that might be displayed
                    art_reg.set_linestyle('dotted')

                    # Makes sure that the region cannot be 'picked'
                    art_reg.set_picker(False)
                    # Sets the standard linewidth
                    art_reg.set_linewidth(self._sel_reg_line_width)
                    # And actually adds the artist to the data axis
                    self._im_ax.add_artist(art_reg)
                    # Also makes sure this artist is on the ignore list, as it's a constant and shouldn't be redrawn
                    #  or be able to be modified
                    self._ignore_arts.append(art_reg)

        def dynamic_view(self):
            """
            The simplest view method of this class, enables the turning on and off of regions.
            """
            # Draws on any regions associated with this instance
            self._draw_regions()

            # I THINK that activating this is what turns on automatic refreshing
            plt.ion()
            plt.show()

        def edit_view(self):
            """
            An extremely useful view method of this class - allows for direct interaction with and editing of
            regions, as well as the ability to add new regions. If a save path for region files was passed on
            declaration of this object, then it will be possible to save new region files in RA-Dec coordinates.
            """
            # This mode we DO want to be able to interact with regions
            self._interacting_on = True

            # Add two buttons to the figure to enable the adding of new elliptical and circular regions
            new_ell_loc = plt.axes([0.045, 0.191, 0.075, 0.075])
            self._new_ell_button = Button(new_ell_loc, "ELL")
            self._new_ell_button.on_clicked(self._new_ell_src)

            new_circ_loc = plt.axes([0.045, 0.111, 0.075, 0.075])
            self._new_circ_button = Button(new_circ_loc, "CIRC")
            self._new_circ_button.on_clicked(self._new_circ_src)

            # This sets up a button that saves an updated region list to a file path that was passed in on the
            #  declaration of this instance of the class. If no path was passed, then the button doesn't
            #  even appear.
            if self._reg_save_path is not None:
                save_loc = plt.axes([self._ax_loc.x0, self._ax_loc.y0 - 0.08, 0.075, 0.075])
                self._save_button = Button(save_loc, "SAVE", color=self._but_inact_col)
                self._save_button.on_clicked(self._save_region_files)

            # Draws on any regions associated with this instance
            self._draw_regions()

            plt.ion()
            plt.show(block=True)

        def _replot_data(self):
            """
            This method updates the currently plotted data using the relevant class attributes. Such attributes
            are updated and edited by other parts of the class. The plot mask is always applied to data, but when
            not turned on by the relevant button it will be all ones so will make no difference.
            """
            # This removes the existing image data without removing the region artists
            if self._im_plot is not None:
                self._im_plot.remove()

            # This does the actual plotting bit, saving the output in an attribute, so it can be
            #  removed when re-plotting
            self._im_plot = self._im_ax.imshow(self._plot_data*self._plot_mask, norm=self._norm, origin="lower",
                                               cmap=self._cmap)

        def _renorm(self) -> ImageNormalize:
            """
            Re-calculates the normalisation of the plot data with current interval and stretch settings. Takes into
            account the mask if applied. The plot mask is always applied to data, but when not turned on by the
            relevant button it will be all ones so will make no difference.

            :return: The normalisation object.
            :rtype: ImageNormalize
            """
            # We calculate the normalisation using masked data, but mask will be all ones if that
            #  feature is not currently turned on
            norm = ImageNormalize(data=self._plot_data*self._plot_mask, interval=self._interval,
                                  stretch=self._stretch)

            return norm

        def _draw_regions(self):
            """
            This method is called by an _InteractiveView instance when regions need to be drawn on top of the
            data view axis (i.e. the image/ratemap). Either for the first time or as an update due to a button
            click, region changing, or new region being added.
            """
            # These artists are the ones that represent regions, the ones in self._ignore_arts are there
            #  just for visualisation (for instance showing an analysis/background region) and can't be
            #  turned on or off, can't be edited, and shouldn't be saved.
            # rel_artists = [arty for arty in self._im_ax.artists if arty not in self._ignore_arts]
            rel_artists = [arty for arty in self._im_ax.patches if arty not in self._ignore_arts]

            # This will trigger in initial cases where there ARE regions associated with the photometric product
            #  that has spawned this InteractiveView, but they haven't been added as artists yet. ALSO, this will
            #  always run prior to any artists being added that are just there to indicate analysis regions, see
            #  toward the end of the __init__ for what I mean.

            if len(rel_artists) == 0 and len([r for o, rl in self._regions.items() for r in rl]) != 0:
                for o in self._regions:
                    for region in self._regions[o]:
                        # Uses the region module's convenience function to turn the region into a matplotlib artist
                        art_reg = region.as_artist()
                        # Makes sure that the region can be 'picked', which enables selecting regions to modify
                        art_reg.set_picker(True)
                        # Sets the standard linewidth
                        art_reg.set_linewidth(self._reg_line_width)
                        # And actually adds the artist to the data axis
                        self._im_ax.add_artist(art_reg)
                        # Adds an entry to the shape dictionary. If a region from the parent Image is elliptical but
                        #  has the same height and width then I define it as a circle.
                        if type(art_reg) == Circle or (type(art_reg) == Ellipse and art_reg.height == art_reg.width):
                            self._shape_dict[art_reg] = 'circle'
                        elif type(art_reg) == Ellipse:
                            self._shape_dict[art_reg] = 'ellipse'
                        else:
                            raise NotImplementedError("This method does not currently support regions other than "
                                                      "circles or ellipses, but please get in touch to discuss "
                                                      "this further.")
                        # Add entries in the dictionary that keeps track of whether a region has been edited or
                        #  not. All entries start out being False of course.
                        self._edited_dict[art_reg] = False
                        # Here we save the knowledge of which artists belong to which ObsID, and vice versa
                        self._obsid_artists[o].append(art_reg)
                        self._artist_obsids[art_reg] = o
                        # This allows us to lookup the original regions from their artist
                        self._artist_region[art_reg] = region

                # Need to update this in this case
                # rel_artists = [arty for arty in self._im_ax.artists if arty not in self._ignore_arts]
                rel_artists = [arty for arty in self._im_ax.patches if arty not in self._ignore_arts]

            # This chunk controls which regions will be drawn when this method is called. The _cur_act_reg_type
            #  dictionary has keys representing the four toggle buttons, and their values are True or False. This
            #  first option is triggered if all entries are True and thus draws all regions
            if all(self._cur_act_reg_type.values()):
                allowed_colours = list(self._colour_convert.keys())

            # This checks individual entries in the dictionary, and adds allowed colours to the colour checking
            #  list which the method uses to identify the regions its allowed to draw for a particular call of this
            #  method.
            else:
                allowed_colours = []
                if self._cur_act_reg_type['EXT']:
                    allowed_colours.append(self._inv_colour_convert['green'])
                if self._cur_act_reg_type['PNT']:
                    allowed_colours.append(self._inv_colour_convert['red'])
                if self._cur_act_reg_type['CUST']:
                    allowed_colours.append(self._inv_colour_convert['white'])
                if self._cur_act_reg_type['OTH']:
                    allowed_colours += [self._inv_colour_convert[c] for c in self._inv_colour_convert
                                        if c not in ['green', 'red', 'white']]

            # This iterates through all the artists currently added to the data axis, setting their linewidth
            #  to zero if their colour isn't in the approved list
            for artist in rel_artists:
                if artist.get_edgecolor() in allowed_colours:
                    # If we're here then the region type of this artist is enabled by a button, and thus it should
                    #  be visible. We also use set_picker to make sure that this artist is allowed to be clicked on.
                    artist.set_linewidth(self._reg_line_width)
                    artist.set_picker(True)

                    # Slightly ugly nested if statement, but this just checks to see whether the current artist
                    #  is one that the user has selected. If yes then the line width should be different.
                    if self._cur_pick is not None and self._cur_pick == artist:
                        artist.set_linewidth(self._sel_reg_line_width)

                else:
                    # This part is triggered if the artist colour isn't 'allowed' - the button for that region type
                    #  hasn't been toggled on. And thus the width is set to 0 and the region becomes invisible
                    artist.set_linewidth(0)
                    # We turn off 'picker' to make sure that invisible regions can't be selected accidentally
                    artist.set_picker(False)
                    # We also make sure that if this artist (which is not currently being displayed) was the one
                    #  selected by the user, it is de-selected, so they don't accidentally make changes to an invisible
                    #  region.
                    if self._cur_pick is not None and self._cur_pick == artist:
                        self._cur_pick = None

        def _change_stretch(self, stretch_name: str):
            """
            Triggered when any of the stretch change buttons are pressed - acts as a generator for the response
            functions that are actually triggered when the separate buttons are pressed. Written this way to
            allow me to just write one of these functions rather than one function for each stretch.

            :param str stretch_name: The name of the stretch associated with a specific button.
            :return: A function matching the input stretch_name that will change the stretch applied to the data.
            """
            def gen_func(event):
                """
                A generated function to change the data stretch.

                :param event: The event passed by clicking the button associated with this function
                """
                # This changes the colours of the buttons so the active button has a different colour
                self._stretch_buttons[stretch_name].color = self._but_act_col
                # And this sets the previously active stretch button colour back to inactive
                self._stretch_buttons[self._active_stretch_name].color = self._but_inact_col
                # Now I change the currently active stretch stored in this class
                self._active_stretch_name = stretch_name

                # This alters the currently selected stretch stored by this class. Fetches the appropriate stretch
                #  object by using the stretch name passed when this function was generated.
                self._stretch = stretch_dict[stretch_name]
                # Performs the renormalisation that takes into account the newly selected stretch
                self._norm = self._renorm()
                # Performs the actual re-plotting that takes into account the newly calculated normalisation
                self._replot_data()

            return gen_func

        def _change_interval(self, boundaries: Tuple):
            """
            This method is called when a change is made to the RangeSlider that controls the interval range
            of the data that is displayed.

            :param Tuple boundaries: The lower and upper boundary currently selected by the RangeSlider
                controlling the interval.
            """
            # Creates a new interval, manually defined this time, with boundaries taken from the RangeSlider
            self._interval = ManualInterval(*boundaries)
            # Recalculate the normalisation with this new interval
            self._norm = self._renorm()
            # And finally replot the data.
            self._replot_data()

        def _apply_smooth(self):
            """
            This very simple function simply sets the internal data to a smooth version, making using of the
            currently stored information on the kernel radius. The smoothing is with a 2D Gaussian kernel, but
            the kernel is symmetric.
            """
            # Sets up the kernel instance - making use of Astropy because I've used it before
            the_kernel = Gaussian2DKernel(self._kernel_rad, self._kernel_rad)
            # Using an FFT convolution for now, I think this should be okay as this is purely for visual
            #  use and so I don't care much about edge effects
            self._plot_data = convolve_fft(self._plot_data, the_kernel)

        def _toggle_smooth(self, event):
            """
            This method is triggered by toggling the smooth button, and will turn smoothing on or off.

            :param event: The button event that triggered this toggle.
            """
            # If the current colour is the active button colour then smoothing is turned on already. Don't
            #  know why I didn't think of doing it this way before
            if self._smooth_button.color == self._but_act_col:
                # Put the button colour back to inactive
                self._smooth_button.color = self._but_inact_col
                # Sets the plot data back to the original unchanged version
                self._plot_data = self._parent_phot_obj.data.copy()
            else:
                # Set the button colour to active
                self._smooth_button.color = self._but_act_col
                # This runs the symmetric 2D Gaussian smoothing, then stores the result in the data
                #  attribute of the class
                self._apply_smooth()

            # Runs re-normalisation on the data and then re-plots it, necessary for either option of the toggle.
            self._renorm()
            self._replot_data()

        def _change_smooth(self, new_rad: float):
            """
            This method is triggered by a change of the slider, and sets a new smoothing kernel radius
            from the slider value. This will trigger a change if smoothing is currently turned on.

            :param float new_rad: The new radius for the smoothing kernel.
            """
            # Sets the kernel radius attribute to the new value
            self._kernel_rad = new_rad
            # But if the smoothing button is the active colour (i.e. smoothing is on), then we update the smooth
            if self._smooth_button.color == self._but_act_col:
                # Need to reset the data even though we're still smoothing, otherwise the smoothing will be
                #  applied on top of other smoothing
                self._plot_data = self._parent_phot_obj.data.copy()
                # Same deal as the else part of _toggle_smooth
                self._apply_smooth()
                self._renorm()
                self._replot_data()

        def _toggle_mask(self, event):
            """
            A method triggered by a button press that toggles whether the currently displayed image is
            masked or not.

            :param event: The event passed by the button that triggers this toggle method.
            """
            # In this case we know that masking is already applied because the button is the active colour and
            #  we set about to return everything to non-masked
            if self._mask_button.color == self._but_act_col:
                # Set the button colour to inactive
                self._mask_button.color = self._but_inact_col
                # Reset the plot mask to just ones, meaning nothing is masked
                self._plot_mask = np.ones(self._parent_phot_obj.shape)
            else:
                # Set the button colour to active
                self._mask_button.color = self._but_act_col
                # Generate a mask from the current regions
                self._plot_mask = self._gen_cur_mask()

            # Run renorm and replot, which will both now apply the current mask, whether it's been set to all ones
            #  or one generated from the current regions
            self._renorm()
            self._replot_data()

        def _gen_cur_mask(self):
            """
            Uses the current region list to generate a mask for the parent image that can be applied to the data.

            :return: The current mask.
            :rtype: np.ndarray
            """
            masks = []
            # Because the user might have added regions, we have to generate an updated region dictionary. However,
            #  we don't want to save the updated region list in the existing _regions attribute as that
            #  might break things
            cur_regs = self._update_reg_list()
            # Iterating through the flattened region dictionary
            for r in [r for o, rl in cur_regs.items() for r in rl]:
                # If the rotation angle is zero then the conversion to mask by the regions module will be upset,
                #  so I perturb the angle by 0.1 degrees
                if isinstance(r, EllipsePixelRegion) and r.angle.value == 0:
                    r.angle += Quantity(0.1, 'deg')
                masks.append(r.to_mask().to_image(self._parent_phot_obj.shape))

            interlopers = sum([m for m in masks if m is not None])
            mask = np.ones(self._parent_phot_obj.shape)
            mask[interlopers != 0] = 0

            return mask

        def _toggle_ext(self, event):
            """
            Method triggered by the extended source toggle button, either causes extended sources to be displayed
            or not, depending on the existing state.

            :param event: The matplotlib event passed through from the button press that triggers this method.
            """
            # Need to save the new state of this type of region being displayed in the dictionary thats used
            #  to keep track of such things. The invert function just switches whatever entry was already there
            #  (True or False) to the opposite (False or True).
            self._cur_act_reg_type['EXT'] = np.invert(self._cur_act_reg_type['EXT'])

            # Then the colour of the button is switched to indicate whether its toggled on or not
            if self._cur_act_reg_type['EXT']:
                self._ext_src_button.color = self._but_act_col
            else:
                self._ext_src_button.color = self._but_inact_col

            # Then the currently displayed regions are updated with this method
            self._draw_regions()

        def _toggle_pnt(self, event):
            """
            Method triggered by the point source toggle button, either causes point sources to be displayed
            or not, depending on the existing state.

            :param event: The matplotlib event passed through from the button press that triggers this method.
            """
            # See the _toggle_ext method for comments explaining
            self._cur_act_reg_type['PNT'] = np.invert(self._cur_act_reg_type['PNT'])
            if self._cur_act_reg_type['PNT']:
                self._pnt_src_button.color = self._but_act_col
            else:
                self._pnt_src_button.color = self._but_inact_col

            self._draw_regions()

        def _toggle_oth(self, event):
            """
            Method triggered by the other source toggle button, either causes other (i.e. not extended,
            point, or custom) sources to be displayed or not, depending on the existing state.

            :param event: The matplotlib event passed through from the button press that triggers this method.
            """
            # See the _toggle_ext method for comments explaining
            self._cur_act_reg_type['OTH'] = np.invert(self._cur_act_reg_type['OTH'])
            if self._cur_act_reg_type['OTH']:
                self._oth_src_button.color = self._but_act_col
            else:
                self._oth_src_button.color = self._but_inact_col

            self._draw_regions()

        def _toggle_cust(self, event):
            """
            Method triggered by the custom source toggle button, either causes custom sources to be displayed
            or not, depending on the existing state.

            :param event: The matplotlib event passed through from the button press that triggers this method.
            """
            # See the _toggle_ext method for comments explaining
            self._cur_act_reg_type['CUST'] = np.invert(self._cur_act_reg_type['CUST'])
            if self._cur_act_reg_type['CUST']:
                self._cust_src_button.color = self._but_act_col
            else:
                self._cust_src_button.color = self._but_inact_col

            self._draw_regions()

        def _new_ell_src(self, event):
            """
            Makes a new elliptical region on the data axis.

            :param event: The matplotlib event passed through from the button press that triggers this method.
            """
            # This matplotlib patch is what we add as an 'artist' to the data (i.e. image) axis and is the
            #  visual representation of our new region. This creates the matplotlib instance for an extended
            #  source, which is an Ellipse.
            new_patch = Ellipse(self._last_click, 36, 28)
            # Now the face and edge colours are set up. Face colour is completely see through as I want regions
            #  to just be denoted by their edges. The edge colour is set to white, fetching the colour definition
            #  set up in the class init.
            new_patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
            new_patch.set_edgecolor(self._inv_colour_convert['white'])
            # This enables 'picking' of the artist. When enabled picking will trigger an event when the
            #  artist is clicked on
            new_patch.set_picker(True)
            # Setting up the linewidth of the new region
            new_patch.set_linewidth(self._reg_line_width)
            # And adds the artist into the axis. As this is a new artist we don't call _draw_regions for this one.
            self._im_ax.add_artist(new_patch)
            # Updates the shape dictionary
            self._shape_dict[new_patch] = 'ellipse'
            # Adds an entry to the dictionary that keeps track of whether regions have been modified or not. In
            #  this case the region in question is brand new so the entry will always be True.
            self._edited_dict[new_patch] = True

        def _new_circ_src(self, event):
            """
            Makes a new circular region on the data axis.

            :param event: The matplotlib event passed through from the button press that triggers this method.
            """
            # This matplotlib patch is what we add as an 'artist' to the data (i.e. image) axis and is the
            #  visual representation of our new region. This creates the instance, a circle in this case.
            new_patch = Circle(self._last_click, 8)
            # Now the face and edge colours are set up. Face colour is completely see through as I want regions
            #  to just be denoted by their edges. The edge colour is set to white, fetching the colour definition
            #  set up in the class init.
            new_patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
            new_patch.set_edgecolor(self._inv_colour_convert['white'])
            # This enables 'picking' of the artist. When enabled picking will trigger an event when the
            #  artist is clicked on
            new_patch.set_picker(True)
            # Setting up the linewidth of the new region
            new_patch.set_linewidth(self._reg_line_width)
            # And adds the artist into the axis. As this is a new artist we don't call _draw_regions for this one.
            self._im_ax.add_artist(new_patch)
            # Updates the shape dictionary
            self._shape_dict[new_patch] = 'circle'
            # Adds an entry to the dictionary that keeps track of whether regions have been modified or not. In
            #  this case the region in question is brand new so the entry will always be True.
            self._edited_dict[new_patch] = True

        def _click_event(self, event):
            """
            This method is triggered by clicking somewhere on the data axis.

            :param event: The click event that triggered this method.
            """
            # Checks whether the click was 'in axis' - so whether it was actually on the image being displayed
            #  If it wasn't then we don't care about it
            if event.inaxes == self._im_ax:
                # This saves the position that the user clicked as the 'last click', as the user may now which
                #  to insert a new region there
                self._last_click = (event.xdata, event.ydata)

        def _on_region_pick(self, event):
            """
            This is triggered by selecting a region

            :param event: The event triggered on 'picking' an artist. Contains information about which artist
                triggered the event, location, etc.
            """
            # If interacting is turned off then we don't want this to do anything, likewise if a region that
            #  is just there for visualisation is clicked ons
            if not self._interacting_on or event.artist in self._ignore_arts:
                return

            # The _cur_pick attribute references which artist is currently selected, which we can grab from the
            #  artist picker event that triggered this method
            self._cur_pick = event.artist
            # Makes sure the instance knows a region is selected right now, set to False again when the click ends
            self._select = True
            # Stores the current position of the current pick
            # self._history.append([self._cur_pick, self._cur_pick.center])

            # Redraws the regions so that thicker lines are applied to the newly selected region
            self._draw_regions()

        def _on_release(self, event):
            """
            Method triggered when button released.

            :param event: Event triggered by releasing a button click.
            """
            # This method's one purpose is to set this to False, meaning that the currently picked artist
            #  (as referenced in self._cur_pick) isn't currently being clicked and held on
            self._select = False

        def _on_motion(self, event):
            """
            This is triggered when someone clicks and holds an artist, and then drags it around.

            :param event: Event triggered by motion of the mouse.
            """
            # Makes sure that an artist is actually clicked and held on, to make sure something should be
            #  being moved around right now
            if self._select is False:
                return

            # Set the new position of the currently picked artist to the new position of the event
            self._cur_pick.center = (event.xdata, event.ydata)

            # Changes the entry in the edited dictionary to True, as the region in question has been moved
            self._edited_dict[self._cur_pick] = True

        def _key_press(self, event):
            """
            A method triggered by the press of a key (or combination of keys) on the keyboard. For most keys
            this method does absolutely nothing, but it does enable the resizing and rotation of regions.

            :param event: The keyboard press event that triggers this method.
            """
            # if event.key == "ctrl+z":
            #     if len(self._history) != 0:
            #         self._history[-1][0].center = self._history[-1][1]
            #         self._history[-1][0].figure.canvas.draw()
            #         self._history.pop(-1)

            if event.key == "w" and self._cur_pick is not None:
                if type(self._cur_pick) == Circle:
                    self._cur_pick.radius += self._size_step
                # It is possible for actual artist type to be an Ellipse but for the region to be circular when
                #  it was taken from the parent Image of this instance, and in that case we still want it to behave
                #  like a circle for resizing.
                elif self._shape_dict[self._cur_pick] == 'circle':
                    self._cur_pick.height += self._size_step
                    self._cur_pick.width += self._size_step
                else:
                    self._cur_pick.height += self._size_step
                self._cur_pick.figure.canvas.draw()
                # The region has had its size changed, thus we make sure the class knows the region has been edited
                self._edited_dict[self._cur_pick] = True

            # For comments for the rest of these, see the event key 'w' one, they're the same but either shrinking
            #  or growing different axes
            if event.key == "s" and self._cur_pick is not None:
                if type(self._cur_pick) == Circle:
                    self._cur_pick.radius -= self._size_step
                elif self._shape_dict[self._cur_pick] == 'circle':
                    self._cur_pick.height -= self._size_step
                    self._cur_pick.width -= self._size_step
                else:
                    self._cur_pick.height -= self._size_step
                self._cur_pick.figure.canvas.draw()
                # The region has had its size changed, thus we make sure the class knows the region has been edited
                self._edited_dict[self._cur_pick] = True

            if event.key == "d" and self._cur_pick is not None:
                if type(self._cur_pick) == Circle:
                    self._cur_pick.radius += self._size_step
                elif self._shape_dict[self._cur_pick] == 'circle':
                    self._cur_pick.height += self._size_step
                    self._cur_pick.width += self._size_step
                else:
                    self._cur_pick.width += self._size_step
                self._cur_pick.figure.canvas.draw()
                # The region has had its size changed, thus we make sure the class knows the region has been edited
                self._edited_dict[self._cur_pick] = True

            if event.key == "a" and self._cur_pick is not None:
                if type(self._cur_pick) == Circle:
                    self._cur_pick.radius -= self._size_step
                elif self._shape_dict[self._cur_pick] == 'circle':
                    self._cur_pick.height -= self._size_step
                    self._cur_pick.width -= self._size_step
                else:
                    self._cur_pick.width -= self._size_step
                self._cur_pick.figure.canvas.draw()
                # The region has had its size changed, thus we make sure the class knows the region has been edited
                self._edited_dict[self._cur_pick] = True

            if event.key == "q" and self._cur_pick is not None:
                self._cur_pick.angle += self._rot_step
                self._cur_pick.figure.canvas.draw()
                # The region has had its size changed, thus we make sure the class knows the region has been edited
                self._edited_dict[self._cur_pick] = True

            if event.key == "e" and self._cur_pick is not None:
                self._cur_pick.angle -= self._rot_step
                self._cur_pick.figure.canvas.draw()
                # The region has had its size changed, thus we make sure the class knows the region has been edited
                self._edited_dict[self._cur_pick] = True

        def _update_reg_list(self) -> Dict:
            """
            This method goes through the current artists, checks whether any represent new or updated regions, and
            generates a new list of region objects from them.

            :return: The updated region dictionary.
            :rtype: Dict
            """
            # Here we use the edited dictionary to note that there have been changes to regions
            if any(self._edited_dict.values()):
                # Setting up the dictionary to store the altered regions in. We include keys for each of the ObsIDs
                #  associated with the parent product, and then another list with the key 'new'; for regions
                #  that have been added during the editing.
                new_reg_dict = {o: [] for o in self._parent_phot_obj.obs_ids}
                new_reg_dict['new'] = []

                # These artists are the ones that represent regions, the ones in self._ignore_arts are there
                #  just for visualisation (for instance showing an analysis/background region) and can't be
                #  turned on or off, can't be edited, and shouldn't be saved.
                # rel_artists = [arty for arty in self._im_ax.artists if arty not in self._ignore_arts]
                rel_artists = [arty for arty in self._im_ax.patches if arty not in self._ignore_arts]
                for artist in rel_artists:
                    # Fetches the boolean variable that describes if the region was edited
                    altered = self._edited_dict[artist]
                    # The altered variable is True if an existing region has changed or if a new artist exists
                    if altered and type(artist) == Ellipse:
                        # As instances of this class are always declared internally by an Image class, and I know
                        #  the image class always turns SkyRegions into PixelRegions, we know that its valid to
                        #  output PixelRegions here
                        cen = PixCoord(x=artist.center[0], y=artist.center[1])
                        # Creating the equivalent region object from the artist
                        new_reg = EllipsePixelRegion(cen, artist.width, artist.height, Quantity(artist.angle, 'deg'))
                        # Fetches and sets the colour of the region, converting from matplotlib colour
                        new_reg.visual['edgecolor'] = self._colour_convert[artist.get_edgecolor()]
                        new_reg.visual['facecolor'] = self._colour_convert[artist.get_edgecolor()]
                    elif altered and type(artist) == Circle:
                        cen = PixCoord(x=artist.center[0], y=artist.center[1])
                        # Creating the equivalent region object from the artist
                        new_reg = CirclePixelRegion(cen, artist.radius)
                        # Fetches and sets the colour of the region, converting from matplotlib colour
                        new_reg.visual['edgecolor'] = self._colour_convert[artist.get_edgecolor()]
                        new_reg.visual['facecolor'] = self._colour_convert[artist.get_edgecolor()]
                    else:
                        # Looking up the region because if we get to this point we know its an original region that
                        #  hasn't been altered
                        # Note that in this case its not actually a new reg, its just called that
                        new_reg = self._artist_region[artist]

                    # Checks to see whether it's an artist that has been modified or a new one
                    if artist in self._artist_obsids:
                        new_reg_dict[self._artist_obsids[artist]].append(new_reg)
                    else:
                        new_reg_dict['new'].append(new_reg)

            # In this case none of the entries in the dictionary that stores whether regions have been
            #  edited (or added) is True, so the new region list is exactly the same as the old one
            else:
                new_reg_dict = self._regions

            return new_reg_dict

        def _save_region_files(self, event=None):
            """
            This just creates the updated region dictionary from any modifications, converts the separate ObsID
            entries to individual region files, and then saves them to disk. All region files are output in RA-Dec
            coordinates, making use of the parent photometric objects WCS information.

            :param event: If triggered by a button, this is the event passed.
            """
            if self._reg_save_path is not None:
                # If the event is not the default None then this function has been triggered by the save button
                if event is not None:
                    # In the case of this button being successfully clicked I want it to turn green. Really I wanted
                    #  it to just flash green, but that doesn't seem to be working so turning green will be fine
                    self._save_button.color = 'green'

                # Runs the method that updates the list of regions with any alterations that the user has made
                final_regions = self._update_reg_list()
                for o in final_regions:
                    # Read out the regions for the current ObsID
                    rel_regs = final_regions[o]
                    # Convert them to degrees
                    rel_regs = [r.to_sky(self._parent_phot_obj.radec_wcs) for r in rel_regs]
                    # Construct a save_path
                    rel_save_path = self._reg_save_path.replace('.reg', '_{o}.reg'.format(o=o))
                    # write_ds9(rel_regs, rel_save_path, 'image', radunit='')
                    # This function is a part of the regions module, and will write out a region file.
                    #  Specifically RA-Dec coordinate system in units of degrees.
                    Regions(rel_regs).write(rel_save_path, format='ds9')

            else:
                raise ValueError('No save path was passed, so region files cannot be output.')


class ExpMap(Image):
    """
    A very simple subclass of the Image product class - designed to allow for easy interaction with exposure maps.

    :param str path: The path to where the product file SHOULD be located.
    :param str obs_id: The ObsID related to the ExpMap being declared.
    :param str instrument: The instrument related to the ExpMap being declared.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The command used to generate the product.
    :param Quantity lo_en: The lower energy bound used to generate this product.
    :param Quantity hi_en: The upper energy bound used to generate this product.
    :param List[List] obs_inst_combs: Supply a list of lists of ObsID-Instrument combinations if the image
        is combined and wasn't made by emosaic (e.g. [['0404910601', 'pn'], ['0404910601', 'mos1'],
        ['0404910601', 'mos2'], ['0201901401', 'pn'], ['0201901401', 'mos1'], ['0201901401', 'mos2']].
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, lo_en: Quantity, hi_en: Quantity, obs_inst_combs: List[List] = None):
        """
        Init of the ExpMap class.
        """
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, lo_en, hi_en,
                         obs_inst_combs=obs_inst_combs)
        self._prod_type = "expmap"
        # Need to overwrite the data unit attribute set by the Image init
        self._data_unit = Unit("s")

    def get_exp(self, at_coord: Quantity) -> Quantity:
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

    :param Image xga_image: The image component of the RateMap.
    :param ExpMap xga_expmap: The exposure map component of the RateMap.
    :param str/List[SkyRegion/PixelRegion]/dict regs: A region list file path, a list of region objects, or a
        dictionary of region lists with ObsIDs as dictionary keys.
    :param dict/SkyRegion/PixelRegion matched_regs: Similar to the regs argument, but in this case for a region
        that has been designated as 'matched', i.e. is the subject of a current analysis. This should either be
        supplied as a single region object, or as a dictionary of region objects with ObsIDs as keys, or None values
        if there is no match. Such a dictionary can be retrieved from a source using the 'matched_regions'
        property. Default is None.
    """
    def __init__(self, xga_image: Image, xga_expmap: ExpMap,
                 regs: Union[str, List[Union[SkyRegion, PixelRegion]], dict] = '',
                 matched_regs: Union[SkyRegion, PixelRegion, dict] = None):
        """
        This initialises a RateMap instance, where a count-rate image is divided by an exposure map, to create a map
        of X-ray counts.
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
        self._data_unit = Unit("ct/s")

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
        self._im_obj.regions = regs
        self._im_obj.matched_regions = matched_regs

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
        return Quantity(rate, "ct/s")

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
        :param float redshift: The redshift of the source that we wish to find the X-ray peak of.
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
        A signal-to-noise calculation method which takes information on source and background regions, then uses
        that to calculate a signal-to-noise for the source. This was primarily motivated by the desire to produce
        valid SNR values for combined data, where uneven exposure times across the combined field of view could
        cause issues with the usual approach of just summing the counts in the region images and scaling by area.
        This method can also measure signal to noises without exposure time correction.

        :param np.ndarray source_mask: The mask which defines the source region, ideally with interlopers removed.
        :param np.ndarray back_mask: The mask which defines the background region, ideally with interlopers removed.
        :param bool exp_corr: Should signal-to-noise be measured with exposure time correction, default is True. I
            recommend that this be true for combined observations, as exposure time could change quite dramatically
            across the combined product.
        :param bool allow_negative: Should pixels in the background subtracted count map be allowed to go below
            zero, which results in a lower signal-to-noise (and can result in a negative signal-to-noise).
        :return: A signal-to-noise value for the source region.
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

    def background_subtracted_counts(self, source_mask: np.ndarray, back_mask: np.ndarray) -> Quantity:
        """
        This method uses a user-supplied source and background mask (alongside knowledge of the sensor layout
        drawn from the exposure map) to calculate the number of background-subtracted counts within the source
        region of the image used to construct this RateMap.

        The exposure map is used to construct a sensor mask, so that we know where the chip gaps are and take
        them into account when calculating the ratio of areas of the source region to the background region. This
        is why this method is built into the RateMap rather than Image class.

        :param np.ndarray source_mask: The mask which defines the source region, ideally with interlopers removed.
        :param np.ndarray back_mask: The mask which defines the background region, ideally with interlopers removed.
        :return: The background subtracted counts in the source region.
        :rtype: Quantity
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
        src_area = (source_mask * self.sensor_mask).sum()
        back_area = (back_mask * self.sensor_mask).sum()

        # Calculate an area normalisation so the background counts can be scaled to the source counts properly
        area_norm = src_area / back_area
        # Find the total counts within the source area
        tot_cnt = (self.image.data * source_mask).sum()
        # Find the total counts within the background area
        bck_cnt = (self.image.data * back_mask).sum()

        # Simple calculation, re-normalising the background counts with the area ratio and subtracting background
        #  from the source. Then storing it in an astropy quantity
        cnts = Quantity(tot_cnt - (bck_cnt*area_norm), 'ct')

        return cnts

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

    @property
    def regions(self) -> Dict:
        """
        Property getter for regions associated with this ratemap.

        :return: Returns a dictionary of regions, if they have been associated with this object.
        :rtype: Dict[PixelRegion]
        """
        return self._regions

    @regions.setter
    def regions(self, new_reg: Union[str, List[Union[SkyRegion, PixelRegion]], dict]):
        """
        A setter for regions associated with this object, a region file path or a list/dict of regions is passed, then
        that file/set of regions is processed into the required format. If a list of regions is passed, it will
        be assumed that they are for the ObsID of the image. In the case of passing a dictionary of regions to a
        combined image we require that each ObsID that goes into the image has an entry in the dictionary.

        :param str/List[SkyRegion/PixelRegion]/dict new_reg: A new region file path, a list of region objects, or a
            dictionary of region lists with ObsIDs as dictionary keys.
        """
        if not isinstance(new_reg, (str, list, dict)):
            raise TypeError("Please pass either a path to a region file, a list of "
                            "SkyRegion/PixelRegion objects, or a dictionary of lists of SkyRegion/PixelRegion objects "
                            "with ObsIDs as keys.")

        # Checks to make sure that a region file path exists, if passed, then processes the file
        if isinstance(new_reg, str) and new_reg != '' and os.path.exists(new_reg):
            self._reg_file_path = new_reg
            self._regions = self._process_regions(new_reg)
        # Possible for an empty string to be passed in which case nothing happens
        elif isinstance(new_reg, str) and new_reg == '':
            pass
        elif isinstance(new_reg, str):
            warnings.warn("That region file path does not exist")
        # If an existing list of regions are passed then we just process them and assign them to regions attribute
        elif isinstance(new_reg, List) and all([isinstance(r, (SkyRegion, PixelRegion)) for r in new_reg]):
            self._reg_file_path = ""
            self._regions = self._process_regions(reg_objs=new_reg)
        elif isinstance(new_reg, dict) and all([all([isinstance(r, (SkyRegion, PixelRegion)) for r in rl])
                                                for o, rl in new_reg.items()]):
            self._reg_file_path = ""
            self._regions = self._process_regions(reg_objs=new_reg)
        else:
            raise ValueError("That value of new_reg is not valid, please pass either a path to a region file or "
                             "a list/dictionary of SkyRegion/PixelRegion objects")

        # This is the only part that's different from the implementation in the superclass. Here we make sure that
        #  the same attribute is set for the Image, so if the user were to access the image from the RateMap
        #  they would still see any regions that have been added. No doubt there is a more elegant solution but this
        #  is what you're getting right now because I am exhausted
        self._im_obj.regions = new_reg

    @property
    def matched_regions(self) -> Dict:
        """
        Property getter for any regions which have been designated a 'match' in the current analysis, if
        they have been set.

        :return: Returns a dictionary of matched regions, if they have been associated with this object.
        :rtype: Dict[PixelRegion]
        """
        return self._matched_regions

    @matched_regions.setter
    def matched_regions(self, new_reg: Union[str, List[Union[SkyRegion, PixelRegion]], dict]):
        """
        A setter for matched regions associated with this object, with a new single matched region or dictionary of
        matched regions (with keys being ObsIDs and one entry for each ObsID associated with this object) being passed.
        If a single region is passed then it will be assumed that it is associated with the current ObsID of this
        object.

        :param dict/SkyRegion/PixelRegion new_reg: A region that has been designated as 'matched', i.e. is the
            subject of a current analysis. This should either be supplied as a single region object, or as a
            dictionary of region objects with ObsIDs as keys.
        """
        if new_reg is not None and not isinstance(new_reg, (PixelRegion, SkyRegion, dict)):
            raise TypeError("Please pass either a dictionary of SkyRegion/PixelRegion objects with ObsIDs as "
                            "keys, or a single SkyRegion/PixelRegion object. Alternatively pass None for no match.")

        self._matched_regions = self._process_matched_regions(new_reg)

        # This is the only part that's different from the implementation in the superclass. Here we make sure that
        #  the same attribute is set for the Image, so if the user were to access the image from the RateMap
        #  they would still see any regions that have been added. No doubt there is a more elegant solution but this
        #  is what you're getting right now because I am exhausted
        self._im_obj.matched_regions = new_reg


class PSF(Image):
    """
    A subclass of image that is a wrapper for 2D images of PSFs that can be generated by SAS. This can be used to
    view the PSF and is used in other analyses to correct images.

    :param str path: The path to where the product file SHOULD be located.
    :param str psf_model: The model used for the generation of the PSF.
    :param str obs_id: The ObsID related to the PSF being declared.
    :param str instrument: The instrument related to the PSF being declared.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The command used to generate the product.
    """
    def __init__(self, path: str, psf_model: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str):
        """
        The init method for PSF class.
        """
        lo_en = Quantity(0, 'keV')
        hi_en = Quantity(100, 'keV')
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, lo_en, hi_en)
        self._prod_type = "psf"
        self._data_unit = Unit('')
        self._psf_centre = None
        self._psf_model = psf_model

    def get_val(self, at_coord: Quantity) -> float:
        """
        A simple method that converts the given coordinates to pixels, then finds the PSF value
        at those coordinates.

        :param Quantity at_coord: A pair of coordinates to find the PSF value time for.
        :return: The PSF value at the supplied coordinates.
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
    """
    :param list file_paths: The file paths of the individual PSF files for this grid.
    :param int bins: The number of bins per side of the grid.
    :param str psf_model: The model used to generate PSFs.
    :param np.ndarray x_bounds: The upper and lower x boundaries of the bins in image pixel coordinates.
    :param np.ndarray y_bounds: The upper and lower y boundaries of the bins in image pixel coordinates.
    :param str obs_id: The ObsID for which this PSFGrid was generated.
    :param str instrument: The instrument for which this PSFGrid was generated.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The commands used to generate the products.
    """
    def __init__(self, file_paths: list, bins: int, psf_model: str, x_bounds: np.ndarray, y_bounds: np.ndarray,
                 obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str):
        """
        The init of the PSFGrid class - a subclass of BaseAggregateProduct that wraps a set of PSFs that have been
        generated at different points on the detector.
        """
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


