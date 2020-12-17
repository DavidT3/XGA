#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 14/12/2020, 17:00. Copyright (c) David J Turner


import warnings
from typing import Tuple, List, Union

import numpy as np
from astropy import wcs
from astropy.units import Quantity, UnitBase, UnitsError, deg, pix, UnitConversionError
from astropy.visualization import LogStretch, MinMaxInterval, ImageNormalize, BaseStretch
from fitsio import read, read_header, FITSHDR
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from scipy.cluster.hierarchy import fclusterdata
from scipy.signal import fftconvolve

from . import BaseProduct, BaseAggregateProduct
from ..exceptions import FailedProductError, RateMapPairError, NotPSFCorrectedError, IncompatibleProductError
from ..sourcetools import ang_to_rad
from ..utils import xmm_sky, find_all_wcs


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

        # This is a flag to let XGA know that the Image object has been PSF corrected
        self._psf_corrected = False
        # These give extra information about the PSF correction, but can't be set unless PSF
        #  corrected is true
        self._psf_correction_algorithm = None
        self._psf_num_bins = None
        self._psf_num_iterations = None
        self._psf_model = None

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

    def coord_conv(self, coords: Quantity, output_unit: UnitBase) -> Quantity:
        """
        This will use the loaded WCSes, and astropy coordinates (including custom ones defined for this module),
        to perform common coordinate conversions for this product object.
        :param coords: The input coordinates quantity to convert, in units of either deg,
        pix, xmm_sky, or xmm_det (xmm_sky and xmm_det are defined for this module).
        :param output_unit: The astropy unit to convert to, can be either deg, pix, xmm_sky, or
        xmm_det (xmm_sky and xmm_det are defined for this module).
        :return: The converted coordinates.
        :rtype: Quantity
        """
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
                out_coord = Quantity(self.radec_wcs.all_world2pix(coords, 0), output_unit).astype(int)

            elif input_unit == "pix" and out_name == "deg":
                out_coord = Quantity(self.radec_wcs.all_pix2world(coords, 0), output_unit)
                # print(out_coord - Quantity(self.radec_wcs.wcs_pix2world(coords, 0), output_unit))

            # These go between degrees and XMM sky XY coordinates
            elif input_unit == "deg" and out_name == "xmm_sky":
                interim = self.radec_wcs.all_world2pix(coords, 0)
                out_coord = Quantity(self.skyxy_wcs.all_pix2world(interim, 0), xmm_sky)
            elif input_unit == "xmm_sky" and out_name == "deg":
                interim = self.skyxy_wcs.all_world2pix(coords, 0)
                out_coord = Quantity(self.radec_wcs.all_pix2world(interim, 0), deg)

            # These go between XMM sky XY and pixel coordinates
            elif input_unit == "xmm_sky" and out_name == "pix":
                out_coord = Quantity(self.skyxy_wcs.all_world2pix(coords, 0), output_unit).astype(int)
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
                out_coord = Quantity(self.detxy_wcs.all_world2pix(coords, 0), output_unit).astype(int)
            elif input_unit == "pix" and out_name == "xmm_det":
                out_coord = Quantity(self.detxy_wcs.all_pix2world(coords, 0), output_unit)

            # It is possible to convert between XMM coordinates and pixel and supply coordinates
            # outside the range covered by an image, but we can at least catch the error
            if out_name == "pix" and np.any(out_coord < 0) and self._prod_type != "psf":
                raise ValueError("You've converted to pixel coordinates, and some elements are less than zero.")

            # If there was only pair passed in, we'll return a flat numpy array
            if out_coord.shape == (1, 2):
                out_coord = out_coord[0, :]

            # if out_coord.shape ==
        elif input_unit == out_name and out_name == 'pix':
            out_coord = coords.astype(int)
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

    def get_view(self, ax: Axes, cross_hair: Quantity = None, mask: np.ndarray = None,
                 chosen_points: np.ndarray = None, other_points: List[np.ndarray] = None, zoom_in: bool = False,
                 manual_zoom_xlims: tuple = None, manual_zoom_ylims: tuple = None,
                 radial_bins_pix: np.ndarray = np.array([]), back_bin_pix: np.ndarray = None,
                 stretch: BaseStretch = LogStretch()) -> Axes:
        """
        The method that creates and populates the view axes, separate from actual view so outside methods
        can add a view to other matplotlib axes.
        :param Axes ax: The matplotlib axes on which to show the image.
        :param Quantity cross_hair: An optional parameter that can be used to plot a cross hair at
        the coordinates.
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
        only be triggered if a cross_hair coordinate is also specified.
        :param np.ndarray back_bin_pix: The inner and outer radii (in pixel units) of the annulus used to measure
        the background value for a given profile, will only be triggered if a cross_hair coordinate is also specified.
        :param BaseStretch stretch: The astropy scaling to use for the image data, default is log.
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
        if type(self) == RateMap:
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

        title = "{n} - {i} {l}-{u}keV {t}".format(n=self.src_name, i=ident, l=self._energy_bounds[0].to("keV").value,
                                                  u=self._energy_bounds[1].to("keV").value, t=self.type)
        # Its helpful to be able to distinguish PSF corrected image/ratemaps from the title
        if self.psf_corrected:
            title += ' - PSF Corrected'

        ax.set_title(title)

        # As this is a very quick view method, users will not be offered a choice of scaling
        #  There will be a more in depth way of viewing cluster data eventually
        norm = ImageNormalize(data=plot_data, interval=MinMaxInterval(), stretch=stretch)
        # I normalize with a log stretch, and use gnuplot2 colormap (pretty decent for clusters imo)

        if chosen_points is not None:
            ax.plot(chosen_points[:, 0], chosen_points[:, 1], '+', color='black', label="Chosen Point Cluster")
            ax.legend(loc="best")

        if other_points is not None:
            for cl in other_points:
                ax.plot(cl[:, 0], cl[:, 1], 'D')

        if cross_hair is not None:
            pix_coord = self.coord_conv(cross_hair, pix).value
            ax.axvline(pix_coord[0], color="white", linewidth=0.5)
            ax.axhline(pix_coord[1], color="white", linewidth=0.5)

            for ann_rad in radial_bins_pix:
                artist = Circle(pix_coord, ann_rad, fill=False, ec='white', linewidth=1.5)
                ax.add_artist(artist)

            if back_bin_pix is not None:
                inn_artist = Circle(pix_coord, back_bin_pix[0], fill=False, ec='white', linewidth=1.6,
                                    linestyle='dashed')
                out_artist = Circle(pix_coord, back_bin_pix[1], fill=False, ec='white', linewidth=1.6,
                                    linestyle='dashed')
                ax.add_artist(inn_artist)
                ax.add_artist(out_artist)

        ax.imshow(plot_data, norm=norm, origin="lower", cmap="gnuplot2")

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
             other_points: List[np.ndarray] = None, figsize: Tuple = (7, 6), zoom_in: bool = False,
             manual_zoom_xlims: tuple = None, manual_zoom_ylims: tuple = None,
             radial_bins_pix: np.ndarray = np.array([]), back_bin_pix: np.ndarray = None,
             stretch: BaseStretch = LogStretch()):
        """
        Powerful method to view this Image/RateMap/Expmap, with different options that can be used for eyeballing
        and producing figures for publication.
        :param Quantity cross_hair: An optional parameter that can be used to plot a cross hair at
        the coordinates.
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
        only be triggered if a cross_hair coordinate is also specified.
        :param np.ndarray back_bin_pix: The inner and outer radii (in pixel units) of the annulus used to measure
        the background value for a given profile, will only be triggered if a cross_hair coordinate is also specified.
        :param BaseStretch stretch: The astropy scaling to use for the image data, default is log.
        """

        # Create figure object
        fig = plt.figure(figsize=figsize)

        # Turns off any ticks and tick labels, we don't want them in an image
        ax = plt.gca()

        ax = self.get_view(ax, cross_hair, mask, chosen_points, other_points, zoom_in, manual_zoom_xlims,
                           manual_zoom_ylims, radial_bins_pix, back_bin_pix, stretch)
        plt.colorbar(ax.images[0])
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

        self._im_data = None
        self._ex_data = None
        self._data = None
        # TODO Could I combine these two, and just make edges = 2, on sensor = 1 etc?
        self._edge_mask = None
        self._on_sensor_mask = None

    def _construct_on_demand(self):
        """
        This method is complimentary to the _read_on_demand method of the base Image class, and ensures that
        the ratemap array is only created if the user actually asks for it. Otherwise a lot of time is wasted
        reading in files for individual images and exposure maps that are rarely used.
        """
        # Runs read on demand to grab the data for the image, as this was the input path to the super init call
        self._read_on_demand()
        # That reads in the WCS information (important), and stores the im data in _data
        # That is read out into this variable
        self._im_data = self.data
        # Then the path is changed, so that the exposure map file becomes the focus
        self._path = self._expmap_path
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

        # Store that edge mask as an attribute.
        self._edge_mask = comb

        # TODO Add another attribute that describes how many sensors a particular pixel falls on for combined
        #  ratemaps
        self._on_sensor_mask = det_map
        # And another mask for whether on or off the sensor, very simple for individual ObsID-Instrument combos
        # if self._obs_id != "combined":
        #     self._on_sensor_mask = det_map
        # # MUCH more complicated for combined ratemaps however, as what is on one detector might not be on another
        # else:
        #     for entry in self.header:
        #         if "EMSCF" in entry:
        #             print(self.header[entry])
        #

        # Re-setting some paths to make more sense
        self._path = self._im_path

        del self._im_data
        del self._ex_data

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
        return self._data

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

    def simple_peak(self, mask: np.ndarray, out_unit: UnitBase = deg) -> Tuple[Quantity, bool]:
        """
        Simplest possible way to find the position of the peak of X-ray emission in a ratemap. This method
        takes a mask in the form of a numpy array, which allows the user to mask out parts of the ratemap
        that shouldn't be searched (outside of a certain region, or within point sources for instance).
        :param np.ndarray mask: A numpy array used to weight the data. It should be 0 for pixels that
        aren't to be searched, and 1 for those that are.
        :param UnitBase out_unit: The desired output unit of the peak coordinates, the default is degrees.
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
            peak_conv = peak_pix

        # Find if the peak coordinates sit near an edge/chip gap
        edge_flag = self.near_edge(peak_pix)

        return peak_conv, edge_flag

    def clustering_peak(self, mask: np.ndarray, out_unit: UnitBase = deg, top_frac: float = 0.05,
                        max_dist: float = 5) -> Tuple[Quantity, bool, np.ndarray, List[np.ndarray]]:
        """
        An experimental peak finding function that cuts out the top 5% (by default) of array elements
        (by value), and runs a hierarchical clustering algorithm on their positions. The motivation
        for this is that the cluster peak will likely be contained in that top 5%, and the only other
        pixels that might be involved are remnants of poorly removed point sources. So when clusters have
        been formed, we can take the one with the most entries, and find the maximal pixel of that cluster.
        Should be consistent with simple_peak under ideal circumstances.
        :param np.ndarray mask: A numpy array used to weight the data. It should be 0 for pixels that
        aren't to be searched, and 1 for those that are.
        :param UnitBase out_unit: The desired output unit of the peak coordinates, the default is degrees.
        :param float top_frac: The fraction of the elements (ordered in descending value) that should be used
        to generate clusters, and thus be considered for the cluster centre.
        :param float max_dist: The maximum distance criterion for the hierarchical clustering algorithm, in pixels.
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

        return edge_flag

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
                 gen_cmd: str, raise_properly: bool = True):
        lo_en = Quantity(0, 'keV')
        hi_en = Quantity(100, 'keV')
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, lo_en, hi_en, raise_properly)
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
                 obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 raise_properly: bool = True):
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
            interim = PSF(f, psf_model, obs_id, instrument, stdout_str, stderr_str, gen_cmd, raise_properly)
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


