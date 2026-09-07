#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/23/26, 3:17 PM. Copyright (c) The Contributors.

from typing import Union, Optional

import tqdm
import numpy as np
from astropy.convolution import Kernel, convolve, convolve_fft, Gaussian2DKernel

from xga import OUTPUT
from xga.exceptions import ProductGenerationError, XGADeveloperError
from xga.products import Image, RateMap, ExpMap


def general_smooth(prod: Union[Image, RateMap], kernel: Kernel, mask: Optional[np.ndarray] = None, fft: bool = False,
                   ratemap_smooth_im: bool = True, force_resmooth: bool = False, norm_kernel: bool = True,
                   boundary: Union[str, None] = 'fill', fill_value: Union[int, float] = 0.0,
                   nan_treatment: str = 'interpolate', preserve_nan: bool = False,
                   normalization_zero_tol: Union[float, int] = 1e-8) -> Union[Image, RateMap]:
    """
    Applies Astropy's smoothing kernels to instances of the XGA Image and RateMap classes, returning a new
    instance of the same class with smoothing applied. If the input is a RateMap instance, then you may choose
    whether to smooth the image component or the image/expmap (using sm_im); if you choose the former, then the
    final smoothed RateMap will be produced by dividing the smoothed Image by the original ExpMap.

    Note that, as this function acts directly on XGA data products rather than on XGA sources or samples, the
    new XGA product instances it produces are NOT added to the storage structure of a source.The returned product
    instance can be manually added to a source by calling the <source instance>.update_products(<return from this function>) method.

    :param Image/RateMap prod: The XGA Image/RateMap to be smoothed. If you pass a RateMap, please see the 'sm_im'
        argument for extra options.
    :param Kernel kernel: The kernel with which to smooth the input data. Should be an instance of an Astropy Kernel.
    :param np.ndarray mask: A mask to apply to the data while smoothing (removing point source contaminants, for
        instance). The default is None, which means no mask is applied. This function expects a mask with 1s where
        the data you wish to keep is, and 0s where the data you wish to remove is - the style of mask produced by XGA.
    :param bool fft: If set to True, then a fast fourier transform method will be used for kernel convolution.
        The default is False.
    :param bool ratemap_smooth_im: If a RateMap is passed, should the image component be smoothed rather than the actual
        RateMap. Default is True, where the Image will be smoothed and divided by the original ExpMap. If set
        to False, the resulting RateMap will be bodged, with the ExpMap all 1s on the sensor.
    :param bool force_resmooth: Force a second smoothing convolution on an already-smoothed Image/RateMap.
        Default is False, in which case an error will be raised if the `prod` input is already smoothed.
    :param bool norm_kernel: Whether to normalize the kernel to have a sum of one, passed to the Astropy convolution
        function's `norm_kernel` argument. Default is True.
    :param Any boundary: A flag indicating how to handle boundaries, passed through to the Astropy convolution
        function's `boundary` argument, see Astropy documentation. The default value is 'fill'.
    :param float/int fill_value: The value to use outside the array when using` boundary='fill'`, passed through
        to the Astropy convolution function's `fill_value` argument. The default value is 0.0.
    :param str nan_treatment: The method that Astropy uses to handle NaNs in the input array, passed through
        to the Astropy convolution function's `nan_treatment` argument, see Astropy documentation. The default value is 'interpolate'.
    :param bool preserve_nan: After performing convolution, should pixels that were originally NaN again become NaN?
        Pass through to the Astropy convolution function's `preserve_nan` argument. The default is False.
    :param float/int normalization_zero_tol: The absolute tolerance on whether the kernel is different from zero. If
        the kernel sums to zero to within this precision, it cannot be normalized. Passed through to the Astropy
        convolution function's `normalization_zero_tol` argument. The default is 1e-8.
    :return: An XGA product with the smoothed Image or RateMap.
    :rtype: Image/RateMap
    """
    # Yes, we know we could/should have used kwargs to pass through Astropy smoothing function
    #  configuration. However, in this case we would rather have the docstring entries present in our
    #  docstring, rather than not mentioning them or appending the convolution function's docstring
    #  to ours at run time.

    # First off, we check the type of the product that has been passed in for smoothing
    if not isinstance(prod, Image) or type(prod) == ExpMap:
        raise TypeError("Only an XGA Image or RateMap instance can be passed to the 'prod' argument.")

    # Also need to check that the kernel has the right number of dimensions
    if len(kernel.shape) != 2:
        raise ValueError("The smoothing kernel must be two-dimensional, e.g. an "
                         "astropy.convolution.Gaussian2DKernel instance.")

    # While we ask for masks in the style XGA produces (0s where you don't want data, 1s where you do), unfortunately,
    #  the smoothing functions seem to want the opposite, so I'll quickly invert the mask here
    if mask is not None:
        mask *= -1

    # By default, we raise an error if the input product has already been smoothed, but
    #  we do also include an argument that allows the user to override the
    #  behavior and smooth again.
    if prod.smoothed and not force_resmooth:
        raise ProductGenerationError("Input XGA Image or RateMap has already been smoothed, and "
                                     "will not be smoothed again. To override this check you may pass "
                                     "`force_resmooth=True`.")

    # Now we figure out what exactly needs to be smoothed.
    # If the input product is an Image, then it is very straightforward - we just copy
    #  the data array and will apply smoothing to that
    if type(prod) == Image:
        data_to_smth = prod.data.copy()

    # In this case the user has passed a RateMap for smoothing and also requested that we
    #  directly smooth the count-rate array (as opposed to extracting the image array, smoothing
    #  that, then re-dividing by the exposure map).
    elif type(prod) == RateMap and not ratemap_smooth_im:
        raise NotImplementedError("XGA RateMaps can currently only be constructed from separate "
                                  "Image and ExpMap instances, and as such a new RateMap cannot"
                                  "yet be created from a smoothed count-rate array.")
        data_to_smth = prod.data.copy()

    # In this instance the user has passed a RateMap, but specified that we should smooth
    #  the IMAGE data, then create a new RateMap by dividing the smoothed image by the
    #  original exposure map.
    elif type(prod) == RateMap and ratemap_smooth_im:
        data_to_smth = prod.image.data.copy()

    # Catch all else statement, meant to raise a vaguely useful error if the user
    #  has passed some sub-class of Image that we don't directly support here.
    else:
        raise XGADeveloperError("Only Image and RateMap instances are directly supported, contact "
                                "the developers if you wish for us to support a sub-class of the Image class.")


    # We now apply the Astropy smoothing kernel to the data, using FFT or non-FFT methods depending
    #  on what the user specified in their call to this function
    if fft:
        sm_data = convolve_fft(data_to_smth, kernel, mask=mask, normalize_kernel=norm_kernel, boundary=boundary,
                           fill_value=fill_value, nan_treatment=nan_treatment, preserve_nan=preserve_nan,
                           normalization_zero_tol=normalization_zero_tol)
    else:
        sm_data = convolve(data_to_smth, kernel, mask=mask, normalize_kernel=norm_kernel, boundary=boundary,
                           fill_value=fill_value, nan_treatment=nan_treatment, preserve_nan=preserve_nan,
                           normalization_zero_tol=normalization_zero_tol)

    # Now we construct the new XGA product instance that houses the smoothed data
    #  In the case of an Image being passed in, we make an image to send back out
    if type(prod) == Image:
        sm_prod = Image({'data': sm_data, 'wcs': prod.radec_wcs, 'header': prod.header}, prod.obs_id,
                        prod.instrument, "", "", "", lo_en=prod.energy_bounds[0],
                        hi_en=prod.energy_bounds[1], telescope=prod.telescope, check_exists=False,
                        smoothed=True, smoothed_info=kernel)

    # User requested that the count-rate array of a RateMap be smoothed
    elif type(prod) == RateMap and not ratemap_smooth_im:
        raise NotImplementedError("XGA RateMaps can't yet be constructed from a single count-rate array.")

    # User requested that the Image of a RateMap be smoothed, and a new RateMap constructed using
    #  the original exposure map.
    elif type(prod) == RateMap and ratemap_smooth_im:
        # Construct the Image instance
        sm_im_prod = Image({'data': sm_data, 'wcs': prod.radec_wcs, 'header': prod.header}, prod.obs_id,
                           prod.instrument, "", "", "", lo_en=prod.energy_bounds[0],
                           hi_en=prod.energy_bounds[1], telescope=prod.telescope, check_exists=False,
                           smoothed=True, smoothed_info=kernel)

        # Now we make a new RateMap instance using that smoothed image
        sm_prod = RateMap(sm_im_prod, prod.expmap)

    else:
        raise XGADeveloperError("Only Image and RateMap instances are directly supported, contact "
                                "the developers if you wish for us to support a sub-class of the Image class.")

    return sm_prod


def adaptive_smooth(
        prod: Union[Image, RateMap], 
        target_snr: Union[int, float], 
        max_size: Union[int, float] = 30,
        min_size: Union[int, float] = 0.5,
        num_widths: int = 40,
        mask: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        variance: Optional[np.ndarray] = None,
        fft: bool = True,
        ratemap_smooth_im: bool = True,
        force_resmooth: bool = False, 
        norm_kernel: bool = True,
        boundary: Union[str, None] = 'fill', 
        fill_value: Union[int, float] = 0.0,
        nan_treatment: str = 'interpolate', 
        preserve_nan: bool = False,
        normalization_zero_tol: Union[float, int] = 1e-8
    ) -> Union[Image, RateMap]:
    """
    Performs adaptive smoothing on XGA image data using a technique similar to XMM SAS's asmooth function. Uses a grid of possible
    kernel widths to find the kernel width for each pixel which would give an SNR for that pixel close to some target SNR as set by 
    the user. 
    
    Applies to instances of the XGA Image and RateMap classes, returning a new instance of the same class with smoothing applied. 
    If the input is a RateMap instance, then you may choose whether to smooth the image component or the image/expmap (using sm_im);
    if you choose the former, then the final smoothed RateMap will be produced by dividing the smoothed Image by the original ExpMap.

    Note that, as this function acts directly on XGA data products rather than on XGA sources or samples, the
    new XGA product instances it produces are NOT added to the storage structure of a source.The returned product
    instance can be manually added to a source by calling the <source instance>.update_products(<return from this function>) method.

    :param Image/RateMap prod: The XGA Image/RateMap to be smoothed. If you pass a RateMap, please see the 'sm_im'
        argument for extra options.
    :param int/float target_snr: The target SNR for each pixel to have after smoothing.
    :param int/float max_size: The maxmimum width (in pixels) of the Gaussian kernel to use when smoothing. 
        If the target_snr cannot be recovered for a given pixel, this value will be used.
        The default is 30 pixels
    :param int/float min_size: The minimum width (in pixels) of the Gaussian kernel to use when smoothing.
        The default is 0.5 pixels.
    :param int num_widths: The number of widths to have in the grid. These will be logarithmically spaced between min_size and max_size.
        More widths will give a better estimate but will be slower.
        The default is 40.
    :param np.ndarray mask: A mask to apply to the data while smoothing (removing point source contaminants, for
        instance). The default is None, which means no mask is applied. This function expects a mask with 1s where
        the data you wish to keep is, and 0s where the data you wish to remove is - the style of mask produced by XGA.
    :param np.ndarray weights: How much should each pixel be weighted when evaluating the SNR. 
        The default is an array of ones with shape(Image.data).
    :param np.ndarray variance: The variance map of the Image. Since this will mostly be used for low-count X-ray data
        the image can generally be assumed as Poisson-like and so the variance map will just be the photon map.
        The default is therefore a copy of Image.data
    :param bool fft: If set to True, then a fast fourier transform method will be used for kernel convolution. This is the
        recommended behaviour for speed as there are many convolutions that need to happen.
        The default is True.
    :param bool ratemap_smooth_im: If a RateMap is passed, should the image component be smoothed rather than the actual
        RateMap. Default is True, where the Image will be smoothed and divided by the original ExpMap. If set
        to False, the resulting RateMap will be bodged, with the ExpMap all 1s on the sensor.
    :param bool force_resmooth: Force a second smoothing convolution on an already-smoothed Image/RateMap.
        Default is False, in which case an error will be raised if the `prod` input is already smoothed.
    :param bool norm_kernel: Whether to normalize the kernel to have a sum of one, passed to the Astropy convolution
        function's `norm_kernel` argument. Default is True.
    :param Any boundary: A flag indicating how to handle boundaries, passed through to the Astropy convolution
        function's `boundary` argument, see Astropy documentation. The default value is 'fill'.
    :param float/int fill_value: The value to use outside the array when using` boundary='fill'`, passed through
        to the Astropy convolution function's `fill_value` argument. The default value is 0.0.
    :param str nan_treatment: The method that Astropy uses to handle NaNs in the input array, passed through
        to the Astropy convolution function's `nan_treatment` argument, see Astropy documentation. The default value is 'interpolate'.
    :param bool preserve_nan: After performing convolution, should pixels that were originally NaN again become NaN?
        Pass through to the Astropy convolution function's `preserve_nan` argument. The default is False.
    :param float/int normalization_zero_tol: The absolute tolerance on whether the kernel is different from zero. If
        the kernel sums to zero to within this precision, it cannot be normalized. Passed through to the Astropy
        convolution function's `normalization_zero_tol` argument. The default is 1e-8.
    :return: An XGA product with the smoothed Image or RateMap.
    :rtype: Image/RateMap
    """
    # Yes, we know we could/should have used kwargs to pass through Astropy smoothing function
    #  configuration. However, in this case we would rather have the docstring entries present in our
    #  docstring, rather than not mentioning them or appending the convolution function's docstring
    #  to ours at run time.

    # First off, we check the type of the product that has been passed in for smoothing
    if not isinstance(prod, Image) or type(prod) == ExpMap:
        raise TypeError("Only an XGA Image or RateMap instance can be passed to the 'prod' argument.")

    # While we ask for masks in the style XGA produces (0s where you don't want data, 1s where you do), unfortunately,
    #  the smoothing functions seem to want the opposite, so I'll quickly invert the mask here
    if mask is not None:
        mask *= -1

    # By default, we raise an error if the input product has already been smoothed, but
    #  we do also include an argument that allows the user to override the
    #  behavior and smooth again.
    if prod.smoothed and not force_resmooth:
        raise ProductGenerationError("Input XGA Image or RateMap has already been smoothed, and "
                                        "will not be smoothed again. To override this check you may pass "
                                        "`force_resmooth=True`.")

    # Now we figure out what exactly needs to be smoothed.
    # If the input product is an Image, then it is very straightforward - we just copy
    #  the data array and will apply smoothing to that
    if type(prod) == Image:
        data_to_smth = prod.data.copy()

    # In this case the user has passed a RateMap for smoothing and also requested that we
    #  directly smooth the count-rate array (as opposed to extracting the image array, smoothing
    #  that, then re-dividing by the exposure map).
    elif type(prod) == RateMap and not ratemap_smooth_im:
        raise NotImplementedError("XGA RateMaps can currently only be constructed from separate "
                                    "Image and ExpMap instances, and as such a new RateMap cannot"
                                    "yet be created from a smoothed count-rate array.")
        data_to_smth = prod.data.copy()

    # In this instance the user has passed a RateMap, but specified that we should smooth
    #  the IMAGE data, then create a new RateMap by dividing the smoothed image by the
    #  original exposure map.
    elif type(prod) == RateMap and ratemap_smooth_im:
        data_to_smth = prod.image.data.copy()

    # Catch all else statement, meant to raise a vaguely useful error if the user
    #  has passed some sub-class of Image that we don't directly support here.
    else:
        raise XGADeveloperError("Only Image and RateMap instances are directly supported, contact "
                                "the developers if you wish for us to support a sub-class of the Image class.")
    
    weights = weights if weights is not None else np.ones_like(data_to_smth)
    variance = variance if variance is not None else data_to_smth.copy()

    # pad the arrays by 4 times the maximum width (this is the radius of a gaussian 2D kernel)
    pad_width = int(np.ceil(4*max_size))
    img_pad = np.pad(data_to_smth, pad_width, 'reflect')
    weight_pad = np.pad(weights, pad_width, 'reflect')
    var_pad = np.pad(variance, pad_width, 'reflect')

    # And now crop to the area specified by the lims (and account for pad width offset so kernel never runs away)

    # Precompute the constant arrays
    weight_pad = weight_pad
    num_base = img_pad * weight_pad
    den_base = (weight_pad**2) * var_pad

    # Make a grid of widths
    widths = np.geomspace(max_size, min_size, num_widths)

    # Caches
    best_diff = np.full(num_base.shape, np.inf)
    kernel_sizes = np.full(num_base.shape, max_size)
    num_cache = np.empty((len(widths),)+num_base.shape)
    final_num = np.zeros_like(num_base)
    final_norm = np.zeros_like(num_base)

    # Iterate over widths
    for wid, width in enumerate(tqdm.tqdm(widths)):
        kernel = Gaussian2DKernel(width).array

        if fft:
            num = convolve_fft(num_base, kernel, mask=mask, normalize_kernel=norm_kernel, boundary=boundary,
                               fill_value=fill_value, nan_treatment=nan_treatment, preserve_nan=preserve_nan,
                               normalization_zero_tol=normalization_zero_tol)
            denom_square = convolve_fft(den_base, kernel**2, mask=mask, normalize_kernel=False, boundary=boundary,
                                        fill_value=fill_value, nan_treatment=nan_treatment, preserve_nan=preserve_nan,
                                        normalization_zero_tol=normalization_zero_tol)
        else:
            num = convolve(num_base, kernel, mask=mask, normalize_kernel=norm_kernel, boundary=boundary,
                           fill_value=fill_value, nan_treatment=nan_treatment, preserve_nan=preserve_nan,
                           normalization_zero_tol=normalization_zero_tol)
            denom_square = convolve(den_base, kernel**2, mask=mask, normalize_kernel=False, boundary=boundary,
                                    fill_value=fill_value, nan_treatment=nan_treatment, preserve_nan=preserve_nan,
                                    normalization_zero_tol=normalization_zero_tol)
        denom = np.sqrt(denom_square)

        snr = num/denom
        diff = np.abs(snr-target_snr)

        # If pixel SNR is closer to target SNR, update width in kernel sizes.
        better = diff < best_diff
        best_diff = np.where(better, diff, best_diff)
        kernel_sizes = np.where(better, width, kernel_sizes)
        num_cache[wid] = num

    # Assign pixels to widths
    nearest_idx = np.argmin(np.abs(kernel_sizes[None] - widths[:, None, None]), axis=0)
    
    # Iterate over widths again and select the best pixels for the final image
    for wid, width in enumerate(tqdm.tqdm(widths)):
        id_mask = (nearest_idx == wid)
        if not np.any(id_mask):
            continue  # if no pixels use this width (unlikely)
        kernel = Gaussian2DKernel(width).array
        final_num[id_mask] = num_cache[wid][id_mask]
        if fft:
            norm = convolve_fft(weight_pad, kernel, mask=mask, normalize_kernel=norm_kernel, boundary=boundary,
                                fill_value=fill_value, nan_treatment=nan_treatment, preserve_nan=preserve_nan,
                                normalization_zero_tol=normalization_zero_tol)
        else:
            norm = convolve(weight_pad, kernel, mask=mask, normalize_kernel=norm_kernel, boundary=boundary,
                            fill_value=fill_value, nan_treatment=nan_treatment, preserve_nan=preserve_nan,
                            normalization_zero_tol=normalization_zero_tol)
        final_norm[id_mask] = norm[id_mask]

    # Normalise the final image
    final_image = final_num / final_norm

    # Remove the padded borders (only half padwidth because of where xlo, xhi, ylo, yhi are defined)
    sm_data = final_image[int(pad_width/2):int(-pad_width/2), int(pad_width/2):int(-pad_width/2)]

    # Now we construct the new XGA product instance that houses the smoothed data
    #  In the case of an Image being passed in, we make an image to send back out
    if type(prod) == Image:
        sm_prod = Image({'data': sm_data, 'wcs': prod.radec_wcs, 'header': prod.header}, prod.obs_id,
                        prod.instrument, "", "", "", lo_en=prod.energy_bounds[0],
                        hi_en=prod.energy_bounds[1], telescope=prod.telescope, check_exists=False,
                        smoothed=True, smoothed_info=kernel)

    # User requested that the count-rate array of a RateMap be smoothed
    elif type(prod) == RateMap and not ratemap_smooth_im:
        raise NotImplementedError("XGA RateMaps can't yet be constructed from a single count-rate array.")

    # User requested that the Image of a RateMap be smoothed, and a new RateMap constructed using
    #  the original exposure map.
    elif type(prod) == RateMap and ratemap_smooth_im:
        # Construct the Image instance
        sm_im_prod = Image({'data': sm_data, 'wcs': prod.radec_wcs, 'header': prod.header}, prod.obs_id,
                            prod.instrument, "", "", "", lo_en=prod.energy_bounds[0],
                            hi_en=prod.energy_bounds[1], telescope=prod.telescope, check_exists=False,
                            smoothed=True, smoothed_info=kernel)

        # Now we make a new RateMap instance using that smoothed image
        sm_prod = RateMap(sm_im_prod, prod.expmap)

    else:
        raise XGADeveloperError("Only Image and RateMap instances are directly supported, contact "
                                "the developers if you wish for us to support a sub-class of the Image class.")

    return sm_prod
