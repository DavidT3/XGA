#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/11/26, 12:56 PM. Copyright (c) The Contributors.

from typing import Literal

import numpy as np
import tqdm
from astropy.convolution import Gaussian2DKernel, Kernel, convolve, convolve_fft

from xga.exceptions import ProductGenerationError, XGADeveloperError
from xga.products import ExpMap, Image, RateMap


def general_smooth(
    prod: Image | RateMap,
    kernel: Kernel,
    mask: np.ndarray | None = None,
    fft: bool = False,
    ratemap_smooth_im: bool = True,
    force_resmooth: bool = False,
    norm_kernel: bool = True,
    boundary: str | None = "fill",
    fill_value: int | float = 0.0,
    nan_treatment: str = "interpolate",
    preserve_nan: bool = False,
    normalization_zero_tol: float | int = 1e-8,
) -> Image | RateMap:
    """
    Applies Astropy's smoothing kernels to instances of the XGA Image and RateMap classes, returning a new
    instance of the same class with smoothing applied. If the input is a RateMap instance, then you may choose
    whether to smooth the image component or the image/expmap (using sm_im); if you choose the former, then the
    final smoothed RateMap will be produced by dividing the smoothed Image by the original ExpMap.

    Note that, as this function acts directly on XGA data products rather than on XGA sources or samples, the
    new XGA product instances it produces are NOT added to the storage structure of a source.The returned product
    instance can be manually added to a source by calling the
    <source instance>.update_products(<return from this function>) method.

    :param Image/RateMap prod: The XGA Image/RateMap to be smoothed. If you pass a RateMap, please see the 'sm_im'
        argument for extra options.
    :param Kernel kernel: The kernel with which to smooth the input data. Should be an instance of an Astropy Kernel.
    :param np.ndarray mask: A mask to apply to the data while smoothing (removing point source contaminants, for
        instance). The default is None, which means no mask is applied. This function expects a mask with 1s where
        the data you wish to keep is, and 0s where the data you wish to remove is - the style of mask produced by XGA.
    :param bool fft: If set to True, then a fast fourier transform method will be used for kernel convolution.
        The default is False.
    :param bool ratemap_smooth_im: If a RateMap is passed, should the image component be smoothed rather than the
        actual RateMap. Default is True, where the Image will be smoothed and divided by the original ExpMap. If set
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
        to the Astropy convolution function's `nan_treatment` argument, see Astropy documentation. The default value is
        'interpolate'.
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
    if not isinstance(prod, Image) or type(prod) is ExpMap:
        raise TypeError("Only an XGA Image or RateMap instance can be passed to the 'prod' argument.")

    # Also need to check that the kernel has the right number of dimensions
    if len(kernel.shape) != 2:
        raise ValueError(
            "The smoothing kernel must be two-dimensional, e.g. an astropy.convolution.Gaussian2DKernel instance."
        )

    # While we ask for masks in the style XGA produces (0s where you don't want data, 1s where you do), unfortunately,
    #  the smoothing functions seem to want the opposite, so I'll quickly invert the mask here
    if mask is not None:
        mask[mask == 0] = -1
        mask[mask == 1] = 0
        mask[mask == -1] = 1

    # By default, we raise an error if the input product has already been smoothed, but
    #  we do also include an argument that allows the user to override the
    #  behavior and smooth again.
    if prod.smoothed and not force_resmooth:
        raise ProductGenerationError(
            "Input XGA Image or RateMap has already been smoothed, and "
            "will not be smoothed again. To override this check you may pass "
            "`force_resmooth=True`."
        )

    # Now we figure out what exactly needs to be smoothed.
    # If the input product is an Image, then it is very straightforward - we just copy
    #  the data array and will apply smoothing to that
    if type(prod) is Image:
        data_to_smth = prod.data.copy()

    # In this case the user has passed a RateMap for smoothing and also requested that we
    #  directly smooth the count-rate array (as opposed to extracting the image array, smoothing
    #  that, then re-dividing by the exposure map).
    elif type(prod) is RateMap and not ratemap_smooth_im:
        raise NotImplementedError(
            "XGA RateMaps can currently only be constructed from separate "
            "Image and ExpMap instances, and as such a new RateMap cannot"
            "yet be created from a smoothed count-rate array."
        )
        data_to_smth = prod.data.copy()

    # In this instance the user has passed a RateMap, but specified that we should smooth
    #  the IMAGE data, then create a new RateMap by dividing the smoothed image by the
    #  original exposure map.
    elif type(prod) is RateMap and ratemap_smooth_im:
        data_to_smth = prod.image.data.copy()

    # Catch all else statement, meant to raise a vaguely useful error if the user
    #  has passed some sub-class of Image that we don't directly support here.
    else:
        raise XGADeveloperError(
            "Only Image and RateMap instances are directly supported, contact "
            "the developers if you wish for us to support a sub-class of the Image class."
        )

    # We now apply the Astropy smoothing kernel to the data, using FFT or non-FFT methods depending
    #  on what the user specified in their call to this function
    if fft:
        sm_data = convolve_fft(
            data_to_smth,
            kernel,
            mask=mask,
            normalize_kernel=norm_kernel,
            boundary=boundary,
            fill_value=fill_value,
            nan_treatment=nan_treatment,
            preserve_nan=preserve_nan,
            normalization_zero_tol=normalization_zero_tol,
        )
    else:
        sm_data = convolve(
            data_to_smth,
            kernel,
            mask=mask,
            normalize_kernel=norm_kernel,
            boundary=boundary,
            fill_value=fill_value,
            nan_treatment=nan_treatment,
            preserve_nan=preserve_nan,
            normalization_zero_tol=normalization_zero_tol,
        )

    # Now we construct the new XGA product instance that houses the smoothed data
    #  In the case of an Image being passed in, we make an image to send back out
    if type(prod) is Image:
        sm_prod = Image(
            {"data": sm_data, "wcs": prod.radec_wcs, "header": prod.header},
            prod.obs_id,
            prod.instrument,
            "",
            "",
            "",
            lo_en=prod.energy_bounds[0],
            hi_en=prod.energy_bounds[1],
            telescope=prod.telescope,
            check_exists=False,
            smoothed=True,
            smoothed_info=kernel,
        )

    # User requested that the count-rate array of a RateMap be smoothed
    elif type(prod) is RateMap and not ratemap_smooth_im:
        raise NotImplementedError("XGA RateMaps can't yet be constructed from a single count-rate array.")

    # User requested that the Image of a RateMap be smoothed, and a new RateMap constructed using
    #  the original exposure map.
    elif type(prod) is RateMap and ratemap_smooth_im:
        # Construct the Image instance
        sm_im_prod = Image(
            {"data": sm_data, "wcs": prod.radec_wcs, "header": prod.header},
            prod.obs_id,
            prod.instrument,
            "",
            "",
            "",
            lo_en=prod.energy_bounds[0],
            hi_en=prod.energy_bounds[1],
            telescope=prod.telescope,
            check_exists=False,
            smoothed=True,
            smoothed_info=kernel,
        )

        # Now we make a new RateMap instance using that smoothed image
        sm_prod = RateMap(sm_im_prod, prod.expmap)

    else:
        raise XGADeveloperError(
            "Only Image and RateMap instances are directly supported, contact "
            "the developers if you wish for us to support a sub-class of the Image class."
        )

    return sm_prod


def adaptive_smooth(
    prod: Image | RateMap,
    target_snr: int | float,
    method: Literal["asmooth", "csmooth"] = "asmooth",
    max_size: int | float = 30,
    min_size: int | float = 0.5,
    num_widths: int = 40,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    variance: np.ndarray | None = None,
    fft: bool = True,
    ratemap_smooth_im: bool = True,
    force_resmooth: bool = False,
    norm_kernel: bool = True,
    boundary: str | None = "fill",
    fill_value: int | float = 0.0,
    nan_treatment: str = "interpolate",
    preserve_nan: bool = False,
    normalization_zero_tol: float | int = 1e-8,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
) -> Image | RateMap:
    """
    Performs adaptive smoothing on XGA image data using a technique similar to XMM SAS's asmooth function. Uses a
    grid of possible kernel widths to find the kernel width for each pixel which would give an SNR for that pixel
    close to some target SNR as set by the user.

    Applies to instances of the XGA Image and RateMap classes, returning a new instance of the same class with
    smoothing applied.
    If the input is a RateMap instance, then you may choose whether to smooth the image component or the image/expmap
    (using sm_im); if you choose the former, then the final smoothed RateMap will be produced by dividing the smoothed
    Image by the original ExpMap.

    Note that, as this function acts directly on XGA data products rather than on XGA sources or samples, the
    new XGA product instances it produces are NOT added to the storage structure of a source.The returned product
    instance can be manually added to a source by calling the
    <source instance>.update_products(<return from this function>) method.

    :param Image/RateMap prod: The XGA Image/RateMap to be smoothed. If you pass a RateMap, please see the 'sm_im'
        argument for extra options.
    :param int/float target_snr: The target SNR for each pixel to have after smoothing.
    :param Literal["asmooth", "csmooth"] method: Specify the adaptive smoothing method to use. "asmooth" uses a method
        similar XMM-SAS's asmooth (https://xmm-tools.cosmos.esa.int/external/sas/current/doc/asmooth/index.html),
        starting from largest kernel to smallest and applying the kernel which is gives SNR closest to the target.
        "csmooth" is more like the method in Ciao (https://cxc.cfa.harvard.edu/ciao/ahelp/csmooth.html), works
        from smallest to largest kernel size and applies the smallest kernel with SNR >= the target SNR; if none of
        the tested kernel sizes reach the target SNR, then the largest available kernel is used. csmooth avoids
        oversmoothing the data, for better or worse.
    :param int/float max_size: The maximum width (in pixels) of the Gaussian kernel to use when smoothing.
        If the target_snr cannot be recovered for a given pixel, this value will be used.
        The default is 30 pixels
    :param int/float min_size: The minimum width (in pixels) of the Gaussian kernel to use when smoothing.
        The default is 0.5 pixels.
    :param int num_widths: The number of widths to have in the grid. These will be logarithmically spaced between
        min_size and max_size. More widths will give a better estimate but will be slower. The default is 40.
    :param np.ndarray mask: A mask to apply to the data while smoothing (removing point source contaminants, for
        instance). The default is None, which means no mask is applied. This function expects a mask with 1s where
        the data you wish to keep is, and 0s where the data you wish to remove is - the style of mask produced by XGA.
    :param np.ndarray weights: How much should each pixel be weighted when evaluating the SNR.
        The default is an array of ones with shape(Image.data).
    :param np.ndarray variance: The variance map of the Image. Since this will mostly be used for low-count X-ray data
        the image can generally be assumed as Poisson-like and so the variance map will just be the photon map.
        The default is therefore a copy of Image.data
    :param bool fft: If set to True, then a fast fourier transform method will be used for kernel convolution. This
        is the recommended behaviour for speed as there are many convolutions that need to happen. The default is True.
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
        to the Astropy convolution function's `nan_treatment` argument, see Astropy documentation. The default value
        is 'interpolate'.
    :param bool preserve_nan: After performing convolution, should pixels that were originally NaN again become NaN?
        Pass through to the Astropy convolution function's `preserve_nan` argument. The default is False.
    :param float/int normalization_zero_tol: The absolute tolerance on whether the kernel is different from zero. If
        the kernel sums to zero to within this precision, it cannot be normalized. Passed through to the Astropy
        convolution function's `normalization_zero_tol` argument. The default is 1e-8.
    :param tuple xlim: A tuple of the minimum and maximum x-limits to use for this smoothing in pixel coordinates. The
        smoothing will only be applied within these limits (plus needed padding to deal with boundary conditions).
        Default is None, which will smooth the whole image instead of just a cutout.
    :param tuple ylim: A tuple of the minimum and maximum y-limits to use for this smoothing in pixel coordinates. The
        smoothing will only be applied within these limits (plus needed padding to deal with boundary conditions).
        Default is None, which will smooth the whole image instead of just a cutout.
    :return: An XGA product with the smoothed Image or RateMap.
    :rtype: Image/RateMap
    """
    # Yes, we know we could/should have used kwargs to pass through Astropy smoothing function
    #  configuration. However, in this case we would rather have the docstring entries present in our
    #  docstring, rather than not mentioning them or appending the convolution function's docstring
    #  to ours at run time.

    # First off, we check the type of the product that has been passed in for smoothing
    if not isinstance(prod, Image) or type(prod) is ExpMap:
        raise TypeError("Only an XGA Image or RateMap instance can be passed to the 'prod' argument.")

    # While we ask for masks in the style XGA produces (0s where you don't want data, 1s where you do), unfortunately,
    #  the smoothing functions seem to want the opposite, so I'll quickly invert the mask here
    if mask is not None:
        mask[mask == 0] = -1
        mask[mask == 1] = 0
        mask[mask == -1] = 1

    # By default, we raise an error if the input product has already been smoothed, but
    #  we do also include an argument that allows the user to override the
    #  behavior and smooth again.
    if prod.smoothed and not force_resmooth:
        raise ProductGenerationError(
            "Input XGA Image or RateMap has already been smoothed, and "
            "will not be smoothed again. To override this check you may pass "
            "`force_resmooth=True`."
        )

    # Catch for the smoothing method, raise a Value error if it is not an eligible method
    #   (csmooth or asmooth)
    if method not in ["csmooth", "asmooth"]:
        raise ValueError(
            f"Smoohting method '{method}' is not a valid adaptive smoothing method. Please use 'asmooth' "
            "for XMM-SAS like smoothing, or 'csmooth' for CIAO-like smoothing."
        )

    # Now we figure out what exactly needs to be smoothed.
    # If the input product is an Image, then it is very straightforward - we just copy
    #  the data array and will apply smoothing to that
    if type(prod) is Image:
        data_to_smth = prod.data.copy()

    # In this case the user has passed a RateMap for smoothing and also requested that we
    #  directly smooth the count-rate array (as opposed to extracting the image array, smoothing
    #  that, then re-dividing by the exposure map).
    elif type(prod) is RateMap and not ratemap_smooth_im:
        raise NotImplementedError(
            "XGA RateMaps can currently only be constructed from separate "
            "Image and ExpMap instances, and as such a new RateMap cannot"
            "yet be created from a smoothed count-rate array."
        )
        data_to_smth = prod.data.copy()

    # In this instance the user has passed a RateMap, but specified that we should smooth
    #  the IMAGE data, then create a new RateMap by dividing the smoothed image by the
    #  original exposure map.
    elif type(prod) is RateMap and ratemap_smooth_im:
        data_to_smth = prod.image.data.copy()

    # Catch all else statement, meant to raise a vaguely useful error if the user
    #  has passed some sub-class of Image that we don't directly support here.
    else:
        raise XGADeveloperError(
            "Only Image and RateMap instances are directly supported, contact "
            "the developers if you wish for us to support a sub-class of the Image class."
        )

    # Set default values for weights (ones of the same shape as the input image data) and the variance map (assuming
    #   that the data are poissonian so using the input data as the variance).
    weights = weights if weights is not None else np.ones_like(data_to_smth)
    variance = variance if variance is not None else data_to_smth.copy()

    # Set the x and y limits if provided, or set to smooth over full image if not explicitly passed by the user.
    xlim = xlim if xlim is not None else (0, len(data_to_smth[0]))
    ylim = ylim if ylim is not None else (0, len(data_to_smth))
    xlo, xhi = xlim[0], xlim[1]
    ylo, yhi = ylim[0], ylim[1]

    # Pad the arrays by 4 times the maximum width (this is the radius of a gaussian 2D kernel) to avoid boundary issues
    #   when applying the smoothing.
    pad_width = int(np.ceil(4 * max_size))
    img_pad = np.pad(data_to_smth, pad_width, "reflect")
    weight_pad = np.pad(weights, pad_width, "reflect")
    var_pad = np.pad(variance, pad_width, "reflect")

    # Crop the padded arrays to only be applied within the provided x and y limits, accounting for the kernel required
    #   pad widths as well.
    cropped_im = img_pad[ylo : yhi + 2 * pad_width, xlo : xhi + 2 * pad_width].copy()
    cropped_weight = weight_pad[ylo : yhi + 2 * pad_width, xlo : xhi + 2 * pad_width].copy()
    cropped_var = var_pad[ylo : yhi + 2 * pad_width, xlo : xhi + 2 * pad_width].copy()

    # ===== Precompute the constant arrays =====
    # num_base is the basic numerator for calculating the SNR and what will be convolved by each of the kernels to get
    #   the final smoothed image. It is the product of the input image and the weights array.
    num_base = cropped_im * cropped_weight

    # den_base is the equivalent of num_base for the denominator when calculating SNR for a given kernel. It is
    #   convolved with the kernel^2 when calculating SNR and is the product of the weights array squared and the
    #   provided variance map.
    den_base = (cropped_weight**2) * cropped_var

    # Make a list of geometrically spaced kernel widths to use for smoothing. Sort in descending
    #   order if using 'asmooth' smoothing method, and ascending order if using 'csmooth'.
    if method == "asmooth":
        widths = np.geomspace(max_size, min_size, num_widths)
    else:
        widths = np.geomspace(min_size, max_size, num_widths)
        # We also need a boolean array for the csmooth-like approach to flag which pixels have already met the
        #   target SNR
        target_not_met = np.full_like(cropped_im, True)

    # ====== Caches ======
    # Stores the "best difference" between kernel SNR and target SNR at each pixel. In 'asmooth', this is the
    #   absolute difference between target SNR and the actual SNR provided by a given kernel. Used to check if
    #   currently tested kernel is an improvement on previous kernels when using the 'asmooth' method.
    best_diff = np.full(num_base.shape, np.inf)

    # Store the kernel sizes for each pixel that give the current "best difference" in SNR for that point in the loop.
    #   Used later to specify which numerator index to use when selecting final pixel value.
    kernel_sizes = np.full(num_base.shape, max_size)

    # Holding arrays for now, will be updated to produce the final image using values from the numerator cache above,
    #   and for the normalisation array to maintain consistent flux before and after smoothing.
    final_image = np.zeros_like(num_base)
    final_snrs = np.zeros_like(num_base)

    # Iterate over kernel widths
    for _wid, width in enumerate(tqdm.tqdm(widths)):
        # Create astropy Guassian kernel, but take only the actual values from the array.
        kernel = Gaussian2DKernel(width).array

        # Apply the astropy smoothing to the numerator base and the denominator base. Optionally uses fft (recommended)
        #   to speed up the smoothing and reduce memory use. For the denominator, we are getting the square of the
        #   final result needed from this convolution step, so it needs to be rooted after.
        #
        # Note that we also do not normalise the denominator kernel here and override the user-specified value of
        #   `norm_kernel`. Re-normalising the kernel would change the results of the SNR calculation and thus select
        #   the wrong kernel sizes to use for the final smoothing.
        if fft:
            num = convolve_fft(
                num_base,
                kernel,
                mask=mask,
                normalize_kernel=norm_kernel,
                boundary=boundary,
                fill_value=fill_value,
                nan_treatment=nan_treatment,
                preserve_nan=preserve_nan,
                normalization_zero_tol=normalization_zero_tol,
            )
            denom_square = convolve_fft(
                den_base,
                kernel**2,
                mask=mask,
                normalize_kernel=False,
                boundary=boundary,
                fill_value=fill_value,
                nan_treatment=nan_treatment,
                preserve_nan=preserve_nan,
                normalization_zero_tol=normalization_zero_tol,
            )
            norm = convolve_fft(
                cropped_weight,
                kernel,
                mask=mask,
                normalize_kernel=norm_kernel,
                boundary=boundary,
                fill_value=fill_value,
                nan_treatment=nan_treatment,
                preserve_nan=preserve_nan,
                normalization_zero_tol=normalization_zero_tol,
            )
        else:
            num = convolve(
                num_base,
                kernel,
                mask=mask,
                normalize_kernel=norm_kernel,
                boundary=boundary,
                fill_value=fill_value,
                nan_treatment=nan_treatment,
                preserve_nan=preserve_nan,
                normalization_zero_tol=normalization_zero_tol,
            )
            denom_square = convolve(
                den_base,
                kernel**2,
                mask=mask,
                normalize_kernel=False,
                boundary=boundary,
                fill_value=fill_value,
                nan_treatment=nan_treatment,
                preserve_nan=preserve_nan,
                normalization_zero_tol=normalization_zero_tol,
            )
            norm = convolve(
                cropped_weight,
                kernel,
                mask=mask,
                normalize_kernel=norm_kernel,
                boundary=boundary,
                fill_value=fill_value,
                nan_treatment=nan_treatment,
                preserve_nan=preserve_nan,
                normalization_zero_tol=normalization_zero_tol,
            )

        # It is possible some divide by zero/invalid value warnings will be raised during square rooting
        #  and dividing. The outcomes of these warnings are handled by the rest of the function, so we
        #  just use the numpy error context manager to suppress them.
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.sqrt(denom_square)

            # Calculate the SNR at each pixel for this kernel as the ratio of numerator and denominator
            snr = np.divide(num, denom)

        # Differentiate handling based on the smoothing method. First using csmooth-like smoothing (first kernel
        #  to meet or exceed the target snr)
        if method == "csmooth":
            if width != widths[-1]:
                # Get an array of new pixels that meet the target SNR with a finite result if it is not the last kernel
                pixels_meeting_target = (snr >= target_snr) & (target_not_met == True) & np.isfinite(snr)
            else:
                # Otherwise, use the largest kernel for all pixels that have not yet met the target SNR.
                pixels_meeting_target = target_not_met == True

            # Update the final image, and other holding arrays (e.g. kernel_sizes)
            final_image[pixels_meeting_target] = num[pixels_meeting_target] / norm[pixels_meeting_target]
            kernel_sizes[pixels_meeting_target] = width
            final_snrs[pixels_meeting_target] = snr[pixels_meeting_target]

            # Update the target_met array to mark which pixels have met their goal.
            target_not_met[pixels_meeting_target] = False

        # Now handle for the asmooth case (taking the kernel with width closest to the target SNR)
        else:
            # Find the absolute difference between the target SNR and the SNR at each pixel.
            diff = np.abs(snr - target_snr)

            # If pixel SNR is closer to target SNR, update the best_diff array to store that new diffence
            better = diff < best_diff
            best_diff = np.where(better, diff, best_diff)

            # Update final image and other holding arrays
            final_image[better] = num[better] / norm[better]
            kernel_sizes = np.where(better, width, kernel_sizes)
            final_snrs[better] = snr[better]

    # Reinsert smoothed data on final image
    img_pad[ylo : yhi + 2 * pad_width, xlo : xhi + 2 * pad_width] = final_image

    # Remove the padded borders
    sm_data = img_pad[int(pad_width) : int(-pad_width), int(pad_width) : int(-pad_width)]

    # Now we construct the new XGA product instance that houses the smoothed data
    #  In the case of an Image being passed in, we make an image to send back out
    smoothed_info = {
        "method": f"adaptive-{method}",
        "pars": {
            "kernel_widths": widths,
            "num_widths": num_widths,
            "min_size": min_size,
            "max_size": max_size,
        },
    }

    if type(prod) is Image:
        sm_prod = Image(
            {"data": sm_data, "wcs": prod.radec_wcs, "header": prod.header},
            prod.obs_id,
            prod.instrument,
            "",
            "",
            "",
            lo_en=prod.energy_bounds[0],
            hi_en=prod.energy_bounds[1],
            telescope=prod.telescope,
            check_exists=False,
            smoothed=True,
            smoothed_info=smoothed_info,
        )

    # User requested that the count-rate array of a RateMap be smoothed
    elif type(prod) is RateMap and not ratemap_smooth_im:
        raise NotImplementedError("XGA RateMaps can't yet be constructed from a single count-rate array.")

    # User requested that the Image of a RateMap be smoothed, and a new RateMap constructed using
    #  the original exposure map.
    elif type(prod) is RateMap and ratemap_smooth_im:
        # Construct the Image instance
        sm_im_prod = Image(
            {"data": sm_data, "wcs": prod.radec_wcs, "header": prod.header},
            prod.obs_id,
            prod.instrument,
            "",
            "",
            "",
            lo_en=prod.energy_bounds[0],
            hi_en=prod.energy_bounds[1],
            telescope=prod.telescope,
            check_exists=False,
            smoothed=True,
            smoothed_info=smoothed_info,
        )

        # Now we make a new RateMap instance using that smoothed image
        sm_prod = RateMap(sm_im_prod, prod.expmap)

    else:
        raise XGADeveloperError(
            "Only Image and RateMap instances are directly supported, contact "
            "the developers if you wish for us to support a sub-class of the Image class."
        )

    return sm_prod
