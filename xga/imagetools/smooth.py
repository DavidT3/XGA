#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/22/26, 5:59 PM. Copyright (c) The Contributors.

from typing import Union, Optional

import numpy as np
from astropy.convolution import Kernel, convolve, convolve_fft

from xga import OUTPUT
from xga.exceptions import ProductGenerationError, XGADeveloperError
from xga.products import Image, RateMap, ExpMap


def general_smooth(prod: Union[Image, RateMap], kernel: Kernel, mask: Optional[np.ndarray] = None, fft: bool = False,
                   norm_kernel: bool = True, sm_im: bool = True, force_resmooth: bool = False) -> Union[Image, RateMap]:
    """
    Simple function to apply (in theory) any Astropy smoothing to an XGA Image/RateMap and create a new smoothed
    XGA data product. This general function will produce XGA Image and RateMap
    objects from any instance of an Astropy Kernel, and if a RateMap is passed as the input then you may choose
    whether to smooth the image component or the image/expmap (using sm_im); if you choose the former then the final
    smoothed RateMap will be produced by dividing the smoothed Image by the original ExpMap.

    :param Image/RateMap prod: The XGA Image/RateMap to be smoothed. If you pass a RateMap, please see the 'sm_im'
        argument for extra options.
    :param Kernel kernel: The kernel with which to smooth the input data. Should be an instance of an Astropy Kernel.
    :param np.ndarray mask: A mask to apply to the data while smoothing (removing point source contaminants, for
        instance). The default is None, which means no mask is applied. This function expects a mask with 1s where
        the data you wish to keep is, and 0s where the data you wish to remove is - the style of mask produced by XGA.
    :param bool fft: Should a fast fourier transform method be used for convolution. The default is False.
    :param bool norm_kernel: Whether to normalize the kernel to have a sum of one.
    :param bool sm_im: If a RateMap is passed, should the image component be smoothed rather than the actual
        RateMap. Default is True, where the Image will be smoothed and divided by the original ExpMap. If set
        to False, the resulting RateMap will be bodged, with the ExpMap all 1s on the sensor.
    :param bool force_resmooth: Force a second smoothing convolution on an already-smoothed Image/RateMap.
        Default is False, in which case an error will be raised if the `prod` input is already smoothed.
    :return: An XGA product with the smoothed Image or RateMap.
    :rtype: Image/RateMap
    """
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
        mask[mask == 0] = -1
        mask[mask == 1] = 0
        mask[mask == -1] = 1

    # By default, we raise an error if the input product has already been smoothed, but
    #  we do also include an argument that allows the user to override the
    #  behavior and smooth again.
    if prod.smoothed and not force_resmooth:
        raise ProductGenerationError("Input XGA Image or RateMap has already been smoothed, and "
                                     "will not be smoothed again. To override this check you may pass "
                                     "`force_resmooth=True`.")

    # We implemented the capability to parse an Astropy smoothing kernel into the information
    #  we want to add to XGA Image class properties into a static method of the Image class.
    # So we can just call the parse_smoothing method and pull the name and parameters of the
    # kernel out
    smooth_name, smooth_pars = Image.parse_smoothing(kernel)
    smooth_pars_str = "_".join([str(k) + str(v) for k, v in smooth_pars.items()])

    # Now we figure out what exactly needs to be smoothed.
    # If the input product is an Image, then it is very straightforward - we just copy
    #  the data array, and will apply smoothing to that
    if type(prod) == Image:
        data_to_smth = prod.data.copy()

    # In this case the user has passed a RateMap for smoothing and also requested that we
    #  directly smooth the count-rate array (as opposed to extracting the image array, smoothing
    #  that, then re-dividing by the exposure map).
    elif type(prod) == RateMap and not sm_im:
        raise NotImplementedError("XGA RateMaps can currently only be constructed from separate "
                                  "Image and ExpMap instances, and as such a new RateMap cannot"
                                  "yet be created from a smoothed count-rate array.")
        data_to_smth = prod.data.copy()

    # In this instance the user has passed a RateMap, but specified that we should smooth
    #  the IMAGE data, then create a new RateMap by dividing the smoothed image by the
    #  original exposure map.
    elif type(prod) == RateMap and sm_im:
        data_to_smth = prod.image.data.copy()

    # Catch all else statement, meant to raise a vaguely useful error if the user
    #  has passed some sub-class of Image that we don't directly support here.
    else:
        raise XGADeveloperError("Only Image and RateMap instances are directly supported, contact "
                                "the developers if you wish for us to support a sub-class of the Image class.")


    # We now apply the Astropy smoothing kernel to the data, using FFT or non-FFT methods depending
    #  on what the user specified in their call to this function
    if fft:
        sm_data = convolve_fft(data_to_smth, kernel, normalize_kernel=norm_kernel, mask=mask)
    else:
        sm_data = convolve(data_to_smth, kernel, normalize_kernel=norm_kernel, mask=mask)

    # Now we construct the new XGA product instance that houses the smoothed data
    #  In the case of an Image being passed in, we make an image to send back out
    if type(prod) == Image:
        sm_prod = Image({'data': sm_data, 'wcs': prod.radec_wcs, 'header': prod.header}, prod.obs_id,
                        prod.instrument, "", "", "", lo_en=prod.energy_bounds[0],
                        hi_en=prod.energy_bounds[1], telescope=prod.telescope, check_exists=False,
                        smoothed=True, smoothed_info=kernel)

    # User requested that the count-rate array of a RateMap be smoothed
    elif type(prod) == RateMap and not sm_im:
        raise NotImplementedError("XGA RateMaps can't yet be constructed from a single count-rate array.")

    # User requested that the Image of a RateMap be smoothed, and a new RateMap constructed using
    #  the original exposure map.
    elif type(prod) == RateMap and sm_im:
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
