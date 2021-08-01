#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 01/08/2021, 22:56. Copyright (c) David J Turner

from typing import Union

import numpy as np
from astropy.convolution import Kernel, convolve, convolve_fft
from astropy.units import Quantity

from ..products import Image, RateMap, ExpMap


def general_smooth(prod: Union[Image, RateMap], kernel: Kernel, mask: np.ndarray = None, fft: bool = False,
                   norm_kernel: bool = True, sm_im: bool = True) -> Union[Image, RateMap]:
    """
    Simple function to apply (in theory) any Astropy smoothing to an XGA Image/RateMap  and create a new smoothed
    XGA data product. This general function will produce XGA Image and RateMap
    objects from any instance of an Astropy Kernel, and if a RateMap is passed as the input then you may choose
    whether to smooth the image component or the image/expmap (using sm_im); if you choose the former then the final
    smoothed RateMap will be produced by dividing the smoothed Image by the original ExpMap.

    :param Image/RateMap prod: The image/ratemap to be smoothed. If a RateMap is passed please see the 'sm_im'
        parameter for extra options.
    :param Kernel kernel: The kernel with which to smooth the input data. Should be an instance of an Astropy Kernel.
    :param np.ndarray mask: A mask to apply to the data while smoothing (removing point source interlopers for
        instance). The default is None, which means no mask is applied.
    :param bool fft: Should a fast fourier transform method be used for convolution, default is False.
    :param bool norm_kernel: Whether to normalize the kernel to have a sum of one.
    :param bool sm_im: If a RateMap is passed, should the image component be smoothed rather than the actual
        RateMap. Default is True, where the Image will be smoothed and divided by the original ExpMap. If set
        to False, the resulting RateMap will be bodged, with the ExpMap all 1s on the sensor.
    :return: An XGA product with the smoothed Image or RateMap.
    :rtype: Image/RateMap
    """

    raise NotImplementedError("Realised part-way through that the way XGA handles images, Ratemaps, and expmaps needs "
                              "to be updated for this to work.")

    # First off we check the type of the product that has been passed in for smoothing
    if not isinstance(prod, (Image, Quantity)) or type(prod) != ExpMap:
        raise TypeError("This function can only smooth data if input in the form of an XGA Image/RateMap, or a numpy"
                        "array.")

    # Now to get into it properly, if its an XGA product then we need to retrieve the data as an array. Only check
    #  whether its an instance of Image as that is the superclass for RateMap as well.
    if type(prod) == Image:
        data = prod.data.copy()
    elif type(prod) == RateMap and not sm_im:
        raise NotImplementedError("I haven't yet made sure that the rest of XGA will like this.")
        data = prod.data.copy()
    elif type(prod) == RateMap and sm_im:
        data = prod.image.data

    # Now we see which type of convolution the user has requested - entirely down to their discretion
    if not fft:
        sm_data = convolve(data, kernel, normalize_kernel=norm_kernel, mask=mask)
    else:
        sm_data = convolve_fft(data, kernel, normalize_kernel=norm_kernel, mask=mask)

    # Now that Astropy has done the heavy lifting part of the smoothing, we assemble the output XGA product
    if type(prod) == Image:
        data = prod.data.copy()
    elif type(prod) == RateMap and not sm_im:
        raise NotImplementedError("I haven't yet made sure that the rest of XGA will like this.")
    elif type(prod) == RateMap and sm_im:
        pass










