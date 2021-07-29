#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 29/07/2021, 11:12. Copyright (c) David J Turner

from typing import Union
from warnings import warn

import numpy as np
from astropy.units import Quantity

from ..products import Image, RateMap

CONT_BIN_METRICS = ['counts', 'snr']


def contour_bin_masks(prod: Union[Image, RateMap], src_mask: np.ndarray = None, bck_mask: np.ndarray = None,
                      metric: str = 'counts', max_val: Quantity = Quantity(1000, 'ct')):
    """
    This method implements different spins on Jeremy Sanders' contour binning
    method (https://doi.org/10.1111/j.1365-2966.2006.10716.x) to split the 2D ratemap into bins that
    are spatially and morphologically connected (in theory). This can make some nice images, and also allows
    you to use those new regions to measure projected spectral quantities (temperature, metallicity, density)
    and make a 2D projected property map.

    Current allowable metric choices:
      * 'counts' - Stop adding to bin when total background subtracted counts are over max_val
      * 'snr' - Stop adding to bin when signal to noise is over max_val.

    :param Image/RateMap prod: The image or ratemap to apply the contour binning process to.
    :param np.ndarray src_mask: A mask that removes emission from regions not associated with the source you're
        analysing, including removing interloper sources. Default is None, in which case no mask will be applied.
    :param np.ndarray bck_mask: A mask defining the background region. Default is None in which case no background
        subtraction will be used.
    :param str metric: The metric by which to judge when to stop adding new pixels to a bin (see docstring).
    :param Quantity max_val: The max value for the chosen metric, above which a new bin is started.
    :return: A 3D array of bin masks, the first two dimensions are
    :rtype:
    """
    if metric not in CONT_BIN_METRICS:
        cont_av = ", ".join(CONT_BIN_METRICS)
        raise ValueError("{m} is not a recognised contour binning metric, please use one of the "
                         "following; {a}".format(m=metric, a=cont_av))

    if src_mask is None and bck_mask is None:
        warn("You have not passed a src or bck mask, the whole image will be binned and no background subtraction "
             "will be applied")

    if not isinstance(prod, (Image, RateMap)):
        raise TypeError("Only XGA Image and RateMap products can be binned with this function.")
    elif isinstance(prod, Image):
        raise NotImplementedError("The method for images has not been implemented yet")
    else:
        pass



