#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 01/09/2020, 16:14. Copyright (c) David J Turner


from typing import Tuple, List, Union

import numpy as np
from astropy.units import Quantity, pix, deg, UnitConversionError
from astropy.wcs import WCS

from xga.products import Image, RateMap


def pix_deg_scale(coord: Quantity, input_wcs: WCS, small_offset: Quantity = Quantity(1, 'arcmin')) -> float:
    """
    Very heavily inspired by the regions version of this function, just tweaked to work better for
    my use case. Perturbs the given coordinates with the small_offset value, converts the changed ra-dec
    coordinates to pixel, then calculates the difference between the new and original coordinates in pixel.
    Then small_offset is converted to degrees and  divided by the pixel distance to calculate a pixel to degree
    factor.
    :param Quantity coord: The starting coordinates.
    :param WCS input_wcs: The to calculate the pixel to degree scale
    :param Quantity small_offset: The amount you wish to peturb the original coordinates
    :return: Factor that can be used to convert pixel distances to degree distances.
    :rtype: float
    """
    if coord.unit != pix and coord.unit != deg:
        raise UnitConversionError("This function can only be used with radec or pixel coordinates as input")
    elif coord.shape != (2,):
        raise ValueError("coord input must only contain 1 pair.")
    elif not small_offset.unit.is_equivalent("deg"):
        raise UnitConversionError("small_offset must be convertable to degrees")

    if coord.unit == deg:
        pix_coord = Quantity(input_wcs.all_world2pix(*coord.value, 0), pix)
        deg_coord = coord
    else:
        deg_coord = Quantity(input_wcs.all_pix2world(*coord.value, 0), deg)
        pix_coord = coord

    perturbed_coord = deg_coord + Quantity([0, small_offset.to("deg").value], 'deg')
    perturbed_pix_coord = Quantity(input_wcs.all_world2pix(*perturbed_coord.value, 0), pix)

    diff = abs(perturbed_pix_coord - pix_coord)
    pix_dist = np.hypot(*diff)

    scale = small_offset.to('deg').value / pix_dist.value

    return scale


def data_limits(im_prod: Union[Image, RateMap, np.ndarray]) -> Tuple[List[int], List[int]]:
    """
    A function that finds the pixel coordinates that bound where data is present in
    Image or RateMap object.
    :param Union[Image, RateMap, ndarray] im_prod: An Image, RateMap, or numpy array that you wish to find
    boundary coordinates for.
    :return: Two lists, the first with the x lower and upper bounding coordinates, and the second with
    the y lower and upper bounding coordinates.
    :rtype: Tuple[List[int, int], List[int, int]]
    """
    if isinstance(im_prod, Image):
        # For the XGA Image products
        # This just finds out where the zeros in the data are
        locations = np.where(im_prod.data != 0)
    else:
        # For numpy arrays
        locations = np.where(im_prod != 0)

    # Finds the maximum and minimum locations of zeros in both x and y spaces - these are the boundary coordinates
    # Adds and subtracts 1 to give a very small border.
    x_min = locations[1].min() - 1
    x_max = locations[1].max() + 1

    y_min = locations[0].min() - 1
    y_max = locations[0].max() + 1

    # Returns the boundary coordinates in pairs of min, max.
    return [x_min, x_max], [y_min, y_max]

