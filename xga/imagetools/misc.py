#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 04/01/2021, 21:18. Copyright (c) David J Turner


from typing import Tuple, List, Union

import numpy as np
from astropy.units import Quantity, pix, deg, UnitConversionError, UnitBase
from astropy.wcs import WCS

from ..products import Image, RateMap, ExpMap
from ..sourcetools import ang_to_rad, rad_to_ang


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
        raise UnitConversionError("small_offset must be convertible to degrees")

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


def pix_rad_to_physical(im_prod: Union[Image, RateMap, ExpMap], pix_rad: Quantity, out_unit: UnitBase,
                        coord: Quantity, z: Union[float, int] = None, cosmo=None) -> Quantity:
    """
    Pure convenience function to convert a list of pixel radii to whatever unit we might want at the end. Used
    quite a lot in the imagetools.profile functions, which is why it was split off into its own function. Redshift
    and cosmology must be supplied if proper distance units (like kpc) are chosen for out_unit.

    :param Union[Image, RateMap] im_prod: The image/ratemap product for which the conversion is taking place.
    :param Quantity pix_rad: The array of pixel radii to convert to out_unit.
    :param UnitBase out_unit: The desired output unit for the radii.
    :param Quantity coord: The position of the object being analysed.
    :param Union[float, int] z: The redshift of the object (only required for proper distance units like kpc).
    :param cosmo: The chosen cosmology for the analysis (only required for proper distance units like kpc).
    :return: An astropy Quantity with the radii in units of out_unit.
    :rtype: Quantity
    """
    if pix_rad.unit != pix:
        raise UnitConversionError("pix_rads must be in units of pixels")

    deg_rads = Quantity(pix_deg_scale(coord, im_prod.radec_wcs) * pix_rad.value, 'deg')

    if out_unit.is_equivalent("kpc") and z is not None and cosmo is not None:
        # Quick convert to kpc with my handy function and then go to whatever unit the user requested
        # Wham bam pixels to proper distance
        conv_rads = ang_to_rad(deg_rads, z, cosmo).to(out_unit)
    elif out_unit.is_equivalent("kpc") and (z is None or cosmo is None):
        raise ValueError("If you wish to convert to physical units such as kpc, you must supply "
                         "a redshift and cosmology")
    elif out_unit.is_equivalent("deg"):
        conv_rads = deg_rads.to(out_unit)
    elif out_unit == pix:
        # Commenting out the sassy warning for now...
        # warn("You're converting pixel radii to pixels...")
        conv_rads = pix_rad
    else:
        conv_rads = None
        raise UnitConversionError("cen_rad_units doesn't appear to be a distance or angular unit.")

    return conv_rads


def physical_rad_to_pix(im_prod: Union[Image, RateMap, ExpMap], physical_rad: Quantity,
                        coord: Quantity, z: Union[float, int] = None, cosmo=None) -> Quantity:
    """
    Another convenience function, this time to convert physical radii to pixels. It can deal with both angular and
    proper radii, so long as redshift and cosmology information is provided for the conversion from proper radii
    to pixels.

    :param Union[Image, RateMap, ExpMap] im_prod:
    :param Quantity physical_rad: The physical radius to be converted to pixels.
    :param Quantity coord: The position of the object being analysed.
    :param Union[float, int] z: The redshift of the object (only required for input proper distance units like kpc).
    :param cosmo: The chosen cosmology for the analysis (only required for input proper distance units like kpc).
    :return: The converted radii, in an astropy Quantity with pix units.
    :rtype: Quantity
    """

    if physical_rad.unit.is_equivalent("kpc") and z is not None and cosmo is not None:
        conv_rads = rad_to_ang(physical_rad, z, cosmo).to('deg')
    elif physical_rad.unit.is_equivalent("kpc") and (z is None or cosmo is None):
        raise ValueError("If you wish to convert to convert from proper distance units such as kpc, you must supply "
                         "a redshift and cosmology")
    elif physical_rad.unit.is_equivalent("deg"):
        conv_rads = physical_rad.to('deg')
    elif physical_rad.unit == pix:
        raise UnitConversionError("You are trying to convert from pixel units to pixel units.")
    else:
        conv_rads = None
        raise UnitConversionError("cen_rad_units doesn't appear to be a distance or angular unit.")

    phys_to_pix = 1 / pix_deg_scale(coord, im_prod.radec_wcs)
    conv_rads = Quantity(conv_rads.value * phys_to_pix, 'pix')

    return conv_rads


def data_limits(im_prod: Union[Image, RateMap, ExpMap, np.ndarray]) -> Tuple[List[int], List[int]]:
    """
    A function that finds the pixel coordinates that bound where data is present in
    Image or RateMap object.

    :param Image/RateMap/ndarray im_prod: An Image, RateMap, or numpy array that you wish to find
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

