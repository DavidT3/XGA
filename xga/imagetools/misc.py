#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors


from typing import Tuple, List, Union

import numpy as np
from astropy.units import Quantity, pix, deg, UnitConversionError, UnitBase, Unit
from astropy.wcs import WCS

from ..products import Image, RateMap, ExpMap
from ..sourcetools import ang_to_rad, rad_to_ang
from ..utils import xmm_sky


def pix_deg_scale(coord: Quantity, input_wcs: WCS, small_offset: Quantity = Quantity(1, 'arcmin')) -> Quantity:
    """
    Very heavily inspired by the regions module version of this function, just tweaked to work better for
    my use case. Perturbs the given coordinates with the small_offset value, converts the changed ra-dec
    coordinates to pixel, then calculates the difference between the new and original coordinates in pixel.
    Then small_offset is converted to degrees and  divided by the pixel distance to calculate a pixel to degree
    factor.

    :param Quantity coord: The starting coordinates.
    :param WCS input_wcs: The world coordinate system used to calculate the pixel to degree scale
    :param Quantity small_offset: The amount you wish to perturb the original coordinates
    :return: Factor that can be used to convert pixel distances to degree distances, returned as an astropy
        quantity with units of deg/pix.
    :rtype: Quantity
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
    elif coord.unit == pix:
        deg_coord = Quantity(input_wcs.all_pix2world(*coord.value, 0), deg)
        pix_coord = coord
    else:
        raise UnitConversionError('{} is not a recognised position unit'.format(coord.unit.to_string()))

    perturbed_coord = deg_coord + Quantity([0, small_offset.to("deg").value], 'deg')
    perturbed_pix_coord = Quantity(input_wcs.all_world2pix(*perturbed_coord.value, 0), pix)

    diff = abs(perturbed_pix_coord - pix_coord)
    pix_dist = np.hypot(*diff)

    scale = small_offset.to('deg').value / pix_dist.value

    return Quantity(scale, 'deg/pix')


def sky_deg_scale(im_prod: Union[Image, RateMap, ExpMap], coord: Quantity,
                  small_offset: Quantity = Quantity(1, 'arcmin')) -> Quantity:
    """
    This is equivelant to pix_deg_scale, but instead calculates the conversion factor between
    XMM's XY sky coordinate system and degrees.

    :param Image/Ratemap/ExpMap im_prod: The image product to calculate the conversion factor for.
    :param Quantity coord: The starting coordinates.
    :param Quantity small_offset: The amount you wish to perturb the original coordinates
    :return: A scaling factor to convert sky distances to degree distances, returned as an astropy
        quantity with units of deg/xmm_sky.
    :rtype: Quantity
    """
    # Now really this function probably isn't necessary at, because there is a fixed scaling from degrees
    #  to this coordinate system, but I do like to be general

    if coord.shape != (2,):
        raise ValueError("coord input must only contain 1 pair.")
    elif not small_offset.unit.is_equivalent("deg"):
        raise UnitConversionError("small_offset must be convertible to degrees")

    # Seeing as we're taking an image product input on this one, I can leave the checking
    #  of inputs to coord_conv
    # We need the degree and xmm_sky original coordinates
    deg_coord = im_prod.coord_conv(coord, deg)
    sky_coord = im_prod.coord_conv(coord, xmm_sky)

    perturbed_coord = deg_coord + Quantity([0, small_offset.to("deg").value], 'deg')
    perturbed_sky_coord = im_prod.coord_conv(perturbed_coord, xmm_sky)

    diff = abs(perturbed_sky_coord - sky_coord)
    sky_dist = np.hypot(*diff)

    scale = small_offset.to('deg').value / sky_dist.value

    return Quantity(scale, deg/xmm_sky)


def pix_rad_to_physical(im_prod: Union[Image, RateMap, ExpMap], pix_rad: Quantity, out_unit: Union[UnitBase, str],
                        coord: Quantity, z: Union[float, int] = None, cosmo=None) -> Quantity:
    """
    Pure convenience function to convert a list of pixel radii to whatever unit we might want at the end. Used
    quite a lot in the imagetools.profile functions, which is why it was split off into its own function. Redshift
    and cosmology must be supplied if proper distance units (like kpc) are chosen for out_unit.

    :param Image/RateMap/ExpMap im_prod: The image/ratemap product for which the conversion is taking place.
    :param Quantity pix_rad: The array of pixel radii to convert to out_unit.
    :param UnitBase/str out_unit: The desired output unit for the radii, either an astropy unit object or a name string.
    :param Quantity coord: The position of the object being analysed.
    :param float/int z: The redshift of the object (only required for proper distance units like kpc).
    :param cosmo: The chosen cosmology for the analysis (only required for proper distance units like kpc).
    :return: An astropy Quantity with the radii in units of out_unit.
    :rtype: Quantity
    """
    if pix_rad.unit != pix:
        raise UnitConversionError("pix_rads must be in units of pixels")

    # See what type of unit input was given
    if isinstance(out_unit, str):
        out_unit = Unit(out_unit)

    deg_rads = Quantity(pix_deg_scale(coord, im_prod.radec_wcs).value * pix_rad.value, 'deg')

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

    :param Image/RateMap/ExpMap im_prod:
    :param Quantity physical_rad: The physical radius to be converted to pixels.
    :param Quantity coord: The position of the object being analysed.
    :param float/int z: The redshift of the object (only required for input proper distance units like kpc).
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

    phys_to_pix = 1 / pix_deg_scale(coord, im_prod.radec_wcs).value
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
    if isinstance(im_prod, Image) and im_prod.data.sum() != 0:
        # For the XGA Image products
        # This just finds out where the zeros in the data are
        locations = np.where(im_prod.data != 0)
    elif isinstance(im_prod, np.ndarray) and im_prod.sum() != 0:
        # For numpy arrays
        locations = np.where(im_prod != 0)
    else:
        raise ValueError("Supplied data only contains zeros, data limits cannot be found in this case.")

    # Finds the maximum and minimum locations of zeros in both x and y spaces - these are the boundary coordinates
    # Adds and subtracts 1 to give a very small border.
    x_min = locations[1].min() - 1
    x_max = locations[1].max() + 1

    y_min = locations[0].min() - 1
    y_max = locations[0].max() + 1

    # Returns the boundary coordinates in pairs of min, max.
    return [x_min, x_max], [y_min, y_max]


def edge_finder(data: Union[RateMap, ExpMap, np.ndarray], keep_corners: bool = True,
                border: bool = False) -> np.ndarray:
    """
    A simple edge finding algorithm designed to locate 'edges' in binary data, or in special cases produce a detector
    map of an instrument using an exposure map. The algorithm takes the difference of one column from the next, over
    the entire array, then does the same with rows. Different difference values indicate where edges are in the array,
    and when added together all edges should be located.

    Depending on how the 'border' option is set, the returned array will either represent the exact edge, or
    a boundary that is 1 pixel outside the actual edge.

    :param RateMap/ExpMap/ndarray data: The 2D array or exposure map to run edge detection on. If an array is
        passed it must only consist of 0s and 1s.
    :param bool keep_corners: Should corner information be kept in the output array. If True then 2s in the
        output will indicate vertices.
    :param bool border: If True, then the returned array will represent a border running around the boundary of the
        true edge, rather than the outer boundary of the edge itself.
    :return: An array of 0s and 1s. 1s indicate a detected edge.
    :rtype: np.ndarray
    """
    if not isinstance(data, (RateMap, ExpMap, np.ndarray)) or (isinstance(data, np.ndarray)
                                                               and np.where((data != 1) & (data != 0))[0].any()):
        raise TypeError("This simple edge-finding algorithm only works on exposure maps (whether passed directly or "
                        "accessed from a RateMap), or arrays of ones and zeros.")
    elif isinstance(data, ExpMap):
        dat_map = data.data.copy()
        # Turn the exposure map into something simpler, either on a detector or not
        dat_map[dat_map != 0] = 1
    elif isinstance(data, RateMap):
        dat_map = data.expmap.data.copy()
        # Turn the exposure map from the ratemap into something simpler, either on a detector or not
        dat_map[dat_map != 0] = 1
    elif isinstance(data, np.ndarray):
        dat_map = data.copy()

    # Do the diff from top to bottom of the image, the append option adds a line of zeros at the end
    #  otherwise the resulting array would only be N-1 elements 'high'.
    hori_edges = np.diff(dat_map, axis=0, append=0)

    if not border:
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
    else:
        need_corr_y, need_corr_x = np.where(hori_edges == -1)
        # So that's why we add one to those y coordinates (as this is the vertical pass of np.diff
        new_y = need_corr_y + 1
        # Then make sure chip edge = 1, and everything else = 0
        hori_edges[need_corr_y, need_corr_x] = 0
        hori_edges[new_y, need_corr_x] = 1

    # The same process is repeated here, but in the x direction, so you're finding vertical edges
    vert_edges = np.diff(dat_map, axis=1, append=0)

    if not border:
        need_corr_y, need_corr_x = np.where(vert_edges == 1)
        new_x = need_corr_x + 1
        vert_edges[need_corr_y, need_corr_x] = 0
        vert_edges[need_corr_y, new_x] = 1
        vert_edges[vert_edges == -1] = 1
    else:
        need_corr_y, need_corr_x = np.where(vert_edges == -1)
        new_x = need_corr_x + 1
        vert_edges[need_corr_y, need_corr_x] = 0
        vert_edges[need_corr_y, new_x] = 1

    # Both passes are combined into one, with possible values of 0 (no edge), 1 (edge detected in one pass),
    #  and 2 (edge detected in both pass).
    comb = hori_edges + vert_edges

    # If we don't want to keep information on intersections between two edges (corner values of 2), then we just
    #  set them to one
    if not keep_corners:
        comb[np.where((comb != 1) & (comb != 0))] = 1

    return comb

