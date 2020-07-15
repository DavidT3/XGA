#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 15/07/2020, 10:42. Copyright (c) David J Turner


from typing import Tuple

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity, UnitBase, pix, deg, arcsec, UnitConversionError

from xga.products import Image
from xga.sourcetools import ang_to_rad, rad_to_ang
from .misc import pix_deg_scale


def annular_mask(centre: Quantity, inn_rad: np.ndarray, out_rad: np.ndarray, shape: tuple,
                 start_ang: Quantity = Quantity(0, 'deg'),
                 stop_ang: Quantity = Quantity(360, 'deg')) -> np.ndarray:
    """
    A handy little function to generate annular (or circular) masks in the form of numpy arrays.
    It produces the src_mask for a given shape of image, centered at supplied coordinates, and with inner and
    outer radii supplied by the user also. Angular limits can also be supplied to give the src_mask an annular
    dependence. This function should be properly vectorised, and accepts inner and outer radii in
    the form of arrays.
    The result will be an len_y, len_x, N dimensional array, where N is equal to the length of inn_rad.
    :param Quantity centre: Astropy pix quantity of the form Quantity([x, y], pix).
    :param np.ndarray inn_rad: Pixel radius for the inner part of the annular src_mask.
    :param np.ndarray out_rad: Pixel radius for the outer part of the annular src_mask.
    :param Quantity start_ang: Lower angular limit for the src_mask.
    :param Quantity stop_ang: Upper angular limit for the src_mask.
    :param tuple shape: The output from the shape property of the numpy array you are generating masks for.
    :return: The generated src_mask array.
    :rtype: np.ndarray
    """
    # Split out the centre coordinates
    cen_x = centre[0].value
    cen_y = centre[1].value

    # Making use of the astropy units module, check that we are being pass actual angle values
    if start_ang.unit not in ['deg', 'rad']:
        raise ValueError("start_angle unit type {} is not an accepted angle unit, "
                         "please use deg or rad.".format(start_ang.unit))
    elif stop_ang.unit not in ['deg', 'rad']:
        raise ValueError("stop_angle unit type {} is not an accepted angle unit, "
                         "please use deg or rad.".format(stop_ang.unit))
    # Enforcing some common sense rules on the angles
    elif start_ang >= stop_ang:
        raise ValueError("start_ang cannot be greater than or equal to stop_ang.")
    elif start_ang > Quantity(360, 'deg') or stop_ang > Quantity(360, 'deg'):
        raise ValueError("start_ang and stop_ang cannot be greater than 360 degrees.")
    elif stop_ang < Quantity(0, 'deg'):
        raise ValueError("stop_ang cannot be less than 0 degrees.")
    else:
        # Don't want to pass astropy objects to numpy functions, but do need the angles in radians
        start_ang = start_ang.to('rad').value
        stop_ang = stop_ang.to('rad').value

    # Check that if the inner and outer radii are arrays, they're the same length
    if isinstance(inn_rad, (np.ndarray, list)) and len(inn_rad) != len(out_rad):
        raise ValueError("inn_rad and out_rad are not the same length")
    elif isinstance(inn_rad, list) and len(inn_rad) == len(out_rad):
        # If it is a list, just quickly transform to a numpy array
        inn_rad = np.array(inn_rad)
        out_rad = np.array(out_rad)

    # This sets up the cartesian coordinate grid of x and y values
    arr_y, arr_x = np.ogrid[:shape[0], :shape[1]]

    # Go to polar coordinates
    rec_x = arr_x - cen_x
    rec_y = arr_y - cen_y
    # Leave this as r**2 to avoid square rooting and involving floats
    init_r_squared = rec_x**2 + rec_y**2

    # arctan2 does just perform arctan on two values, but then uses the signs of those values to
    # decide the quadrant of the output
    init_arr_theta = (np.arctan2(rec_x, rec_y) - start_ang) % (2*np.pi)  # Normalising to 0-2pi range

    # If the radius limits are an array, the arrays that describe the space we have constructed are copied
    #  into a third dimension - This allows masks for different radii to be generated in a vectorised fashion
    if isinstance(inn_rad, np.ndarray):
        arr_r_squared = np.repeat(init_r_squared[:, :, np.newaxis], len(inn_rad), axis=2)
        arr_theta = np.repeat(init_arr_theta[:, :, np.newaxis], len(inn_rad), axis=2)
    else:
        arr_r_squared = init_r_squared
        arr_theta = init_arr_theta

    # This will deal properly with inn_rad and out_rads that are arrays
    if np.greater(inn_rad, out_rad).any():
        raise ValueError("inn_rad value cannot be greater than out_rad")
    else:
        rad_mask = (arr_r_squared < out_rad ** 2) & (arr_r_squared >= inn_rad ** 2)

    # Finally, puts a cut on the allowed angle, and combined the radius and angular cuts into the final src_mask
    ang_mask = arr_theta <= (stop_ang - start_ang)
    ann_mask = rad_mask * ang_mask

    # Should ensure that the central pixel will be 0 for annular masks that are bounded by zero.
    #  Sometimes they aren't because of custom angle choices
    if 0 in inn_rad:
        where_zeros = np.where(inn_rad == 0)[0]
        ann_mask[cen_y, cen_x, where_zeros] = 1
    # Returns the annular src_mask(s), in the form of a len_y, len_x, N dimension np array
    return ann_mask


def ann_radii(im_prod: Image, centre: Quantity, rad: Quantity, z: float = None, cen_rad_units: UnitBase = arcsec,
              cosmo=Planck15) -> Tuple[np.ndarray, np.ndarray, Quantity]:
    """
    Will probably only ever be called by an internal brightness calculation, but two different methods
    need it so it gets its own method.
    :param Image im_prod: An Image or RateMap product object that you wish to calculate annuli for.
    :param Quantity centre: The coordinates of the centre of the set of annuli.
    :param Quantity rad: The outer radius of the set of annuli.
    :param float z: The redshift of the source of interest, required if the output radius units are
    a proper radius.
    :param UnitBase cen_rad_units: The output units for the centres of the annulli returned by
    this function. The inner and outer radii will always be in pixels.
    :param cosmo: An instance of an astropy cosmology, the default is Planck15.
    :return: Returns the inner and outer radii of the annuli (in pixels), and the centres of the annuli
    in cen_rad_units.
    :rtype: Tuple[np.ndarray, np.ndarray, Quantity]
    """
    # Quickly get the central coordinates in degrees as well
    deg_cen = im_prod.coord_conv(centre, deg)
    pix_cen = im_prod.coord_conv(centre, pix)

    # If the radius is passed in a proper distance unit, this will convert it or raise an exception
    #  if it can't
    if rad.unit.is_equivalent('kpc') and z is None:
        raise UnitConversionError("Cannot use a radius in kpc without redshift information")
    elif rad.unit.is_equivalent('kpc') and z is not None:
        rad = rad_to_ang(rad, z, cosmo)

    # If the radius was passed as an angle, or has just been converted to one from a proper distance unit
    if rad.unit.is_equivalent('deg'):
        to_add = Quantity([rad.to('deg').value, 0], 'deg')
        rad_coord = im_prod.coord_conv(deg_cen+to_add, pix)
        rad = Quantity(abs((rad_coord - pix_cen).value[0]), 'pix')

    rad = np.ceil(rad)

    # By this point, the rad should be in pixels
    rads = np.arange(0, rad.value + 1).astype(int)
    inn_rads = rads[:len(rads) - 1]
    out_rads = rads[1:len(rads)]

    deg_rads = Quantity(pix_deg_scale(deg_cen, im_prod.radec_wcs) * out_rads, 'deg')

    if cen_rad_units.is_equivalent("kpc"):
        # Quick convert to kpc with my handy function and add a zero at the beginning
        kpc_rads = ang_to_rad(deg_rads, z, cosmo).insert(0, Quantity(0, "kpc"))
        # Wham-bam now have the centres of the bins in kilo-parsecs
        cen_rads = (kpc_rads[1:].to(cen_rad_units) + kpc_rads[:-1].to(cen_rad_units)) / 2
    elif cen_rad_units.is_equivalent("deg"):
        deg_rads.insert(0, Quantity(0, cen_rad_units))
        cen_rads = (deg_rads[1:].to(cen_rad_units) +
                    deg_rads[:-1].to(cen_rad_units)) / 2
    else:
        cen_rads = None
        raise UnitConversionError("cen_rad_units doesn't appear to be a distance or angular unit.")
    return inn_rads, out_rads, cen_rads


def radial_brightness(im_prod: Image, src_mask: np.ndarray, back_mask: np.ndarray,
                      centre: Quantity, rad: Quantity, z: float = None, cen_rad_units: UnitBase = arcsec,
                      cosmo=Planck15) -> Tuple[np.ndarray, Quantity, np.float64]:
    """
    A simple method to calculate the average brightness in circular annuli upto the radius of
    the chosen region. The annuli are one pixel in width, and as this uses the masks that were generated
    earlier, interloper sources should be removed.
    :param Image im_prod: An Image or RateMap object that you wish to construct a brightness profile from.
    :param np.ndarray src_mask: A numpy array that masks out everything but the source, including interlopers.
    :param np.ndarray back_mask: A numpy array that masks out everything but the background, including interlopers.
    :param Quantity centre: The coordinates for the centre of the brightness profile.
    :param Quantity rad: The outer radius of the brightness profile (THIS SHOULD BE THE SAME RADIUS AS THE REGION
    YOUR SRC_MASK IS BASED ON, OTHERWISE YOU'LL GET AN INVALID BACKGROUND MEASUREMENT).
    :param float z: The redshift of the source of interest.
    :param BaseUnit cen_rad_units: The desired output units for the central radii of the annulli.
    :param cosmo: An astropy cosmology object for source coordinate conversions.
    :return: The brightness is returned in a flat numpy array, then the radii at the centre of the bins are
    returned in units of kpc, and finally the average brightness in the background region is returned.
    :rtype: Tuple[np.ndarray, Quantity, np.float64]
    """
    if im_prod.shape != src_mask.shape:
        raise ValueError("The shape of the src_mask array ({0}) must be the same as that of im_prod "
                         "({1}).".format(src_mask.shape, im_prod.shape))

    # Just making sure we have the centre in pixel coordinates
    pix_cen = im_prod.coord_conv(centre, pix)

    # This sets up the annular bin radii, as well as finding the central radii of the bins in the chosen units.
    inn_rads, out_rads, cen_rads = ann_radii(im_prod, centre, rad, z, cen_rad_units, cosmo)

    # Using the ellipse adds enough : to get all the dimensions in the array, then the None adds an empty
    #  dimension. Helps with broadcasting the annular masks with the region src_mask that gets rid of interlopers
    if im_prod.type == 'image':
        masks = annular_mask(pix_cen, inn_rads, out_rads, im_prod.shape) * src_mask[..., None]
    elif im_prod.type == 'ratemap':
        masks = annular_mask(pix_cen, inn_rads, out_rads, im_prod.shape) * src_mask[..., None] \
                * im_prod.sensor_mask[..., None]

    # Creates a 3D array of the masked data
    masked_data = masks * im_prod.data[..., None]
    # Calculates the average for each radius, use the masks array as weights to only include unmasked
    #  areas in the average for each radius.
    br = np.average(masked_data, axis=(0, 1), weights=masks)

    # Finds the average of the background region
    bg = np.average(im_prod.data * back_mask, axis=(0, 1), weights=back_mask)

    return br, cen_rads, bg


def pizza_brightness(im_prod: Image, src_mask: np.ndarray, back_mask: np.ndarray,
                     centre: Quantity, rad: Quantity, num_slices: int = 4,
                     z: float = None, cen_rad_units: UnitBase = arcsec,
                     cosmo=Planck15) -> Tuple[np.ndarray, Quantity, Quantity, np.float64]:
    """
    A different type of brightness profile that allows you to divide the cluster up azimuthally as
    well as radially. It performs the same calculation as radial_brightness, but for N angular bins,
    and as such returns N separate profiles.
    :param Image im_prod: An Image or RateMap object that you wish to construct a brightness profile from.
    :param np.ndarray src_mask: A numpy array that masks out everything but the source, including interlopers.
    :param np.ndarray back_mask: A numpy array that masks out everything but the background, including interlopers.
    :param Quantity centre: The coordinates for the centre of the brightness profile.
    :param Quantity rad: The outer radius of the brightness profile (THIS SHOULD BE THE SAME RADIUS AS THE REGION
    YOUR SRC_MASK IS BASED ON, OTHERWISE YOU'LL GET AN INVALID BACKGROUND MEASUREMENT).
    :param int num_slices: The number of pizza slices to cut the cluster into. The size of each
    :param float z: The redshift of the source of interest.
    :param BaseUnit cen_rad_units: The desired output units for the central radii of the annulli.
    :param cosmo: An astropy cosmology object for source coordinate conversions.
    slice will be 360 / num_slices degrees.
    :return: The brightness is returned in a numpy array with a column per pizza slice, then the
    radii at the centre of the bins are returned in units of kpc, then the angle boundaries of each slice,
    and finally the average brightness in the background region is returned.
    :rtype: Tuple[ndarray, Quantity, Quantity, np.float64]
    """
    if im_prod.shape != src_mask.shape:
        raise ValueError("The shape of the src_mask array ({0}) must be the same as that of im_prod "
                         "({1}).".format(src_mask.shape, im_prod.shape))

    # Just making sure we have the centre in pixel coordinates
    pix_cen = im_prod.coord_conv(centre, pix)

    # This sets up the annular bin radii, as well as finding the central radii of the bins in the chosen units.
    inn_rads, out_rads, cen_rads = ann_radii(im_prod, centre, rad, z, cen_rad_units, cosmo)

    # Setup the angular limits for the slices
    angs = Quantity(np.linspace(0, 360, int(num_slices)+1), deg)
    start_angs = angs[:-1]
    stop_angs = angs[1:]

    br = np.zeros((len(inn_rads), len(start_angs)))
    # TODO Find a way to fail gracefully if weights are all zeros maybe - hopefully shouldn't
    #  happen anymore but can't promise
    for ang_ind in range(len(start_angs)):
        if im_prod.type == 'image':
            masks = annular_mask(pix_cen, inn_rads, out_rads, im_prod.shape, start_angs[ang_ind],
                                 stop_angs[ang_ind]) * src_mask[..., None]
        elif im_prod.type == 'ratemap':
            masks = annular_mask(pix_cen, inn_rads, out_rads, im_prod.shape, start_angs[ang_ind],
                                 stop_angs[ang_ind]) * src_mask[..., None] * im_prod.sensor_mask[..., None]

        masked_data = masks * im_prod.data[..., None]

        # Calculates the average for each radius, use the masks array as weights to only include unmasked
        #  areas in the average for each radius.
        br[:, ang_ind] = np.average(masked_data, axis=(0, 1), weights=masks)

    # Finds the average of the background region
    bg = np.average(im_prod.data * back_mask, axis=(0, 1), weights=back_mask)

    # Just packaging the angles nicely
    return_angs = Quantity(np.stack([start_angs.value, stop_angs.value]).T, deg)

    return br, cen_rads, return_angs, bg



