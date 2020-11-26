#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 26/11/2020, 17:24. Copyright (c) David J Turner


from typing import Tuple

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity, UnitBase, pix, deg, arcsec, UnitConversionError

from .misc import pix_deg_scale, pix_rad_to_physical, physical_rad_to_pix
from ..products import Image, RateMap
from ..products.profile import SurfaceBrightness1D


def annular_mask(centre: Quantity, inn_rad: np.ndarray, out_rad: np.ndarray, shape: tuple,
                 start_ang: Quantity = Quantity(0, 'deg'), stop_ang: Quantity = Quantity(360, 'deg')) -> np.ndarray:
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

    if ann_mask.shape[-1] == 1:
        ann_mask = np.squeeze(ann_mask)

    # Returns the annular src_mask(s), in the form of a len_y, len_x, N dimension np array
    return ann_mask


def ann_radii(im_prod: Image, centre: Quantity, rad: Quantity, z: float = None, pix_step: int = 1,
              cen_rad_units: UnitBase = arcsec, cosmo=Planck15, min_central_pix_rad: int = 3,
              start_pix_rad: int = 0) -> Tuple[np.ndarray, np.ndarray, Quantity]:
    """
    Will probably only ever be called by an internal brightness calculation, but two different methods
    need it so it gets its own method.
    :param Image im_prod: An Image or RateMap product object that you wish to calculate annuli for.
    :param Quantity centre: The coordinates of the centre of the set of annuli.
    :param Quantity rad: The outer radius of the set of annuli.
    :param float z: The redshift of the source of interest, required if the output radius units are
    a proper radius.
    :param int pix_step: The width (in pixels) of each annular bin, default is 1.
    :param UnitBase cen_rad_units: The output units for the centres of the annulli returned by
    this function. The inner and outer radii will always be in pixels.
    :param cosmo: An instance of an astropy cosmology, the default is Planck15.
    :param int start_pix_rad: The pixel radius at which the innermost annulus starts, default is zero.
    :param int min_central_pix_rad: The minimum radius of the innermost circular annulus (will only
    be used if start_pix_rad is 0, otherwise the innermost annulus is not a circle), default is three.
    :return: Returns the inner and outer radii of the annuli (in pixels), and the centres of the annuli
    in cen_rad_units.
    :rtype: Tuple[np.ndarray, np.ndarray, Quantity]
    """
    # Quickly get the central coordinates in degrees as well
    deg_cen = im_prod.coord_conv(centre, deg)
    pix_cen = im_prod.coord_conv(centre, pix)

    # If the radius is passed in a proper distance unit, this will convert it or raise an exception
    #  if it can't
    if rad.unit.is_equivalent('deg') or rad.unit.is_equivalent('kpc'):
        rad = physical_rad_to_pix(im_prod, rad, deg_cen, z, cosmo)
    elif rad.unit == pix:
        # If the radius is already in pixels then all is well
        pass
    else:
        raise UnitConversionError('{} is not a recognised distance unit'.format(rad.unit.to_string()))

    rad = np.ceil(rad)

    # So I'm adding a safety feature here, and ensuring the the central circle is a minimum radius
    if pix_step < 3 and start_pix_rad == 0:
        central_circ = np.array([0, min_central_pix_rad], dtype=int)
        ann_rads = np.arange(min_central_pix_rad+pix_step, rad.value + 1, pix_step).astype(int)
        rads = np.concatenate([central_circ, ann_rads])
    else:
        # By this point, the rad should be in pixels
        rads = np.arange(start_pix_rad, rad.value + 1, pix_step).astype(int)

    inn_rads = rads[:len(rads) - 1]
    out_rads = rads[1:len(rads)]

    pix_cen_rads = Quantity((inn_rads + out_rads)/2, pix)
    cen_rads = pix_rad_to_physical(im_prod, pix_cen_rads, cen_rad_units, deg_cen, z, cosmo)

    # If the innermost radius is zero then the innermost annulus is actually a circle and we don't want
    #  the central radius to be between it and the next radius, as that wouldn't be strictly accurate
    if rads[0] == 0:
        cen_rads[0] = Quantity(0, cen_rad_units)

    return inn_rads, out_rads, cen_rads


def radial_brightness(rt: RateMap, centre: Quantity, outer_rad: Quantity, back_inn_rad_factor: float = 1.05,
                      back_out_rad_factor: float = 1.5, interloper_mask: np.ndarray = None,
                      z: float = None, pix_step: int = 1, cen_rad_units: UnitBase = arcsec,
                      cosmo=Planck15, min_snr: float = 0.0, min_central_pix_rad: int = 3,
                      start_pix_rad: int = 0) -> Tuple[SurfaceBrightness1D, bool]:
    """
    A simple method to calculate the average brightness in circular annuli upto the radius of
    the chosen region. The annuli are one pixel in width, and as this uses the masks that were generated
    earlier, interloper sources should be removed.
    :param RateMap rt: A RateMap object to construct a brightness profile from.
    :param Quantity centre: The coordinates for the centre of the brightness profile.
    :param Quantity outer_rad: The outer radius of the brightness profile.
    :param float back_inn_rad_factor: This factor is multiplied by the outer pixel radius, which gives the inner
    radius for the background mask.
    :param float back_out_rad_factor: This factor is multiplied by the outer pixel radius, which gives the outer
    radius for the background mask.
    :param np.ndarray interloper_mask: A numpy array that masks out any interloper sources.
    :param float z: The redshift of the source of interest.
    :param int pix_step: The width (in pixels) of each annular bin, default is 1.
    :param BaseUnit cen_rad_units: The desired output units for the central radii of the annuli.
    :param cosmo: An astropy cosmology object for source coordinate conversions.
    :param float min_snr: The minimum signal to noise allowed for each bin in the profile. If any point is
    below this threshold the profile will be rebinned. Default is 0.0
    :param int start_pix_rad: The pixel radius at which the innermost annulus starts, default is zero.
    :param int min_central_pix_rad: The minimum radius of the innermost circular annulus (will only
    be used if start_pix_rad is 0, otherwise the innermost annulus is not a circle), default is three.
    :return: The brightness is returned in a flat numpy array, then the radii at the centre of the bins are
    returned in units of kpc, the width of the bins, and finally the average brightness in the background region is
    returned.
    :rtype: Tuple[SurfaceBrightness1D, bool]
    """
    if interloper_mask is not None and rt.shape != interloper_mask.shape:
        raise ValueError("The shape of the src_mask array {0} must be the same as that of im_prod "
                         "{1}.".format(interloper_mask.shape, rt.shape))
    elif interloper_mask is None:
        interloper_mask = np.ones(rt.shape)

    cr_err_map = np.divide(np.sqrt(rt.image.data), rt.expmap.data, out=np.zeros_like(rt.image.data),
                           where=rt.expmap.data != 0)

    # Returns conversion factor to degrees, so multiplying by 60 goes to arcminutes
    # Getting this because we want to be able to convert pixel distance into arcminutes for dividing by the area
    to_arcmin = pix_deg_scale(centre, rt.radec_wcs) * 60

    # Just making sure we have the centre in pixel coordinates
    pix_cen = rt.coord_conv(centre, pix)

    # This sets up the initial annular bin radii, as well as finding the central radii of the bins in the chosen units.
    inn_rads, out_rads, init_cen_rads = ann_radii(rt, centre, outer_rad, z, pix_step, cen_rad_units, cosmo,
                                                  min_central_pix_rad, start_pix_rad)

    # These calculate the inner and out pixel radii for the background mask - placed in arrays because the
    #  annular mask function expects iterable radii. As pixel radii have to be integer for generating a mask,
    #  I've used ceil.
    back_inn_rad = np.array([np.ceil(out_rads[-1]*back_inn_rad_factor).astype(int)])
    back_out_rad = np.array([np.ceil(out_rads[-1]*back_out_rad_factor).astype(int)])

    # Using my annular mask function to make a nice background region, which will be corrected for instrumental
    #  stuff and interlopers in a second
    back_mask = annular_mask(pix_cen, back_inn_rad, back_out_rad, rt.shape)

    # Adds any intersecting chip gaps into the background region mask
    corr_back_mask = back_mask * rt.sensor_mask * interloper_mask

    # Calculates the area of the background region in arcmin^2
    back_area = np.sum(corr_back_mask, axis=(0, 1)) * to_arcmin**2
    if back_area == 0:
        raise ValueError("The background mask combined with the sensor mask is is all zeros, this is probably"
                         " because you're looking at a large cluster at a low redshift.")
    # Finds the emission per arcmin^2 of the background region (accounting for removed sources and chip gaps)
    bg = np.sum(rt.data * corr_back_mask, axis=(0, 1)) / back_area

    # Using the ellipse adds enough : to get all the dimensions in the array, then the None adds an empty
    #  dimension. Helps with broadcasting the annular masks with the region src_mask that gets rid of interlopers
    masks = annular_mask(pix_cen, inn_rads, out_rads, rt.shape) * interloper_mask[..., None] * rt.sensor_mask[..., None]
    # This calculates the area of each annulus mask
    num_pix = np.sum(masks, axis=(0, 1))
    areas = num_pix * to_arcmin**2

    # Creates a 3D array of the masked data
    masked_data = masks * rt.data[..., None]
    # Calculates the sum of the pixel count rates for each annular radius, masking out other known sources
    cr = np.sum(masked_data, axis=(0, 1))
    cr_errs = np.sqrt(np.sum(masks * cr_err_map[..., None]**2, axis=(0, 1)))

    # Then calculates the actual brightness profile by dividing by the area of each annulus
    br = cr / areas
    br_errs = cr_errs / areas

    # Calculates the signal to noise profile, defined as the ratio between brightness profile and background
    snr_prof = br / bg

    # Finds the elements in the the SNR profile that do not meet the minimum requirements provided by the user
    #  Flatten it just because I know this will always a be a 1D array and flattening makes it nicer to work with
    below = np.argwhere(snr_prof < min_snr).flatten()

    # Making copies in case rebinning fails
    init_br = br.copy()
    init_br_errs = br_errs.copy()
    init_inn = inn_rads.copy()
    init_out = out_rads.copy()

    # Our task here is to combine radial bins until the minimum SNR requirements are met for all bins
    # Using a while loop for this doesn't feel super efficient, but as a first attempt hopefully it'll be fast enough
    # If the shape of below is (0,) then there are no bins at which the SNR is causing a problem
    while below.shape != (0,):
        # This is triggered if the first index where SNR is too low IS NOT the last bin in the profile
        if below[0] != br.shape[0] - 1:
            # We deal with and modify the count rates and areas separately as you have to combine bins in the count
            #  rate and area regimes separately, then calculate brightness by dividing cr by area.
            cr[below[0]] = cr[below[0]] + cr[below[0] + 1]
            # ADDING ERRORS IN QUADRATURE FOR NOW, DON'T KNOW IF I'LL KEEP IT LIKE THIS
            cr_errs[below[0]] = np.sqrt(cr_errs[below[0]]**2 + cr_errs[below[0] + 1]**2)
            areas[below[0]] = areas[below[0]] + areas[below[0] + 1]
            # Then the donor bin that was added to the problem bin is deleted
            cr = np.delete(cr, below[0] + 1)
            cr_errs = np.delete(cr_errs, below[0] + 1)
            areas = np.delete(areas, below[0] + 1)
            # The new outer radius of the problem bin is set to the outer radius of the donor bin
            out_rads[below[0]] = out_rads[below[0] + 1]
            # The outer radius entry of the donor bin is deleted
            out_rads = np.delete(out_rads, below[0] + 1)
            # The inner radius of the donor bin is deleted as there is only one bin now from problem inner
            #  to donor outer radii
            inn_rads = np.delete(inn_rads, below[0] + 1)
        # This is triggered if the first index where SNR is to low IS the last bin in the profile
        else:
            # As this is the last bin in the profile, there is no extra bin in the outward direction to add to our
            #  problem bin. As such the problem bin is added inwards, to the N-1th bin in the profile
            cr[below[0] - 1] = cr[below[0] - 1] + cr[below[0]]
            cr_errs[below[0] - 1] = np.sqrt(cr_errs[below[0] - 1]**2 + cr_errs[below[0]]**2)
            areas[below[0] - 1] = areas[below[0] - 1] + areas[below[0]]
            cr = np.delete(cr, below[0])
            cr_errs = np.delete(cr_errs, below[0])
            areas = np.delete(areas, below[0])
            out_rads[below[0] - 1] = out_rads[below[0]]
            out_rads = np.delete(out_rads, below[0])
            inn_rads = np.delete(inn_rads, below[0])

        # Calculate the new brightness with our combined count rates and areas
        br = cr/areas
        br_errs = cr_errs / areas
        # Recalculate the SNR profile after the re-binning in this iteration
        snr_prof = br / bg
        # Find out which bins are still below the SNR threshold (if any)
        below = np.argwhere(snr_prof < min_snr).flatten()

    if len(inn_rads) == 0:
        inn_rads = init_inn
        out_rads = init_out
        br = init_br
        br_errs = init_br_errs
        succeeded = False
    else:
        succeeded = True

    final_inn_rads = pix_rad_to_physical(rt, Quantity(inn_rads, pix), cen_rad_units, centre, z, cosmo)
    final_out_rads = pix_rad_to_physical(rt, Quantity(out_rads, pix), cen_rad_units, centre, z, cosmo)
    cen_rads = (final_inn_rads + final_out_rads) / 2
    if final_inn_rads[0].value == 0:
        cen_rads[0] = Quantity(0, cen_rads.unit)
    rad_err = (final_out_rads-final_inn_rads) / 2

    # Now I've finally implemented some profile product classes I can just smoosh everything into a convenient product
    br_prof = SurfaceBrightness1D(rt, cen_rads, Quantity(br, 'ct/(s*arcmin**2)'), centre, pix_step, min_snr, outer_rad,
                                  rad_err, Quantity(br_errs, 'ct/(s*arcmin**2)'), Quantity(bg, 'ct/(s*arcmin**2)'),
                                  np.insert(out_rads, 0, inn_rads[0]), Quantity(areas, 'arcmin**2'))
    # Set the success property
    br_prof.min_snr_succeeded = succeeded

    return br_prof, succeeded


# TODO At some point implement minimum SNR for this also
def pizza_brightness(im_prod: Image, src_mask: np.ndarray, back_mask: np.ndarray,
                     centre: Quantity, rad: Quantity, num_slices: int = 4,
                     z: float = None, pix_step: int = 1, cen_rad_units: UnitBase = arcsec,
                     cosmo=Planck15) -> Tuple[np.ndarray, Quantity, Quantity, np.float64, np.ndarray, np.ndarray]:
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
    :param int pix_step: The width (in pixels) of each annular bin, default is 1.
    :param BaseUnit cen_rad_units: The desired output units for the central radii of the annuli.
    :param cosmo: An astropy cosmology object for source coordinate conversions.
    slice will be 360 / num_slices degrees.
    :return: The brightness is returned in a numpy array with a column per pizza slice, then the
    radii at the centre of the bins are returned in units of kpc, then the angle boundaries of each slice,
    and finally the average brightness in the background region is returned.
    :rtype: Tuple[ndarray, Quantity, Quantity, np.float64, ndarray, ndarray]
    """
    raise NotImplementedError("The supporting infrastructure to allow pizza profile product objects hasn't been"
                              " written yet sorry!")
    if im_prod.shape != src_mask.shape:
        raise ValueError("The shape of the src_mask array ({0}) must be the same as that of im_prod "
                         "({1}).".format(src_mask.shape, im_prod.shape))

    # Just making sure we have the centre in pixel coordinates
    pix_cen = im_prod.coord_conv(centre, pix)

    # This sets up the annular bin radii, as well as finding the central radii of the bins in the chosen units.
    inn_rads, out_rads, cen_rads = ann_radii(im_prod, centre, rad, z, pix_step, cen_rad_units, cosmo)

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

    return br, cen_rads, return_angs, bg, inn_rads, out_rads


