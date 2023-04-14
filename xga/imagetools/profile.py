#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/04/2023, 23:33. Copyright (c) The Contributors


from typing import Tuple

import numpy as np
from astropy.cosmology import Cosmology
from astropy.units import Quantity, UnitBase, pix, deg, arcsec, UnitConversionError

from .misc import pix_deg_scale, pix_rad_to_physical, physical_rad_to_pix, rad_to_ang
from .. import DEFAULT_COSMO
from ..products import Image, RateMap
from ..products.profile import SurfaceBrightness1D


def annular_mask(centre: Quantity, inn_rad: np.ndarray, out_rad: np.ndarray, shape: tuple,
                 start_ang: Quantity = Quantity(0, 'deg'), stop_ang: Quantity = Quantity(360, 'deg')) -> np.ndarray:
    """
    A handy little function to generate annular (or circular) masks in the form of numpy arrays.
    It produces the src_mask for a given shape of image, centered at supplied coordinates, and with inner and
    outer radii supplied by the user also. Angular limits can also be supplied to give the src_mask an annular
    dependence. This function should be properly vectorised, and accepts inner and outer radii in
    the form of arrays. The result will be an len_y, len_x, N dimensional array, where N is equal to
    the length of inn_rad.

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
              rad_units: UnitBase = arcsec, cosmo: Cosmology = DEFAULT_COSMO, min_central_pix_rad: int = 3,
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
    :param UnitBase rad_units: The output units for the centres of the annulli returned by
        this function. The inner and outer radii will always be in pixels.
    :param Cosmology cosmo: An instance of an astropy cosmology, the default is a concordance flat LambdaCDM model.
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
    cen_rads = pix_rad_to_physical(im_prod, pix_cen_rads, rad_units, deg_cen, z, cosmo)

    # If the innermost radius is zero then the innermost annulus is actually a circle and we don't want
    #  the central radius to be between it and the next radius, as that wouldn't be strictly accurate
    if rads[0] == 0:
        cen_rads[0] = Quantity(0, rad_units)

    return inn_rads, out_rads, cen_rads


def radial_brightness(rt: RateMap, centre: Quantity, outer_rad: Quantity, back_inn_rad_factor: float = 1.05,
                      back_out_rad_factor: float = 1.5, interloper_mask: np.ndarray = None,
                      z: float = None, pix_step: int = 1, rad_units: UnitBase = arcsec,
                      cosmo: Cosmology = DEFAULT_COSMO, min_snr: float = 0.0, min_central_pix_rad: int = 3,
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
    :param BaseUnit rad_units: The desired output units for the central radii of the annuli.
    :param Cosmology cosmo: An astropy cosmology object for source coordinate conversions.
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

    def _iterative_profile(annulus_masks: np.ndarray, inner_rads: np.ndarray, outer_rads: np.ndarray) \
            -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal function to calculate and re-bin (ONCE) a surface brightness profile. The profile, along with
        modified (if rebinned) masks and radii arrays are returned to the user. This can be called once, or iteratively
        by a loop.

        :param np.ndarray annulus_masks: 512x512xN numpy array of annular masks, where N is the number of annuli
        :param np.ndarray inner_rads: The inner radii of the annuli.
        :param np.ndarray outer_rads: The outer radii of the annuli.
        :return: Boolean variable that describes whether another re-binning iteration is required, the
            brightness profile and uncertainties, the modified annular masks, inner radii, outer radii, and annulus areas.
        :rtype:
        """
        # These are annular masks with interloper sources removed, sensor and edge masks applied
        corr_ann_masks = annulus_masks * interloper_mask[..., None] * rt.sensor_mask[..., None] \
                         * rt.edge_mask[..., None]

        # This calculates the area of each annulus mask
        num_pix = np.sum(corr_ann_masks, axis=(0, 1))
        ann_areas = num_pix * to_arcmin ** 2

        # Applying the annular masks, with interlopers removed, and chip edges
        masked_countrate_data = corr_ann_masks * rt.data[..., None]
        masked_countrate_error_data = corr_ann_masks * count_rate_err_map[..., None]

        # Defining the brightness profile with the current annuli
        bright_profile = np.sum(masked_countrate_data, axis=(0, 1)) / ann_areas

        # And the uncertainties on the profiles. Adding pixels in quadrature
        bright_profile_errors = np.sqrt(np.sum(masked_countrate_error_data**2, axis=(0, 1))) / ann_areas

        # We want to check if all parts of the profile are above the defined minimum signal to noise
        snr_prof = bright_profile / countrate_bg_per_area

        # Checking if we are below the minimum signal to noise anywhere
        below = np.argwhere(snr_prof < min_snr).flatten()
        if below.shape != (0,) and below[0] != (bright_profile.shape[0] - 1):
            annulus_masks[:, :, below[0]] = annulus_masks[:, :, below[0]] + annulus_masks[:, :, below[0] + 1]
            annulus_masks = np.delete(annulus_masks, below[0] + 1, axis=2)

            outer_rads[below[0]] = outer_rads[below[0] + 1]
            outer_rads = np.delete(outer_rads, below[0] + 1)
            inner_rads = np.delete(inner_rads, below[0] + 1)

            another_pass = True

        elif below.shape != (0,) and below[0] == (bright_profile.shape[0] - 1):
            annulus_masks[:, :, below[0] - 1] = annulus_masks[:, :, below[0] - 1] + annulus_masks[:, :, below[0]]
            annulus_masks = np.delete(annulus_masks, below[0], axis=2)

            outer_rads[below[0] - 1] = outer_rads[below[0]]
            outer_rads = np.delete(outer_rads, below[0])
            inner_rads = np.delete(inner_rads, below[0])

            another_pass = True

        else:
            another_pass = False

        return another_pass, bright_profile, bright_profile_errors, annulus_masks, inner_rads, outer_rads, ann_areas

    if interloper_mask is not None and rt.shape != interloper_mask.shape:
        raise ValueError("The shape of the src_mask array {0} must be the same as that of im_prod "
                         "{1}.".format(interloper_mask.shape, rt.shape))
    elif interloper_mask is None:
        interloper_mask = np.ones(rt.shape)

    # Returns conversion factor to degrees, so multiplying by 60 goes to arcminutes
    # Getting this because we want to be able to convert pixel distance into arcminutes for dividing by the area
    to_arcmin = pix_deg_scale(centre, rt.radec_wcs).value * 60

    # Just making sure we have the centre in pixel coordinates
    pix_cen = rt.coord_conv(centre, pix)

    # This sets up the initial annular bin radii, as well as finding the central radii of the bins in the chosen units.
    inn_rads, out_rads, cen_rads = ann_radii(rt, centre, outer_rad, z, pix_step, rad_units, cosmo,
                                             min_central_pix_rad, start_pix_rad)

    # These calculate the inner and out pixel radii for the background mask - placed in arrays because the
    #  annular mask function expects iterable radii. As pixel radii have to be integer for generating a mask,
    #  I've used ceil.
    back_inn_rad = np.array([np.ceil(out_rads[-1]*back_inn_rad_factor).astype(int)])
    back_out_rad = np.array([np.ceil(out_rads[-1]*back_out_rad_factor).astype(int)])

    # Using my annular mask function to make a nice background region, which will be corrected for instrumental
    #  stuff and interlopers in a second
    back_mask = annular_mask(pix_cen, back_inn_rad, back_out_rad, rt.shape)

    # Includes chip gaps, interloper removal, and edge removal (to try and avoid artificially bright pixels)
    #  in the background mask
    corr_back_mask = back_mask * rt.sensor_mask * rt.edge_mask * interloper_mask

    # Calculates the area of the background region in arcmin^2
    back_pix = np.sum(corr_back_mask, axis=(0, 1))
    back_area = back_pix * to_arcmin**2
    if back_area == 0:
        raise ValueError("The background mask combined with the sensor mask is is all zeros, this is probably"
                         " because you're looking at a large cluster at a low redshift.")
    # Per pix is the background which has been divided by the number of pixels
    count_bg_per_pix = np.sum(rt.image.data * corr_back_mask, axis=(0, 1)) / back_pix
    # This is where the pixels have been converted into an actual area measurement in arcmin units
    countrate_bg_per_area = np.sum(rt.data * corr_back_mask, axis=(0, 1)) / back_area

    # Defined here so we can use the where argument in np.sqrt
    err_calc = (rt.image.data - count_bg_per_pix) + 2*count_bg_per_pix
    # Errors on the raw photon counts
    count_err_map = np.zeros(rt.shape)
    # Turning those into count rate errors
    np.sqrt(err_calc, where=err_calc > 0, out=count_err_map)

    # I divide by a copy of the expmap data array here, otherwise it breaks, and if I'm honest I don't know why...
    count_rate_err_map = np.divide(count_err_map, rt.expmap.data.copy(), where=rt.expmap.data != 0) * rt.edge_mask

    # Using the ellipse adds enough : to get all the dimensions in the array, then the None adds an empty
    #  dimension. Helps with broadcasting the annular masks with the region src_mask that gets rid of interlopers
    ann_masks = annular_mask(pix_cen, inn_rads, out_rads, rt.shape)

    # Make copies of originals in case re-binning fails
    init_inn = inn_rads.copy()
    init_out = out_rads.copy()

    # This gets switched to false if the signal to noise conditions are not fulfilled
    calculate_profile = True
    while calculate_profile:
        calculate_profile, br_prof, br_errs, ann_masks, inn_rads, \
            out_rads, areas = _iterative_profile(ann_masks, inn_rads, out_rads)

    if len(inn_rads) == 0:
        # This means that rebinning failed, and the profile couldn't be made greater than the minimum signal to noise
        inn_rads = init_inn
        out_rads = init_out
        ann_masks = annular_mask(pix_cen, inn_rads, out_rads, rt.shape)
        # Set the min_snr to 0 to get an non-rebinned profile
        min_snr = 0
        prof_results = _iterative_profile(ann_masks, inn_rads, out_rads)
        br_prof, br_errs = prof_results[1:3]
        areas = prof_results[-1]
        succeeded = False
    else:
        succeeded = True

    final_inn_rads = pix_rad_to_physical(rt, Quantity(inn_rads, pix), rad_units, centre, z, cosmo)
    final_out_rads = pix_rad_to_physical(rt, Quantity(out_rads, pix), rad_units, centre, z, cosmo)

    # Need these simply because the brightness profile must always be aware of the radii in degrees in order to
    #  to assemble a storage key properly
    deg_inn_rads = pix_rad_to_physical(rt, Quantity(inn_rads, pix), deg, centre, z, cosmo)
    deg_out_rads = pix_rad_to_physical(rt, Quantity(out_rads, pix), deg, centre, z, cosmo)

    cen_rads = (final_inn_rads + final_out_rads) / 2
    deg_cen_rads = (deg_inn_rads + deg_out_rads) / 2
    if final_inn_rads[0].value == 0:
        cen_rads[0] = Quantity(0, cen_rads.unit)
        deg_cen_rads[0] = Quantity(0, 'deg')
    rad_err = (final_out_rads-final_inn_rads) / 2

    if not outer_rad.unit.is_equivalent('deg'):
        deg_outer_rad = rad_to_ang(outer_rad, z, cosmo).to('deg')
    else:
        deg_outer_rad = outer_rad.to("deg")

    # Now I've finally implemented some profile product classes I can just smoosh everything into a convenient product
    br_prof = SurfaceBrightness1D(rt, cen_rads, Quantity(br_prof, 'ct/(s*arcmin**2)'), centre, pix_step, min_snr,
                                  deg_outer_rad, rad_err, Quantity(br_errs, 'ct/(s*arcmin**2)'),
                                  Quantity(countrate_bg_per_area, 'ct/(s*arcmin**2)'),
                                  np.insert(out_rads, 0, inn_rads[0]), np.concatenate([back_inn_rad, back_out_rad]),
                                  Quantity(areas, 'arcmin**2'), deg_cen_rads, succeeded)

    return br_prof, succeeded


# TODO REWRITE THESE AT SOME POINT
def pizza_brightness(im_prod: Image, src_mask: np.ndarray, back_mask: np.ndarray,
                     centre: Quantity, rad: Quantity, num_slices: int = 4,
                     z: float = None, pix_step: int = 1, cen_rad_units: UnitBase = arcsec,
                     cosmo=DEFAULT_COSMO) -> Tuple[np.ndarray, Quantity, Quantity, np.float64, np.ndarray, np.ndarray]:

    raise NotImplementedError("The supporting infrastructure to allow pizza profile product objects hasn't been"
                              " written yet sorry!")


