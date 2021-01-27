#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 27/01/2021, 17:18. Copyright (c) David J Turner

from typing import Tuple

import numpy as np
from astropy.units import Quantity

from ..exceptions import XGAPoorDataError
from ..imagetools.misc import pix_deg_scale
from ..imagetools.profile import annular_mask
from ..sources import BaseSource


def _snr_bins(source: BaseSource, outer_rad: Quantity, min_snr: float, min_width: Quantity, lo_en: Quantity,
              hi_en: Quantity, obs_id: str = None, inst: str = None, psf_corr: bool = False, psf_model: str = "ELLBETA",
              psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15,
              allow_negative: bool = False) -> Tuple[Quantity, int]:
    """
    An internal function that will find the radii required to create annuli with a certain minimum signal to noise
    and minimum annulus width.

    :param BaseSource source: The source object which needs annuli generating for it.
    :param Quantity outer_rad: The outermost radius of the source region we will generate annuli within.
    :param float min_snr: The minimum signal to noise which is allowable in a given annulus.
    :param Quantity min_width: The minimum allowable width of the annuli. This can be set to try and avoid
        PSF effects.
    :param Quantity lo_en: The lower energy bound of the ratemap to use for the signal to noise calculation.
    :param Quantity hi_en: The upper energy bound of the ratemap to use for the signal to noise calculation.
    :param str obs_id: An ObsID of a specific ratemap to use for the SNR calculations. Default is None, which
            means the combined ratemap will be used. Please note that inst must also be set to use this option.
    :param str inst: The instrument of a specific ratemap to use for the SNR calculations. Default is None, which
        means the combined ratemap will be used.
    :param bool psf_corr: Sets whether you wish to use a PSF corrected ratemap or not.
    :param str psf_model: If the ratemap you want to use is PSF corrected, this is the PSF model used.
    :param int psf_bins: If the ratemap you want to use is PSF corrected, this is the number of PSFs per
        side in the PSF grid.
    :param str psf_algo: If the ratemap you want to use is PSF corrected, this is the algorithm used.
    :param int psf_iter: If the ratemap you want to use is PSF corrected, this is the number of iterations.
    :param bool allow_negative: Should pixels in the background subtracted count map be allowed to go below
        zero, which results in a lower signal to noise (and can result in a negative signal to noise).
    :return: The radii of the requested annuli, and the original maximum number based on min_width.
    :rtype: Tuple[Quantity, int]
    """
    # Parsing the ObsID and instrument options, see if they want to use a specific ratemap
    if all([obs_id is None, inst is None]):
        # Here the user hasn't set ObsID or instrument, so we use the combined data
        rt = source.get_combined_ratemaps(lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)
        interloper_mask = source.get_interloper_mask()
    elif all([obs_id is not None, inst is not None]):
        # Both ObsID and instrument have been set by the user
        rt = source.get_ratemaps(obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)
        interloper_mask = source.get_interloper_mask(obs_id)

    # Just making sure our relevant distances are in the same units, so that we can convert to pixels
    outer_rad = source.convert_radius(outer_rad, 'deg')
    min_width = source.convert_radius(min_width, 'deg')

    # Using the ratemap to get a conversion factor from pixels to degrees, though we will use it
    #  the other way around
    pix_to_deg = pix_deg_scale(source.default_coord, rt.radec_wcs)

    # Making sure to go up to the whole number, pixels have to be integer of course and I think its
    #  better to err on the side of caution here and make things slightly wider than requested
    outer_rad = int(np.ceil(outer_rad/pix_to_deg).value)
    min_width = int(np.ceil(min_width/pix_to_deg).value)

    # The maximum possible number of annuli, based on the input outer radius and minimum width
    # We have already made sure that the outer radius and minimum width allowed are integers by using
    #  np.ceil, so we know max_ann is going to be a whole number of annuli
    max_ann = int(outer_rad/min_width)

    # These are the initial bins, with imposed minimum width, I have to add one to max_ann because linspace wants the
    #  total number of values to generate, and while there are max_ann annuli, there are max_ann+1 radial boundaries
    init_rads = np.linspace(0, outer_rad, max_ann+1).astype(int)

    # Converts the source's default analysis coordinates to pixels
    pix_centre = rt.coord_conv(source.default_coord, 'pix')
    # Sets up a mask to correct for interlopers and weird edge effects
    corr_mask = interloper_mask*rt.edge_mask

    # Setting up our own background region
    back_inn_rad = np.array([np.ceil(source.background_radius_factors[0] * outer_rad)]).astype(int)
    back_out_rad = np.array([np.ceil(source.background_radius_factors[1] * outer_rad)]).astype(int)

    # Using my annular mask function to make a nice background region, which will be corrected for instrumental
    #  stuff and interlopers in a second
    back_mask = annular_mask(pix_centre, back_inn_rad, back_out_rad, rt.shape) * corr_mask

    # Generates the requested annular masks, making sure to apply the correcting mask
    ann_masks = annular_mask(pix_centre, init_rads[:-1], init_rads[1:], rt.shape)*corr_mask[..., None]

    # This will be modified by the loop until it describes annuli which all have an acceptable signal to noise
    cur_rads = init_rads.copy()
    acceptable = False
    while not acceptable:
        # How many annuli are there at this point in the loop?
        cur_num_ann = ann_masks.shape[2]

        # Just a list for the snrs to live in
        snrs = []
        for i in range(cur_num_ann):
            # We're calling the signal to noise calculation method of the ratemap for all of our annuli
            snrs.append(rt.signal_to_noise(ann_masks[:, :, i], back_mask, allow_negative=allow_negative))
        # Becomes a numpy array because they're nicer to work with
        snrs = np.array(snrs)
        # We find any indices of the array (== annuli) where the signal to noise is not above our minimum
        bad_snrs = np.where(snrs < min_snr)[0]

        # If there are no annuli below our signal to noise threshold then all is good and joyous and we accept
        #  the current radii
        if len(bad_snrs) == 0:
            acceptable = True
        # We work from the outside of the bad list inwards, and if the outermost bad bin is the one right on the
        #  end of the SNR profile, then we merge that leftwards into the N-1th annuli
        elif len(bad_snrs) != 0 and bad_snrs[-1] == cur_num_ann-1:
            cur_rads = np.delete(cur_rads, -2)
            ann_masks = annular_mask(pix_centre, cur_rads[:-1], cur_rads[1:], rt.shape) * corr_mask[..., None]
        # Otherwise if the outermost bad annulus is NOT right at the end of the profile, we merge to the right
        else:
            cur_rads = np.delete(cur_rads, bad_snrs[-1])
            ann_masks = annular_mask(pix_centre, cur_rads[:-1], cur_rads[1:], rt.shape) * corr_mask[..., None]

        if ann_masks.shape[2] < 4:
            raise XGAPoorDataError("The requested annuli for {s} cannot be created, the data quality is too "
                                   "low".format(s=source.name))

    # Now of course, pixels must become a more useful unit again
    final_rads = (Quantity(cur_rads, 'pix') * pix_to_deg).to("arcsec")

    return final_rads, max_ann











