#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 16/06/2021, 14:57. Copyright (c) David J Turner

from typing import Tuple, Union, List
from warnings import warn

import numpy as np
from astropy.units import Quantity

from .deproj import shell_ann_vol_intersect
from .. import NUM_CORES, ABUND_TABLES
from ..exceptions import NoProductAvailableError
from ..imagetools.misc import pix_deg_scale
from ..imagetools.profile import annular_mask
from ..products.profile import GasTemperature3D
from ..samples import BaseSample, ClusterSample
from ..sas import region_setup
from ..sources import BaseSource, GalaxyCluster
from ..xspec.fit import single_temp_apec_profile

ALLOWED_ANN_METHODS = ['min_snr', 'growth']


def _snr_bins(source: BaseSource, outer_rad: Quantity, min_snr: float, min_width: Quantity, lo_en: Quantity,
              hi_en: Quantity, obs_id: str = None, inst: str = None, psf_corr: bool = False, psf_model: str = "ELLBETA",
              psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15,
              allow_negative: bool = False, exp_corr: bool = True) -> Tuple[Quantity, np.ndarray, int]:
    """
    An internal function that will find the radii required to create annuli with a certain minimum signal to noise
    and minimum annulus width.

    :param BaseSource source: The source object which needs annuli generating for it.
    :param Quantity outer_rad: The outermost radius of the source region we will generate annuli within.
    :param float min_snr: The minimum signal to noise which is allowable in a given annulus.
    :param Quantity min_width: The minimum allowable width of the annuli. This can be set to try and avoid
        PSF effects.
    :param Quantity lo_en: The lower energy bound of the ratemap to use for the signal to noise calculations.
    :param Quantity hi_en: The upper energy bound of the ratemap to use for the signal to noise calculations.
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
    :param bool exp_corr: Should signal to noises be measured with exposure time correction, default is True. I
            recommend that this be true for combined observations, as exposure time could change quite dramatically
            across the combined product.
    :return: The radii of the requested annuli, the final snr values, and the original maximum number
        based on min_width.
    :rtype: Tuple[Quantity, np.ndarray, int]
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

    cur_rads = init_rads.copy()
    if max_ann > 4:
        # This will be modified by the loop until it describes annuli which all have an acceptable signal to noise
        acceptable = False
    else:
        # If there are already 4 or less annuli present then we don't do the reduction while loop, and just take it
        #  as they are, while also issuing a warning
        acceptable = True
        warn("The min_width combined with the outer radius of the source means that there are only {} initial"
             " annuli, normally four is the minimum number I will allow, so I will do no re-binning.".format(max_ann))
        cur_num_ann = ann_masks.shape[2]
        snrs = []
        for i in range(cur_num_ann):
            # We're calling the signal to noise calculation method of the ratemap for all of our annuli
            snrs.append(rt.signal_to_noise(ann_masks[:, :, i], back_mask, exp_corr, allow_negative))
        # Becomes a numpy array because they're nicer to work with
        snrs = np.array(snrs)

    while not acceptable:
        # How many annuli are there at this point in the loop?
        cur_num_ann = ann_masks.shape[2]

        # Just a list for the snrs to live in
        snrs = []
        for i in range(cur_num_ann):
            # We're calling the signal to noise calculation method of the ratemap for all of our annuli
            snrs.append(rt.signal_to_noise(ann_masks[:, :, i], back_mask, exp_corr, allow_negative))
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

        if ann_masks.shape[2] == 4 and not acceptable:
            warn("The requested annuli for {s} cannot be created, the data quality is too low. As such a set "
                 "of four annuli will be returned".format(s=source.name))
            break

    # Now of course, pixels must become a more useful unit again
    final_rads = (Quantity(cur_rads, 'pix') * pix_to_deg).to("arcsec")

    return final_rads, snrs, max_ann


def min_snr_proj_temp_prof(sources: Union[GalaxyCluster, ClusterSample], outer_radii: Union[Quantity, List[Quantity]],
                           min_snr: float = 20, min_width: Quantity = Quantity(20, 'arcsec'), use_combined: bool = True,
                           use_worst: bool = False, lo_en: Quantity = Quantity(0.5, 'keV'),
                           hi_en: Quantity = Quantity(2, 'keV'), psf_corr: bool = False, psf_model: str = "ELLBETA",
                           psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, allow_negative: bool = False,
                           exp_corr: bool = True, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                           over_sample: float = None, one_rmf: bool = True, abund_table: str = "angr",
                           num_cores: int = NUM_CORES) -> List[Quantity]:
    """
    This is a convenience function that allows you to quickly and easily start measuring projected
    temperature profiles of galaxy clusters, deciding on the annular bins using signal to noise measurements
    from photometric products. This function calls single_temp_apec_profile, but doesn't expose all of the more
    in depth variables, so if you want more control then use single_temp_apec_profile directly. The projected
    temperature profiles which are generated are added to their source's storage structure.

    :param GalaxyCluster/ClusterSample sources: An individual or sample of sources to measure projected
        temperature profiles for.
    :param str/Quantity outer_radii: The name or value of the outer radius to use for the generation of
        the spectra (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored. If you are
        generating for multiple sources then you can also pass a Quantity with one entry per source.
    :param float min_snr: The minimum signal to noise which is allowable in a given annulus.
    :param Quantity min_width: The minimum allowable width of an annulus. The default is set to 20 arcseconds to try
        and avoid PSF effects.
    :param bool use_combined: If True then the combined RateMap will be used for signal to noise annulus
        calculations, this is overridden by use_worst.
    :param bool use_worst: If True then the worst observation of the cluster (ranked by global signal to noise) will
        be used for signal to noise annulus calculations.
    :param Quantity lo_en: The lower energy bound of the ratemap to use for the signal to noise calculations.
    :param Quantity hi_en: The upper energy bound of the ratemap to use for the signal to noise calculations.
    :param bool psf_corr: Sets whether you wish to use a PSF corrected ratemap or not.
    :param str psf_model: If the ratemap you want to use is PSF corrected, this is the PSF model used.
    :param int psf_bins: If the ratemap you want to use is PSF corrected, this is the number of PSFs per
        side in the PSF grid.
    :param str psf_algo: If the ratemap you want to use is PSF corrected, this is the algorithm used.
    :param int psf_iter: If the ratemap you want to use is PSF corrected, this is the number of iterations.
    :param bool allow_negative: Should pixels in the background subtracted count map be allowed to go below
        zero, which results in a lower signal to noise (and can result in a negative signal to noise).
    :param bool exp_corr: Should signal to noises be measured with exposure time correction, default is True. I
            recommend that this be true for combined observations, as exposure time could change quite dramatically
            across the combined product.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param str abund_table: The abundance table to use during the XSPEC fits.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :return: A list of non-scalar astropy quantities containing the annular radii used to generate the
        projected temperature profiles created by this function. Each Quantity element of the list corresponds
        to a source.
    :rtype: List[Quantity]
    """

    if outer_radii != 'region':
        inn_rad_vals, out_rad_vals = region_setup(sources, outer_radii, Quantity(0, 'arcsec'), True, '')[1:]
    else:
        raise NotImplementedError("I don't currently support fitting region spectra")

    if all([use_combined, use_worst]):
        warn("You have passed both use_combined and use_worst as True. use_worst overrides use_combined, so the "
             "worst observation for each source will be used to decide on the annuli.")
        use_combined = False
    elif all([not use_combined, not use_worst]):
        warn("You have passed both use_combined and use_worst as False. One of them must be True, so here we default"
             " to using the combined data to decide on the annuli.")
        use_combined = True

    if abund_table not in ABUND_TABLES:
        avail_abund = ", ".join(ABUND_TABLES)
        raise ValueError("{a} is not a valid abundance table choice, please use one of the "
                         "following; {av}".format(a=abund_table, av=avail_abund))

    if isinstance(sources, BaseSource):
        sources = [sources]

    all_rads = []
    for src_ind, src in enumerate(sources):
        if use_combined:
            # This is the simplest option, we just use the combined ratemap to decide on the annuli with minimum SNR
            rads, snrs, ma = _snr_bins(src, out_rad_vals[src_ind], min_snr, min_width, lo_en, hi_en, psf_corr=psf_corr,
                                       psf_model=psf_model, psf_bins=psf_bins, psf_algo=psf_algo, psf_iter=psf_iter,
                                       allow_negative=allow_negative, exp_corr=exp_corr)
        else:
            # This way is slightly more complicated, but here we use the worst observation (ranked by global
            #  signal to noise).
            # The return for this function is ranked worst to best, so we grab the first row (which is an ObsID and
            #  instrument), then call _snr_bins with that one
            lowest_ranked = src.snr_ranking(out_rad_vals[src_ind], lo_en, hi_en, allow_negative)[0][0, :]
            rads, snrs, ma = _snr_bins(src, out_rad_vals[src_ind], min_snr, min_width, lo_en, hi_en, lowest_ranked[0],
                                       lowest_ranked[1], psf_corr, psf_model, psf_bins, psf_algo, psf_iter,
                                       allow_negative, exp_corr)

        # Shoves the annuli we've decided upon into a list for single_temp_apec_profile to use
        all_rads.append(rads)

    if len(sources) == 1:
        sources = sources[0]

    single_temp_apec_profile(sources, all_rads, group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                             over_sample=over_sample, one_rmf=one_rmf, num_cores=num_cores, abund_table=abund_table)

    return all_rads


def grow_ann_proj_temp_prof(sources: Union[BaseSource, BaseSample], outer_radii: Union[Quantity, List[Quantity]],
                            growth_factor: float = 1.3, start_radius: Quantity = Quantity(20, 'arcsec'),
                            num_ann: int = None, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                            over_sample: float = None, one_rmf: bool = True, num_cores: int = NUM_CORES):
    """
    This is a convenience function that allows you to quickly and easily start measuring projected temperature
    profiles of galaxy clusters where the outer radius of each annulus is some factor larger than that of the
    last annulus:
        .. math::
             R_{i+1} = R_{i}F

    If a growth factor is passed then the start radius and outer radius of a particular source will be used to solve
    for the number of annuli which should be generated. However if a number of annuli is passed (through num_ann),
    then this function will again use the start and outer radii and solve for the growth factor instead, over-riding
    any growth factor that may have been passed in.

    This function calls single_temp_apec_profile, but doesn't expose all of the more
    in depth variables, so if you want more control then use single_temp_apec_profile directly. The projected
    temperature profiles which are generated are added to their source's storage structure.

    :param GalaxyCluster/ClusterSample sources: An individual or sample of sources to measure projected
        temperature profiles for.
    :param str/Quantity outer_radii: The name or value of the outer radius to use for the generation of
        the spectra (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored. If you are
        generating for multiple sources then you can also pass a Quantity with one entry per source.
    :param float growth_factor: The factor by which the outer radius of the Nth annulus should be larger than
        the outer radius of the N-1th annulus. This will be over-ridden by a re-calculated value if a value
        is passed to num_ann.
    :param Quantity start_radius: The radius of the innermost circular annulus, the default is 20 arcseconds, which
        was chosen to try and avoid PSF effects.
    :param int num_ann: The number of annuli which should be used, default is None, in which case the value will be
        calculated using the growth factor, outer radius, and start radius. If this parameter is passed then
        growth_factor will be overridden by a recalculated value.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    """

    raise NotImplementedError("This doesn't work yet because I got bored")

    if outer_radii != 'region':
        inn_rad_vals, out_rad_vals = region_setup(sources, outer_radii, Quantity(0, 'arcsec'), True, '')[1:]
    else:
        raise NotImplementedError("I don't currently support fitting region spectra")

    all_rads = []
    for src_ind, src in enumerate(sources):
        cur_start = src.convert_radius(start_radius, 'arcsec')
        if num_ann is None:
            cur_num_ann = int(np.ceil(np.log(out_rad_vals[src_ind].to('arcsec').value / cur_start.value)
                                      / np.log(growth_factor)))
            cur_growth_factor = growth_factor
        else:
            cur_growth_factor = np.power(out_rad_vals[src_ind].to('arcsec').value / cur_start.value, 1/num_ann)
            cur_num_ann = num_ann

        rads = [cur_start.value]
        rads += [cur_start.value*ann_ind*cur_growth_factor for ann_ind in range(1, cur_num_ann+1)]
        print(Quantity(rads, 'arcsec'))
        print('')

    single_temp_apec_profile(sources, all_rads, group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                             over_sample=over_sample, one_rmf=one_rmf, num_cores=num_cores)


def onion_deproj_temp_prof(sources: Union[GalaxyCluster, ClusterSample], outer_radii: Union[Quantity, List[Quantity]],
                           annulus_method: str = 'min_snr', min_snr: float = 30,
                           min_width: Quantity = Quantity(20, 'arcsec'), use_combined: bool = True,
                           use_worst: bool = False, lo_en: Quantity = Quantity(0.5, 'keV'),
                           hi_en: Quantity = Quantity(2, 'keV'), psf_corr: bool = False, psf_model: str = "ELLBETA",
                           psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, allow_negative: bool = False,
                           exp_corr: bool = True, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                           over_sample: float = None, one_rmf: bool = True, abund_table: str = "angr",
                           num_data_real: int = 300, sigma: int = 1, num_cores: int = NUM_CORES) \
        -> List[GasTemperature3D]:
    """
    This function will generate de-projected, three-dimensional, gas temperature profiles of galaxy clusters using
    the 'onion peeling' deprojection method. It will also generate any projected temperature profiles that may be
    necessary, using the method specified in the function call (the default is the minimum signal to noise annuli
    method). As a side effect of that process APEC normalisation profiles will also be created, as well as Emission
    Measure profiles. The function is an implementation of a fairly old technique, though it has been used recently
    in https://doi.org/10.1051/0004-6361/201731748. For a more in depth discussion of this technique and its uses
    I would currently recommend https://doi.org/10.1051/0004-6361:20020905.

    :param GalaxyCluster/ClusterSample sources: An individual or sample of sources to calculate 3D temperature
        profiles for.
    :param str/Quantity outer_radii: The name or value of the outer radius to use for the generation of
        the spectra (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored. If you are
        generating for multiple sources then you can also pass a Quantity with one entry per source.
    :param str annulus_method: The method by which the annuli are designated, this can be 'min_snr' (which will use
        the min_snr_proj_temp_prof function), or 'growth' (which will use the grow_ann_proj_temp_prof function).
    :param float min_snr: The minimum signal to noise which is allowable in a given annulus.
    :param Quantity min_width: The minimum allowable width of an annulus. The default is set to 20 arcseconds to try
        and avoid PSF effects.
    :param bool use_combined: If True then the combined RateMap will be used for signal to noise annulus
        calculations, this is overridden by use_worst.
    :param bool use_worst: If True then the worst observation of the cluster (ranked by global signal to noise) will
        be used for signal to noise annulus calculations.
    :param Quantity lo_en: The lower energy bound of the ratemap to use for the signal to noise calculations.
    :param Quantity hi_en: The upper energy bound of the ratemap to use for the signal to noise calculations.
    :param bool psf_corr: Sets whether you wish to use a PSF corrected ratemap or not.
    :param str psf_model: If the ratemap you want to use is PSF corrected, this is the PSF model used.
    :param int psf_bins: If the ratemap you want to use is PSF corrected, this is the number of PSFs per
        side in the PSF grid.
    :param str psf_algo: If the ratemap you want to use is PSF corrected, this is the algorithm used.
    :param int psf_iter: If the ratemap you want to use is PSF corrected, this is the number of iterations.
    :param bool allow_negative: Should pixels in the background subtracted count map be allowed to go below
        zero, which results in a lower signal to noise (and can result in a negative signal to noise).
    :param bool exp_corr: Should signal to noises be measured with exposure time correction, default is True. I
            recommend that this be true for combined observations, as exposure time could change quite dramatically
            across the combined product.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param str abund_table: The abundance table to use both for the conversion from n_exn_p to n_e^2 during density
        calculation, and the XSPEC fit.
    :param int num_data_real: The number of random realisations to generate when propagating profile uncertainties.
    :param int sigma: What sigma uncertainties should newly created profiles have, the default is 1σ.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :return: A list of the 3D temperature profiles measured by this function, though if the measurement was not
        successful an entry of None will be added to the list.
    :rtype: List[GasTemperature3D]
    """
    if annulus_method not in ALLOWED_ANN_METHODS:
        a_meth = ", ".join(ALLOWED_ANN_METHODS)
        raise ValueError("That is not a valid method for deciding where to place annuli, please use one of "
                         "these; {}".format(a_meth))

    if annulus_method == 'min_snr':
        # This returns the boundary radii for the annuli
        ann_rads = min_snr_proj_temp_prof(sources, outer_radii, min_snr, min_width, use_combined, use_worst, lo_en,
                                          hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter, allow_negative,
                                          exp_corr, group_spec, min_counts, min_sn, over_sample, one_rmf, abund_table,
                                          num_cores)
    elif annulus_method == "growth":
        raise NotImplementedError("This method isn't implemented yet")

    # So we can iterate through sources without worrying if there's more than one cluster
    if not isinstance(sources, (BaseSample, list)):
        sources = [sources]

    all_3d_temp_profs = []
    # Don't need to check abundance table input because that happens in min_snr_proj_temp_prof
    for src_ind, src in enumerate(sources):
        cur_rads = ann_rads[src_ind]

        try:
            # The projected temperature profile we're going to use
            proj_temp = src.get_proj_temp_profiles(cur_rads, group_spec, min_counts, min_sn, over_sample)
            # The normalisation profile(s) from the fit that produced the projected temperature profile.
            apec_norm_prof = src.get_apec_norm_profiles(cur_rads, group_spec, min_counts, min_sn, over_sample)
        except NoProductAvailableError:
            warn("{s} doesn't have a matching projected temperature profile, skipping.")
            all_3d_temp_profs.append(None)
            continue

        # We need to check if a matching 3D temperature profile has already been generated, as then we
        #  just use that one rather than making another (which would be silly and also eat up more storage
        #  because they are automatically saved to disk).
        try:
            existing_3d_temp_prof = src.get_3d_temp_profiles(set_id=proj_temp.set_ident)
            all_3d_temp_profs.append(existing_3d_temp_prof)
            continue
        except NoProductAvailableError:
            pass

        obs_id = 'combined'
        inst = 'combined'
        # There are reasons that a projected temperature profile can be considered unusable, so we must check. Also
        #  make sure to only use those profiles that have a minimum of 4 annuli. The len operator retrieves the number
        #  of radial data points a profile has
        if proj_temp.usable and len(proj_temp) > 3:
            # Also make an Emission Measure profile, used for weighting the contributions from different
            #  shells to annuli
            em_prof = apec_norm_prof.emission_measure_profile(src.redshift, src.cosmo, abund_table,
                                                              num_data_real, sigma)
            src.update_products(em_prof)

            # Need to make sure the annular boundaries are a) in a proper distance unit rather than degrees, and b)
            #  in units of centimeters
            cur_rads = src.convert_radius(cur_rads, 'cm')
            # Use a handy function I wrote to calculate the volume intersections of spherical shells and
            #  projected annuli
            vol_intersects = shell_ann_vol_intersect(cur_rads, cur_rads)

            # Then its a simple inverse problem to recover the 3D temperatures
            temp_3d = (np.linalg.inv(vol_intersects.T)@(proj_temp.values*em_prof.values)) / (np.linalg.inv(
                vol_intersects.T)@em_prof.values)

            # I generate random realisations of the projected temperature profile and the emission measure profile
            #  to help me with error propagation
            proj_temp_reals = proj_temp.generate_data_realisations(num_data_real)
            em_reals = em_prof.generate_data_realisations(num_data_real)

            # Set up an N x R array for the random realisations of the 3D temperature, where N is the number
            #  of realisations and R is the number of radius data points
            temp_3d_reals = Quantity(np.zeros(proj_temp_reals.shape), proj_temp_reals.unit)
            for i in range(0, num_data_real):
                # Calculate and store the 3D temperature profile realisations
                interim = (np.linalg.inv(vol_intersects.T)@(proj_temp_reals[i, :]*em_reals[i, :])) / (np.linalg.inv(
                    vol_intersects.T)@em_reals[i, :])
                temp_3d_reals[i, :] = interim

            # Calculate a standard deviation for each bin to use as the uncertainty
            temp_3d_sigma = np.std(temp_3d_reals, axis=0)*sigma

            # And finally actually set up a 3D temperature profile
            temp_3d_prof = GasTemperature3D(proj_temp.radii, temp_3d, proj_temp.centre, src.name, obs_id, inst,
                                            proj_temp.radii_err, temp_3d_sigma, proj_temp.set_ident,
                                            proj_temp.associated_set_storage_key, proj_temp.deg_radii)
            src.update_products(temp_3d_prof)
            all_3d_temp_profs.append(temp_3d_prof)

        else:
            warn("The projected temperature profile for {src} is not considered usable by XGA".format(src=src.name))
            all_3d_temp_profs.append(None)

    return all_3d_temp_profs



