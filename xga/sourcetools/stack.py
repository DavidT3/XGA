#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 24/11/2020, 16:09. Copyright (c) David J Turner

from multiprocessing.dummy import Pool
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
from astropy.units import Quantity, pix, kpc
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from ..exceptions import NoRegionsError, NoProductAvailableError
from ..imagetools.profile import radial_brightness
from ..samples.extended import ClusterSample
from ..sas import evselect_spectrum
from ..sources import GalaxyCluster
from ..utils import NUM_CORES, COMPUTE_MODE
from ..xspec.fakeit import cluster_cr_conv
from ..xspec.fit import single_temp_apec


def radial_data_stack(sources: ClusterSample, scale_radius: str = "r200", use_peak: bool = True,
                      pix_step: int = 1, radii: np.ndarray = np.linspace(0.01, 1, 20), min_snr: float = 0.0,
                      lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                      custom_temps: Quantity = None, psf_corr: bool = False, psf_model: str = "ELLBETA",
                      psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, num_cores: int = NUM_CORES) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Creates and scales radial brightness profiles for a set of galaxy clusters so that they can be combined
    and compared, like for like. This particular function does not fit models, and outputs a mean brightness
    profile, as well as the scaled stack data and covariance matrices. This is based on the method in
    https://doi.org/10.1093/mnras/stv1366, though modified to work with profiles rather than 2D images.
    :param List[GalaxyCluster] sources: The source objects that will contribute to the stacked brightness profile.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
    have an entry for this radius.
    :param bool use_peak: Controls whether the peak position is used as the centre of the brightness profile
    for each GalaxyCluster object.
    :param int pix_step: The width (in pixels) of each annular bin for the individual profiles, default is 1.
    :param ndarray radii: The radii (in units of scale_radius) at which to measure and stack surface brightness.
    :param Union[int, float] min_snr: The minimum allowed signal to noise for individual cluster profiles. Default is
    0, which disables automatic rebinning.
    :param Quantity lo_en: The lower energy limit of the data that goes into the stacked profiles.
    :param Quantity hi_en: The upper energy limit of the data that goes into the stacked profiles.
    :param Quantity custom_temps: Temperatures at which to calculate conversion factors for each cluster
    in sources, they will overwrite any temperatures measured by XGA. A single temperature can be passed to be used
    for all clusters in sources. If None, appropriate temperatures will be retrieved from the source objects.
    :param bool psf_corr: If True, PSF corrected ratemaps will be used to make the brightness profile stack.
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    :param int num_cores: The number of cores to use when calculating the brightness profiles, the default is 90%
    of available cores.
    :return: This function returns the average profile, the scaled brightness profiles with the cluster
    changing along the y direction and the bin changing along the x direction, an array of the radii at which the
    brightness was measured (in units of scale_radius), and finally the covariance matrix and normalised
    covariance matrix. I also return a list of source names that WERE included in the stack.
    :rtype: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, list]
    """

    def construct_profile(src_obj: GalaxyCluster, src_id: int, lower: Quantity, upper: Quantity) -> Tuple[Quantity, int]:
        """
        Constructs a brightness profile for the given galaxy cluster, and interpolates to find values
        at the requested radii in units of scale_radius.
        :param GalaxyCluster src_obj: The GalaxyCluster to construct a profile for.
        :param int src_id: An identifier that enables the constructed profile to be placed
        correctly in the results array.
        :param Quantity lower: The lower energy limit to use.
        :param Quantity upper: The higher energy limit to use.
        :return: The scaled profile, the cluster identifier, and the original generated
        surface brightness profile.
        :rtype: Tuple[Quantity, int]
        """
        # The storage key is different based on whether the user wishes to generate profiles from PSF corrected
        #  ratemaps or not.
        if not psf_corr:
            storage_key = "bound_{l}-{u}".format(l=lower.value, u=upper.value)
        else:
            storage_key = "bound_{l}-{u}_{m}_{n}_{a}{i}".format(l=lower.value, u=upper.value, m=psf_model,
                                                                n=psf_bins, a=psf_algo, i=psf_iter)

        # Retrieving the relevant ratemap object, as well as masks
        rt = [r[-1] for r in src_obj.get_products("combined_ratemap", just_obj=False) if storage_key in r][0]

        # The user can choose to use the original user passed coordinates, or the X-ray centroid
        if use_peak:
            pix_peak = rt.coord_conv(src_obj.peak, pix)
            source_mask, background_mask = src_obj.get_mask(scale_radius, central_coord=src_obj.peak)
            # TODO Should I use my source radius mask or just remove interlopers?
            source_mask = src_obj.get_interloper_mask()
        else:
            pix_peak = rt.coord_conv(src_obj.ra_dec, pix)
            source_mask, background_mask = src_obj.get_mask(scale_radius, central_coord=src_obj.ra_dec)
            source_mask = src_obj.get_interloper_mask()

        rad = src_obj.get_radius(scale_radius, kpc)

        # This fetches any profiles that might have already been generated to our required specifications
        prof_prods = src_obj.get_products("combined_brightness_profile")
        if len(prof_prods) == 1:
            matching_profs = [p for p in list(prof_prods[0].values()) if p.check_match(rt, pix_peak, pix_step,
                                                                                       min_snr, rad)]
        else:
            matching_profs = []

        # This is because a ValueError can be raised by radial_brightness when there is a problem with the
        #  background mask
        try:
            if len(matching_profs) == 0:
                sb_prof, success = radial_brightness(rt, source_mask, background_mask, pix_peak, rad, src_obj.redshift,
                                                     pix_step, kpc, src_obj.cosmo, min_snr=min_snr)
                src_obj.update_products(sb_prof)
            elif len(matching_profs) == 1:
                sb_prof = matching_profs[0]
            elif len(matching_profs) > 1:
                raise ValueError("This shouldn't be possible.")
            # Calculates the value of pixel radii in terms of the scale radii
            scaled_radii = (sb_prof.radii / rad).value
            # Interpolating brightness profile values at the radii passed by the user
            interp_brightness = np.interp(radii, scaled_radii, (sb_prof.values - sb_prof.background).value)
        except ValueError as ve:
            # This will mean that the profile is thrown away in a later step
            interp_brightness = np.full(radii.shape, np.NaN)
            # But will also raise a warning so the user knows
            warn(str(ve).replace("you're looking at", "{s} is".format(s=src_obj.name)).replace(".", "")
                 + " - profile set to NaNs.")

        return interp_brightness, src_id

    # This function isn't split out to be submitted to HPC jobs, unlike SAS tasks, so I make sure the num
    #  of cores is set to 1 to minimise resource usage.
    if COMPUTE_MODE != "local":
        num_cores = 1

    # Checking that all the sources are GalaxyClusters
    if not all([isinstance(s, GalaxyCluster) for s in sources]):
        raise TypeError("Currently only GalaxyCluster source objects may be analysed in this way.")

    # Checking that every single GalaxyCluster object was supplied with the scale radius chosen by the user
    if scale_radius.lower() == "r200":
        rad_check = [s.r200 is not None for s in sources]
    elif scale_radius.lower() == "r500":
        rad_check = [s.r500 is not None for s in sources]
    elif scale_radius.lower() == "r2500":
        rad_check = [s.r2500 is not None for s in sources]
    else:
        raise ValueError("{0} is not an acceptable overdensity radius, please use r200, r500, or "
                         "r2500.".format(scale_radius))

    if not all(rad_check):
        raise NoRegionsError("Some GalaxyCluster objects are missing the {} region".format(scale_radius))

    if psf_corr:
        psf_key = "bound_{l}-{u}_{m}_{n}_{a}{i}".format(l=lo_en.value, u=hi_en.value, m=psf_model, n=psf_bins,
                                                        a=psf_algo, i=psf_iter)
        psf_corr_avail = [len(source.get_products("combined_ratemap", extra_key=psf_key)) != 0 for source in sources]
        if False in psf_corr_avail:
            raise NoProductAvailableError("At least one source does not have PSF corrected "
                                          "image products available.")

    sb = np.zeros((len(sources), len(radii)))
    # Sets up a multiprocessing pool
    with tqdm(total=len(sources), desc="Generating Brightness Profiles") as onwards, Pool(num_cores) as pool:
        def callback(results):
            nonlocal sb
            nonlocal onwards
            b, s_id = results
            sb[s_id, :] = b
            onwards.update(1)

        def err_callback(err):
            onwards.update()
            raise err

        for s_ind, s in enumerate(sources):
            pool.apply_async(construct_profile, callback=callback, error_callback=err_callback,
                             args=(s, s_ind, lo_en, hi_en))
        pool.close()
        pool.join()
        onwards.close()

    # Now, we have all the brightness values at common radii (in units of R200 so scaled properly), now we have
    #  to weight the SB values so they are directly comparable. This accounts for redshift, nH, and sort-of for
    #  the temperature of each cluster.
    # First must make sure we have generated spectra for all the clusters, as we need the ARFs and RMFs to
    #  simulate spectra and calculate conversion values
    evselect_spectrum(sources, scale_radius)  # Use our standard setting for spectra generation
    # Use a simple single_temp_apec to fit said spectra, with all the default settings
    single_temp_apec(sources, reg_type=scale_radius)

    # Calculate all the conversion factors
    if custom_temps is not None:
        cluster_cr_conv(sources, scale_radius, custom_temps)
    else:
        temps = Quantity([source.get_temperature(scale_radius, "tbabs*apec")[0] for source in sources], 'keV')
        cluster_cr_conv(sources, scale_radius, temps)

    combined_factors = []
    # Now to generate a combined conversion factor from count rate to luminosity
    for source in sources:
        combined_factors.append(source.combined_lum_conv_factor(scale_radius, lo_en, hi_en).value)

    # Check for NaN values in the brightness profiles we've retrieved - very bad if they exist
    no_nan = np.where(~np.isnan(sb.sum(axis=1)))[0]
    # Selects only those clusters that don't have nans in their brightness profiles
    combined_factors = np.array(combined_factors)[no_nan]

    # Multiplies each cluster profile by the matching conversion factor to go from countrate to luminosity
    luminosity = (sb[no_nan, :].T * combined_factors).T

    # Finds the highest value in the profile of each cluster
    max_lums = np.max(luminosity, axis=1)
    # Finds the mean of the maximum values and calculates scaling factors so that the maximum
    #  value in each profile is now equal to the average
    scale_factors = max_lums.mean() / max_lums
    # Applied the rescaling factors
    scaled_luminosity = (luminosity.T * scale_factors).T

    # Calculates normalised and the usual covariance matrices
    norm_cov = np.corrcoef(scaled_luminosity, rowvar=False)
    cov = np.cov(scaled_luminosity, rowvar=False)

    average_profile = np.mean(scaled_luminosity, axis=0)
    stack_names = []
    for src_ind, src in enumerate(sources):
        if src_ind not in no_nan:
            warn("A NaN value was detected in {}'s brightness profile, and as such it has been excluded from the "
                 "stack.".format(src.name))
        else:
            stack_names.append(src.name)
    return average_profile, scaled_luminosity, radii, cov, norm_cov, stack_names


def view_radial_data_stack(sources: ClusterSample, scale_radius: str = "r200", use_peak: bool = True,
                           pix_step: int = 1, radii: np.ndarray = np.linspace(0.01, 1, 20),
                           min_snr: Union[int, float] = 0.0, lo_en: Quantity = Quantity(0.5, 'keV'),
                           hi_en: Quantity = Quantity(2.0, 'keV'), custom_temps: Quantity = None,
                           psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4,
                           psf_algo: str = "rl", psf_iter: int = 15, num_cores: int = NUM_CORES,
                           show_images: bool = False, figsize: tuple = (14, 14)):
    """
    A convenience function that calls radial_data_stack and makes plots of the average profile, individual profiles,
    covariance, and normalised covariance matrix.
    :param List[GalaxyCluster] sources: The source objects that will contribute to the stacked brightness profile.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
    have an entry for this radius.
    :param bool use_peak: Controls whether the peak position is used as the centre of the brightness profile
    for each GalaxyCluster object.
    :param int pix_step: The width (in pixels) of each annular bin for the individual profiles, default is 1.
    :param ndarray radii: The radii (in units of scale_radius) at which to measure and stack surface brightness.
    :param Union[int, float] min_snr: The minimum allowed signal to noise for individual cluster profiles. Default is
    0, which disables automatic rebinning.
    :param Quantity lo_en: The lower energy limit of the data that goes into the stacked profiles.
    :param Quantity hi_en: The upper energy limit of the data that goes into the stacked profiles.
    :param Quantity custom_temps: Temperatures at which to calculate conversion factors for each cluster
    in sources, they will overwrite any temperatures measured by XGA. A single temperature can be passed to be used
    for all clusters in sources. If None, appropriate temperatures will be retrieved from the source objects.
    :param bool psf_corr: If True, PSF corrected ratemaps will be used to make the brightness profile stack.
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    :param int num_cores: The number of cores to use when calculating the brightness profiles, the default is 90%
    of available cores.
    :param bool show_images: If true then for each source in the stack an image and profile will be displayed
    side by side, with annuli overlaid on the image.
    :param tuple figsize: The desired figure size for the plot.
    """
    # Calls the stacking function
    results = radial_data_stack(sources, scale_radius, use_peak, pix_step, radii, min_snr, lo_en, hi_en,
                                custom_temps, psf_corr, psf_model, psf_bins, psf_algo, psf_iter, num_cores)

    # Gets the average profile from the results
    av_prof = results[0]
    # Gets the individual scaled profiles from results
    all_prof = results[1]

    # The covariance matrix
    cov = results[3]
    # The normalised covariance matrix
    norm_cov = results[4]
    # Finds the standard deviations by diagonalising the covariance matrix and taking the sqrt
    # TODO This gives a slightly different answer to np.std for some reason
    sd = np.sqrt(np.diagonal(cov))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    ax[0, 0].set_title("Average Profile")
    ax[0, 0].set_xlabel("Radius [{}]".format(scale_radius))
    ax[0, 1].set_title("All Profiles")
    ax[0, 1].set_xlabel("Radius [{}]".format(scale_radius))

    ax[0, 0].plot(radii, av_prof, color="black", label="Average Profile")
    ax[0, 0].errorbar(radii, av_prof, fmt="kx", yerr=sd, capsize=2)
    for i in range(0, all_prof.shape[0]):
        ax[0, 1].plot(radii, all_prof[i, :])

    ax[0, 0].set_xscale("log")
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_yscale("log")

    ax[0, 0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0, 1].xaxis.set_major_formatter(ScalarFormatter())

    ax[1, 0].set_title("Covariance Matrix")
    ax[1, 0].tick_params(axis='both', direction='in', which='both', top=False, right=False)
    ax[1, 0].xaxis.set_ticklabels([])
    ax[1, 0].yaxis.set_ticklabels([])
    im = ax[1, 0].imshow(cov, cmap="gnuplot2", origin="lower")
    fig.colorbar(im, ax=ax[1, 0])

    ax[1, 1].set_title("Normalised Covariance Matrix")
    ax[1, 1].tick_params(axis='both', direction='in', which='both', top=False, right=False)
    ax[1, 1].xaxis.set_ticklabels([])
    ax[1, 1].yaxis.set_ticklabels([])
    im = ax[1, 1].imshow(norm_cov, cmap="gnuplot2", origin="lower")
    fig.colorbar(im, ax=ax[1, 1])

    fig.tight_layout()
    plt.show()

    if show_images:
        for name_ind, name in enumerate(results[5]):
            cur_src = sources[name]
            if not psf_corr:
                storage_key = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
            else:
                storage_key = "bound_{l}-{u}_{m}_{n}_{a}{i}".format(l=lo_en.value, u=hi_en.value, m=psf_model,
                                                                    n=psf_bins, a=psf_algo, i=psf_iter)

            rt = cur_src.get_products('combined_ratemap', extra_key=storage_key)[0]

            # The user can choose to use the original user passed coordinates, or the X-ray centroid
            if use_peak:
                pix_peak = rt.coord_conv(cur_src.peak, pix)
            else:
                pix_peak = rt.coord_conv(cur_src.ra_dec, pix)
            inter_mask = cur_src.get_interloper_mask()
            rad = cur_src.get_radius(scale_radius, kpc)

            prof_prods = cur_src.get_products("combined_brightness_profile")
            matching_profs = [p for p in list(prof_prods[0].values()) if p.check_match(rt, pix_peak, pix_step,
                                                                                       min_snr, rad)]
            pr = matching_profs[0]
            fig, ax_arr = plt.subplots(ncols=2, figsize=(figsize[0], figsize[0]*0.5))

            plt.sca(ax_arr[0])

            multiplier = 1.3
            custom_xlims = (pr.centre[0].value - pr.pixel_bins[-1]*multiplier,
                            pr.centre[0].value + pr.pixel_bins[-1]*multiplier)
            custom_ylims = (pr.centre[1].value - pr.pixel_bins[-1]*multiplier,
                            pr.centre[1].value + pr.pixel_bins[-1]*multiplier)
            # This populates ones of the axes with a view of the image
            im_ax = rt.get_view(ax_arr[0], pr.centre, inter_mask, radial_bins_pix=pr.pixel_bins, zoom_in=True,
                                manual_zoom_xlims=custom_xlims, manual_zoom_ylims=custom_ylims)

            ax_arr[1].set_xscale("log")
            ax_arr[1].set_yscale("log")
            ax_arr[1].xaxis.set_major_formatter(ScalarFormatter())
            ax_arr[1].plot(radii, all_prof[name_ind, :])
            ax_arr[1].set_xlabel("Radius [{}]".format(scale_radius))
            ax_arr[1].set_title("{} - Luminosity Profile".format(cur_src.name))
            ax_arr[1].set_ylabel("L$_x$ [erg$s^{-1}$]")

            plt.tight_layout()
            plt.show()
            plt.close('all')





