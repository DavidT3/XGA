#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

from multiprocessing.dummy import Pool
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
from astropy.units import Quantity, pix, kpc
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from ..exceptions import NoRegionsError, NoProductAvailableError, XGAFitError, ModelNotAssociatedError, \
    ParameterNotAssociatedError
from ..imagetools.profile import radial_brightness
from ..samples.extended import ClusterSample
from ..sources import GalaxyCluster
from ..utils import NUM_CORES
from ..xspec.fakeit import cluster_cr_conv
from ..xspec.fit import single_temp_apec


def _stack_setup_checks(sources: ClusterSample, scale_radius: str = "r200", lo_en: Quantity = Quantity(0.5, 'keV'),
                        hi_en: Quantity = Quantity(2.0, 'keV'), psf_corr: bool = False, psf_model: str = "ELLBETA",
                        psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15):
    """
    Internal function that was originally split off from radial data stack. This performs checks to make sure passed
    in values are valid for all types of stacking available in this part of XGA.

    :param ClusterSample sources: The source objects that will contribute to the stacked brightness profile.
    :param str scale_radius: The over-density radius to scale the cluster radii by, all GalaxyCluster objects must
        have an entry for this radius.
    :param Quantity lo_en: The lower energy limit of the data that goes into the stacked profiles.
    :param Quantity hi_en: The upper energy limit of the data that goes into the stacked profiles.
    :param bool psf_corr: If True, PSF corrected ratemaps will be used to make the brightness profile stack.
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    """
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


def _create_stack(sb: np.ndarray, sources: ClusterSample, scale_radius: str, lo_en: Quantity, hi_en: Quantity,
                  custom_temps: Quantity, sim_met: Union[float, List] = 0.3, abund_table: str = 'angr') \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Internal function that was originally split off from radial data stack. Takes the surface brightness profiles
    that have been generated for radii as a fraction of the scale radius. It then calculates the scaling factors and
    combines them into a single stacked profile.

    :param np.ndarray sb: The surface brightness data output for all the sources.
    :param ClusterSample sources: The source objects that will contribute to the stacked brightness profile.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
        have an entry for this radius.
    :param Quantity lo_en: The lower energy limit of the data that goes into the stacked profiles.
    :param Quantity hi_en: The upper energy limit of the data that goes into the stacked profiles.
    :param Quantity custom_temps: Temperatures at which to calculate conversion factors for each cluster
        in sources, they will overwrite any temperatures measured by XGA. A single temperature can be passed to be used
        for all clusters in sources. If None, appropriate temperatures will be retrieved from the source objects.
    :param float/List sim_met: The metallicity(s) to use when calculating the conversion factor. Pass a
        single float to use the same value for all sources, or pass a list to use a different value for each.
    :param str abund_table: The abundance table to use for the temperature fit and conversion factor calculation.
    :return: The average profile, all scaled profiles, the covariance matrix, normalised covariance, and names
        of successful profiles.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]
    """
    # Now, we have all the brightness values at common radii (in units of R200 so scaled properly), now we have
    #  to weight the SB values so they are directly comparable. This accounts for redshift, nH, and sort-of for
    #  the temperature of each cluster.

    # Calculate all the conversion factors
    if custom_temps is not None:
        # I'm not going to give the user the ability to choose the specifics of the spectra that are
        #  being used to calculate conversion factors - I'll use standard settings. This function will
        #  also make sure that they've actually been generated.
        cluster_cr_conv(sources, scale_radius, custom_temps, sim_met=sim_met, abund_table=abund_table)
    else:
        # Use a simple single_temp_apec to fit said spectra, but only if we haven't had custom temperatures
        #  passed in
        single_temp_apec(sources, scale_radius, abund_table=abund_table)
        temp_temps = []
        for src in sources:
            try:
                # A temporary temperature variable
                temp_temp = src.get_temperature(scale_radius, "constant*tbabs*apec")[0]
            except (ModelNotAssociatedError, ParameterNotAssociatedError):
                warn("{s}'s temperature fit is not valid, so I am defaulting to a temperature of "
                     "3keV".format(s=src.name))
                temp_temp = Quantity(3, 'keV')

            temp_temps.append(temp_temp.value)
        temps = Quantity(temp_temps, 'keV')
        cluster_cr_conv(sources, scale_radius, sim_temp=temps, sim_met=sim_met, abund_table=abund_table)

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

    return average_profile, scaled_luminosity, cov, norm_cov, stack_names


def _view_stack(results: Tuple, scale_radius: str, radii: np.ndarray, figsize: Tuple):
    """
    Internal function to plot the results of a stack function.

    :param Tuple results: The results tuple from a stack function, this is what will be plotted.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
        have an entry for this radius.
    :param ndarray radii: The radii (in units of scale_radius) at which to measure and stack surface brightness.
    :param tuple figsize: The desired figure size for the plot.
    """
    # Gets the average profile from the results
    av_prof = results[0]
    # Gets the individual scaled profiles from results
    all_prof = results[1]

    # The covariance matrix
    cov = results[3]
    # The normalised covariance matrix
    norm_cov = results[4]
    # Finds the standard deviations by diagonalising the covariance matrix and taking the sqrt
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


def radial_data_stack(sources: ClusterSample, scale_radius: str = "r200", use_peak: bool = True,
                      pix_step: int = 1, radii: np.ndarray = np.linspace(0.01, 1, 20), min_snr: float = 0.0,
                      lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                      custom_temps: Quantity = None, sim_met: Union[float, List] = 0.3,
                      abund_table: str = 'angr', psf_corr: bool = False, psf_model: str = "ELLBETA",
                      psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, num_cores: int = NUM_CORES) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Creates and scales radial brightness profiles for a set of galaxy clusters so that they can be combined
    and compared, like for like. This particular function does not fit models, and outputs a mean brightness
    profile, as well as the scaled stack data and covariance matrices. This is based on the method in
    https://doi.org/10.1093/mnras/stv1366, though modified to work with profiles rather than 2D images.

    :param ClusterSample sources: The source objects that will contribute to the stacked brightness profile.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
        have an entry for this radius.
    :param bool use_peak: Controls whether the peak position is used as the centre of the brightness profile
        for each GalaxyCluster object.
    :param int pix_step: The width (in pixels) of each annular bin for the individual profiles, default is 1.
    :param ndarray radii: The radii (in units of scale_radius) at which to measure and stack surface brightness.
    :param int/float min_snr: The minimum allowed signal to noise for individual cluster profiles. Default is
        0, which disables automatic rebinning.
    :param Quantity lo_en: The lower energy limit of the data that goes into the stacked profiles.
    :param Quantity hi_en: The upper energy limit of the data that goes into the stacked profiles.
    :param Quantity custom_temps: Temperatures at which to calculate conversion factors for each cluster
        in sources, they will overwrite any temperatures measured by XGA. A single temperature can be passed to be
        used for all clusters in sources. If None, appropriate temperatures will be retrieved from the source objects.
    :param float/List sim_met: The metallicity(s) to use when calculating the conversion factor. Pass a
        single float to use the same value for all sources, or pass a list to use a different value for each.
    :param str abund_table: The abundance table to use for the temperature fit and conversion factor calculation.
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

    def construct_profile(src_obj: GalaxyCluster, src_id: int, lower: Quantity, upper: Quantity) \
            -> Tuple[Quantity, int]:
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
            central_coord = src_obj.peak
        else:
            central_coord = src_obj.ra_dec

        # We obviously want to remove point sources from the profiles we make, so get the mask that removes
        #  interlopers
        int_mask = src_obj.get_interloper_mask()

        # Tells the source object to give us the requested scale radius in units of kpc
        rad = src_obj.get_radius(scale_radius, kpc)

        # This fetches any profiles that might have already been generated to our required specifications
        prof_prods = src_obj.get_products("combined_brightness_profile")
        if len(prof_prods) == 1:
            matching_profs = [p for p in list(prof_prods[0].values()) if p.check_match(rt, central_coord, pix_step,
                                                                                       min_snr, rad)]
        else:
            matching_profs = []

        # This is because a ValueError can be raised by radial_brightness when there is a problem with the
        #  background mask
        try:
            if len(matching_profs) == 0:
                sb_prof, success = radial_brightness(rt, central_coord, rad, float(src_obj.background_radius_factors[0]),
                                                     float(src_obj.background_radius_factors[1]), int_mask,
                                                     src_obj.redshift, pix_step, kpc, src_obj.cosmo, min_snr)
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

    # This is an internal function that does setup checks common to both stacking of data and models
    _stack_setup_checks(sources, scale_radius, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)

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

    average_profile, scaled_luminosity, cov, norm_cov, stack_names = _create_stack(sb, sources, scale_radius, lo_en,
                                                                                   hi_en, custom_temps, sim_met,
                                                                                   abund_table)

    return average_profile, scaled_luminosity, radii, cov, norm_cov, stack_names


def view_radial_data_stack(sources: ClusterSample, scale_radius: str = "r200", use_peak: bool = True,
                           pix_step: int = 1, radii: np.ndarray = np.linspace(0.01, 1, 20),
                           min_snr: Union[int, float] = 0.0, lo_en: Quantity = Quantity(0.5, 'keV'),
                           hi_en: Quantity = Quantity(2.0, 'keV'), custom_temps: Quantity = None,
                           sim_met: Union[float, List] = 0.3, abund_table: str = 'angr',
                           psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4,
                           psf_algo: str = "rl", psf_iter: int = 15, num_cores: int = NUM_CORES,
                           show_images: bool = False, figsize: tuple = (14, 14)):
    """
    A convenience function that calls radial_data_stack and makes plots of the average profile, individual profiles,
    covariance, and normalised covariance matrix.

    :param ClusterSample sources: The source objects that will contribute to the stacked brightness profile.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
        have an entry for this radius.
    :param bool use_peak: Controls whether the peak position is used as the centre of the brightness profile
        for each GalaxyCluster object.
    :param int pix_step: The width (in pixels) of each annular bin for the individual profiles, default is 1.
    :param ndarray radii: The radii (in units of scale_radius) at which to measure and stack surface brightness.
    :param int/float min_snr: The minimum allowed signal to noise for individual cluster profiles. Default is
        0, which disables automatic rebinning.
    :param Quantity lo_en: The lower energy limit of the data that goes into the stacked profiles.
    :param Quantity hi_en: The upper energy limit of the data that goes into the stacked profiles.
    :param Quantity custom_temps: Temperatures at which to calculate conversion factors for each cluster
        in sources, they will overwrite any temperatures measured by XGA. A single temperature can be passed to be used
        for all clusters in sources. If None, appropriate temperatures will be retrieved from the source objects.
    :param float/List sim_met: The metallicity(s) to use when calculating the conversion factor. Pass a
        single float to use the same value for all sources, or pass a list to use a different value for each.
    :param str abund_table: The abundance table to use for the temperature fit and conversion factor calculation.
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
                                custom_temps, sim_met, abund_table, psf_corr, psf_model, psf_bins, psf_algo,
                                psf_iter, num_cores)

    # Gets the individual scaled profiles from results
    all_prof = results[1]

    # Call this internal function that contains all the plotting code. I've set it up this way because there
    #  is another stacking method and viewing function - and code duplication is a very serious crime!
    _view_stack(results, scale_radius, radii, figsize)

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
            fig, ax_arr = plt.subplots(ncols=2, figsize=(figsize[0], figsize[0] * 0.5))

            plt.sca(ax_arr[0])

            multiplier = (pr.back_pixel_bin[-1] / pr.pixel_bins[-1]) * 1.05
            custom_xlims = (pr.centre[0].value - pr.pixel_bins[-1] * multiplier,
                            pr.centre[0].value + pr.pixel_bins[-1] * multiplier)
            custom_ylims = (pr.centre[1].value - pr.pixel_bins[-1] * multiplier,
                            pr.centre[1].value + pr.pixel_bins[-1] * multiplier)
            # This populates ones of the axes with a view of the image
            im_ax = rt.get_view(ax_arr[0], pr.centre, inter_mask, radial_bins_pix=pr.pixel_bins,
                                back_bin_pix=pr.back_pixel_bin, zoom_in=True, manual_zoom_xlims=custom_xlims,
                                manual_zoom_ylims=custom_ylims)

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


def radial_model_stack(sources: ClusterSample, model: str, scale_radius: str = "r200", fit_method: str = 'mcmc',
                       use_peak: bool = True, model_priors: list = None, model_start_pars: list = None,
                       pix_step: int = 1, radii: np.ndarray = np.linspace(0.01, 1, 20), min_snr: float = 0.0,
                       lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                       custom_temps: Quantity = None, sim_met: Union[float, List] = 0.3, abund_table: str = 'angr',
                       psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                       psf_iter: int = 15, num_cores: int = NUM_CORES, model_realisations: int = 500,
                       conf_level: int = 90, num_walkers: int = 20, num_steps: int = 20000) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Creates, fits, and scales radial brightness profiles for a set of galaxy clusters so that they can be combined
    and compared, like for like. This function fits models of a user's choice, and then uses the models to retrieve
    brightness values at user-defined radii as a fraction of the scale radius. From that point it functions much
    as radial_data_stack does.

    :param ClusterSample sources: The source objects that will contribute to the stacked brightness profile.
    :param str model: The model to fit to the brightness profiles.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
        have an entry for this radius.
    :param str fit_method: The method to use when fitting the model to the profile.
    :param bool use_peak: Controls whether the peak position is used as the centre of the brightness profile
        for each GalaxyCluster object.
    :param list model_priors: A list of priors to use when fitting the model with MCMC, default is None in which
        case the default priors for the selected model are used.
    :param list model_start_pars: A list of start parameters to use when fitting with methods like curve_fit, default
        is None in which case the default start parameters for the selected model are used.
    :param int pix_step: The width (in pixels) of each annular bin for the individual profiles, default is 1.
    :param ndarray radii: The radii (in units of scale_radius) at which to measure and stack surface brightness.
    :param int/float min_snr: The minimum allowed signal to noise for individual cluster profiles. Default is
        0, which disables automatic rebinning.
    :param Quantity lo_en: The lower energy limit of the data that goes into the stacked profiles.
    :param Quantity hi_en: The upper energy limit of the data that goes into the stacked profiles.
    :param Quantity custom_temps: Temperatures at which to calculate conversion factors for each cluster
        in sources, they will overwrite any temperatures measured by XGA. A single temperature can be passed to be used
        for all clusters in sources. If None, appropriate temperatures will be retrieved from the source objects.
    :param float/List sim_met: The metallicity(s) to use when calculating the conversion factor. Pass a
        single float to use the same value for all sources, or pass a list to use a different value for each.
    :param str abund_table: The abundance table to use for the temperature fit and conversion factor calculation.
    :param bool psf_corr: If True, PSF corrected ratemaps will be used to make the brightness profile stack.
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    :param int num_cores: The number of cores to use when calculating the brightness profiles, the default is 90%
        of available cores.
    :param int model_realisations: The number of random realisations of a model to generate.
    :param int conf_level: The confidence level at which to measure uncertainties of parameters and profiles.
    :param int num_walkers: The number of walkers in the MCMC ensemble sampler.
    :param int num_steps: The number of steps in the chain that each walker should take.
    :return: This function returns the average profile, the scaled brightness profiles with the cluster
        changing along the y direction and the bin changing along the x direction, an array of the radii at which the
        brightness was measured (in units of scale_radius), and finally the covariance matrix and normalised
        covariance matrix. I also return a list of source names that WERE included in the stack.
    :rtype: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, list]
    """

    def construct_profile(src_obj: GalaxyCluster, src_id: int, lower: Quantity, upper: Quantity) \
            -> Tuple[Quantity, int]:
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
            central_coord = src_obj.peak
        else:
            central_coord = src_obj.ra_dec

        # We obviously want to remove point sources from the profiles we make, so get the mask that removes
        #  interlopers
        int_mask = src_obj.get_interloper_mask()

        # Tells the source object to give us the requested scale radius in units of kpc
        rad = src_obj.get_radius(scale_radius, kpc)

        # This fetches any profiles that might have already been generated to our required specifications
        prof_prods = src_obj.get_products("combined_brightness_profile")
        if len(prof_prods) == 1:
            matching_profs = [p for p in list(prof_prods[0].values()) if p.check_match(rt, central_coord, pix_step,
                                                                                       min_snr, rad)]
        else:
            matching_profs = []

        # This is because a ValueError can be raised by radial_brightness when there is a problem with the
        #  background mask
        try:
            if len(matching_profs) == 0:
                sb_prof, success = radial_brightness(rt, central_coord, rad,
                                                     float(src_obj.background_radius_factors[0]),
                                                     float(src_obj.background_radius_factors[1]), int_mask,
                                                     src_obj.redshift, pix_step, kpc, src_obj.cosmo, min_snr)
                src_obj.update_products(sb_prof)
            elif len(matching_profs) == 1:
                sb_prof = matching_profs[0]
            elif len(matching_profs) > 1:
                raise ValueError("This shouldn't be possible.")

            # The model was not fit in terms of the scale radius, so I need to convert the chosen global
            #  radii to values I can pass into the model
            model_radii = radii * src_obj.get_radius(scale_radius, kpc)

            sb_prof.fit(model, progress_bar=False, show_errors=False, method=fit_method, priors=model_priors,
                        start_pars=model_start_pars, conf_level=conf_level, num_walkers=num_walkers,
                        num_steps=num_steps, model_real=model_realisations)
            try:
                fitted_model = sb_prof.get_model_fit(model)
                model_brightness = fitted_model['model_func'](model_radii.value, *fitted_model['par'])

            except XGAFitError:
                model_brightness = np.full(radii.shape, np.NaN)
                warn('Model fit for {s} failed - profile set to NaNs'.format(s=src_obj.name))

        except ValueError as ve:
            # This will mean that the profile is thrown away in a later step
            model_brightness = np.full(radii.shape, np.NaN)
            # But will also raise a warning so the user knows
            warn(str(ve).replace("you're looking at", "{s} is".format(s=src_obj.name)).replace(".", "")
                 + " - profile set to NaNs.")

        return model_brightness, src_id

    # This is an internal function that does setup checks common to both stacking of data and models
    _stack_setup_checks(sources, scale_radius, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)

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

    average_profile, scaled_luminosity, cov, norm_cov, stack_names = _create_stack(sb, sources, scale_radius, lo_en,
                                                                                   hi_en, custom_temps, sim_met,
                                                                                   abund_table)

    return average_profile, scaled_luminosity, radii, cov, norm_cov, stack_names


def view_radial_model_stack(sources: ClusterSample, model: str, scale_radius: str = "r200", fit_method: str = 'mcmc',
                            use_peak: bool = True, model_priors: List = None, model_start_pars: list = None,
                            pix_step: int = 1, radii: np.ndarray = np.linspace(0.01, 1, 20), min_snr: float = 0.0,
                            lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                            custom_temps: Quantity = None, sim_met: Union[float, List] = 0.3,
                            abund_table: str = 'angr', psf_corr: bool = False, psf_model: str = "ELLBETA",
                            psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, num_cores: int = NUM_CORES,
                            model_realisations: int = 500, conf_level: int = 90, ml_mcmc_start: bool = True,
                            ml_rand_dev: float = 1e-4, num_walkers: int = 30, num_steps: int = 20000,
                            show_images: bool = False, figsize: tuple = (14, 14)):
    """
    A convenience function that calls radial_model_stack and makes plots of the average profile, individual profiles,
    covariance, and normalised covariance matrix.

    :param ClusterSample sources: The source objects that will contribute to the stacked brightness profile.
    :param str model: The model to fit to the brightness profiles.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
        have an entry for this radius.
    :param str fit_method: The method to use when fitting the model to the profile.
    :param bool use_peak: Controls whether the peak position is used as the centre of the brightness profile
        for each GalaxyCluster object.
    :param list model_priors: A list of priors to use when fitting the model with MCMC, default is None in which
        case the default priors for the selected model are used.
    :param list model_start_pars: A list of start parameters to use when fitting with methods like curve_fit, default
        is None in which case the default start parameters for the selected model are used.
    :param int pix_step: The width (in pixels) of each annular bin for the individual profiles, default is 1.
    :param ndarray radii: The radii (in units of scale_radius) at which to measure and stack surface brightness.
    :param int/float min_snr: The minimum allowed signal to noise for individual cluster profiles. Default is
        0, which disables automatic rebinning.
    :param Quantity lo_en: The lower energy limit of the data that goes into the stacked profiles.
    :param Quantity hi_en: The upper energy limit of the data that goes into the stacked profiles.
    :param Quantity custom_temps: Temperatures at which to calculate conversion factors for each cluster
        in sources, they will overwrite any temperatures measured by XGA. A single temperature can be passed to be used
        for all clusters in sources. If None, appropriate temperatures will be retrieved from the source objects.
    :param float/List sim_met: The metallicity(s) to use when calculating the conversion factor. Pass a
        single float to use the same value for all sources, or pass a list to use a different value for each.
    :param str abund_table: The abundance table to use for the temperature fit and conversion factor calculation.
    :param bool psf_corr: If True, PSF corrected ratemaps will be used to make the brightness profile stack.
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    :param int num_cores: The number of cores to use when calculating the brightness profiles, the default is 90%
        of available cores.
    :param int model_realisations: The number of random realisations of a model to generate.
    :param int conf_level: The confidence level at which to measure uncertainties of parameters and profiles.
    :param bool ml_mcmc_start: If True then maximum likelihood estimation will be used to generate start parameters for
        MCMC fitting, otherwise they will be randomly drawn from parameter priors
    :param float ml_rand_dev: The scale of the random deviation around start parameters used for starting the
        different walkers in the MCMC ensemble sampler.
    :param int num_walkers: The number of walkers in the MCMC ensemble sampler.
    :param int num_steps: The number of steps in the chain that each walker should take.
    :param bool show_images: If true then for each source in the stack an image and profile will be displayed
        side by side, with annuli overlaid on the image.
    :param tuple figsize: The desired figure size for the plot.
    """
    # Calls the stacking function
    results = radial_model_stack(sources, model, scale_radius, fit_method, use_peak, model_priors, model_start_pars,
                                 pix_step, radii, min_snr, lo_en, hi_en, custom_temps, sim_met, abund_table, psf_corr,
                                 psf_model, psf_bins, psf_algo, psf_iter, num_cores, model_realisations, conf_level,
                                 ml_mcmc_start, ml_rand_dev, num_walkers, num_steps)

    # Gets the individual scaled profiles from results
    all_prof = results[1]

    # Call this internal function that contains all the plotting code. I've set it up this way because there
    #  is another stacking method and viewing function - and code duplication is a very serious crime!
    _view_stack(results, scale_radius, radii, figsize)

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
            fig, ax_arr = plt.subplots(ncols=3, figsize=(figsize[0], figsize[0] * 0.32))

            plt.sca(ax_arr[0])
            multiplier = (pr.back_pixel_bin[-1] / pr.pixel_bins[-1]) * 1.05
            custom_xlims = (pr.centre[0].value - pr.pixel_bins[-1] * multiplier,
                            pr.centre[0].value + pr.pixel_bins[-1] * multiplier)
            custom_ylims = (pr.centre[1].value - pr.pixel_bins[-1] * multiplier,
                            pr.centre[1].value + pr.pixel_bins[-1] * multiplier)
            # This populates ones of the axes with a view of the image
            im_ax = rt.get_view(ax_arr[0], pr.centre, inter_mask, radial_bins_pix=pr.pixel_bins,
                                back_bin_pix=pr.back_pixel_bin, zoom_in=True, manual_zoom_xlims=custom_xlims,
                                manual_zoom_ylims=custom_ylims)

            ax_arr[1].set_xscale("log")
            ax_arr[1].set_yscale("log")
            ax_arr[1].xaxis.set_major_formatter(ScalarFormatter())
            ax_arr[1].plot(radii, all_prof[name_ind, :])
            ax_arr[1].set_xlabel("Radius [{}]".format(scale_radius))
            ax_arr[1].set_title("{} - Luminosity Profile".format(cur_src.name))
            ax_arr[1].set_ylabel("L$_x$ [erg$s^{-1}$]")

            # This plots a basic representation of the SB data and the model fit, for validation purposes
            ax_arr[2].set_xscale("log")
            ax_arr[2].set_yscale("log")
            ax_arr[2].xaxis.set_major_formatter(ScalarFormatter())
            ax_arr[2].set_xlabel("Radius [{}]".format(pr.radii.unit.to_string()))
            ax_arr[2].set_title("{} - Fitted Surface Brightness Profile".format(cur_src.name))
            y_unit = r"$\left[" + pr.values_unit.to_string("latex").strip("$") + r"\right]$"
            ax_arr[2].set_ylabel("Surface Brightness " + y_unit)

            ax_arr[2].errorbar(pr.radii.value, pr.values.value-pr.background.value, xerr=pr.radii_err.value,
                               yerr=pr.values_err.value, fmt="x", capsize=2,)

            mod_fit = pr.get_model_fit(model)
            mod_plot_radii = np.linspace(pr.radii.value[0], pr.radii.value[-1], 300)
            ax_arr[2].plot(mod_plot_radii, mod_fit['model_func'](mod_plot_radii, *mod_fit['par']))

            plt.tight_layout()
            plt.show()
            plt.close('all')





