#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 25/08/2020, 12:57. Copyright (c) David J Turner

from multiprocessing.dummy import Pool
from typing import List, Tuple

from astropy.units import Quantity, pix
from numpy import ndarray, linspace, zeros, interp
from tqdm import tqdm

from xga.exceptions import NoRegionsError
from xga.imagetools.profile import radial_brightness
from xga.sas import evselect_spectrum
from xga.sources import GalaxyCluster
from xga.utils import NUM_CORES, COMPUTE_MODE
from xga.xspec.fakeit import cluster_cr_conv


# TODO As I currently have to do a background region, I'll only go out to the scale radius as the outer boundary
#  However if Paul's background model works out, I may be able to abandon that.
def radial_data_stack(sources: List[GalaxyCluster], scale_radius: str = "r200", use_peak: bool = True,
                      radii: ndarray = linspace(0, 1, 20), psf_corr: bool = False, psf_model: str = "ELLBETA",
                      psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15,
                      num_cores: int = NUM_CORES) -> Tuple[ndarray]:
    """

    :param List[GalaxyCluster] sources: The source objects that will contribute to the stacked brightness profile.
    :param str scale_radius: The overdensity radius to scale the cluster radii by, all GalaxyCluster objects must
    have an entry for this radius.
    :param bool use_peak: Controls whether the peak position is used as the centre of the brightness profile for each
    GalaxyCluster object.
    :param ndarray radii: The radii (in units of scale_radius) at which to measure and stack surface brightness.
    :param bool psf_corr: If True, PSF corrected ratemaps will be used to make the brightness profile stack.
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    :param int num_cores: The number of cores to use when calculating the brightness profiles, the default is 90%
    of available cores.
    :return: The brightness and radius arrays.
    :rtype: Tuple[ndarray]
    """

    def construct_profile(src, src_id, rad_type, lo_en, hi_en, use_p, rads, p_corr: bool = False,
                          p_model: str = "ELLBETA", p_bins: int = 4, p_algo: str = "rl", p_iter: int = 15):
        """

        """
        if not p_corr:
            storage_key = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
        else:
            storage_key = "bound_{l}-{u}_{m}_{n}_{a}{i}".format(l=lo_en.value, u=hi_en.value, m=p_model,
                                                                n=p_bins, a=p_algo, i=p_iter)

        rt = [rt[-1] for rt in src.get_products("combined_ratemap", just_obj=False) if storage_key in rt][0]
        source_mask, background_mask = src.get_mask(rad_type)
        # Get combined peak - basically the only peak internal methods will use
        pix_peak = rt.coord_conv(src.peak, pix)
        rad = Quantity(src.get_source_region(rad_type)[0].to_pixel(rt.radec_wcs).radius, pix)
        brightness, cen_rad, bck = radial_brightness(rt, source_mask, background_mask, pix_peak,
                                                     rad, src.redshift, pix, src.cosmo)
        # Subtracting the background in the simplest way possible
        brightness -= bck
        brightness[brightness < 0] = 0

        scaled_radii = (cen_rad / rad).value

        interp_brightness = interp(rads, scaled_radii, brightness)

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

    # TODO This will need to change somehow, either to allow for user choice, or redshifting the image energy limits
    #  to be the same in the cluster frames
    en_key = "bound_0.5-2.0"

    sb = zeros((len(sources), len(radii)))
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

        temp_lo = Quantity(0.5, "keV")
        temp_hi = Quantity(2.0, "keV")
        for s_ind, s in enumerate(sources):
            pool.apply_async(construct_profile, callback=callback, error_callback=err_callback,
                             args=(s, s_ind, scale_radius, temp_lo, temp_hi, use_peak, radii))
        pool.close()
        pool.join()
        onwards.close()

    # Now, we have all the brightness values at common radii (in units of R200 so scaled properly), now we have
    #  to weight the SB values so they are directly comparable. This accounts for redshift, nH, and sort-of for
    #  the temperature of each cluster.
    # First must make sure we have generated spectra for all the clusters, as we need the ARFs and RMFs to
    #  simulate spectra and calculate conversion values
    evselect_spectrum(sources, scale_radius)  # Use our standard setting for spectra generation

    # Calculate all the conversion factors
    # TODO Replace this with the cluster's own temperature once I allow users to add a temperature themselves
    #  on source declaration
    cluster_cr_conv(sources, scale_radius, Quantity(3.0, 'keV'))





