#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 30/09/2020, 15:59. Copyright (c) David J Turner

from typing import Union, List, Tuple
from warnings import warn

import numpy as np
from abel.direct import direct_transform
from astropy.constants import m_p, m_e
from astropy.units import Quantity, pix, cm

from ..exceptions import NoProductAvailableError
from ..imagetools.profile import radial_brightness
from ..products import RateMap
from ..samples.extended import ClusterSample
from ..sas.spec import evselect_spectrum
from ..sources import GalaxyCluster, BaseSource
from ..sourcetools import ang_to_rad
from ..utils import NHC, ABUND_TABLES
from ..utils import NUM_CORES
from ..xspec.fakeit import cluster_cr_conv
from ..xspec.fit import single_temp_apec

# I know this is practically pointless, I could just use m_p, but I like doing things properly.
HY_MASS = m_p + m_e


def _dens_setup(sources: Union[GalaxyCluster, ClusterSample], reg_type: str, abund_table: str, lo_en: Quantity,
                hi_en: Quantity, num_cores: int = NUM_CORES) -> Tuple[Union[ClusterSample, List], np.ndarray]:
    """
    An internal function which exists because all the density profile methods that I have planned
    need the same product checking and setup steps. This function checks that all necessary spectra/fits have
    been generated/run, then uses them to calculate the conversion factors from count-rate/volume to squared
    hydrogen number density.
    :param Union[GalaxyCluster, ClusterSample] sources: The source objects/sample object for which the density profile
    is being found.
    :param str reg_type: The region type to use for the spectrum, XSPEC temperature fit, and FakeIt run.
    :param str abund_table: Which abundance table should be used for the XSPEC fit, FakeIt run, and for the
    electron/hydrogen number density ratio.
    :param Quantity lo_en: The lower energy limit of the combined ratemap used to calculate density.
    :param Quantity hi_en: The upper energy limit of the combined ratemap used to calculate density.
    :param int num_cores: The number of cores that the evselect call and XSPEC functions are allowed to use.
    :return: The source object(s)/sample that was passed in, an array of the calculated conversion factors.
    :rtype: Tuple[Union[ClusterSample, List], np.ndarray]
    """
    # If its a single source I shove it in a list so I can just iterate over the sources parameter
    #  like I do when its a Sample object
    if isinstance(sources, BaseSource):
        sources = [sources]

    if not all([type(src) == GalaxyCluster for src in sources]):
        raise TypeError("Only GalaxyCluster sources can be passed to cluster_density_profile.")

    # Triggers an exception if the abundance table name passed isn't recognised
    if abund_table not in ABUND_TABLES:
        ab_list = ", ".join(ABUND_TABLES)
        raise ValueError("{0} is not in the accepted list of abundance tables; {1}".format(abund_table, ab_list))

    # This will eventually become obselete, but I haven't yet implemented ne/nh ratios for all allowed
    #  abundance tables - so this just checks whether the chosen table has an ratio associated.
    try:
        hy_to_elec = NHC[abund_table]
    except KeyError:
        raise NotImplementedError("That is an acceptable abundance table, but I haven't added the conversion factor "
                                  "to the dictionary yet")

    # Check that spectra of the passed reg_type exist
    evselect_spectrum(sources, reg_type, num_cores=num_cores)
    # Check that said spectra have been fitted
    single_temp_apec(sources, reg_type, abund_table=abund_table, num_cores=num_cores)

    # Then we need to grab the temperatures and pass them through to the cluster conversion factor
    #  calculator - this may well change as I intend to let cluster_cr_conv grab temperatures for
    #  itself at some point
    temps = Quantity([src.get_temperature(reg_type, "tbabs*apec")[0] for src in sources], 'keV')
    cluster_cr_conv(sources, reg_type, temps, abund_table=abund_table)

    # This where the combined conversion factor that takes a count-rate/volume to a squared number density
    #  of hydrogen
    to_dens_convs = []
    # These are from the distance and redshift, also the normalising 10^-14 (see my paper for
    #  more of an explanation)
    for src in sources:
        # Both the angular_diameter_distance and redshift are guaranteed to be present here because redshift
        #  is REQUIRED to define GalaxyCluster objects
        factor = ((4 * np.pi * (src.angular_diameter_distance.to("cm") * (1 + src.redshift)) ** 2) / (
                    hy_to_elec * 10 ** -14)).value
        to_dens_convs.append(factor * src.combined_norm_conv_factor(reg_type, lo_en, hi_en).value)

    # Just convert to numpy array for shits and gigs
    to_dens_convs = np.array(to_dens_convs)

    return sources, to_dens_convs


# TODO Come up with some way of propagating the SB profile uncertainty to density
def data_abel_profile(sources: Union[GalaxyCluster, ClusterSample], reg_type: str = "r500", use_peak: bool = True,
                      pix_step: int = 1, min_snr: Union[int, float] = 0.0, abund_table: str = "angr",
                      lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                      psf_corr: bool = True, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                      psf_iter: int = 15, num_cores: int = NUM_CORES) -> List[List[np.ndarray]]:
    """
    This is the most basic method for measuring the baryonic density profile of a Galaxy Cluster, and is not
    recommended for serious use due to the often unstable results from applying numerical inverse abel
    transforms to data rather than a model.
    :param Union[GalaxyCluster, ClusterSample] sources:
    :param str reg_type: The region type to use for the spectrum, XSPEC temperature fit, and FakeIt run.
    :param bool use_peak: If true the measured peak will be used as the central coordinate of the profile.
    :param int pix_step: The width (in pixels) of each annular bin for the profiles, default is 1.
    :param Union[int, float] min_snr: The minimum allowed signal to noise for the surface brightness
    profiles. Default is 0, which disables automatic re-binning.
    :param str abund_table: Which abundance table should be used for the XSPEC fit, FakeIt run, and for the
    electron/hydrogen number density ratio.
    :param Quantity lo_en: The lower energy limit of the combined ratemap used to calculate density.
    :param Quantity hi_en: The upper energy limit of the combined ratemap used to calculate density.
    :param bool psf_corr: Default True, whether PSF corrected ratemaps will be used to make the
    surface brightness profile, and thus the density (if False density results could be incorrect).
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    :param int num_cores: The number of cores that the evselect call and XSPEC functions are allowed to use.
    :return: A hopefully temporary list of lists of arrays (I knooooow). Each entry in the outer list
    is for a source, then the nested list contains the radii, and the density measurements.
    :rtype: List[List[np.ndarray]]
    """
    # Run the setup function, calculates the factors that translate 3D countrate to density
    #  Also checks parameters and runs any spectra/fits that need running
    sources, conv_factors = _dens_setup(sources, reg_type, abund_table, lo_en, hi_en, num_cores=num_cores)

    if psf_corr:
        storage_key = "bound_{l}-{u}_{m}_{n}_{a}{i}".format(l=lo_en.value, u=hi_en.value, m=psf_model, n=psf_bins,
                                                            a=psf_algo, i=psf_iter)
    else:
        storage_key = "bound_{0}-{1}".format(lo_en.to("keV").value, hi_en.to("keV").value)

    densities = []
    for src_ind, src in enumerate(sources):
        comb_rts = src.get_products("combined_ratemap", extra_key=storage_key)
        if len(comb_rts) != 1 and psf_corr:
            raise NoProductAvailableError("No matching PSF corrected combined ratemap is available for {}, don't "
                                          "forget to run a PSF correction function first!".format(src.name))
        elif len(comb_rts) != 1 and not psf_corr:
            raise NoProductAvailableError("No matching combined ratemap is available for {}.".format(src.name))
        else:
            comb_rt = comb_rts[0]
            comb_rt: RateMap

        if use_peak:
            pix_centre = comb_rt.coord_conv(src.peak, pix)
            source_mask, background_mask = src.get_mask(reg_type, central_coord=src.peak)
        else:
            pix_centre = comb_rt.coord_conv(src.ra_dec, pix)
            source_mask, background_mask = src.get_mask(reg_type, central_coord=src.ra_dec)

        # This is because I actually only want to mask interloper point sources right now.
        source_mask = src.get_interloper_mask()

        rad = Quantity(src.source_back_regions(reg_type)[0].to_pixel(comb_rt.radec_wcs).radius, pix)
        sb, sb_err, cen_rad, rad_bins, bck, success = radial_brightness(comb_rt, source_mask, background_mask,
                                                                        pix_centre, rad, src.redshift, pix_step,
                                                                        cm, src.cosmo, min_snr)
        if not success:
            warn("Minimum SNR rebinning failed for {}".format(src.name))

        # The returned SB profile is in count/s/arcmin^2, this converts it to count/s/cm^2 for the abel transform
        conv = (ang_to_rad(Quantity(1, 'arcmin'), src.redshift, src.cosmo).to("cm"))**2
        # Just very clumsily subtracting the background
        sb -= bck
        # Applying the conversion to /cm^2
        sb /= conv.value

        # The direct_transform takes us from surface brightness to countrate/volume, then the conv_factors goes
        #  from that to squared hydrogen number density, and from there the square root goes to just hydrogen
        #  number density.
        num_density = np.sqrt(direct_transform(sb, r=cen_rad.value, backend="python") * conv_factors[src_ind])
        # Now we convert to an actual mass
        density = (Quantity(num_density, "1/cm^3") * HY_MASS).to("Msun/Mpc^3")
        # The densities list is gonna be a right mess, but I may change it in the future.
        densities.append([cen_rad, density])

    return densities


