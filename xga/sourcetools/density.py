#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 02/10/2020, 15:31. Copyright (c) David J Turner

import inspect
from typing import Union, List, Tuple
from warnings import warn

import numpy as np
from abel.direct import direct_transform
from astropy.constants import m_p, m_e
from astropy.units import Quantity, pix, kpc
from scipy.integrate import trapz
from scipy.optimize import curve_fit
from tqdm import tqdm

from ..exceptions import NoProductAvailableError
from ..imagetools.profile import radial_brightness
from ..models.sb import SB_MODELS, SB_MODELS_STARTS
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


def _run_sb(src, reg_type, use_peak, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter, pix_step,
            min_snr):
    if psf_corr:
        storage_key = "bound_{l}-{u}_{m}_{n}_{a}{i}".format(l=lo_en.value, u=hi_en.value, m=psf_model, n=psf_bins,
                                                            a=psf_algo, i=psf_iter)
    else:
        storage_key = "bound_{0}-{1}".format(lo_en.to("keV").value, hi_en.to("keV").value)

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
                                                                    kpc, src.cosmo, min_snr)
    if not success:
        warn("Minimum SNR rebinning failed for {}".format(src.name))

    # Just very clumsily subtracting the background
    sb -= bck

    return sb, sb_err, cen_rad, rad_bins


def _sample_sb_model(max_r, model, model_par, model_cov, num_samp=1000, num_rad_steps=300, conf_level=90):
    # Converting to the input expected by numpy's percentile function
    upper = 50 + (conf_level / 2)
    lower = 50 - (conf_level / 2)

    # Calculate the uncertainties on the model_cov
    model_par_err = np.sqrt(np.diagonal(model_cov))

    # Copying the model pars and errors into a new array with the length of the y axis being the number
    #  of samples we want to take from the parameter distributions.
    ext_model_par = np.repeat(model_par[..., None], num_samp, axis=1).T
    ext_model_par_err = np.repeat(model_par_err[..., None], num_samp, axis=1).T

    # This generates num_samp random samples from the passed model parameters, assuming they are Gaussian
    model_par_dists = np.random.normal(ext_model_par, ext_model_par_err)

    # No longer need these now we've drawn the random samples
    del ext_model_par
    del ext_model_par_err

    # Setting up some radii between 0 and the maximum radius to sample the model at
    model_radii = np.linspace(0, max_r, num_rad_steps)
    # Copies the chosen radii num_samp times, much as with the ext_model_par definition
    model_radii = np.repeat(model_radii[..., None], num_samp, axis=1)  # .T

    # Generates num_samp realisations of the model at the model_radii
    model_realisations = model(model_radii, *model_par_dists.T)

    # Calculates the mean model value at each radius step
    model_mean = np.mean(model_realisations, axis=1)
    # Then calculates the values for the upper and lower limits (defined by the
    #  confidence level) for each radii
    model_lower = np.percentile(model_realisations, lower, axis=1)
    model_upper = np.percentile(model_realisations, upper, axis=1)

    return model_radii[:, 0], model_realisations.T, model_mean, model_lower, model_upper


# TODO Come up with some way of propagating the SB profile uncertainty to density
def inv_abel_data(sources: Union[GalaxyCluster, ClusterSample], reg_type: str = "r500", use_peak: bool = True,
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

    densities = []
    for src_ind, src in enumerate(sources):
        sb, sb_err, cen_rad, rad_bins = _run_sb(src, reg_type, use_peak, lo_en, hi_en, psf_corr, psf_model, psf_bins,
                                                psf_algo, psf_iter, pix_step, min_snr)

        # Convert the cen_rad and rad_bins to cm
        cen_rad = cen_rad.to("cm")
        rad_bins = rad_bins.to("cm")

        # The returned SB profile is in count/s/arcmin^2, this converts it to count/s/cm^2 for the abel transform
        conv = (ang_to_rad(Quantity(1, 'arcmin'), src.redshift, src.cosmo).to("cm")) ** 2
        # Applying the conversion to /cm^2
        sb /= conv.value
        sb_err /= conv.value

        # The direct_transform takes us from surface brightness to countrate/volume, then the conv_factors goes
        #  from that to squared hydrogen number density, and from there the square root goes to just hydrogen
        #  number density.
        num_density = np.sqrt(direct_transform(sb, r=cen_rad.value, backend="python") * conv_factors[src_ind])
        # Now we convert to an actual mass
        density = (Quantity(num_density, "1/cm^3") * HY_MASS).to("Msun/Mpc^3")
        # The densities list is gonna be a right mess, but I may change it in the future.
        densities.append([cen_rad, density])

    return densities


def inv_abel_fitted_model(sources: Union[GalaxyCluster, ClusterSample], model: str, fit_method: str = "mcmc",
                          model_start_pars: list = None, reg_type: str = "r500", use_peak: bool = True,
                          pix_step: int = 1, min_snr: Union[int, float] = 0.0, abund_table: str = "angr",
                          lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                          psf_corr: bool = True, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                          psf_iter: int = 15, model_realisations: int = 500, model_rad_steps: int = 300,
                          conf_level: int = 90, num_cores: int = NUM_CORES):
    fit_methods = ["curve_fit", "mcmc"]

    if fit_method == "mcmc":
        raise NotImplementedError("I haven't actually written the mcmc fitting part yet...")

    if fit_method not in fit_methods:
        raise ValueError("{0} is not an accepted method, please choose one of these; "
                         "{1}".format(fit_method, ", ".join(fit_methods)))

    if model not in SB_MODELS:
        raise ValueError("{0} is not an accepted surface brightness model, please choose one of these; "
                         "{1}".format(model, ", ".join(list(SB_MODELS.keys()))))

    model_sig = inspect.signature(SB_MODELS[model])
    model_par_names = [p.name for p in list(model_sig.parameters.values())[1:]]
    if model_start_pars is not None and len(model_start_pars) != len(model_par_names):
        raise ValueError("model_start_pars must either be None, or have an entry for each parameter expected by"
                         " the chosen model; {0} expects {1}".format(model, ", ".join(model_par_names)))
    elif model_start_pars is None:
        model_start_pars = SB_MODELS_STARTS[model]

    # Run the setup function, calculates the factors that translate 3D countrate to density
    #  Also checks parameters and runs any spectra/fits that need running
    sources, conv_factors = _dens_setup(sources, reg_type, abund_table, lo_en, hi_en, num_cores=num_cores)

    masses = {}
    masses_mi = {}
    masses_pl = {}
    densities = {}
    onwards = tqdm(desc="GOING FOR IT", total=len(sources))
    for src_ind, src in enumerate(sources):
        sb, sb_err, cen_rad, rad_bins = _run_sb(src, reg_type, use_peak, lo_en, hi_en, psf_corr, psf_model, psf_bins,
                                                psf_algo, psf_iter, pix_step, min_snr)
        try:
            fit_par, fit_cov = curve_fit(SB_MODELS[model], cen_rad.value, sb, p0=model_start_pars, sigma=sb_err)
            max_rad = (cen_rad[-1] + rad_bins[-1]).value
            radii, realisations, mean, lower, upper = _sample_sb_model(max_rad, SB_MODELS[model], fit_par, fit_cov,
                                                                       model_realisations, model_rad_steps, conf_level)

            # The returned SB profile is in count/s/arcmin^2, this converts it to count/s/cm^2 for the abel transform
            conv = (ang_to_rad(Quantity(1, 'arcmin'), src.redshift, src.cosmo).to("cm")) ** 2

            # Convert those radii to cm
            radii = Quantity(radii, "kpc").to("cm")
            realisations /= conv.value
            mean /= conv.value
            lower /= conv.value
            upper /= conv.value

            num_density = np.zeros(realisations.shape)
            for r_ind, realisation in enumerate(realisations):
                num_density[r_ind, :] = np.sqrt(direct_transform(realisation, r=radii.value, backend="python")
                                                * conv_factors[src_ind])

            # Now we convert to an actual mass
            density = (Quantity(num_density, "1/cm^3") * HY_MASS).to("Msun/Mpc^3")

            upper = 50 + (conf_level / 2)
            lower = 50 - (conf_level / 2)

            #mass_profiles = Quantity(cumtrapz(density.value, radii.to("Mpc").value), "Msun").to("10^13 Msun")

            mass = Quantity(trapz(density.value, radii.to("Mpc").value), "Msun").to("10^13 Msun")

            m = mass.mean()
            l = m - np.percentile(mass, lower)
            u = np.percentile(mass, upper) - m

            if not np.isnan(m.value):
                densities[src.name] = density
                masses[src.name] = m
                masses_mi[src.name] = l
                masses_pl[src.name] = u

        except RuntimeError:
            pass

        onwards.update(1)
    onwards.close()
    return density






