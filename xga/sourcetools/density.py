#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 05/03/2021, 13:05. Copyright (c) David J Turner

from typing import Union, List, Tuple
from warnings import warn

import numpy as np
from abel.direct import direct_transform
from astropy.units import Quantity, kpc
from tqdm import tqdm

from .temperature import min_snr_proj_temp_prof, ALLOWED_ANN_METHODS
from ..exceptions import NoProductAvailableError, ModelNotAssociatedError, ParameterNotAssociatedError
from ..imagetools.profile import radial_brightness
from ..products.profile import SurfaceBrightness1D, GasDensity3D
from ..samples.extended import ClusterSample
from ..sas.spec import region_setup
from ..sources import GalaxyCluster, BaseSource
from ..sourcetools import ang_to_rad
from ..utils import NHC, ABUND_TABLES, HY_MASS, NUM_CORES
from ..xspec.fakeit import cluster_cr_conv
from ..xspec.fit import single_temp_apec


def _dens_setup(sources: Union[GalaxyCluster, ClusterSample], outer_radius: Union[str, Quantity],
                inner_radius: Union[str, Quantity], abund_table: str, lo_en: Quantity,
                hi_en: Quantity, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                over_sample: float = None, obs_id: Union[str, list] = None, inst: Union[str, list] = None,
                conv_temp: Quantity = None, conv_outer_radius: Quantity = "r500",
                num_cores: int = NUM_CORES) -> Tuple[Union[ClusterSample, List], np.ndarray, list, list]:
    """
    An internal function which exists because all the density profile methods that I have planned
    need the same product checking and setup steps. This function checks that all necessary spectra/fits have
    been generated/run, then uses them to calculate the conversion factors from count-rate/volume to squared
    hydrogen number density.

    :param Union[GalaxyCluster, ClusterSample] sources: The source objects/sample object for which the density profile
    is being found.
    :param str/Quantity outer_radius: The name or value of the outer radius of the spectra that should be used
        to calculate conversion factors (for instance 'r200' would be acceptable for a GalaxyCluster, or
        Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any
        inner radius will be ignored.
    :param str/Quantity inner_radius: The name or value of the inner radius of the spectra that should be used
        to calculate conversion factors (for instance 'r500' would be acceptable for a GalaxyCluster, or
        Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a circular spectrum.
    :param str abund_table: Which abundance table should be used for the XSPEC fit, FakeIt run, and for the
        electron/hydrogen number density ratio.
    :param Quantity lo_en: The lower energy limit of the combined ratemap used to calculate density.
    :param Quantity hi_en: The upper energy limit of the combined ratemap used to calculate density.
    :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
    :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
        desired result were grouped by minimum counts.
    :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
        desired result were grouped by minimum signal to noise.
    :param float over_sample: The level of oversampling applied on the spectra that were fitted.
    :param str/list obs_id: A specific ObsID(s) to measure the density from. This should be a string if a single
        source is being analysed, and a list of ObsIDs the same length as the number of sources otherwise. The
        default is None, in which case the combined data will be used to measure the density profile.
    :param str/list inst: A specific instrument(s) to measure the density from. This can either be passed as a
        single string (e.g. 'pn') if only one source is being analysed, or the same instrument should be used for
        every source in a sample, or a list of strings if different instruments are required for each source. The
        default is None, in which case the combined data will be used to measure the density profile.
    :param Quantity conv_temp: If set this will override XGA measured temperatures within the conv_outer_radius, and
        the fakeit run to calculate the normalisation conversion factor will use these temperatures. The quantity
         should have an entry for each cluster being analysed. Default is None.
    :param str/Quantity conv_outer_radius: The outer radius within which to generate spectra and measure temperatures
        for the conversion factor calculation, default is 'r500'. An astropy quantity may also be passed, with either
        a single value or an entry for each cluster being analysed.
    :param int num_cores: The number of cores that the evselect call and XSPEC functions are allowed to use.
    :return: The source object(s)/sample that was passed in, an array of the calculated conversion factors,
        the parsed obs_id variable, and the parsed inst variable.
    :rtype: Tuple[Union[ClusterSample, List], np.ndarray, list, list]
    """
    # If its a single source I shove it in a list so I can just iterate over the sources parameter
    #  like I do when its a Sample object
    if isinstance(sources, BaseSource):
        sources = [sources]

    # Perform some checks on the ObsID and instrument parameters to make sure that they are in the correct
    #  format if they have been set. We don't need to check that the ObsIDs are associated with the sources
    #  here, because that will happen when the ratemaps are retrieved from the source objects.
    if all([obs_id is not None, inst is not None]):
        if isinstance(obs_id, str):
            obs_id = [obs_id]
        if isinstance(inst, str):
            inst = [inst]

        if len(obs_id) != len(sources):
            raise ValueError("If you set the obs_id argument there must be one entry per source being analysed.")

        if len(inst) != len(sources) and len(inst) != 1:
            raise ValueError("The value passed for inst must either be a single instrument name, or a list "
                             "of instruments the same length as the number of sources being analysed.")
    elif all([obs_id is None, inst is None]):
        obs_id = [None]*len(sources)
        inst = [None]*len(sources)
    else:
        raise ValueError("If a value is supplied for obs_id, then a value must be supplied for inst as well, and "
                         "vice versa.")

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

    if conv_temp is not None and not conv_temp.isscalar and len(conv_temp) != len(sources):
        raise ValueError("If multiple there are multiple entries in conv_temp, then there must be the same number"
                         " of entries as there are sources being analysed.")
    elif conv_temp is not None:
        temps = conv_temp
    else:
        # Check that the spectra we will be relying on for conversion calculation have been fitted, calling
        #  this function will also make sure that they are generated
        single_temp_apec(sources, outer_radius, inner_radius, abund_table=abund_table, num_cores=num_cores,
                         group_spec=group_spec, min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)

        # Then we need to grab the temperatures and pass them through to the cluster conversion factor
        #  calculator - this may well change as I intend to let cluster_cr_conv grab temperatures for
        #  itself at some point
        temp_temps = []
        for src in sources:
            try:
                # A temporary temperature variable
                temp_temp = src.get_temperature("tbabs*apec", outer_radius, inner_radius, group_spec, min_counts,
                                                min_sn, over_sample)[0]
            except (ModelNotAssociatedError, ParameterNotAssociatedError):
                warn("{s}'s temperature fit is not valid, so I am defaulting to a temperature of "
                     "3keV".format(s=src.name))
                temp_temp = Quantity(3, 'keV')
            temp_temps.append(temp_temp.value)
        temps = Quantity(temp_temps, 'keV')

    # This call actually does the fakeit calculation of the conversion factors, then stores them in the
    #  XGA Spectrum objects
    cluster_cr_conv(sources, conv_outer_radius, inner_radius, temps, abund_table=abund_table, num_cores=num_cores,
                    group_spec=group_spec, min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)

    # This where the combined conversion factor that takes a count-rate/volume to a squared number density
    #  of hydrogen
    to_dens_convs = []
    # These are from the distance and redshift, also the normalising 10^-14 (see my paper for
    #  more of an explanation)
    for src_ind, src in enumerate(sources):
        src: GalaxyCluster
        # Both the angular_diameter_distance and redshift are guaranteed to be present here because redshift
        #  is REQUIRED to define GalaxyCluster objects
        factor = ((4 * np.pi * (src.angular_diameter_distance.to("cm") * (1 + src.redshift)) ** 2) / (
                hy_to_elec * 10 ** -14)).value
        total_factor = factor * src.norm_conv_factor(outer_radius, lo_en, hi_en, inner_radius, group_spec, min_counts,
                                                     min_sn, over_sample, obs_id[src_ind], inst[src_ind]).value
        to_dens_convs.append(total_factor)

    # Just convert to numpy array for shits and gigs
    to_dens_convs = np.array(to_dens_convs)

    return sources, to_dens_convs, obs_id, inst


def _run_sb(src: GalaxyCluster, outer_radius: Quantity, use_peak: bool, lo_en: Quantity, hi_en: Quantity,
            psf_corr: bool, psf_model: str, psf_bins: int, psf_algo: str, psf_iter: int, pix_step: int,
            min_snr: float, obs_id: str = None, inst: str = None) -> SurfaceBrightness1D:
    """
    An internal function for the Surface Brightness based density functions, which just quickly assembles the
    requested surface brightness profile.

    :param GalaxyCluster src: A GalaxyCluster object to generate a brightness profile for.
    :param Quantity outer_radius: The desired outer radius of the brightness profile.
    :param bool use_peak: If true the measured peak will be used as the central coordinate of the profile.
    :param Quantity lo_en: The lower energy limit of the combined ratemap used to calculate density.
    :param Quantity hi_en: The upper energy limit of the combined ratemap used to calculate density.
    :param bool psf_corr: Default True, whether PSF corrected ratemaps will be used to make the
        surface brightness profile, and thus the density (if False density results could be incorrect).
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    :param int pix_step: The width (in pixels) of each annular bin for the profiles, default is 1.
    :param int/float min_snr: The minimum allowed signal to noise for the surface brightness
        profiles. Default is 0, which disables automatic re-binning.
    :param str obs_id: The ObsID of the ratemap that should be used to generate the brightness profile, default
        is None in which case the combined ratemap will be used.
    :param str inst: The instrument of the ratemap that should be used to generate the brightness profile, default
        is None in which case the combined ratemap will be used.
    :return: The requested surface brightness profile.
    :rtype: SurfaceBrightness1D
    """
    if all([obs_id is None, inst is None]):
        rt = src.get_combined_ratemaps(lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)
        # Grabs the mask which will remove interloper sources
        int_mask = src.get_interloper_mask()
        comb = True
    elif all([obs_id is not None, inst is not None]):
        rt = src.get_ratemaps(obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)
        # Grabs the mask which will remove interloper sources
        int_mask = src.get_interloper_mask(obs_id=obs_id)
        comb = False
    else:
        raise ValueError("If an ObsID is supplied, an instrument must be supplied as well, and vice versa.")

    if use_peak:
        centre = src.peak
    else:
        centre = src.ra_dec

    rad = src.convert_radius(outer_radius, 'kpc')

    try:
        sb_prof = src.get_1d_brightness_profile(rad, obs_id, inst, centre, lo_en=lo_en, hi_en=hi_en, combined=comb,
                                                pix_step=pix_step, min_snr=min_snr, psf_corr=psf_corr,
                                                psf_model=psf_model, psf_bins=psf_bins, psf_algo=psf_algo,
                                                psf_iter=psf_iter)
    except NoProductAvailableError:
        sb_prof, success = radial_brightness(rt, centre, rad, src.background_radius_factors[0],
                                             src.background_radius_factors[1], int_mask, src.redshift, pix_step, kpc,
                                             src.cosmo, min_snr)
        if not success:
            warn("Minimum SNR rebinning failed for {}".format(src.name))

    return sb_prof


# TODO Come up with some way of propagating the SB profile uncertainty to density
def inv_abel_data(sources: Union[GalaxyCluster, ClusterSample], outer_radius: Union[str, Quantity] = 'r500',
                  use_peak: bool = True, pix_step: int = 1, min_snr: Union[int, float] = 0.0, abund_table: str = "angr",
                  lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                  psf_corr: bool = True, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                  psf_iter: int = 15, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                  over_sample: float = None, obs_id: Union[str, list] = None, inst: Union[str, list] = None,
                  conv_temp: Quantity = None, conv_outer_radius: Quantity = "r500",
                  num_cores: int = NUM_CORES) -> Union[GalaxyCluster, ClusterSample]:
    """
    This is the most basic method for measuring the baryonic density profile of a Galaxy Cluster, and is not
    recommended for serious use due to the often unstable results from applying numerical inverse abel
    transforms to data rather than a model.

    :param GalaxyCluster/ClusterSample sources: A GalaxyCluster or ClusterSample object to measure density
        profiles for.
    :param str/Quantity outer_radius: The name or value of the outer radius of the spectra that should be used
        to calculate conversion factors (for instance 'r200' would be acceptable for a GalaxyCluster, or
        Quantity(1000, 'kpc')).
    :param bool use_peak: If true the measured peak will be used as the central coordinate of the profile.
    :param int pix_step: The width (in pixels) of each annular bin for the profiles, default is 1.
    :param int/float min_snr: The minimum allowed signal to noise for the surface brightness
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
    :param bool group_spec: Whether the spectra that were used for fakeit were grouped.
    :param float min_counts: The minimum counts per channel, if the spectra that were used for fakeit
        were grouped by minimum counts.
    :param float min_sn: The minimum signal to noise per channel, if the spectra that were used for fakeit
        were grouped by minimum signal to noise. THIS IS FOR THE SPECTRUM GENERATION.
    :param float over_sample: The level of oversampling applied on the spectra that were used for fakeit.
    :param str/list obs_id: A specific ObsID(s) to measure the density from. This should be a string if a single
        source is being analysed, and a list of ObsIDs the same length as the number of sources otherwise. The
        default is None, in which case the combined data will be used to measure the density profile.
    :param str/list inst: A specific instrument(s) to measure the density from. This can either be passed as a
        single string (e.g. 'pn') if only one source is being analysed, or the same instrument should be used for
        every source in a sample, or a list of strings if different instruments are required for each source. The
        default is None, in which case the combined data will be used to measure the density profile.
    :param Quantity conv_temp: If set this will override XGA measured temperatures within the conv_outer_radius, and
        the fakeit run to calculate the normalisation conversion factor will use these temperatures. The quantity
         should have an entry for each cluster being analysed. Default is None.
    :param str/Quantity conv_outer_radius: The outer radius within which to generate spectra and measure temperatures
        for the conversion factor calculation, default is 'r500'. An astropy quantity may also be passed, with either
        a single value or an entry for each cluster being analysed.
    :param int num_cores: The number of cores that the evselect call and XSPEC functions are allowed to use.
    :return: A source or sample of sources, with the density profile added to its storage structure.
    :rtype: Union[GalaxyCluster, ClusterSample]
    """
    # Run the setup function, calculates the factors that translate 3D countrate to density
    #  Also checks parameters and runs any spectra/fits that need running. _dens_setup takes an inner_radius
    #  parameter, but I don't currently want people to be able to generate conversion factors from spectra
    #  which are non-circular, so I just pass 0 arcseconds
    sources, conv_factors, obs_id, inst = _dens_setup(sources, outer_radius, Quantity(0, 'arcsec'), abund_table, lo_en,
                                                      hi_en, group_spec, min_counts, min_sn, over_sample, obs_id, inst,
                                                      conv_temp, conv_outer_radius, num_cores)

    # Calls the handy spectrum region setup function to make a predictable set of outer radius values
    out_rads = region_setup(sources, outer_radius, Quantity(0, 'arcsec'), False, '')[-1]

    dens_prog = tqdm(desc="Inverse Abel transforming data and measuring densities", total=len(sources))
    for src_ind, src in enumerate(sources):
        sb_prof = _run_sb(src, out_rads[src_ind], use_peak, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo,
                          psf_iter, pix_step, min_snr, obs_id[src_ind], inst[src_ind])
        src.update_products(sb_prof)

        # Convert the cen_rad and rad_bins to cm
        cen_rad = sb_prof.radii.to("cm")
        rad_bins = sb_prof.radii_err.to("cm")

        # The returned SB profile is in count/s/arcmin^2, this converts it to count/s/cm^2 for the abel transform
        conv = (ang_to_rad(Quantity(1, 'arcmin'), src.redshift, src.cosmo).to("cm")) ** 2
        # Applying the conversion to /cm^2
        sb = sb_prof.values.value / conv.value
        sb_err = sb_prof.values_err.value / conv.value

        # The direct_transform takes us from surface brightness to countrate/volume, then the conv_factors goes
        #  from that to squared hydrogen number density, and from there the square root goes to just hydrogen
        #  number density.
        num_density = np.sqrt(direct_transform(sb, r=cen_rad.value, backend="python") * conv_factors[src_ind])
        # Now we convert to an actual mass
        density = (Quantity(num_density, "1/cm^3") * HY_MASS).to("Msun/Mpc^3")
        deg_radii = src.convert_radius(cen_rad, 'deg')

        # TODO Figure out how to convert the surface brightness uncertainties
        if obs_id[src_ind] is None:
            dens_prof = GasDensity3D(cen_rad.to("kpc"), density, sb_prof.centre, src.name, "combined", "combined",
                                     rad_bins.to("kpc"), deg_radii=deg_radii)
        else:
            dens_prof = GasDensity3D(cen_rad.to("kpc"), density, sb_prof.centre, src.name, obs_id[src_ind],
                                     inst[src_ind], rad_bins.to("kpc"), deg_radii=deg_radii)
        src.update_products(dens_prof)

        dens_prog.update(1)
    dens_prog.close()

    return sources


def inv_abel_fitted_model(sources: Union[GalaxyCluster, ClusterSample], model: str, fit_method: str = "mcmc",
                          model_priors: list = None, model_start_pars: list = None,
                          outer_radius: Union[str, Quantity] = "r500", use_peak: bool = True, pix_step: int = 1,
                          min_snr: Union[int, float] = 0.0, abund_table: str = "angr",
                          lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                          psf_corr: bool = True, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                          psf_iter: int = 15, model_realisations: int = 500, model_rad_steps: int = 300,
                          conf_level: int = 90, num_walkers: int = 20, num_steps: int = 20000, group_spec: bool = True,
                          min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                          obs_id: Union[str, list] = None, inst: Union[str, list] = None, conv_temp: Quantity = None,
                          conv_outer_radius: Quantity = "r500", num_cores: int = NUM_CORES):
    """
    A more sophisticated method of calculating density profiles than inv_abel_data, this fits a model
    to the surface brightness profile of each cluster, and the model is then numerically inverse Abel
    transformed. Tends to result in a more stable and smoother density profile.

    :param GalaxyCluster/ClusterSample sources: A GalaxyCluster or ClusterSample object to measure density
        profiles for.
    :param str model: The model to fit to the surface brightness profiles.
    :param str fit_method: The method for the profile object to use to fit the model, default is mcmc.
    :param list model_priors: If supplied these will be used as priors for the model fit (if mcmc is the
        fitting method), otherwise model defaults will be used.
    :param list model_start_pars: If supplied these will be used as start pars for the model fit (if mcmc is
        NOT the fit method), otherwise model defaults will be used.
    :param str/Quantity outer_radius: The name or value of the outer radius of the spectra that should be used
        to calculate conversion factors (for instance 'r200' would be acceptable for a GalaxyCluster, or
        Quantity(1000, 'kpc')).
    :param bool use_peak: If true the measured peak will be used as the central coordinate of the profile.
    :param int pix_step: The width (in pixels) of each annular bin for the profiles, default is 1.
    :param int/float min_snr: The minimum allowed signal to noise for the surface brightness
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
    :param int model_realisations: The number of realisations of the fitted model to generate for
        error propagation, default is 500.
    :param int model_rad_steps: The number of radius points at which to sample the model for the
        realisations, the default is 300.
    :param int conf_level: The confidence level at which to calculate uncertainties on the density
        profiles, default is 90%.
    :param int num_walkers: If using mcmc fitting, the number of walkers to use. Default is 20.
    :param int num_steps: If using mcmc fitting, the number of steps each walker should take. Default is 20000.
    :param bool group_spec: Whether the spectra that were used for fakeit were grouped.
    :param float min_counts: The minimum counts per channel, if the spectra that were used for fakeit
        were grouped by minimum counts.
    :param float min_sn: The minimum signal to noise per channel, if the spectra that were used for fakeit
        were grouped by minimum signal to noise.
    :param float over_sample: The level of oversampling applied on the spectra that were used for fakeit.
    :param str/list obs_id: A specific ObsID(s) to measure the density from. This should be a string if a single
        source is being analysed, and a list of ObsIDs the same length as the number of sources otherwise. The
        default is None, in which case the combined data will be used to measure the density profile.
    :param str/list inst: A specific instrument(s) to measure the density from. This can either be passed as a
        single string (e.g. 'pn') if only one source is being analysed, or the same instrument should be used for
        every source in a sample, or a list of strings if different instruments are required for each source. The
        default is None, in which case the combined data will be used to measure the density profile.
    :param Quantity conv_temp: If set this will override XGA measured temperatures within the conv_outer_radius, and
        the fakeit run to calculate the normalisation conversion factor will use these temperatures. The quantity
         should have an entry for each cluster being analysed. Default is None.
    :param str/Quantity conv_outer_radius: The outer radius within which to generate spectra and measure temperatures
        for the conversion factor calculation, default is 'r500'. An astropy quantity may also be passed, with either
        a single value or an entry for each cluster being analysed.
    :param int num_cores: The number of cores that the evselect call and XSPEC functions are allowed to use.
    :return: The source/sample object passed in to this function.
    :rtype: GalaxyCluster/ClusterSample
    """
    # Run the setup function, calculates the factors that translate 3D countrate to density
    #  Also checks parameters and runs any spectra/fits that need running
    sources, conv_factors, obs_id, inst = _dens_setup(sources, outer_radius, Quantity(0, 'arcsec'), abund_table, lo_en,
                                                      hi_en, group_spec, min_counts, min_sn, over_sample, obs_id, inst,
                                                      conv_temp, conv_outer_radius, num_cores)

    # Calls the handy spectrum region setup function to make a predictable set of outer radius values
    out_rads = region_setup(sources, outer_radius, Quantity(0, 'arcsec'), False, '')[-1]

    dens_prog = tqdm(desc="Fitting data, inverse Abel transforming, and measuring densities",
                     total=len(sources), position=0)

    for src_ind, src in enumerate(sources):
        sb_prof = _run_sb(src, out_rads[src_ind], use_peak, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo,
                          psf_iter, pix_step, min_snr, obs_id[src_ind], inst[src_ind])
        src.update_products(sb_prof)

        # Fit the user chosen model to sb_prof
        sb_prof.fit(model, fit_method, model_priors, model_start_pars, model_realisations, model_rad_steps,
                    conf_level, num_walkers, num_steps, progress_bar=False)

        model_r = sb_prof.get_realisation(model)
        if model_r is not None:
            # The returned SB profile is in count/s/arcmin^2, this converts it to count/s/cm^2 for the abel transform
            conv = (ang_to_rad(Quantity(1, 'arcmin'), src.redshift, src.cosmo).to("cm")) ** 2

            realisation_info = sb_prof.get_realisation(model)

            # Convert those radii to cm
            radii = Quantity(realisation_info["mod_radii"], sb_prof.radii_unit).to("cm")
            realisations = (realisation_info["mod_real"] / conv.value).T
            mean = realisation_info["mod_real_mean"] / conv.value
            lower = realisation_info["mod_real_lower"] / conv.value
            upper = realisation_info["mod_real_upper"] / conv.value

            num_density = np.zeros(realisations.shape)
            for r_ind, realisation in enumerate(realisations):
                num_density[r_ind, :] = np.sqrt(direct_transform(realisation, r=radii.value, backend="python")
                                                * conv_factors[src_ind])

            # Now we convert to an actual mass density
            density = (Quantity(num_density, "1/cm^3") * HY_MASS).to("Msun/Mpc^3").T
            mean_dens = np.mean(density, axis=1)
            deg_radii = src.convert_radius(radii, 'deg')

            if obs_id[src_ind] is None:
                dens_prof = GasDensity3D(radii.to("kpc"), mean_dens, sb_prof.centre, src.name, "combined", "combined",
                                         deg_radii=deg_radii)
            else:
                dens_prof = GasDensity3D(radii.to("kpc"), mean_dens, sb_prof.centre, src.name, obs_id[src_ind],
                                         inst[src_ind], deg_radii=deg_radii)
            dens_prof.add_realisation("inv_abel_model", radii.to("kpc"), density)

            src.update_products(dens_prof)

        dens_prog.update(1)
    dens_prog.close()
    return sources


def ann_spectra_apec_norm(sources: Union[GalaxyCluster, ClusterSample], outer_radii: Union[Quantity, List[Quantity]],
                          annulus_method: str = 'min_snr', min_snr: float = 20,
                          min_width: Quantity = Quantity(20, 'arcsec'), use_combined: bool = True,
                          use_worst: bool = False, lo_en: Quantity = Quantity(0.5, 'keV'),
                          hi_en: Quantity = Quantity(2, 'keV'), psf_corr: bool = False, psf_model: str = "ELLBETA",
                          psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, allow_negative: bool = False,
                          exp_corr: bool = True, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                          over_sample: float = None, one_rmf: bool = True, link_norm: bool = True,
                          abund_table: str = "angr", num_data_real: int = 300, sigma: int = 2,
                          num_cores: int = NUM_CORES):
    """
    A method of measuring density profiles using XSPEC fits of a set of Annular Spectra. First checks whether the
    required annular spectra already exist and have been fit using XSPEC, if not then they are generated and fitted,
    and APEC normalisation profiles will be produced (with projected temperature profiles also being made as a useful
    extra). Then the apec normalisation profile will be used, with knowledge of the source's redshift and chosen
    analysis cosmology, to produce a density profile from the APEC normalisation.

    :param GalaxyCluster/ClusterSample sources: An individual or sample of sources to calculate 3D gas
        density profiles for.
    :param str/Quantity outer_radii: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored. If you are
        generating for multiple sources then you can also pass a Quantity with one entry per source.
    :param str annulus_method:
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
    :param bool link_norm: Sets whether the normalisation parameter is linked across the spectra in an individual
        annulus during the XSPEC fit. Normally the default is False, but here I have set it to True so one global
        normalisation profile is produced rather than separate profiles for individual ObsID-inst combinations.
    :param str abund_table: The abundance table to use both for the conversion from n_exn_p to n_e^2 during density
        calculation, and the XSPEC fit.
    :param int num_data_real: The number of random realisations to generate when propagating profile uncertainties.
    :param int sigma: What sigma uncertainties should newly created profiles have, the default is 2σ.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    """
    if annulus_method not in ALLOWED_ANN_METHODS:
        a_meth = ", ".join(ALLOWED_ANN_METHODS)
        raise ValueError("That is not a valid method for deciding where to place annuli, please use one of "
                         "these; {}".format(a_meth))

    if annulus_method == 'min_snr':
        # This returns the boundary radii for the annuli
        ann_rads = min_snr_proj_temp_prof(sources, outer_radii, min_snr, min_width, use_combined, use_worst, lo_en,
                                          hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter, allow_negative,
                                          exp_corr, group_spec, min_counts, min_sn, over_sample, one_rmf, link_norm,
                                          abund_table, num_cores)
    elif annulus_method == "growth":
        raise NotImplementedError("This method isn't implemented yet")

    # So we can iterate through sources without worrying if there's more than one cluster
    if not isinstance(sources, ClusterSample):
        sources = [sources]

    # Don't need to check abundance table input because that happens in min_snr_proj_temp_prof and the
    #  gas_density_profile method of APECNormalisation1D
    for src_ind, src in enumerate(sources):
        cur_rads = ann_rads[src_ind]

        try:
            # The normalisation profile(s) from the fit that produced the projected temperature profile. Possible
            #  this will be a list of profiles if link_norm == False
            apec_norm_prof = src.get_apec_norm_profiles(cur_rads, link_norm, group_spec, min_counts, min_sn,
                                                        over_sample)
        except NoProductAvailableError:
            warn("{s} doesn't have a matching apec normalisation profile, skipping.")
            continue

        if not link_norm:
            # obs_id =
            # inst =
            raise NotImplementedError("I haven't decided on what the behaviour will be when there are multiple "
                                      "normalisation profiles.")
        else:
            obs_id = 'combined'
            inst = 'combined'

        # Seeing as we're here, I might as well make a  density profile from the apec normalisation profile
        dens_prof = apec_norm_prof.gas_density_profile(src.redshift, src.cosmo, abund_table, num_data_real, sigma)
        # Then I store it in the source
        src.update_products(dens_prof)
