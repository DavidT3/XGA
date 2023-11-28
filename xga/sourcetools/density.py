#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/11/2023, 20:40. Copyright (c) The Contributors

from typing import Union, List, Tuple
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import m_p
from astropy.units import Quantity, kpc
from tqdm import tqdm

from .misc import model_check
from .temperature import min_snr_proj_temp_prof, min_cnt_proj_temp_prof, ALLOWED_ANN_METHODS
from ..exceptions import NoProductAvailableError, ModelNotAssociatedError, ParameterNotAssociatedError
from ..imagetools.profile import radial_brightness
from ..models import BaseModel1D
from ..products.profile import SurfaceBrightness1D, GasDensity3D
from ..samples.extended import ClusterSample
from ..sas._common import region_setup
from ..sources import GalaxyCluster, BaseSource
from ..sourcetools import ang_to_rad
from ..sourcetools.deproj import shell_ann_vol_intersect
from ..utils import NHC, ABUND_TABLES, NUM_CORES, MEAN_MOL_WEIGHT
from ..xspec.fakeit import cluster_cr_conv
from ..xspec.fit import single_temp_apec


def _dens_setup(sources: Union[GalaxyCluster, ClusterSample], outer_radius: Union[str, Quantity],
                inner_radius: Union[str, Quantity], abund_table: str, lo_en: Quantity,
                hi_en: Quantity, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                over_sample: float = None, obs_id: Union[str, list] = None, inst: Union[str, list] = None,
                conv_temp: Quantity = None, conv_outer_radius: Quantity = "r500",
                num_cores: int = NUM_CORES) -> Tuple[Union[ClusterSample, List], List[Quantity], list, list]:
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
    :return: The source object(s)/sample that was passed in, an array of the calculated conversion factors to take the
        count-rate/volume to a number density of hydrogen, the parsed obs_id variable, and the parsed inst variable.
    :rtype: Tuple[Union[ClusterSample, List], List[Quantity], list, list]
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

    # This check will eventually become obselete, but I haven't yet implemented electron to proton ratios for
    #  all allowed abundance tables - so this just checks whether the chosen table has an ratio associated.
    try:
        e_to_p_ratio = NHC[abund_table]
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
        single_temp_apec(sources, conv_outer_radius, inner_radius, abund_table=abund_table, group_spec=group_spec,
                         min_counts=min_counts, min_sn=min_sn, over_sample=over_sample, num_cores=num_cores)

        # Then we need to grab the temperatures and pass them through to the cluster conversion factor
        #  calculator - this may well change as I intend to let cluster_cr_conv grab temperatures for
        #  itself at some point
        temp_temps = []
        for src in sources:
            try:
                # A temporary temperature variable
                temp_temp = src.get_temperature(conv_outer_radius, "constant*tbabs*apec", inner_radius, group_spec,
                                                min_counts, min_sn, over_sample)[0]
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
        factor = ((4 * np.pi * (src.angular_diameter_distance.to("cm") * (1 + src.redshift)) ** 2)
                  / (e_to_p_ratio * 10 ** -14))
        total_factor = factor * src.norm_conv_factor(conv_outer_radius, lo_en, hi_en, inner_radius, group_spec,
                                                     min_counts, min_sn, over_sample, obs_id[src_ind], inst[src_ind])
        to_dens_convs.append(total_factor)

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
    try:
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
    except NoProductAvailableError:
        raise NoProductAvailableError("The RateMap required to measure the density profile has not been generated "
                                      "yet, possibly because you haven't generated PSF corrected image yet.")

    if use_peak:
        centre = src.peak
    else:
        centre = src.ra_dec

    rad = src.convert_radius(outer_radius, 'kpc')

    try:
        sb_prof = src.get_1d_brightness_profile(rad, obs_id, inst, centre, lo_en=lo_en, hi_en=hi_en,
                                                pix_step=pix_step, min_snr=min_snr, psf_corr=psf_corr,
                                                psf_model=psf_model, psf_bins=psf_bins, psf_algo=psf_algo,
                                                psf_iter=psf_iter)
    except NoProductAvailableError:
        try:
            sb_prof, success = radial_brightness(rt, centre, rad, src.background_radius_factors[0],
                                                 src.background_radius_factors[1], int_mask, src.redshift, pix_step,
                                                 kpc, src.cosmo, min_snr)
        except ValueError:
            sb_prof = None
            success = False
            # No longer just background region failure that can set this off
            # warn("Background region for brightness profile is all zeros for {}".format(src.name))

        if sb_prof is not None and not success:
            warn("Minimum SNR rebinning failed for {}".format(src.name))

    return sb_prof


def _onion_peel_data(sources: Union[GalaxyCluster, ClusterSample], outer_radius: Union[str, Quantity] = "r500",
                    num_dens: bool = True, use_peak: bool = True, pix_step: int = 1, min_snr: Union[int, float] = 0.0,
                    abund_table: str = "angr", lo_en: Quantity = Quantity(0.5, 'keV'),
                    hi_en: Quantity = Quantity(2.0, 'keV'), psf_corr: bool = True, psf_model: str = "ELLBETA",
                    psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, num_samples: int = 10000,
                    group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                    obs_id: Union[str, list] = None, inst: Union[str, list] = None, conv_temp: Quantity = None,
                    conv_outer_radius: Quantity = "r500",  num_cores: int = NUM_CORES):

    raise NotImplementedError("This isn't finished at the moment")
    # Run the setup function, calculates the factors that translate 3D countrate to density
    #  Also checks parameters and runs any spectra/fits that need running
    sources, conv_factors, obs_id, inst = _dens_setup(sources, outer_radius, Quantity(0, 'arcsec'), abund_table, lo_en,
                                                      hi_en, group_spec, min_counts, min_sn, over_sample, obs_id, inst,
                                                      conv_temp, conv_outer_radius, num_cores)

    # Calls the handy spectrum region setup function to make a predictable set of outer radius values
    out_rads = region_setup(sources, outer_radius, Quantity(0, 'arcsec'), False, '')[-1]

    final_dens_profs = []
    # I need the ratio of electrons to protons here as well, so just fetch that for the current abundance table
    e_to_p_ratio = NHC[abund_table]
    with tqdm(desc="Generating density profiles based on onion-peeled data", total=len(sources)) as dens_onwards:
        for src_ind, src in enumerate(sources):
            sb_prof = _run_sb(src, out_rads[src_ind], use_peak, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo,
                              psf_iter, pix_step, min_snr, obs_id[src_ind], inst[src_ind])
            if sb_prof is None:
                final_dens_profs.append(None)
                continue
            else:
                src.update_products(sb_prof)

            rad_bounds = sb_prof.annulus_bounds.to("cm")
            vol_intersects = shell_ann_vol_intersect(rad_bounds, rad_bounds)
            print(vol_intersects.min())
            plt.imshow(vol_intersects.value)
            plt.show()
            # Generating random normalisation profile realisations from DATA
            sb_reals = sb_prof.generate_data_realisations(num_samples, truncate_zero=True) * sb_prof.areas

            # Using a loop here is ugly and relatively slow, but it should be okay
            transformed = []
            for i in range(0, num_samples):
                transformed.append(np.linalg.inv(vol_intersects.T) @ sb_reals[i, :])

            transformed = Quantity(transformed)
            print(np.percentile(transformed, 50, axis=0))
            import sys
            sys.exit()

            # We convert the volume element to cm^3 now, this is the unit we expect for the density conversion
            transformed = transformed.to('ct/(s*cm^3)')
            num_dens_dist = np.sqrt(transformed * conv_factors[src_ind]) * (1 + e_to_p_ratio)

            print(np.where(np.isnan(np.sqrt(transformed * conv_factors[src_ind])))[0].shape)
            import sys
            sys.exit()

            med_num_dens = np.percentile(num_dens_dist, 50, axis=1)
            num_dens_err = np.std(num_dens_dist, axis=1)

            # Setting up the instrument and ObsID to pass into the density profile definition
            if obs_id[src_ind] is None:
                cur_inst = "combined"
                cur_obs = "combined"
            else:
                cur_inst = inst[src_ind]
                cur_obs = obs_id[src_ind]

            dens_rads = sb_prof.radii.copy()
            dens_rads_errs = sb_prof.radii_err.copy()
            dens_deg_rads = sb_prof.deg_radii.copy()
            print(np.where(np.isnan(transformed)))
            print(med_num_dens)
            try:
                # I now allow the user to decide if they want to generate number or mass density profiles using
                #  this function, and here is where that distinction is made
                if num_dens:
                    dens_prof = GasDensity3D(dens_rads.to("kpc"), med_num_dens, sb_prof.centre, src.name, cur_obs,
                                             cur_inst, 'onion', sb_prof, dens_rads_errs, num_dens_err,
                                             deg_radii=dens_deg_rads)
                else:
                    # The mean molecular weight multiplied by the proton mass
                    conv_mass = MEAN_MOL_WEIGHT * m_p
                    dens_prof = GasDensity3D(dens_rads.to("kpc"), (med_num_dens * conv_mass).to('Msun/Mpc^3'),
                                             sb_prof.centre, src.name, cur_obs, cur_inst, 'onion', sb_prof,
                                             dens_rads_errs, (num_dens_err * conv_mass).to('Msun/Mpc^3'),
                                             deg_radii=dens_deg_rads)

                src.update_products(dens_prof)
                final_dens_profs.append(dens_prof)

            # If, for some reason, there are some inf/NaN values in any of the quantities passed to the GasDensity3D
            #  declaration, this is where an error will be thrown
            except ValueError:
                final_dens_profs.append(None)
                warn("One or more of the quantities passed to the init of {}'s density profile has a NaN or Inf value"
                     " in it.".format(src.name))

            dens_onwards.update(1)

    return final_dens_profs


def inv_abel_fitted_model(sources: Union[GalaxyCluster, ClusterSample],
                          model: Union[str, List[str], BaseModel1D, List[BaseModel1D]], fit_method: str = "mcmc",
                          outer_radius: Union[str, Quantity] = "r500", num_dens: bool = True, use_peak: bool = True,
                          pix_step: int = 1, min_snr: Union[int, float] = 0.0, abund_table: str = "angr",
                          lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                          psf_corr: bool = True, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                          psf_iter: int = 15, num_walkers: int = 20, num_steps: int = 20000, num_samples: int = 10000,
                          group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                          obs_id: Union[str, list] = None, inst: Union[str, list] = None, conv_temp: Quantity = None,
                          conv_outer_radius: Quantity = "r500", inv_abel_method: str = None, num_cores: int = NUM_CORES,
                          show_warn: bool = True) -> List[GasDensity3D]:
    """
    A photometric galaxy cluster gas density calculation method where a surface brightness profile is fit with
    a model and an inverse abel transform is used to infer the 3D count-rate/volume profile. Then a conversion factor
    calculated from simulated spectra is used to infer the number density profile.

    Depending on the chosen surface brightness model, the inverse abel transform may be performed using an analytical
    solution, or numerical methods.

    :param GalaxyCluster/ClusterSample sources: A GalaxyCluster or ClusterSample object to measure density
        profiles for.
    :param str/List[str]/BaseModel1D/List[BaseModel1D] model: The model(s) to be fit to the cluster surface
        profile(s). You may pass the string name of a model (for single or multiple clusters), a single instance
        of an XGA model class (for single or multiple clusters), a list of string names (one entry for each cluster
        being analysed), or a list of XGA model instances (one entry for each cluster being analysed).
    :param str fit_method: The method for the profile object to use to fit the model, default is mcmc.
    :param str/Quantity outer_radius:
    :param bool num_dens: If True then a number density profile will be generated, otherwise a mass density profile
        will be generated.
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
    :param int num_walkers: If using mcmc fitting, the number of walkers to use. Default is 20.
    :param int num_steps: If using mcmc fitting, the number of steps each walker should take. Default is 20000.
    :param int num_samples: The number of samples drawn from the posterior distributions of model parameters
        after the fitting process is complete.
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
    :param str inv_abel_method: The method which should be used for the inverse abel transform of model which
        is fitted to the surface brightness profile. This overrides the default method for the model, which is either
        'analytical' for models with an analytical solution to the inverse abel transform, or 'direct' for
        models which don't have an analytical solution. Default is None.
    :param int num_cores: The number of cores that the evselect call and XSPEC functions are allowed to use.
    :param bool show_warn: Should fit warnings be shown on screen.
    :return: A list of the 3D gas density profiles measured by this function, though if the measurement was not
        successful an entry of None will be added to the list.
    :rtype: List[GasDensity3D]
    """
    # Run the setup function, calculates the factors that translate 3D countrate to density
    #  Also checks parameters and runs any spectra/fits that need running
    sources, conv_factors, obs_id, inst = _dens_setup(sources, outer_radius, Quantity(0, 'arcsec'), abund_table, lo_en,
                                                      hi_en, group_spec, min_counts, min_sn, over_sample, obs_id, inst,
                                                      conv_temp, conv_outer_radius, num_cores)

    # Calls the handy spectrum region setup function to make a predictable set of outer radius values
    out_rads = region_setup(sources, outer_radius, Quantity(0, 'arcsec'), False, '')[-1]

    # Need to sort out the type of model input that the user chose, and make sure its ready to be passed into
    #  the fit method of the surface brightness profile(s)
    # First we check the number of arguments passed for the model
    model = model_check(sources, model)

    with tqdm(desc="Fitting data, inverse Abel transforming, and measuring densities",
              total=len(sources), position=0) as dens_prog:
        final_dens_profs = []
        # I need the ratio of electrons to protons here as well, so just fetch that for the current abundance table
        e_to_p_ratio = NHC[abund_table]
        for src_ind, src in enumerate(sources):

            sb_prof = _run_sb(src, out_rads[src_ind], use_peak, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo,
                              psf_iter, pix_step, min_snr, obs_id[src_ind], inst[src_ind])
            if sb_prof is None:
                final_dens_profs.append(None)
                continue
            else:
                src.update_products(sb_prof)

            # Fit the user chosen model to sb_prof
            cur_model = model[src_ind]
            sb_prof.fit(cur_model, fit_method, num_samples, num_steps, num_walkers, show_warn=show_warn,
                        progress_bar=False)

            if isinstance(cur_model, str):
                model_r = sb_prof.get_model_fit(cur_model, fit_method)
            else:
                model_r = sb_prof.get_model_fit(cur_model.name, fit_method)

            if model_r.success:
                dens_rads = sb_prof.radii.copy()
                dens_rads_errs = sb_prof.radii_err.copy()
                dens_deg_rads = sb_prof.deg_radii.copy()
                # Run the inverse abel transform for the model, to retrieve distributions for the value of the transformed
                #  model at each r point. If the user hasn't set a method then we use the default method for the current
                #  model, otherwise we pass the user's choice
                if inv_abel_method is None:
                    transformed = model_r.inverse_abel(dens_rads, use_par_dist=True)
                else:
                    transformed = model_r.inverse_abel(dens_rads, use_par_dist=True, method=inv_abel_method)

                # Now need to make sure the units of the transformed model are what we need
                if sb_prof.values_unit.is_equivalent('ct/(s*arcmin**2)'):
                    # If the SB profile is in count/s/arcmin^2 then the abel transform will have
                    #  units of ct/s/(arcmin^2 kpc), so I create a quantity which will convert the arcmin^2 to kpc^2
                    conv = Quantity(ang_to_rad(Quantity(1, 'arcmin'), src.redshift, src.cosmo).to("kpc").value,
                                    'kpc/arcmin')**2
                    transformed /= conv
                elif sb_prof.values_unit.is_equivalent('ct/(s*kpc**2)'):
                    pass
                else:
                    raise NotImplementedError("Haven't yet added support for surface brightness profiles in "
                                              "other units, don't really know how you even got here.")

                # We convert the volume element to cm^3 now, this is the unit we expect for the density conversion
                transformed = transformed.to('ct/(s*cm^3)')

                # We multiply by the conversion factor that is unique to the cluster and calculated earlier to take
                #  the transformed profile to a gas number density (n_gas as seen in Eckert et al. 2016, eq. 2).
                num_dens_dist = np.sqrt(transformed * conv_factors[src_ind])*(1+e_to_p_ratio)

                med_num_dens = np.percentile(num_dens_dist, 50, axis=1)
                num_dens_err = np.std(num_dens_dist, axis=1)

                # Setting up the instrument and ObsID to pass into the density profile definition
                if obs_id[src_ind] is None:
                    cur_inst = "combined"
                    cur_obs = "combined"
                else:
                    cur_inst = inst[src_ind]
                    cur_obs = obs_id[src_ind]

                try:
                    # I now allow the user to decide if they want to generate number or mass density profiles using
                    #  this function, and here is where that distinction is made
                    if num_dens:
                        dens_prof = GasDensity3D(dens_rads.to("kpc"), med_num_dens, sb_prof.centre, src.name, cur_obs,
                                                 cur_inst, model_r.name, sb_prof, dens_rads_errs, num_dens_err,
                                                 deg_radii=dens_deg_rads)
                    else:
                        # TODO Check the origin of the mean molecular weight, see if there are different values for
                        #  different abundance tables
                        # The mean molecular weight multiplied by the proton mass
                        conv_mass = MEAN_MOL_WEIGHT*m_p
                        dens_prof = GasDensity3D(dens_rads.to("kpc"), (med_num_dens*conv_mass).to('Msun/Mpc^3'),
                                                 sb_prof.centre, src.name, cur_obs, cur_inst, model_r.name, sb_prof,
                                                 dens_rads_errs, (num_dens_err*conv_mass).to('Msun/Mpc^3'),
                                                 deg_radii=dens_deg_rads)

                    src.update_products(dens_prof)
                    final_dens_profs.append(dens_prof)

                # If, for some reason, there are some inf/NaN values in any of the quantities passed to the GasDensity3D
                #  declaration, this is where an error will be thrown
                except ValueError:
                    final_dens_profs.append(None)
                    warn("One or more of the quantities passed to the init of {}'s density profile has a NaN or Inf value"
                         " in it.".format(src.name))
            else:
                final_dens_profs.append(None)

            dens_prog.update(1)

    return final_dens_profs


def ann_spectra_apec_norm(sources: Union[GalaxyCluster, ClusterSample], outer_radii: Union[Quantity, List[Quantity]],
                          num_dens: bool = True, annulus_method: str = 'min_snr', min_snr: float = 30,
                          min_cnt: Union[int, Quantity] = Quantity(1000, 'ct'),
                          min_width: Quantity = Quantity(20, 'arcsec'), use_combined: bool = True,
                          use_worst: bool = False, lo_en: Quantity = Quantity(0.5, 'keV'),
                          hi_en: Quantity = Quantity(2, 'keV'), psf_corr: bool = False, psf_model: str = "ELLBETA",
                          psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, allow_negative: bool = False,
                          exp_corr: bool = True, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                          over_sample: float = None, one_rmf: bool = True, freeze_met: bool = True,
                          abund_table: str = "angr", temp_lo_en: Quantity = Quantity(0.3, 'keV'),
                          temp_hi_en: Quantity = Quantity(7.9, 'keV'), num_data_real: int = 10000, sigma: int = 1,
                          num_cores: int = NUM_CORES) -> List[GasDensity3D]:
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
    :param bool num_dens: If True then a number density profile will be generated, otherwise a mass density profile
        will be generated.
    :param str annulus_method: The method by which the annuli are designated, this can be 'min_snr' (which will use
        the min_snr_proj_temp_prof function), or 'min_cnt' (which will use the min_cnt_proj_temp_prof function).
    :param float min_snr: The minimum signal-to-noise which is allowable in a given annulus, used if annulus_method
        is set to 'min_snr'.
    :param int/Quantity min_cnt: The minimum background subtracted counts which are allowable in a given annulus, used
        if annulus_method is set to 'min_cnt'.
    :param Quantity min_width: The minimum allowable width of an annulus. The default is set to 20 arcseconds to try
        and avoid PSF effects.
    :param bool use_combined: If True (and annulus_method is set to 'min_snr') then the combined RateMap will be
        used for signal-to-noise annulus calculations, this is overridden by use_worst. If True (and annulus_method
        is set to 'min_cnt') then combined RateMaps will be used for annulus count calculations, if False then
        the median observation (in terms of counts) will be used.
    :param bool use_worst: If True then the worst observation of the cluster (ranked by global signal-to-noise) will
        be used for signal-to-noise annulus calculations. Used if annulus_method is set to 'min_snr'.
    :param Quantity lo_en: The lower energy bound of the RateMap to use for the signal-to-noise or background
        subtracted count calculations.
    :param Quantity hi_en: The upper energy bound of the RateMap to use for the signal-to-noise or background
        subtracted count calculations.
    :param bool psf_corr: Sets whether you wish to use a PSF corrected RateMap or not.
    :param str psf_model: If the RateMap you want to use is PSF corrected, this is the PSF model used.
    :param int psf_bins: If the RateMap you want to use is PSF corrected, this is the number of PSFs per
        side in the PSF grid.
    :param str psf_algo: If the RateMap you want to use is PSF corrected, this is the algorithm used.
    :param int psf_iter: If the RateMap you want to use is PSF corrected, this is the number of iterations.
    :param bool allow_negative: Should pixels in the background subtracted count map be allowed to go below
        zero, which results in a lower signal-to-noise (and can result in a negative signal-to-noise).
    :param bool exp_corr: Should signal to noises be measured with exposure time correction, default is True. I
            recommend that this be true for combined observations, as exposure time could change quite dramatically
            across the combined product.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param bool freeze_met: Whether the metallicity parameter in the fits to annuli in XSPEC should be frozen.
    :param str abund_table: The abundance table to use both for the conversion from n_exn_p to n_e^2 during density
        calculation, and the XSPEC fit.
    :param Quantity temp_lo_en: The lower energy limit for the XSPEC fits to annular spectra.
    :param Quantity temp_hi_en: The upper energy limit for the XSPEC fits to annular spectra.
    :param int num_data_real: The number of random realisations to generate when propagating profile uncertainties.
    :param int sigma: What sigma uncertainties should newly created profiles have, the default is 2σ.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :return: A list of the 3D gas density profiles measured by this function, though if the measurement was not
        successful an entry of None will be added to the list.
    :rtype: List[GasDensity3D]
    """
    if annulus_method not in ALLOWED_ANN_METHODS:
        a_meth = ", ".join(ALLOWED_ANN_METHODS)
        raise ValueError("That is not a valid method for deciding where to place annuli, please use one of "
                         "these; {}".format(a_meth))

    if annulus_method == 'min_snr':
        # This returns the boundary radii for the annuli
        ann_rads = min_snr_proj_temp_prof(sources, outer_radii, min_snr, min_width, use_combined, use_worst, lo_en,
                                          hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter, allow_negative,
                                          exp_corr, group_spec, min_counts, min_sn, over_sample, one_rmf, freeze_met,
                                          abund_table, temp_lo_en, temp_hi_en, num_cores)
    elif annulus_method == 'min_cnt':
        # This returns the boundary radii for the annuli, based on a minimum number of counts per annulus
        ann_rads = min_cnt_proj_temp_prof(sources, outer_radii, min_cnt, min_width, use_combined, lo_en, hi_en,
                                          psf_corr, psf_model, psf_bins, psf_algo, psf_iter, group_spec, min_counts,
                                          min_sn, over_sample, one_rmf, freeze_met, abund_table, temp_lo_en, temp_hi_en,
                                          num_cores)
    elif annulus_method == "growth":
        raise NotImplementedError("This method isn't implemented yet")

    # So we can iterate through sources without worrying if there's more than one cluster
    if not isinstance(sources, ClusterSample):
        sources = [sources]

    # Don't need to check abundance table input because that happens in min_snr_proj_temp_prof and the
    #  gas_density_profile method of APECNormalisation1D
    final_dens_profs = []
    with tqdm(desc="Generating density profiles from annular spectra", total=len(sources)) as dens_prog:
        for src_ind, src in enumerate(sources):
            cur_rads = ann_rads[src_ind]

            try:
                # The normalisation profile(s) from the fit that produced the projected temperature profile.
                apec_norm_prof = src.get_apec_norm_profiles(cur_rads, group_spec, min_counts, min_sn, over_sample)

                obs_id = 'combined'
                inst = 'combined'
                # Seeing as we're here, I might as well make a density profile from the apec normalisation profile
                dens_prof = apec_norm_prof.gas_density_profile(src.redshift, src.cosmo, abund_table, num_data_real,
                                                               sigma, num_dens)
                # Then I store it in the source
                src.update_products(dens_prof)
                final_dens_profs.append(dens_prof)

            # It is possible that no normalisation profile exists because the spectral fitting failed, we account
            #  for that here
            except NoProductAvailableError:
                warn("{s} doesn't have a matching apec normalisation profile, skipping.")
                final_dens_profs.append(None)

            # It's also possible that the gas_density_profile method of our normalisation profile is going to
            #  throw a ValueError because some values are infinite or NaNs - we have to catch that too
            except ValueError:
                warn("{s}'s density profile has NaN values in it, skipping.", stacklevel=2)
                final_dens_profs.append(None)

            dens_prog.update(1)

    return final_dens_profs
