#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 07/09/2021, 12:00. Copyright (c) David J Turner

from typing import Union, List
from warnings import warn

from astropy.units import Quantity
from tqdm import tqdm

from .misc import model_check
from .. import NUM_CORES
from ..exceptions import ModelNotAssociatedError, XGAFitError
from ..imagetools.psf import rl_psf
from ..models import BaseModel1D
from ..products.profile import HydrostaticMass
from ..samples import ClusterSample
from ..sas import region_setup
from ..sources import BaseSource, GalaxyCluster
from ..sourcetools.density import inv_abel_fitted_model
from ..sourcetools.temperature import onion_deproj_temp_prof
from ..xspec.fit import single_temp_apec


def _setup_global(sources, outer_radius, global_radius, abund_table: str, group_spec: bool, min_counts: int,
                  min_sn: float, over_sample: float, num_cores: int):

    out_rads = region_setup(sources, outer_radius, Quantity(0, 'arcsec'), False, '')[-1]
    global_out_rads = region_setup(sources, global_radius, Quantity(0, 'arcsec'), False, '')[-1]

    # If its a single source I shove it in a list so I can just iterate over the sources parameter
    #  like I do when its a Sample object
    if isinstance(sources, BaseSource):
        sources = [sources]

    # We also want to make sure that everything has a PSF corrected image, using all the default settings
    rl_psf(sources)

    # We do this here (even though its also in the density measurement), because if we can't measure a global
    #  temperature then its absurdly unlikely that we'll be able to measure a temperature profile, so we can avoid
    #  even trying and save some time.
    single_temp_apec(sources, global_radius, min_counts=min_counts, min_sn=min_sn, over_sample=over_sample,
                     num_cores=num_cores, abund_table=abund_table, group_spec=group_spec)

    has_glob_temp = []
    for src_ind, src in enumerate(sources):
        try:
            src.get_temperature(global_out_rads[src_ind], 'constant*tbabs*apec', group_spec=group_spec,
                                min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)
            has_glob_temp.append(True)
        except ModelNotAssociatedError:
            warn("The global temperature fit for {} has failed, and as such we're very unlikely to be able to measure "
                 "a mass and we're not even going to try.".format(src.name))
            has_glob_temp.append(False)

    return sources, out_rads, has_glob_temp


def inv_abel_dens_onion_temp(sources: Union[GalaxyCluster, ClusterSample], outer_radius: Union[str, Quantity],
                             sb_model: Union[str, List[str], BaseModel1D, List[BaseModel1D]],
                             dens_model: Union[str, List[str], BaseModel1D, List[BaseModel1D]],
                             temp_model: Union[str, List[str], BaseModel1D, List[BaseModel1D]], global_radius: Quantity,
                             fit_method: str = "mcmc", num_walkers: int = 20, num_steps: int = 20000,
                             sb_pix_step: int = 1, sb_min_snr: Union[int, float] = 0.0, inv_abel_method: str = None,
                             temp_min_snr: float = 20, abund_table: str = "angr", group_spec: bool = True,
                             spec_min_counts: int = 5, spec_min_sn: float = None, over_sample: float = None,
                             num_cores: int = NUM_CORES, show_warn: bool = True) -> List[HydrostaticMass]:
    """
    A convenience function that should allow the user to easily measure hydrostatic masses of a sample of galaxy
    clusters, elegantly dealing with any sources that have inadequate data or aren't fit properly. For the sake
    of convenience, I have taken away a lot of choices that can be passed into the density and temperature
    measurement routines, and if you would like more control then please manually define a hydrostatic mass profile
    object.

    This function uses the inv_abel_fitted_model density measurement function, and the onion_deproj_temp_prof
    temperature measurement function (with the minimum signal to noise criteria for deciding on the annular
    spectra sizes).

    :param GalaxyCluster/ClusterSample sources: The galaxy cluster, or sample of galaxy clusters, that you wish to
        measure hydrostatic masses for.
    :param str/Quantity outer_radius: The radius out to which you wish to measure gas density and temperature
        profiles. This can either be a string radius name (like 'r500') or an astropy quantity. That quantity should
        have as many entries as there are sources.
    :param str/List[str]/BaseModel1D/List[BaseModel1D] sb_model: The model(s) to be fit to the cluster surface
        profile(s). You may pass the string name of a model (for single or multiple clusters), a single instance
        of an XGA model class (for single or multiple clusters), a list of string names (one entry for each cluster
        being analysed), or a list of XGA model instances (one entry for each cluster being analysed).
    :param str/List[str]/BaseModel1D/List[BaseModel1D] dens_model: The model(s) to be fit to the cluster density
        profile(s). You may pass the string name of a model (for single or multiple clusters), a single instance
        of an XGA model class (for single or multiple clusters), a list of string names (one entry for each cluster
        being analysed), or a list of XGA model instances (one entry for each cluster being analysed).
    :param str/List[str]/BaseModel1D/List[BaseModel1D] temp_model: The model(s) to be fit to the cluster temperature
        profile(s). You may pass the string name of a model (for single or multiple clusters), a single instance
        of an XGA model class (for single or multiple clusters), a list of string names (one entry for each cluster
        being analysed), or a list of XGA model instances (one entry for each cluster being analysed).
    :param str/Quantity global_radius: This is a radius for a 'global' temperature measurement, which is both used as
        an initial check of data quality, and feeds into the conversion factor required for density measurements. This
        may also be passed as either a named radius or a quantity.
    :param str fit_method: The method to use for fitting profiles within this function, default is 'mcmc'.
    :param int num_walkers: If fit_method is 'mcmc' this is the number of walkers to initialise for
        the ensemble sampler.
    :param int num_steps: If fit_method is 'mcmc' this is the number steps for each walker to take.
    :param int sb_pix_step: The width (in pixels) of each annular bin for the surface brightness profiles, default is 1.
    :param int/float sb_min_snr: The minimum allowed signal to noise for the surface brightness profiles. Default
        is 0, which disables automatic re-binning.
    :param str inv_abel_method: The method which should be used for the inverse abel transform of the model which
        is fitted to the surface brightness profile. This overrides the default method for the model, which is either
        'analytical' for models with an analytical solution to the inverse abel transform, or 'direct' for
        models which don't have an analytical solution. Default is None.
    :param int/float temp_min_snr: The minimum signal to noise for a temperature measurement annulus, default is 30.
    :param str abund_table: The abundance table to use for fitting, and the conversion factor required during density
        calculations.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int spec_min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float spec_min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param bool over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param int num_cores: The number of cores on your local machine which this function is allowed, default is
        90% of the cores in your system.
    :param bool show_warn: Should profile fit warnings be shown, or only stored in the profile models.
    :return: A list of the hydrostatic mass profiles measured by this function, though if the measurement was not
        successful an entry of None will be added to the list.
    :rtype: List[HydrostaticMass]
    """
    sources, outer_rads, has_glob_temp = _setup_global(sources, outer_radius, global_radius, abund_table, group_spec,
                                                       spec_min_counts, spec_min_sn, over_sample, num_cores)
    rads_dict = {str(sources[r_ind]): r for r_ind, r in enumerate(outer_rads)}

    # This checks and sets up a predictable structure for the models needed for this measurement.
    sb_model = model_check(sources, sb_model)
    dens_model = model_check(sources, dens_model)
    temp_model = model_check(sources, temp_model)

    # I also set up dictionaries, so that models for specific clusters (as you can pass individual model instances
    #  for different clusters) are assigned to the right source when we start cutting down the sources based on
    #  whether a measurement has been successful
    sb_model_dict = {str(sources[m_ind]): m for m_ind, m in enumerate(sb_model)}
    dens_model_dict = {str(sources[m_ind]): m for m_ind, m in enumerate(dens_model)}
    temp_model_dict = {str(sources[m_ind]): m for m_ind, m in enumerate(temp_model)}

    # Here we take only the sources that have a successful global temperature measurement
    cut_sources = [src for src_ind, src in enumerate(sources) if has_glob_temp[src_ind]]
    cut_rads = Quantity([rads_dict[str(src)] for src in cut_sources])
    if len(cut_sources) == 0:
        raise ValueError("No sources have a successful global temperature measurement.")

    # Attempt to measure their 3D temperature profiles
    temp_profs = onion_deproj_temp_prof(cut_sources, cut_rads, min_snr=temp_min_snr, min_counts=spec_min_counts,
                                        min_sn=spec_min_sn, over_sample=over_sample, abund_table=abund_table,
                                        num_cores=num_cores)
    # This just allows us to quickly lookup the temperature profile we need later
    temp_prof_dict = {str(cut_sources[p_ind]): p for p_ind, p in enumerate(temp_profs)}

    # Now we take only the sources that have successful 3D temperature profiles. We do the temperature profile
    #  stuff first because its more difficult, and why should we waste time on a density profile if the temperature
    #  profile cannot even be measured.
    cut_cut_sources = [cut_sources[prof_ind] for prof_ind, prof in enumerate(temp_profs) if prof is not None]
    cut_cut_rads = Quantity([rads_dict[str(src)] for src in cut_cut_sources])

    # And checking again if this stage of the measurement worked out
    if len(cut_cut_sources) == 0:
        raise ValueError("No sources have a successful temperature profile measurement.")

    # We also need to setup the sb model list for our cut sample
    sb_models_cut = [sb_model_dict[str(src)] for src in cut_cut_sources]
    # Now we run the inverse abel density profile generator
    dens_profs = inv_abel_fitted_model(cut_cut_sources, sb_models_cut, fit_method, cut_cut_rads, pix_step=sb_pix_step,
                                       min_snr=sb_min_snr, abund_table=abund_table, num_steps=num_steps,
                                       num_walkers=num_walkers, group_spec=group_spec, min_counts=spec_min_counts,
                                       min_sn=spec_min_sn, over_sample=over_sample, conv_outer_radius=global_radius,
                                       inv_abel_method=inv_abel_method, num_cores=num_cores, show_warn=show_warn)
    # Set this up to lookup density profiles based on source
    dens_prof_dict = {str(cut_cut_sources[p_ind]): p for p_ind, p in enumerate(dens_profs)}

    # So I can return a list of profiles, a tad more elegant than fetching them from the sources sometimes
    final_mass_profs = []
    # Better to use a with statement for tqdm, so its shut down if something fails inside
    prog_desc = "Generating {} hydrostatic mass profile"
    with tqdm(desc=prog_desc.format("None"), total=len(sources)) as onwards:
        for src in sources:
            onwards.set_description(prog_desc.format(src.name))
            # If every stage of this analysis has worked then we setup the hydro mass profile
            if str(src) in dens_prof_dict and dens_prof_dict[str(src)] is not None:
                # This fetches out the correct density and temperature profiles
                d_prof = dens_prof_dict[str(src)]
                t_prof = temp_prof_dict[str(src)]

                # And the appropriate temperature and density models
                d_model = dens_model_dict[str(src)]
                t_model = temp_model_dict[str(src)]

                # Set up the hydrogen mass profile using the temperature radii as they will tend to be spaced a lot
                #  wider than the density radii.
                try:
                    rads = t_prof.radii.copy()[1:]
                    rad_errs = t_prof.radii_err.copy()[1:]
                    deg_rads = src.convert_radius(rads, 'deg')
                    hy_mass = HydrostaticMass(t_prof, t_model, d_prof, d_model, rads, rad_errs, deg_rads, fit_method,
                                              num_walkers, num_steps, show_warn=show_warn, progress=False)
                    # Add the profile to the source storage structure
                    src.update_products(hy_mass)
                    # Also put it into a list for returning
                    final_mass_profs.append(hy_mass)
                except XGAFitError:
                    warn("A fit failure occurred in the hydrostatic mass profile definition.")
                    final_mass_profs.append(None)
                except ValueError:
                    warn("A mass of less than zero was measured by a hydrostatic mass profile, this is not physical"
                         " and the profile is not valid.")
                    final_mass_profs.append(None)

            # If the density generation failed we give a warning here
            elif str(src) in dens_prof_dict:
                warn("The density profile for {} could not be generated".format(src.name))
                # No density means no mass, so we append None to the list
                final_mass_profs.append(None)
            else:
                # And again this is a failure state, so we append a None to the list
                final_mass_profs.append(None)

            onwards.update(1)
        onwards.set_description("Complete")

    # In case only one source is being analysed
    if len(final_mass_profs) == 1:
        final_mass_profs = final_mass_profs[0]
    return final_mass_profs

















