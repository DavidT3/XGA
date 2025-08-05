#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 08/07/2025, 12:56. Copyright (c) The Contributors

from typing import Union, List, Tuple
from warnings import warn

from astropy.units import Quantity

from .misc import model_check
from .. import NUM_CORES
from ..exceptions import ModelNotAssociatedError, NotAssociatedError
from ..generate.sas._common import region_setup
from ..imagetools.psf import rl_psf
from ..models import BaseModel1D
from ..samples import BaseSample
from ..samples import ClusterSample
from ..sources import BaseSource, GalaxyCluster
from ..xspec.fit import single_temp_apec


def _get_all_telescopes(sources: Union[BaseSource, BaseSample, List[BaseSource]]) -> List[str]:
    """
    Returns a list of all the telescopes associated with at least one source. For most functions within
    xga.sourcetools, the initial sources argument may be a list, so the 'telescopes' attribute can't be
    used.

    :param BaseSource/List[BaseSource]/BaseSample sources: The sources to extract telescope information from.
    :return: A list of telescope names that are associated with at least one of the sources
        that were passed in.
    :rtype: List[str]
    """
    if isinstance(sources, list):
        # This collects all telescopes associated with each source, so there will be duplicates
        all_telescopes_inc_dups = []
        for src in sources:
            all_telescopes_inc_dups.extend(src.telescopes)
        # Now removing the duplicates
        all_telescopes = list(set(all_telescopes_inc_dups))
    else:
        all_telescopes = sources.telescopes
    
    return all_telescopes


def _setup_global(sources, outer_radius, global_radius, abund_table: str, group_spec: bool, 
                  min_counts: int, min_sn: float, over_sample: float, num_cores: int, psf_bins: int,
                  stacked_spectra: bool, telescope: List[str]):
    """
    Internal function to see if a source/sources have a measured global temperature, single_temp_apec
    is run before the check is done. It also runs the region_setup() method to fetch the outer radii
    of the annular bins of profiles. This method is used in _setup_inv_abel_dens_onion_temp, which
    is then used in entropy and mass profile functions.

    :param BaseSource/List[BaseSource]/BaseSample sources: The sources to check whether a global 
        temperature is measured.
    :param str/Quantity outer_radius: The radius out to which you wish to measure gas density and 
        temperature profiles. This can either be a string radius name (like 'r500') or an astropy 
        quantity. That quantity should have as many entries as there are sources.
    :param str/Quantity global_radius: This is a radius for a 'global' temperature measurement, 
        which is both used as an initial check of data quality, and feeds into the conversion factor 
        required for density measurements. This may also be passed as either a named radius or a 
        quantity.
    :param str abund_table: The abundance table to use for fitting, and the conversion factor 
        required during density calculations.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts 
        per channel. To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in 
        each channel. To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. 
        e.g. if over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at 
        that energy.
    :param int num_cores: The number of cores on your local machine which this function is allowed, 
        default is 90% of the cores in your system.
    :param int psf_bins: The number of bins per side when generating a grid of PSFs for image 
        correction prior to surface brightness profile (and thus density) measurements.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be
        used for this XSPEC spectral fit. If a stacking procedure for a particular telescope is not
        supported, this function will instead use individual spectra for an ObsID. The default is
        False
    :param List[str] telescope: The telescopes to set up global absorbed plasma emission fits for.
    :return: A tuple. The first elements are the sources. The second are the Quantity objects
        describing the outer_radii of the regions used for annular bins. The third is a dictionary
        with telescope keys, containing a list of Trues and Falses, depending on if the source 
        has a global temperature or not.
    :rtype Tuple[BaseSource/List[BaseSource]/BaseSample, Tuple[Union[BaseSource, BaseSample], 
        List[Quantity], List[Quantity]], dict]:
    """

    out_rads = region_setup(sources, outer_radius, Quantity(0, 'arcsec'), False, '')[-1]
    global_out_rads = region_setup(sources, global_radius, Quantity(0, 'arcsec'), False, '')[-1]

    # If the user didn't specify a particular telescope, or telescopes, from which we are to
    #  produce temperature profiles, we fetch all associated with at least one source
    if telescope is None:
        src_telescopes = _get_all_telescopes(sources)
    elif isinstance(telescope, str):
        src_telescopes = [telescope]
    else:
        src_telescopes = telescope

    # If it's a single source, we put it in a list so we can iterate over the single source like a sample
    if isinstance(sources, BaseSource):
        sources = [sources]

    # If XMM is associated with at least one source, we'll run PSF correction
    if 'xmm' in src_telescopes:
        # We also want to make sure that everything has a PSF corrected image, using all the default settings
        rl_psf(sources, bins=psf_bins)

    # We do this here (even though its also in the density measurement), because if we can't measure a global
    #  temperature, then its unlikely that we'll be able to measure a temperature profile
    single_temp_apec(sources, global_radius, abund_table=abund_table, group_spec=group_spec, min_counts=min_counts,
                     min_sn=min_sn, over_sample=over_sample, num_cores=num_cores, stacked_spectra=stacked_spectra,
                     telescope=telescope)

    # We want to return a dictionary of telescope keys and values that are a list of len(sources) where
    # each element in the list is a boolean indicated whether a glob temp has been measured
    # ie. has_glob_temp = {'xmm' : [True, True, False], 'erosita' : [True, True, True]}
    has_glob_temp = {key : [] for key in src_telescopes}
    for src_ind, src in enumerate(sources):
        # We cycle over the telescopes in the Sample and not the Source, so that every list in 
        # has_glob_temp is the same length
        for tel in src_telescopes:
            try:
                if tel == 'erosita' and len(src.obs_ids['erosita']) > 1:
                    # A temporary temperature variable
                    src.get_temperature(global_out_rads[src_ind], tel, "constant*tbabs*apec", 
                                        group_spec=group_spec, min_counts=min_counts, min_sn=min_sn, 
                                        over_sample=over_sample, stacked_spectra=stacked_spectra)
                else:
                    src.get_temperature(global_out_rads[src_ind], tel, 'constant*tbabs*apec', 
                                        group_spec=group_spec, min_counts=min_counts, min_sn=min_sn, 
                                        over_sample=over_sample)
                has_glob_temp[tel].append(True)
            except ModelNotAssociatedError:
                warn("The global temperature fit for {} has failed, which means a temperature profile from annular "
                     "spectra is unlikely to be possible, and we will not attempt it.".format(src.name), stacklevel=2)
                has_glob_temp[tel].append(False)
            # If the telescope is not associated with this Source it will raise a NotAssociatedError
            except NotAssociatedError:
                has_glob_temp[tel].append(False)

    return sources, out_rads, has_glob_temp


def _setup_inv_abel_dens_onion_temp(sources: Union[GalaxyCluster, ClusterSample], outer_radius: Union[str, Quantity],
                                    sb_model: Union[str, List[str], BaseModel1D, List[BaseModel1D]],
                                    dens_model: Union[str, List[str], BaseModel1D, List[BaseModel1D]],
                                    temp_model: Union[str, List[str], BaseModel1D, List[BaseModel1D]],
                                    global_radius: Quantity, fit_method: str = "mcmc", num_walkers: int = 20,
                                    num_steps: int = 20000, sb_pix_step: int = 1, sb_min_snr: Union[int, float] = 0.0,
                                    inv_abel_method: str = None, temp_annulus_method: str = 'min_snr',
                                    temp_min_snr: float = 30, temp_min_cnt: Union[int, Quantity] = Quantity(1000, 'ct'),
                                    temp_min_width: Quantity = Quantity(20, 'arcsec'), temp_use_combined: bool = True,
                                    temp_use_worst: bool = False, freeze_met: bool = True, abund_table: str = "angr",
                                    temp_lo_en: Quantity = Quantity(0.3, 'keV'),
                                    temp_hi_en: Quantity = Quantity(7.9, 'keV'), group_spec: bool = True,
                                    spec_min_counts: int = 5, spec_min_sn: float = None, over_sample: float = None,
                                    one_rmf: bool = True, num_cores: int = NUM_CORES, show_warn: bool = True,
                                    psf_bins: int = 4, stacked_spectra: bool = False,
                                    telescope: Union[str, List[str]] = None) \
        -> Tuple[Union[BaseSource, List[BaseSource], BaseSample], dict, dict, dict, dict, List[str]]:
    """
    Internal function to run the common setup functions that are needed for mass and entropy profile
    measurements.

    :param GalaxyCluster/ClusterSample sources: The galaxy cluster, or sample of galaxy clusters, 
        that are having their hydrostatic masses/entropy measured.
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
    :param str temp_annulus_method: The method by which the temperature profile annuli are designated, this can
        be 'min_snr' (which will use the min_snr_proj_temp_prof function), or 'min_cnt' (which will use the
        min_cnt_proj_temp_prof function).
    :param int/float temp_min_snr: The minimum signal-to-noise for a temperature measurement annulus, default is 30.
    :param int/Quantity temp_min_cnt: The minimum background subtracted counts which are allowable in a given
        temperature annulus, used if temp_annulus_method is set to 'min_cnt'.
    :param Quantity temp_min_width: The minimum allowable width of a temperature annulus. The default is set to
        20 arcseconds to try and avoid PSF effects.
    :param bool temp_use_combined: If True (and temp_annulus_method is set to 'min_snr') then the combined
        RateMap will be used for signal-to-noise annulus calculations, this is overridden by temp_use_worst. If
        True (and temp_annulus_method is set to 'min_cnt') then combined RateMaps will be used for temperature
        annulus count calculations, if False then the median observation (in terms of counts) will be used.
    :param bool temp_use_worst: If True then the worst observation of the cluster (ranked by global signal-to-noise)
        will be used for signal-to-noise temperature annulus calculations. Used if temp_annulus_method is set
        to 'min_snr'.
    :param bool freeze_met: Whether the metallicity parameter in the fits to annuli in XSPEC should be frozen.
    :param str abund_table: The abundance table to use for fitting, and the conversion factor required during density
        calculations.
    :param Quantity temp_lo_en: The lower energy limit for the XSPEC fits to annular spectra.
    :param Quantity temp_hi_en: The upper energy limit for the XSPEC fits to annular spectra.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int spec_min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float spec_min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param bool over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores on your local machine which this function is allowed, default is
        90% of the cores in your system.
    :param bool show_warn: Should profile fit warnings be shown, or only stored in the profile models.
    :param int psf_bins: The number of bins per side when generating a grid of PSFs for image correction prior
        to surface brightness profile (and thus density) measurements.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be
        used for this XSPEC spectral fit. If a stacking procedure for a particular telescope is not
        supported, this function will instead use individual spectra for an ObsID. The default is False.
    :param str/List[str] telescope: Telescope(s) that the user wants to use to produce a profile. Default is
        None, in which case profiles will be produced from all telescopes associated with a source.
    :return: A tuple. The first elements are the sources. The second is a dens_prof_dict with source
    strings as keys, and values of dictionaries with telescope keys and values of the density 
    profile objects (ie. {src_string: {tel : dens_prof}}). The third is a temp_prof_dict with source
    strings as keys, and values of dictionaries with telescope keys and values of the temperature
    profile objects (ie. {src_string: {tel : temp_prof}}). The fourth is a dens_model_dict, with 
    source strings as keys and density models as values. The fifth is a temp_model_dict, with 
    source strings as keys and temperature models as values. The sixth is the list of telescopes we're working on.
    :rtype: Tuple[Union[BaseSource, List[BaseSource], BaseSample], dict, dict, dict, dict, List[str]]
    """

    # If the user didn't specify a particular telescope, or telescopes, from which we are to
    #  produce temperature profiles, we fetch all associated with at least one source
    if telescope is None:
        src_telescopes = _get_all_telescopes(sources)
    elif isinstance(telescope, str):
        src_telescopes = [telescope]
    else:
        src_telescopes = telescope

    sources, outer_rads, has_glob_temp = _setup_global(sources, outer_radius, global_radius, abund_table, group_spec,
                                                       spec_min_counts, spec_min_sn, over_sample, num_cores, psf_bins,
                                                       stacked_spectra, src_telescopes)
    
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

    # Here we take only the sources that have a successful global temperature measurement for at 
    # least one of the associated telescopes
    cut_sources = []
    for src_ind, src in enumerate(sources):
        # The format of has_glob_temp is a dictionary with telescope keys, and then an array of booleans
        # ie. has_glob_temp = {'xmm' : [True, True, False], 'erosita' : [True, True, True]}
        # So we need to cycle through each key and collect the correct indicies to the corresponding source
        # has_temp is storing the boolean for every telescope 
        has_temp = []
        for key in has_glob_temp:
            has_temp.append(has_glob_temp[key][src_ind])

        # If a source has a global temperature from at least one telescope, we'll continue analysing
        # If the sum is 0 that means every element in has_temp was False, and we discard this source
        if sum(has_temp) > 0:
            cut_sources.append(src)
    
    # Collecting the abridged radii needed now we have cut the sources
    cut_rads = Quantity([rads_dict[str(src)] for src in cut_sources])
    if len(cut_sources) == 0:
        raise ValueError("No sources have a successful global temperature measurement.")
    
    # I know this looks nasty, but I had to do this to avoid a circular import error
    from ..sourcetools.temperature import onion_deproj_temp_prof
    # Attempt to measure their 3D temperature profiles
    temp_profs = onion_deproj_temp_prof(cut_sources, cut_rads, temp_annulus_method, temp_min_snr, temp_min_cnt,
                                        temp_min_width, temp_use_combined, temp_use_worst, min_counts=spec_min_counts,
                                        min_sn=spec_min_sn, over_sample=over_sample, one_rmf=one_rmf,
                                        freeze_met=freeze_met, abund_table=abund_table, temp_lo_en=temp_lo_en,
                                        temp_hi_en=temp_hi_en, num_cores=num_cores, stacked_spectra=stacked_spectra,
                                        telescope=src_telescopes)
    print("temp_profs")
    print(temp_profs)
    # We are reorganising this temp_profs output so it is easier to cycle through in later functions
    # temp_prof_dict will have sources as keys, then a dictionary value, this dictionary has
    # telescope keys with values that are the profile object, ie. {src1: {'xmm' : Profile}}
    temp_prof_dict = {}
    for p_ind, p in enumerate(cut_sources):
        # this is the nested dict that will have telescope keys and profile object values
        src_dict = {}
        for tel in temp_profs:
            # temp_profs is of the form {'xmm': [Profile, Profile, etc.]}
            src_dict[tel] = temp_profs[tel][p_ind]
        temp_prof_dict[str(cut_sources[p_ind])] = src_dict
    
    print('temp_prof_dict')
    print(temp_prof_dict)
    # Now we take only the sources that have successful 3D temperature profiles. 
    # We do the temperature profile stuff first because its more difficult, and why should we waste 
    # time on a density profile if the temperature profile cannot even be measured.
    # We keep sources that have at least one successfully measured profile from any of the associated telescopes.
    cut_cut_sources = []
    for p_ind, p in enumerate(cut_sources):
        # the string of the source objects is the key in temp_prof_dict
        src_key = str(cut_sources[p_ind])
        # storing the profile objects for every telescope in this list
        has_prof = []
        for tel in temp_prof_dict[src_key]:
            prof = temp_prof_dict[src_key][tel]
            has_prof.append(prof)
        # keeping a source if at least one profile has been measured for any of the telescopes
        if has_prof.count(None) != len(temp_prof_dict[src_key]):
            cut_cut_sources.append(cut_sources[p_ind])

    cut_cut_rads = Quantity([rads_dict[str(src)] for src in cut_cut_sources])

    # And checking again if this stage of the measurement worked out
    if len(cut_cut_sources) == 0:
        raise ValueError("No sources have a successful temperature profile measurement.")

    # We also need to setup the sb model list for our cut sample
    sb_models_cut = [sb_model_dict[str(src)] for src in cut_cut_sources]
     # I know this looks nasty, but I had to do this to avoid a circular import error
    from ..sourcetools.density import inv_abel_fitted_model
    dens_profs = inv_abel_fitted_model(cut_cut_sources, sb_models_cut, fit_method, cut_cut_rads, pix_step=sb_pix_step,
                                       min_snr=sb_min_snr, abund_table=abund_table, psf_bins=psf_bins,
                                       num_walkers=num_walkers, num_steps=num_steps, group_spec=group_spec,
                                       min_counts=spec_min_counts, min_sn=spec_min_sn, over_sample=over_sample,
                                       conv_outer_radius=global_radius, inv_abel_method=inv_abel_method,
                                       num_cores=num_cores, show_warn=show_warn, stacked_spectra=stacked_spectra,
                                       telescope=src_telescopes)
    
    # Once again reformatting this output to lookup density profiles based on source
    # so dens_prof_dict will be of the form: {src_key : {'xmm': prof, 'erosita': prof} etc.}
    dens_prof_dict = {}
    for p_ind, p in enumerate(cut_cut_sources):
        # this is the nested dict that will have telescope keys and profile object values
        src_dict = {}
        for tel in dens_profs:
            # dens_profs is of the form {'xmm': [Profile, Profile, etc.]}
            src_dict[tel] = dens_profs[tel][p_ind]
        dens_prof_dict[str(cut_cut_sources[p_ind])] = src_dict

    return sources, dens_prof_dict, temp_prof_dict, dens_model_dict, temp_model_dict, src_telescopes

