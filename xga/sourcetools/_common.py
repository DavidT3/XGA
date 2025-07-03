#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 03/07/2025, 10:55. Copyright (c) The Contributors

from typing import Union, List
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


def _get_all_telescopes(sources: Union[BaseSource, BaseSample, list]) -> list:
    """
    Returns a list of all the telescopes associated to each Source. For most functions within
    sourcetools, the initial sources argument may be a list, so the telescopes attribute can't be
    used.
    """

    if isinstance(sources, list):
        # This collects all telescopes associated with each source, so there will be duplicates
        all_telescopes_inc_dups = []
        for src in sources:
            all_telescopes_inc_dups.extend(src.telescopes)
        # and now removing the duplicates
        all_telescopes = list(set(all_telescopes_inc_dups))
    
    else:
        all_telescopes = sources.telescopes
    
    return all_telescopes

def _setup_global(sources, outer_radius, global_radius, abund_table: str, group_spec: bool, min_counts: int,
                  min_sn: float, over_sample: float, num_cores: int, psf_bins: int, stacked_spectra: bool):

    out_rads = region_setup(sources, outer_radius, Quantity(0, 'arcsec'), False, '')[-1]
    global_out_rads = region_setup(sources, global_radius, Quantity(0, 'arcsec'), False, '')[-1]

    all_tels = sources.telescopes
    # If it's a single source I shove it in a list, so I can just iterate over the sources parameter
    #  like I do when it's a Sample object
    if isinstance(sources, BaseSource):
        sources = [sources]

    # We also want to make sure that everything has a PSF corrected image, using all the default settings
    rl_psf(sources, bins=psf_bins)

    # We do this here (even though its also in the density measurement), because if we can't measure a global
    #  temperature then its absurdly unlikely that we'll be able to measure a temperature profile, so we can avoid
    #  even trying and save some time.
    single_temp_apec(sources, global_radius, abund_table=abund_table, group_spec=group_spec, min_counts=min_counts,
                     min_sn=min_sn, over_sample=over_sample, num_cores=num_cores, stacked_spectra=stacked_spectra)

    # returning a dictionary of telescope keys and values that are a list of len(sources) where 
    # each element in the list is a boolean indicated whether a glob temp has been measured
    # ie. has_glob_temp = {'xmm' : [True, True, False], 'erosita' : [True, True, True]}
    has_glob_temp = {key : [] for key in all_tels}
    for src_ind, src in enumerate(sources):
        # We cycle over the telescopes in the Sample and not the Source, so that every list in 
        # has_glob_temp is the same length
        for tel in all_tels:
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
                                    global_radius: Quantity,
                                    fit_method: str = "mcmc", num_walkers: int = 20, num_steps: int = 20000,
                                    sb_pix_step: int = 1, sb_min_snr: Union[int, float] = 0.0,
                                    inv_abel_method: str = None,
                                    temp_annulus_method: str = 'min_snr', temp_min_snr: float = 30,
                                    temp_min_cnt: Union[int, Quantity] = Quantity(1000, 'ct'),
                                    temp_min_width: Quantity = Quantity(20, 'arcsec'), temp_use_combined: bool = True,
                                    temp_use_worst: bool = False, freeze_met: bool = True, abund_table: str = "angr",
                                    temp_lo_en: Quantity = Quantity(0.3, 'keV'),
                                    temp_hi_en: Quantity = Quantity(7.9, 'keV'),
                                    group_spec: bool = True, spec_min_counts: int = 5, spec_min_sn: float = None,
                                    over_sample: float = None, one_rmf: bool = True, num_cores: int = NUM_CORES,
                                    show_warn: bool = True, psf_bins: int = 4, stacked_spectra: bool = False):

    sources, outer_rads, has_glob_temp = _setup_global(sources, outer_radius, global_radius, abund_table, group_spec,
                                                       spec_min_counts, spec_min_sn, over_sample, num_cores, psf_bins,
                                                       stacked_spectra)
    
    rads_dict = {str(sources[r_ind]): r for r_ind, r in enumerate(outer_rads)}

    # This checks and sets up a predictable structure for the models needed for this measurement.
    sb_model = model_check(sources, sb_model)
    dens_model = model_check(sources, dens_model)
    temp_model = model_check(sources, temp_model)
    print('temp_model')
    print(temp_model)

    # I also set up dictionaries, so that models for specific clusters (as you can pass individual model instances
    #  for different clusters) are assigned to the right source when we start cutting down the sources based on
    #  whether a measurement has been successful
    sb_model_dict = {str(sources[m_ind]): m for m_ind, m in enumerate(sb_model)}
    dens_model_dict = {str(sources[m_ind]): m for m_ind, m in enumerate(dens_model)}
    temp_model_dict = {str(sources[m_ind]): m for m_ind, m in enumerate(temp_model)}
    print('temp_model_dict')
    print(temp_model_dict)

    # Here we take only the sources that have a successful global temperature measurement for at 
    # least one of the associated telescopes
    cut_sources = []
    for sind, src in enumerate(sources):
        # The format of has_glob_temp is a dictionary with telescope keys, and then an array of booleans
        # ie. has_glob_temp = {'xmm' : [True, True, False], 'erosita' : [True, True, True]}
        # So we need to cycle through each key and collect the correct indicies to the corresponding source
        # has_temp is storing the boolean for every telescope 
        has_temp = []
        for key in has_glob_temp:
            has_temp.append(has_glob_temp[key][sind])

        # If a source has a glob temp in at least one telescope it gets passed on
        # If the sum is 0 that means every element in has_temp was False, and we discard these sources
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
                                        temp_hi_en=temp_hi_en, num_cores=num_cores, stacked_spectra=stacked_spectra)

    # DAVID_QUESTION case where a source has a measured glob temp in one telescope and not the others
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

    # Now we take only the sources that have successful 3D temperature profiles. 
    # We do the temperature profile stuff first because its more difficult, and why should we waste 
    # time on a density profile if the temperature profile cannot even be measured.
    # We keep source that have had at least one successful profile in any of the associated telescopes.
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
                                       min_snr=sb_min_snr, abund_table=abund_table, num_steps=num_steps,
                                       num_walkers=num_walkers, group_spec=group_spec, min_counts=spec_min_counts,
                                       min_sn=spec_min_sn, over_sample=over_sample, conv_outer_radius=global_radius,
                                       inv_abel_method=inv_abel_method, num_cores=num_cores, show_warn=show_warn,
                                       psf_bins=psf_bins, stacked_spectra=stacked_spectra)
    
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

    print('dens_prof_dict')
    print(dens_prof_dict)
    return sources, dens_prof_dict, temp_prof_dict, dens_model_dict, temp_model_dict

