#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 26/03/2025, 19:42. Copyright (c) The Contributors

from typing import Union, List

from astropy.units import Quantity

from .misc import model_check
from .. import NUM_CORES
from ..imagetools.psf import rl_psf
from ..models import BaseModel1D
from ..samples import ClusterSample
from ..sas._common import region_setup
from ..sources import BaseSource, GalaxyCluster
from ..sourcetools.density import inv_abel_fitted_model
from ..sourcetools.temperature import onion_deproj_temp_prof


def _setup_global(sources, outer_radius, global_radius, abund_table: str, group_spec: bool, min_counts: int,
                  min_sn: float, over_sample: float, num_cores: int, psf_bins: int):

    out_rads = region_setup(sources, outer_radius, Quantity(0, 'arcsec'), False, '')[-1]
    global_out_rads = region_setup(sources, global_radius, Quantity(0, 'arcsec'), False, '')[-1]

    # If it's a single source I shove it in a list, so I can just iterate over the sources parameter
    #  like I do when it's a Sample object
    if isinstance(sources, BaseSource):
        sources = [sources]

    # We also want to make sure that everything has a PSF corrected image, using all the default settings
    rl_psf(sources, bins=psf_bins)

    # We do this here (even though its also in the density measurement), because if we can't measure a global
    #  temperature then its absurdly unlikely that we'll be able to measure a temperature profile, so we can avoid
    #  even trying and save some time.
    # single_temp_apec(sources, global_radius, abund_table=abund_table, group_spec=group_spec, min_counts=min_counts,
    #                  min_sn=min_sn, over_sample=over_sample, num_cores=num_cores)

    # has_glob_temp = []
    # for src_ind, src in enumerate(sources):
    #     try:
    #         src.get_temperature(global_out_rads[src_ind], 'constant*tbabs*apec', group_spec=group_spec,
    #                             min_counts=min_counts, min_sn=min_sn, over_sample=over_sample,
    #                             fit_conf={'abund_table': abund_table,})
    #         has_glob_temp.append(True)
    #     except ModelNotAssociatedError:
    #         warn("The global temperature fit for {} has failed, measuring a temperature profile from annular "
    #              "spectra may not succeed.".format(src.name), stacklevel=2)
    #         has_glob_temp.append(False)
    # has_glob_temp

    return sources, out_rads


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
                                    show_warn: bool = True, psf_bins: int = 4):

    # This used to return a 'has_glob_temp' argument, with the thought that if a global temperature measurement was
    #  successful a temperature profile definitely wouldn't work - that was a naive and overly restrictive decision, as
    #  there are reasons a global fit would fail where a profile would not (the data being so high quality that a
    #  single temperature model is a very poor fit, for instance).
    sources, outer_rads = _setup_global(sources, outer_radius, global_radius, abund_table, group_spec,
                                        spec_min_counts, spec_min_sn, over_sample, num_cores, psf_bins)
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
    # cut_sources = [src for src_ind, src in enumerate(sources) if has_glob_temp[src_ind]]
    # cut_rads = Quantity([rads_dict[str(src)] for src in cut_sources])
    # if len(cut_sources) == 0:
    #     raise ValueError("No sources have a successful global temperature measurement.")

    # Attempt to measure their 3D temperature profiles
    temp_profs = onion_deproj_temp_prof(sources, outer_rads, temp_annulus_method, temp_min_snr, temp_min_cnt,
                                        temp_min_width, temp_use_combined, temp_use_worst, min_counts=spec_min_counts,
                                        min_sn=spec_min_sn, over_sample=over_sample, one_rmf=one_rmf,
                                        freeze_met=freeze_met, abund_table=abund_table, temp_lo_en=temp_lo_en,
                                        temp_hi_en=temp_hi_en, num_cores=num_cores)

    # This just allows us to quickly look-up the temperature profile we need later
    temp_prof_dict = {str(sources[p_ind]): p for p_ind, p in enumerate(temp_profs)}

    # Now we take only the sources that have successful 3D temperature profiles. We do the temperature profile
    #  stuff first because its more difficult, and why should we waste time on a density profile if the temperature
    #  profile cannot even be measured.
    cut_sources = [sources[prof_ind] for prof_ind, prof in enumerate(temp_profs) if prof is not None]
    cut_rads = Quantity([rads_dict[str(src)] for src in cut_sources])

    # And checking again if this stage of the measurement worked out
    if len(cut_sources) == 0:
        raise ValueError("No sources have a successful temperature profile measurement.")

    # We also need to setup the sb model list for our cut sample
    sb_models_cut = [sb_model_dict[str(src)] for src in cut_sources]
    # Now we run the inverse abel density profile generator
    dens_profs = inv_abel_fitted_model(cut_sources, sb_models_cut, fit_method, cut_rads, pix_step=sb_pix_step,
                                       min_snr=sb_min_snr, abund_table=abund_table, num_steps=num_steps,
                                       num_walkers=num_walkers, group_spec=group_spec, min_counts=spec_min_counts,
                                       min_sn=spec_min_sn, over_sample=over_sample, conv_outer_radius=global_radius,
                                       inv_abel_method=inv_abel_method, num_cores=num_cores, show_warn=show_warn,
                                       psf_bins=psf_bins)
    # Set this up to lookup density profiles based on source
    dens_prof_dict = {str(cut_sources[p_ind]): p for p_ind, p in enumerate(dens_profs)}

    return sources, dens_prof_dict, temp_prof_dict, dens_model_dict, temp_model_dict
