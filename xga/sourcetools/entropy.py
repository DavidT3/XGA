#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/04/2023, 15:07. Copyright (c) The Contributors

from typing import Union, List
from warnings import warn

from astropy.units import Quantity
from tqdm import tqdm

from .. import NUM_CORES
from ..exceptions import XGAFitError
from ..models import BaseModel1D
from ..products.profile import SpecificEntropy
from ..samples import ClusterSample
from ..sources import GalaxyCluster
from ..sourcetools._common import _setup_inv_abel_dens_onion_temp


def entropy_inv_abel_dens_onion_temp(sources: Union[GalaxyCluster, ClusterSample], outer_radius: Union[str, Quantity],
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
                                     show_warn: bool = True) -> Union[List[SpecificEntropy], SpecificEntropy]:
    """
    A convenience function that should allow the user to easily measure specific entropy profiles for a sample of
    galaxy clusters, elegantly dealing with any sources that have inadequate data or aren't fit properly. For
    the sake of convenience, I have taken away a lot of choices that can be passed into the density and temperature
    measurement routines, and if you would like more control then please manually define a specific entropy profile
    object.

    This function uses the inv_abel_fitted_model density measurement function, and the onion_deproj_temp_prof
    temperature measurement function (with the minimum signal to noise criteria for deciding on the annular
    spectra sizes).

    The bulk of this code is the same as the hydrostatic mass measurement convenience function that also uses the
    inverse Abel density method, and the onion peeling temperature method, as the same physical information is
    required to measure the entropy.

    :param GalaxyCluster/ClusterSample sources: The galaxy cluster, or sample of galaxy clusters, that you wish to
        measure specific entropy profiles for.
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
    :return: A list of the specific entropy profiles measured by this function, though if the measurement was not
        successful an entry of None will be added to the list.
    :rtype: List[SpecificEntropy]/SpecificEntropy
    """
    # Call this common function which checks for whether temperature profiles/density profiles exist, if not creates
    #  them, and tries to fit the requested models to them - implemented like this because it is an identical process
    #  to that required by the hydrostatic mass function of similar name
    sources, dens_prof_dict, temp_prof_dict, dens_model_dict, \
        temp_model_dict = _setup_inv_abel_dens_onion_temp(sources, outer_radius, sb_model, dens_model, temp_model,
                                                          global_radius, fit_method, num_walkers, num_steps,
                                                          sb_pix_step, sb_min_snr, inv_abel_method, temp_annulus_method,
                                                          temp_min_snr, temp_min_cnt, temp_min_width, temp_use_combined,
                                                          temp_use_worst, freeze_met, abund_table, temp_lo_en,
                                                          temp_hi_en, group_spec, spec_min_counts, spec_min_sn,
                                                          over_sample, one_rmf, num_cores, show_warn)

    # So I can return a list of profiles, a tad more elegant than fetching them from the sources sometimes
    final_entropy_profs = []
    # Better to use a with statement for tqdm, so it shut down if something fails inside
    prog_desc = "Generating {} specific entropy profile"
    with tqdm(desc=prog_desc.format("None"), total=len(sources)) as onwards:
        for src in sources:
            onwards.set_description(prog_desc.format(src.name))
            # If every stage of this analysis has worked then we setup the entropy profile
            if str(src) in dens_prof_dict and dens_prof_dict[str(src)] is not None:
                # This fetches out the correct density and temperature profiles
                d_prof = dens_prof_dict[str(src)]
                t_prof = temp_prof_dict[str(src)]

                # And the appropriate temperature and density models
                d_model = dens_model_dict[str(src)]
                t_model = temp_model_dict[str(src)]

                # Set up the specific entropy profile using the temperature radii as they will tend to be spaced a lot
                #  wider than the density radii.
                try:
                    rads = t_prof.radii.copy()[1:]
                    rad_errs = t_prof.radii_err.copy()[1:]
                    deg_rads = src.convert_radius(rads, 'deg')
                    entropy = SpecificEntropy(t_prof, t_model, d_prof, d_model, rads, rad_errs, deg_rads, fit_method,
                                              num_walkers, num_steps, show_warn=show_warn, progress=False)
                    # Add the profile to the source storage structure
                    src.update_products(entropy)
                    # Also put it into a list for returning
                    final_entropy_profs.append(entropy)
                except XGAFitError:
                    warn("A fit failure occurred in the specific entropy profile definition.", stacklevel=2)
                    final_entropy_profs.append(None)

            # If the density generation failed we give a warning here
            elif str(src) in dens_prof_dict:
                warn("The density profile for {} could not be generated".format(src.name), stacklevel=2)
                # No density means no entropy, so we append None to the list
                final_entropy_profs.append(None)
            else:
                # And again this is a failure state, so we append a None to the list
                final_entropy_profs.append(None)

            onwards.update(1)
        onwards.set_description("Complete")

    # In case only one source is being analysed
    if len(final_entropy_profs) == 1:
        final_entropy_profs = final_entropy_profs[0]
    return final_entropy_profs