#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/4/26, 1:07 PM. Copyright (c) The Contributors.
"""
This module implements global XSPEC spectral fitting functions for XGA sources and samples, including both
dedicated convenience functions for common models (e.g. single_temp_apec, power_law) and the generic
general_model_fit function for fitting arbitrary XSPEC model expressions.
"""

from copy import deepcopy
from inspect import Parameter, signature
from warnings import warn

import astropy.units as u
import numpy as np
from astropy.units import Quantity

from ... import NUM_CORES
from ...exceptions import FitConfNotAssociatedError, ModelNotAssociatedError, XGADeveloperError
from ...samples.base import BaseSample
from ...sources import BaseSource
from ..fitconfgen import FIT_FUNC_ARGS, _gen_fit_conf
from ..run import xspec_call
from ._common import (
    _check_inputs,
    _convert_par_val_to_xspec,
    _pregen_spectra,
    _spec_obj_setup,
    _validate_par_hard_limits,
    _write_xspec_script,
)
from .models import validate_and_parse_model


@xspec_call
def single_temp_apec(
    sources: BaseSource | BaseSample | list[BaseSource],
    outer_radius: str | Quantity,
    inner_radius: str | Quantity = Quantity(0, "arcsec"),
    start_temp: Quantity = Quantity(3.0, "keV"),
    start_met: float | list = 0.3,
    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
    freeze_nh: bool = True,
    freeze_met: bool = True,
    freeze_temp: bool = False,
    lo_en: Quantity = Quantity(0.3, "keV"),
    hi_en: Quantity = Quantity(7.9, "keV"),
    par_fit_stat: float = 1.0,
    lum_conf: float = 68.0,
    abund_table: str = "angr",
    fit_method: str = "leven",
    group_spec: bool = True,
    min_counts: int = 5,
    min_sn: int | float | None = None,
    over_sample: float | None = None,
    one_rmf: bool = True,
    num_cores: int = NUM_CORES,
    spectrum_checking: bool = True,
    timeout: Quantity = Quantity(1, "hr"),
    stacked_spectra: bool = False,
    telescope: str | list[str] | None = None,
    force_gen: bool = False,
) -> BaseSource | BaseSample | list[BaseSource]:
    """
    This is a convenience function for fitting an absorbed single temperature apec model (constant*tbabs*apec) to an
    object. If there are no existing spectra with the passed settings, then they will be generated automatically.

    If the spectrum checking step of the XSPEC fit is enabled (using the boolean flag spectrum_checking), then
    each individual spectrum available for a given source will be fitted, and if the measured temperature is less
    than or equal to 0.01keV, or greater than 20keV, or the temperature uncertainty is greater than 15keV, then
    that spectrum will be rejected and not included in the final fit. Spectrum checking also involves rejecting any
    spectra with fewer than 10 noticed channels.

    Freezing the temperature value of the fit is also possible, in cases where the data may not be sufficient to
    constrain it, and an external temperature constraint is used (by passing to the 'start_temp' argument).

    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any value
        passed for inner_radius is ignored, and the fit performed on spectra for the entire region. If you are
        fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). By default this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param Quantity start_temp: The initial temperature for the fit, the default is 3 keV. This value can also be
        a non-scalar Quantity, with an entry for every source in a sample (this is most useful when used with the
        'freeze_temp' argument, to provide some external constraint on temperature for objects with poor data).
    :param float/List start_met: The initial metallicity for the fit (in ZSun). This value can also be
        a list, with an entry for every source in a sample (this is most useful when used with the
        'freeze_met' argument, to provide some external constraint on metallicity for objects with poor data).
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen. Default is True.
    :param bool freeze_met: Whether the metallicity parameter in the fit should be frozen. Default is True.
    :param bool freeze_temp: Whether the temperature parameter in the fit should be frozen. Default is False
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 which should be
        equivalent to 1σ errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSerror.html)
        correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param int/float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param bool spectrum_checking: Should the spectrum checking step of the XSPEC fit (where each spectrum is fit
        individually and tested to see whether it will contribute to the simultaneous fit) be activated?
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be used for this
        XSPEC spectral fit. If a stacking procedure for a particular telescope is not supported, this function will
        instead use individual spectra for an ObsID. The default is False.
    :param str/List[str] telescope: Telescope(s) to perform the XSPEC operations for. Default is None, in which
        case the XSPEC fit will be performed individually for all telescopes associated with a source.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    :return: The source object(s) passed in.
    :rtype: Union[BaseSource, BaseSample, List[BaseSource]]
    """
    # We need to make sure that the spectra we're going to be fitting the model to with XSPEC actually exist
    sources, inn_rad_vals, out_rad_vals, telescope, eff_stack_spec = _pregen_spectra(
        sources,
        outer_radius,
        inner_radius,
        group_spec,
        min_counts,
        min_sn,
        over_sample,
        one_rmf,
        num_cores,
        stacked_spectra,
        telescope,
        force_gen,
    )

    # Confirms that input parameters are legal, and nothing silly has been passed
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # TODO REPLACE THESE REPEATED CHECKS/PREPARATION OF START TEMP/MET WITH A FUNCTION - SEE ISSUE #1509
    # Have to check that every source has a start temperature entry, if the user decided to pass a set of them
    if not start_temp.isscalar and len(start_temp) != len(sources):
        raise ValueError(
            "If a non-scalar Quantity is passed for 'start_temp', it must have one entry for each "
            f"source. It currently has {len(start_temp)} for {len(sources)} sources."
        )
    # Want to make sure that the start_temp variable is always a non-scalar Quantity with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif start_temp.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temps = Quantity([start_temp.value.copy()] * len(sources), start_temp.unit)
    else:
        all_start_temps = start_temp.copy()

    # TODO WHEN IMPLEMENTING THE ABOVE TODO, MAKE THIS CHECK SUPPORT QUANTITY AS WELL
    # We repeat the above, but for the starting metallicity value(s)
    if isinstance(start_met, (list, np.ndarray)) and len(start_met) != len(sources):
        raise ValueError(
            "If a list or array is passed to 'start_met', it must have one entry for each "
            f"source. It currently has {len(start_met)} for {len(sources)} sources."
        )
    elif isinstance(start_met, (list, np.ndarray)):
        all_start_mets = deepcopy(start_met)
    # Want to make sure that the start_met variable is always a list with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif isinstance(start_met, (float, int)):
        all_start_mets = [start_met] * len(sources)
    else:
        raise ValueError("'start_met' must be either a list/array with an entry for each source, or a float/integer.")

    # This function is for a set model, absorbed apec, so I can hard code all of this stuff.
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    model = "constant*tbabs*apec"
    par_names = "{factor nH kT Abundanc Redshift norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS["single_temp_apec"]

    sig = signature(single_temp_apec)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        problem_pars = set(list(cur_args.keys())) - set(list(rel_args.keys()))

        raise XGADeveloperError(
            f"Current keyword arguments of this function and FIT_FUNC_ARGS entry do not "
            f"match - unmatched arguments are {', '.join(problem_pars)}."
        )

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    # This function supports passing multiple sources, so we have to set up a script for all of them.
    for src_ind, source in enumerate(sources):
        # We do not do simultaneous fits with spectra from different telescopes, they are all fit separately - at
        #  least in this current setup
        for tel in source.telescopes:
            if tel not in telescope:
                continue
            # retrieving the spectrum objects needed for each source/ tel combo
            specs, storage_key = _spec_obj_setup(
                stacked_spectra,
                tel,
                source,
                out_rad_vals,
                src_ind,
                inn_rad_vals,
                group_spec,
                min_counts,
                min_sn,
                over_sample,
            )

            # For this model, we have to know the redshift of the source.
            if source.redshift is None:
                raise ValueError("You cannot supply a source without a redshift to this model.")

            # Whatever start temperature is passed gets converted to keV, this will be put in the template
            start_temp = all_start_temps[src_ind].to("keV", equivalencies=u.temperature_energy())
            start_met = all_start_mets[src_ind]
            # Another TCL list, this time of the parameter start values for this model.
            par_values = (
                f"{{{1.0} {source.nH.to('10^22 cm^-2').value} {start_temp.value} {start_met} {source.redshift} {1.0}}}"
            )

            # Set up the TCL list that defines which parameters are frozen, dependent on user input - this can now
            #  include the temperature, if the user wants it fixed at the start value
            freezing = (
                f"{{F {'T' if freeze_nh else 'F'} {'T' if freeze_temp else 'F'} {'T' if freeze_met else 'F'} T F}}"
            )

            # Set up the TCL list that defines which parameters are linked across different spectra, only the
            #  multiplicative constant that accounts for variation in normalisation over different observations is not
            #  linked
            linking = "{F T T T T T}"

            # If the user wants the spectrum cleaning step to be run, then we have to set up some acceptable
            #  limits. For this function they will be hardcoded, for simplicities' sake, and we're only going to
            #  check the temperature, as it's the main thing we're fitting for with constant*tbabs*apec
            if spectrum_checking:
                check_list = "{kT}"
                check_lo_lims = "{0.01}"
                check_hi_lims = "{20}"
                check_err_lims = "{15}"
            else:
                check_list = "{}"
                check_lo_lims = "{}"
                check_hi_lims = "{}"
                check_err_lims = "{}"

            # We generate the fit configuration key here, on a source by source basis, as the start temperature is
            #  allowed to be different for every source
            in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
            fit_conf = _gen_fit_conf(in_fit_conf)

            # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed
            #  luminosities. I am only specifying parameter 2 here (though there will likely be multiple models because
            #  there are likely multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing
            #  one will zero them all.
            nh_to_zero = "{2}"

            # If the fit has already been performed we do not wish to perform it again
            try:
                # We search for the norm parameter, as it is guaranteed to be there for any fit with this model
                source.get_results(
                    out_rad_vals[src_ind],
                    tel,
                    model,
                    inn_rad_vals[src_ind],
                    "norm",
                    group_spec,
                    min_counts,
                    min_sn,
                    over_sample,
                    eff_stack_spec[tel],
                    fit_conf,
                )
            except (ModelNotAssociatedError, FitConfNotAssociatedError):
                out_file, script_file, inv_ent = _write_xspec_script(
                    source,
                    storage_key,
                    model,
                    abund_table,
                    fit_method,
                    specs,
                    lo_en,
                    hi_en,
                    par_names,
                    par_values,
                    linking,
                    freezing,
                    par_fit_stat,
                    lum_low_lims,
                    lum_upp_lims,
                    lum_conf,
                    source.redshift,
                    spectrum_checking,
                    check_list,
                    check_lo_lims,
                    check_hi_lims,
                    check_err_lims,
                    True,
                    tel,
                    fit_conf,
                    nh_to_zero,
                )
                script_paths.append(script_file)
                outfile_paths.append(out_file)
                src_inds.append(src_ind)
                fit_confs.append(fit_conf)
                inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents  # type: ignore[return-value]


@xspec_call
def single_temp_mekal(
    sources: BaseSource | BaseSample | list[BaseSource],
    outer_radius: str | Quantity,
    inner_radius: str | Quantity = Quantity(0, "arcsec"),
    start_temp: Quantity = Quantity(3.0, "keV"),
    start_met: float = 0.3,
    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
    freeze_nh: bool = True,
    freeze_met: bool = True,
    freeze_temp: bool = False,
    lo_en: Quantity = Quantity(0.3, "keV"),
    hi_en: Quantity = Quantity(7.9, "keV"),
    par_fit_stat: float = 1.0,
    lum_conf: float = 68.0,
    abund_table: str = "angr",
    fit_method: str = "leven",
    group_spec: bool = True,
    min_counts: int = 5,
    min_sn: float | None = None,
    over_sample: float | None = None,
    one_rmf: bool = True,
    num_cores: int = NUM_CORES,
    spectrum_checking: bool = True,
    timeout: Quantity = Quantity(1, "hr"),
    stacked_spectra: bool = False,
    telescope: str | list[str] | None = None,
) -> BaseSource | BaseSample | list[BaseSource]:
    """
    This is a convenience function for fitting an absorbed single temperature mekal model(constant*tbabs*mekal) to an
    object. If there are no existing spectra with the passed settings, then they will be generated automatically.

    If the spectrum checking step of the XSPEC fit is enabled (using the boolean flag spectrum_checking), then
    each individual spectrum available for a given source will be fitted, and if the measured temperature is less
    than or equal to 0.01keV, or greater than 20keV, or the temperature uncertainty is greater than 15keV, then
    that spectrum will be rejected and not included in the final fit. Spectrum checking also involves rejecting any
    spectra with fewer than 10 noticed channels.

    Switch is set to 1, so the fit will compute the spectrum by interpolating on a pre-calculated mekal table.

    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any value
        passed for inner_radius is ignored, and the fit performed on spectra for the entire region. If you are
        fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). By default this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param Quantity start_temp: The initial temperature for the fit, the default is 3 keV. This value can also be
        a non-scalar Quantity, with an entry for every source in a sample (this is most useful when used with the
        'freeze_temp' argument, to provide some external constraint on temperature for objects with poor data).
    :param float/List start_met: The initial metallicity for the fit (in ZSun). This value can also be
        a list, with an entry for every source in a sample (this is most useful when used with the
        'freeze_met' argument, to provide some external constraint on metallicity for objects with poor data).
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.
    :param bool freeze_met: Whether the metallicity parameter in the fit should be frozen.
    :param bool freeze_temp: Whether the temperature parameter in the fit should be frozen. Default is False
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 which should be
        equivalent to 1σ errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSerror.html)
        correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param int/float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param bool spectrum_checking: Should the spectrum checking step of the XSPEC fit (where each spectrum is fit
        individually and tested to see whether it will contribute to the simultaneous fit) be activated?
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be used for this
        XSPEC spectral fit. If a stacking procedure for a particular telescope is not supported, this function will
        instead use individual spectra for an ObsID. The default is False.
    :param str/List[str] telescope: Telescope(s) to perform the XSPEC operations for. Default is None, in which
        case the XSPEC fit will be performed individually for all telescopes associated with a source.
    """
    sources, inn_rad_vals, out_rad_vals, telescope, eff_stack_spec = _pregen_spectra(
        sources,
        outer_radius,
        inner_radius,
        group_spec,
        min_counts,
        min_sn,
        over_sample,
        one_rmf,
        num_cores,
        stacked_spectra,
        telescope,
    )
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # TODO REPLACE THESE REPEATED CHECKS/PREPARATION OF START TEMP/MET WITH A FUNCTION - SEE ISSUE #1509
    # Have to check that every source has a start temperature entry, if the user decided to pass a set of them
    if not start_temp.isscalar and len(start_temp) != len(sources):
        raise ValueError(
            "If a non-scalar Quantity is passed for 'start_temp', it must have one entry for each "
            f"source. It currently has {len(start_temp)} for {len(sources)} sources."
        )
    # Want to make sure that the start_temp variable is always a non-scalar Quantity with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif start_temp.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temps = Quantity([start_temp.value.copy()] * len(sources), start_temp.unit)
    else:
        all_start_temps = start_temp.copy()

    # TODO WHEN IMPLEMENTING THE ABOVE TODO, MAKE THIS CHECK SUPPORT QUANTITY AS WELL
    # We repeat the above, but for the starting metallicity value(s)
    if isinstance(start_met, (list, np.ndarray)) and len(start_met) != len(sources):
        raise ValueError(
            "If a list or array is passed to 'start_met', it must have one entry for each "
            f"source. It currently has {len(start_met)} for {len(sources)} sources."
        )
    elif isinstance(start_met, (list, np.ndarray)):
        all_start_mets = deepcopy(start_met)
    # Want to make sure that the start_met variable is always a list with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif isinstance(start_met, (float, int)):
        all_start_mets = [start_met] * len(sources)
    else:
        raise ValueError("'start_met' must be either a list/array with an entry for each source, or a float/integer.")

    # This function is for a set model, absorbed mekal, so I can hard code all of this stuff.
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    model = "constant*tbabs*mekal"
    par_names = "{factor nH kT nH Abundanc Redshift switch norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS["single_temp_mekal"]
    sig = signature(single_temp_mekal)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        problem_pars = set(list(cur_args.keys())) - set(list(rel_args.keys()))
        raise XGADeveloperError(
            f"Current keyword arguments of this function and FIT_FUNC_ARGS entry do not "
            f"match - unmatched arguments are {', '.join(problem_pars)}."
        )

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for src_ind, source in enumerate(sources):
        for tel in source.telescopes:
            if tel not in telescope:
                continue
            # retrieving the spectrum objects needed for each source/ tel combo
            specs, storage_key = _spec_obj_setup(
                stacked_spectra,
                tel,
                source,
                out_rad_vals,
                src_ind,
                inn_rad_vals,
                group_spec,
                min_counts,
                min_sn,
                over_sample,
            )
            # For this model, we have to know the redshift of the source.
            if source.redshift is None:
                raise ValueError("You cannot supply a source without a redshift to this model.")

            # Whatever start temperature is passed gets converted to keV, this will be put in the template
            start_temp = all_start_temps[src_ind].to("keV", equivalencies=u.temperature_energy())
            # We also fetch the start metallicity
            start_met = all_start_mets[src_ind]
            # Another TCL list, this time of the parameter start values for this model.
            par_values = (
                f"{{{1.0} {source.nH.to('10^22 cm^-2').value} {start_temp.value} "
                f"{1} {start_met} {source.redshift} {1} {1.0}}}"
            )

            # Set up the TCL list that defines which parameters are frozen, dependent on user input
            freezing = (
                f"{{F {'T' if freeze_nh else 'F'} {'T' if freeze_temp else 'F'} T {'T' if freeze_met else 'F'} T T F}}"
            )

            # Set up the TCL list that defines which parameters are linked across different spectra, only the
            #  multiplicative constant that accounts for variation in normalisation over different observations is not
            #  linked
            linking = "{F T T T T T T T}"

            # If the user wants the spectrum cleaning step to be run, then we have to setup some acceptable
            #  limits. For this function they will be hardcoded, for simplicities sake, and we're only going to
            #  check the temperature, as its the main thing we're fitting for with constant*tbabs*mekal
            if spectrum_checking:
                check_list = "{kT}"
                check_lo_lims = "{0.01}"
                check_hi_lims = "{20}"
                check_err_lims = "{15}"
            else:
                check_list = "{}"
                check_lo_lims = "{}"
                check_hi_lims = "{}"
                check_err_lims = "{}"

            # We generate the fit configuration key here, on a source by source basis, as the start temperature is
            #  allowed to be different for every source
            in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
            fit_conf = _gen_fit_conf(in_fit_conf)

            # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed
            #  luminosities. I am only specifying parameter 2 here (though there will likely be multiple models because
            #  there are likely multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing
            #  one will zero them all.
            nh_to_zero = "{2}"

            # If the fit has already been performed, we do not wish to perform it again
            try:
                # We search for the norm parameter, as it is guaranteed to be there for any fit with this model
                source.get_results(
                    out_rad_vals[src_ind],
                    tel,
                    model,
                    inn_rad_vals[src_ind],
                    "norm",
                    group_spec,
                    min_counts,
                    min_sn,
                    over_sample,
                    eff_stack_spec[tel],
                    fit_conf,
                )
            except (ModelNotAssociatedError, FitConfNotAssociatedError):
                out_file, script_file, inv_ent = _write_xspec_script(
                    source,
                    storage_key,
                    model,
                    abund_table,
                    fit_method,
                    specs,
                    lo_en,
                    hi_en,
                    par_names,
                    par_values,
                    linking,
                    freezing,
                    par_fit_stat,
                    lum_low_lims,
                    lum_upp_lims,
                    lum_conf,
                    source.redshift,
                    spectrum_checking,
                    check_list,
                    check_lo_lims,
                    check_hi_lims,
                    check_err_lims,
                    True,
                    tel,
                    fit_conf,
                    nh_to_zero,
                )

                script_paths.append(script_file)
                outfile_paths.append(out_file)
                src_inds.append(src_ind)
                fit_confs.append(fit_conf)
                inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents  # type: ignore[return-value]


@xspec_call
def multi_temp_dem_apec(
    sources: BaseSource | BaseSample | list[BaseSource],
    outer_radius: str | Quantity,
    inner_radius: str | Quantity = Quantity(0, "arcsec"),
    start_max_temp: Quantity = Quantity(5.0, "keV"),
    start_met: float = 0.3,
    start_t_rat: float = 0.1,
    start_inv_em_slope: float = 0.25,
    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
    freeze_nh: bool = True,
    freeze_met: bool = True,
    lo_en: Quantity = Quantity(0.3, "keV"),
    hi_en: Quantity = Quantity(7.9, "keV"),
    par_fit_stat: float = 1.0,
    lum_conf: float = 68.0,
    abund_table: str = "angr",
    fit_method: str = "leven",
    group_spec: bool = True,
    min_counts: int = 5,
    min_sn: int | float | None = None,
    over_sample: float | None = None,
    one_rmf: bool = True,
    num_cores: int = NUM_CORES,
    spectrum_checking: bool = True,
    timeout: Quantity = Quantity(1, "hr"),
    stacked_spectra: bool = False,
    telescope: str | list[str] | None = None,
) -> BaseSource | BaseSample | list[BaseSource]:
    """
    This is a convenience function for fitting an absorbed multi temperature apec model (constant*tbabs*wdem) to
    spectra generated for XGA sources. The wdem model uses a power law distribution of the differential emission
    measure distribution. It may be a good empirical approximation for the spectra in cooling cores of
    clusters of galaxies. This implementation sets the 'switch' to 2, which means that the APEC model is used.

    If there are no existing spectra with the passed settings, then they will be generated automatically.

    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any value
        passed for inner_radius is ignored, and the fit performed on spectra for the entire region. If you are
        fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). By default this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param Quantity start_max_temp: The initial maximum temperature for the fit.
    :param float start_met: The initial metallicity for the fit (in ZSun).
    :param float start_t_rat: The initial minimum to maximum temperature ratio (beta) for the fit.
    :param float start_inv_em_slope: The initial inverse slope value of the emission measure for the fit.
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.
    :param bool freeze_met: Whether the metallicity parameter in the fit should be frozen.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 which should be
        equivalent to 1σ errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSerror.html)
        correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param int/float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param bool spectrum_checking: Should the spectrum checking step of the XSPEC fit (where each spectrum is fit
        individually and tested to see whether it will contribute to the simultaneous fit) be activated?
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be used for this
        XSPEC spectral fit. If a stacking procedure for a particular telescope is not supported, this function will
        instead use individual spectra for an ObsID. The default is False.
    :param str/List[str] telescope: Telescope(s) to perform the XSPEC operations for. Default is None, in which
        case the XSPEC fit will be performed individually for all telescopes associated with a source.
    """
    sources, inn_rad_vals, out_rad_vals, telescope, eff_stack_spec = _pregen_spectra(
        sources,
        outer_radius,
        inner_radius,
        group_spec,
        min_counts,
        min_sn,
        over_sample,
        one_rmf,
        num_cores,
        stacked_spectra,
        telescope,
    )
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # This function is for a set model, absorbed apec, so I can hard code all of this stuff.
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    model = "constant*tbabs*wdem"
    par_names = "{factor nH Tmax beta inv_slope nH abundanc Redshift switch norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS["multi_temp_dem_apec"]
    sig = signature(multi_temp_dem_apec)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        problem_pars = set(list(cur_args.keys())) - set(list(rel_args.keys()))
        raise XGADeveloperError(
            f"Current keyword arguments of this function and FIT_FUNC_ARGS entry do not "
            f"match - unmatched arguments are {', '.join(problem_pars)}."
        )

    # We generate the fit configuration key - in this case there are no relevant variables that can have different
    #  values for different sources, so we don't need to put this in the loop. Still, we will pass back  a list of
    #  fit configuration keys because the xspec call decorator will expect a list with one entry per source that
    #  is having a fit run
    in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
    fit_conf = _gen_fit_conf(in_fit_conf)

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for src_ind, source in enumerate(sources):
        for tel in source.telescopes:
            if tel not in telescope:
                continue
            # retrieving the spectrum objects needed for each source/ tel combo
            specs, storage_key = _spec_obj_setup(
                stacked_spectra,
                tel,
                source,
                out_rad_vals,
                src_ind,
                inn_rad_vals,
                group_spec,
                min_counts,
                min_sn,
                over_sample,
            )
            # For this model, we have to know the redshift of the source.
            if source.redshift is None:
                raise ValueError("You cannot supply a source without a redshift to this model.")

            # Whatever start temperature is passed gets converted to keV, this will be put in the template
            t = start_max_temp.to("keV", equivalencies=u.temperature_energy()).value

            # Another TCL list, this time of the parameter start values for this model.
            par_values = (
                f"{{{1.0} {source.nH.to('10^22 cm^-2').value} {t} {start_t_rat} "
                f"{start_inv_em_slope} {1} {start_met} {source.redshift} {2} {1.0}}}"
            )

            # Set up the TCL list that defines which parameters are frozen, dependent on user input
            freezing = f"{{F {'T' if freeze_nh else 'F'} F F F T {'T' if freeze_met else 'F'} T T F}}"

            # Set up the TCL list that defines which parameters are linked across different spectra, only the
            #  multiplicative constant that accounts for variation in normalisation over different observations is not
            #  linked
            linking = "{F T T T T T T T T T}"

            # If the user wants the spectrum cleaning step to be run, then we have to setup some acceptable
            #  limits. The check limits here are somewhat of a guesstimate based on my understanding of the model
            #  rather than on practical experience with it
            if spectrum_checking:
                check_list = "{Tmax beta inv_slope}"
                check_lo_lims = "{0.01 0.01 0.1}"
                check_hi_lims = "{20 1 20}"
                check_err_lims = "{15 5 5}"
            else:
                check_list = "{}"
                check_lo_lims = "{}"
                check_hi_lims = "{}"
                check_err_lims = "{}"

            # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed
            #  luminosities. I am only specifying parameter 2 here (though there will likely be multiple models
            #  because there are likely multiple spectra) because I know that nH of tbabs is linked in this setup, so
            #  zeroing one will zero them all.
            nh_to_zero = "{2}"

            # If the fit has already been performed we do not wish to perform it again
            try:
                source.get_results(
                    out_rad_vals[src_ind],
                    tel,
                    model,
                    inn_rad_vals[src_ind],
                    "Tmax",
                    group_spec,
                    min_counts,
                    min_sn,
                    over_sample,
                    eff_stack_spec[tel],
                    fit_conf,
                )
            except (ModelNotAssociatedError, FitConfNotAssociatedError):
                # This internal function writes out the XSPEC script with all the information we've assembled in this
                #  function - filling out the XSPEC template and writing to disk
                out_file, script_file, inv_ent = _write_xspec_script(
                    source,
                    storage_key,
                    model,
                    abund_table,
                    fit_method,
                    specs,
                    lo_en,
                    hi_en,
                    par_names,
                    par_values,
                    linking,
                    freezing,
                    par_fit_stat,
                    lum_low_lims,
                    lum_upp_lims,
                    lum_conf,
                    source.redshift,
                    spectrum_checking,
                    check_list,
                    check_lo_lims,
                    check_hi_lims,
                    check_err_lims,
                    True,
                    tel,
                    fit_conf,
                    nh_to_zero,
                )
                script_paths.append(script_file)
                outfile_paths.append(out_file)
                src_inds.append(src_ind)
                fit_confs.append(fit_conf)
                inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents  # type: ignore[return-value]


@xspec_call
def power_law(
    sources: BaseSource | BaseSample | list[BaseSource],
    outer_radius: str | Quantity,
    inner_radius: str | Quantity = Quantity(0, "arcsec"),
    redshifted: bool = False,
    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
    start_pho_index: float = 1.0,
    lo_en: Quantity = Quantity(0.3, "keV"),
    hi_en: Quantity = Quantity(7.9, "keV"),
    freeze_nh: bool = True,
    par_fit_stat: float = 1.0,
    lum_conf: float = 68.0,
    abund_table: str = "angr",
    fit_method: str = "leven",
    group_spec: bool = True,
    min_counts: int = 5,
    min_sn: float | None = None,
    over_sample: float | None = None,
    one_rmf: bool = True,
    num_cores: int = NUM_CORES,
    timeout: Quantity = Quantity(1, "hr"),
    stacked_spectra: bool = False,
    telescope: str | list[str] | None = None,
) -> BaseSource | BaseSample | list[BaseSource]:
    """
    This is a convenience function for fitting a tbabs absorbed powerlaw (or zpowerlw if redshifted
    is selected) to source spectra, with a multiplicative constant included to deal with different spectrum
    normalisations (constant*tbabs*powerlaw, or constant*tbabs*zpowerlw).

    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'point' would be acceptable for a PointSource,
        or Quantity(40, 'arcsec')). If 'region' is chosen (to use the regions in region files), then any value
        passed for inner_radius is ignored, and the fit performed on spectra for the entire region. If you are
        fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius of the region that the
        desired spectrum covers (for instance 'point' would be acceptable for a PointSource,
        or Quantity(40, 'arcsec')). By default this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param bool redshifted: Whether the powerlaw that includes redshift (zpowerlw) should be used.
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param float start_pho_index: The starting value for the photon index of the powerlaw.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
        :param bool freeze_nh: Whether the hydrogen column density should be frozen.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 which
        should be equivalent to 1sigma errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec
        /manual/XSerror.html) correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param int/float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be used for this
        XSPEC spectral fit. If a stacking procedure for a particular telescope is not supported, this function will
        instead use individual spectra for an ObsID. The default is False.
    :param str/List[str] telescope: Telescope(s) to perform the XSPEC operations for. Default is None, in which
        case the XSPEC fit will be performed individually for all telescopes associated with a source.
    """
    sources, inn_rad_vals, out_rad_vals, telescope, eff_stack_spec = _pregen_spectra(
        sources,
        outer_radius,
        inner_radius,
        group_spec,
        min_counts,
        min_sn,
        over_sample,
        one_rmf,
        num_cores,
        stacked_spectra,
        telescope,
    )
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # This function is for a set model, either absorbed powerlaw or absorbed zpowerlw
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"
    if redshifted:
        model = "constant*tbabs*zpowerlw"
        par_names = "{factor nH PhoIndex Redshift norm}"
    else:
        model = "constant*tbabs*powerlaw"
        par_names = "{factor nH PhoIndex norm}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS["power_law"]
    sig = signature(power_law)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        problem_pars = set(list(cur_args.keys())) - set(list(rel_args.keys()))
        raise XGADeveloperError(
            f"Current keyword arguments of this function and FIT_FUNC_ARGS entry do not "
            f"match - unmatched arguments are {', '.join(problem_pars)}."
        )

    # We generate the fit configuration key - in this case there are no relevant variables that can have different
    #  values for different sources, so we don't need to put this in the loop. Still, we will pass back  a list of
    #  fit configuration keys because the xspec call decorator will expect a list with one entry per source that
    #  is having a fit run
    in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
    fit_conf = _gen_fit_conf(in_fit_conf)

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    for src_ind, source in enumerate(sources):
        for tel in source.telescopes:
            if tel not in telescope:
                continue
            # retrieving the spectrum objects needed for each source/ tel combo
            specs, storage_key = _spec_obj_setup(
                stacked_spectra,
                tel,
                source,
                out_rad_vals,
                src_ind,
                inn_rad_vals,
                group_spec,
                min_counts,
                min_sn,
                over_sample,
            )
            # For this model, we have to know the redshift of the source.
            if redshifted and source.redshift is None:
                raise ValueError("You cannot supply a source without a redshift if you have elected to fit zpowerlw.")
            elif redshifted and source.redshift is not None:
                par_values = f"{{{1.0} {source.nH.to('10^22 cm^-2').value} {start_pho_index} {source.redshift} {1.0}}}"
            else:
                par_values = f"{{{1.0} {source.nH.to('10^22 cm^-2').value} {start_pho_index} {1.0}}}"

            # Set up the TCL list that defines which parameters are frozen, dependent on user input
            if redshifted and freeze_nh:
                freezing = "{F T F T F}"
            elif not redshifted and freeze_nh:
                freezing = "{F T F F}"
            elif redshifted and not freeze_nh:
                freezing = "{F F F T F}"
            elif not redshifted and not freeze_nh:
                freezing = "{F F F F}"

            # Set up the TCL list that defines which parameters are linked across different spectra,
            #  dependant on user input
            if redshifted:
                linking = "{F T T T T}"
            else:
                linking = "{F T T T}"

            # If the powerlaw with redshift has been chosen, then we use the redshift attached to the source object
            #  If not we just pass a filler redshift and the luminosities are invalid
            if source.redshift is not None:
                z = source.redshift
            else:
                z = 1
                warn(
                    f"{source.name} has no redshift information associated, so luminosities from this fit"
                    f" will be invalid, as redshift has been set to one.",
                    stacklevel=2,
                )

            # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed
            #  luminosities. I am only specifying parameter 2 here (though there will likely be multiple models
            #  because there are likely multiple spectra) because I know that nH of tbabs is linked in this setup, so
            #  zeroing one will zero them all.
            nh_to_zero = "{2}"

            # If the fit has already been performed we do not wish to perform it again
            try:
                source.get_results(
                    out_rad_vals[src_ind],
                    tel,
                    model,
                    inn_rad_vals[src_ind],
                    None,
                    group_spec,
                    min_counts,
                    min_sn,
                    over_sample,
                    eff_stack_spec[tel],
                    fit_conf,
                )
            except (ModelNotAssociatedError, FitConfNotAssociatedError):
                out_file, script_file, inv_ent = _write_xspec_script(
                    source,
                    storage_key,
                    model,
                    abund_table,
                    fit_method,
                    specs,
                    lo_en,
                    hi_en,
                    par_names,
                    par_values,
                    linking,
                    freezing,
                    par_fit_stat,
                    lum_low_lims,
                    lum_upp_lims,
                    lum_conf,
                    z,
                    False,
                    "{}",
                    "{}",
                    "{}",
                    "{}",
                    True,
                    tel,
                    fit_conf,
                    nh_to_zero,
                )
                script_paths.append(script_file)
                outfile_paths.append(out_file)
                src_inds.append(src_ind)
                fit_confs.append(fit_conf)
                inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents  # type: ignore[return-value]


@xspec_call
def blackbody(
    sources: BaseSource | BaseSample | list[BaseSource],
    outer_radius: str | Quantity,
    inner_radius: str | Quantity = Quantity(0, "arcsec"),
    redshifted: bool = False,
    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
    start_temp: Quantity = Quantity(1, "keV"),
    lo_en: Quantity = Quantity(0.3, "keV"),
    hi_en: Quantity = Quantity(7.9, "keV"),
    freeze_nh: bool = True,
    par_fit_stat: float = 1.0,
    lum_conf: float = 68.0,
    abund_table: str = "angr",
    fit_method: str = "leven",
    group_spec: bool = True,
    min_counts: int = 5,
    min_sn: float | None = None,
    over_sample: float | None = None,
    one_rmf: bool = True,
    num_cores: int = NUM_CORES,
    timeout: Quantity = Quantity(1, "hr"),
    stacked_spectra: bool = False,
    telescope: str | list[str] | None = None,
) -> BaseSource | BaseSample | list[BaseSource]:
    """
    This is a convenience function for fitting a tbabs absorbed blackbody (or zbbody if redshifted
    is selected) to source spectra, with a multiplicative constant included to deal with different spectrum
    normalisations (constant*tbabs*bbody, or constant*tbabs*zbbody).

    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'point' would be acceptable for a PointSource,
        or Quantity(40, 'arcsec')). If 'region' is chosen (to use the regions in region files), then any value
        passed for inner_radius is ignored, and the fit performed on spectra for the entire region. If you are
        fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius of the region that the
        desired spectrum covers (for instance 'point' would be acceptable for a PointSource,
        or Quantity(40, 'arcsec')). By default this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param bool redshifted: Whether the powerlaw that includes redshift (zpowerlw) should be used.
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param Quantity start_temp: The starting value for the temperature of the blackbody.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 which
        should be equivalent to 1sigma errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec
        /manual/XSerror.html) correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param int/float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be used for this
        XSPEC spectral fit. If a stacking procedure for a particular telescope is not supported, this function will
        instead use individual spectra for an ObsID. The default is False.
    :param str/List[str] telescope: Telescope(s) to perform the XSPEC operations for. Default is None, in which
        case the XSPEC fit will be performed individually for all telescopes associated with a source.
    """
    sources, inn_rad_vals, out_rad_vals, telescope, eff_stack_spec = _pregen_spectra(
        sources,
        outer_radius,
        inner_radius,
        group_spec,
        min_counts,
        min_sn,
        over_sample,
        one_rmf,
        num_cores,
        stacked_spectra,
        telescope,
    )
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # This function is for a set model, either absorbed blackbody or absorbed zbbody
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"
    if redshifted:
        model = "constant*tbabs*zbbody"
        par_names = "{factor nH kT Redshift norm}"
    else:
        model = "constant*tbabs*bbody"
        par_names = "{factor nH kT norm}"

    # TODO REPLACE THESE REPEATED CHECKS/PREPARATION OF START TEMP/MET WITH A FUNCTION - SEE ISSUE #1509
    # Have to check that every source has a start temperature entry, if the user decided to pass a set of them
    if not start_temp.isscalar and len(start_temp) != len(sources):
        raise ValueError(
            "If a non-scalar Quantity is passed for 'start_temp', it must have one entry for each "
            f"source. It currently has {len(start_temp)} for {len(sources)} sources."
        )
    # Want to make sure that the start_temp variable is always a non-scalar Quantity with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif start_temp.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temps = Quantity([start_temp.value.copy()] * len(sources), start_temp.unit)
    else:
        all_start_temps = start_temp.copy()

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS["blackbody"]
    sig = signature(blackbody)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        problem_pars = set(list(cur_args.keys())) - set(list(rel_args.keys()))
        raise XGADeveloperError(
            f"Current keyword arguments of this function and FIT_FUNC_ARGS entry do not "
            f"match - unmatched arguments are {', '.join(problem_pars)}."
        )

    # We generate the fit configuration key - in this case there are no relevant variables that can have different
    #  values for different sources, so we don't need to put this in the loop. Still, we will pass back  a list of
    #  fit configuration keys because the xspec call decorator will expect a list with one entry per source that
    #  is having a fit run
    in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
    fit_conf = _gen_fit_conf(in_fit_conf)

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    for src_ind, source in enumerate(sources):
        for tel in source.telescopes:
            if tel not in telescope:
                continue
            # retrieving the spectrum objects needed for each source/ tel combo
            specs, storage_key = _spec_obj_setup(
                stacked_spectra,
                tel,
                source,
                out_rad_vals,
                src_ind,
                inn_rad_vals,
                group_spec,
                min_counts,
                min_sn,
                over_sample,
            )

            # Whatever start temperature is passed gets converted to keV, this will be put in the template
            t = all_start_temps[src_ind].to("keV", equivalencies=u.temperature_energy()).value

            # For this model, we have to know the redshift of the source.
            if redshifted and source.redshift is None:
                raise ValueError("You cannot supply a source without a redshift if you have elected to fit zbbody.")
            elif redshifted and source.redshift is not None:
                par_values = f"{{{1.0} {source.nH.to('10^22 cm^-2').value} {t} {source.redshift} {1.0}}}"
            else:
                par_values = f"{{{1.0} {source.nH.to('10^22 cm^-2').value} {t} {1.0}}}"

            # Set up the TCL list that defines which parameters are frozen, dependent on user input
            if redshifted and freeze_nh:
                freezing = "{F T F T F}"
            elif not redshifted and freeze_nh:
                freezing = "{F T F F}"
            elif redshifted and not freeze_nh:
                freezing = "{F F F T F}"
            elif not redshifted and not freeze_nh:
                freezing = "{F F F F}"

            # Set up the TCL list that defines which parameters are linked across different spectra,
            #  dependant on user input
            if redshifted:
                linking = "{F T T T T}"
            else:
                linking = "{F T T T}"

            # If the blackbody with redshift has been chosen, then we use the redshift attached to the source object
            #  If not we just pass a filler redshift and the luminosities are invalid
            if source.redshift is not None:
                z = source.redshift
            else:
                z = 1
                warn(
                    f"{source.name} has no redshift information associated, so luminosities from this fit"
                    f" will be invalid, as redshift has been set to one.",
                    stacklevel=2,
                )

            # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed
            #  luminosities. I am only specifying parameter 2 here (though there will likely be multiple models because
            #  there are likely multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing
            #  one will zero them all.
            nh_to_zero = "{2}"

            # If the fit has already been performed, we do not wish to perform it again
            try:
                source.get_results(
                    out_rad_vals[src_ind],
                    tel,
                    model,
                    inn_rad_vals[src_ind],
                    None,
                    group_spec,
                    min_counts,
                    min_sn,
                    over_sample,
                    eff_stack_spec[tel],
                    fit_conf,
                )
            except (ModelNotAssociatedError, FitConfNotAssociatedError):
                out_file, script_file, inv_ent = _write_xspec_script(
                    source,
                    storage_key,
                    model,
                    abund_table,
                    fit_method,
                    specs,
                    lo_en,
                    hi_en,
                    par_names,
                    par_values,
                    linking,
                    freezing,
                    par_fit_stat,
                    lum_low_lims,
                    lum_upp_lims,
                    lum_conf,
                    z,
                    False,
                    "{}",
                    "{}",
                    "{}",
                    "{}",
                    True,
                    tel,
                    fit_conf,
                    nh_to_zero,
                )
                script_paths.append(script_file)
                outfile_paths.append(out_file)
                src_inds.append(src_ind)
                fit_confs.append(fit_conf)
                inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents  # type: ignore[return-value]


@xspec_call
def double_temp_apec(
    sources: BaseSource | BaseSample | list[BaseSource],
    outer_radius: str | Quantity,
    inner_radius: str | Quantity = Quantity(0, "arcsec"),
    start_temp_one: Quantity = Quantity(3.0, "keV"),
    start_temp_two: Quantity = Quantity(5.0, "keV"),
    start_met_one: float = 0.3,
    start_met_two: float = 0.3,
    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
    freeze_nh: bool = True,
    freeze_met_one: bool = True,
    freeze_met_two: bool = True,
    freeze_temp_one: bool = False,
    freeze_temp_two: bool = False,
    lo_en: Quantity = Quantity(0.3, "keV"),
    hi_en: Quantity = Quantity(7.9, "keV"),
    par_fit_stat: float = 1.0,
    lum_conf: float = 68.0,
    abund_table: str = "angr",
    fit_method: str = "leven",
    group_spec: bool = True,
    min_counts: int = 5,
    min_sn: float | None = None,
    over_sample: float | None = None,
    one_rmf: bool = True,
    num_cores: int = NUM_CORES,
    spectrum_checking: bool = False,
    timeout: Quantity = Quantity(1, "hr"),
    stacked_spectra: bool = False,
    telescope: str | list[str] | None = None,
) -> BaseSource | BaseSample | list[BaseSource]:
    """
    This is a convenience function for fitting an absorbed double-temperature apec model (constant*tbabs*(apec+apec))
    to an object's spectrum. If there are no existing spectra with the passed settings, then they will be
    generated automatically.

    BE AWARE that higher quality data (i.e. higher signal-to-noise spectra, with a greater number of X-ray counts) are
    required to fit models with more free parameters, so this model fit is less likely to converge than single
    temperature apec fits for lower quality data.

    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any value
        passed for inner_radius is ignored, and the fit performed on spectra for the entire region. If you are
        fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). By default this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param Quantity start_temp_one: The initial temperature of the first APEC model, the default is 3 keV. This value
        can also be a non-scalar Quantity, with an entry for every source in a sample (this is most useful when
        used with the 'freeze_temp' argument, to provide some external constraint on temperature for objects with
        poor data).
    :param Quantity start_temp_two: The initial temperature of the second APEC model, the default is 3 keV. This value
        can also be a non-scalar Quantity, with an entry for every source in a sample (this is most useful when
        used with the 'freeze_temp' argument, to provide some external constraint on temperature for objects with
        poor data).
    :param start_met_one: The initial metallicity of the first APEC model (in ZSun).
    :param start_met_two: The initial metallicity of the second APEC model (in ZSun).
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen. Default is True.
    :param bool freeze_met_one: Whether the metallicity of the first APEC model should be frozen. Default is True.
    :param bool freeze_met_two: Whether the metallicity of the second APEC model should be frozen. Default is True.
    :param bool freeze_temp_one: Whether the temperature parameter of the first APEC model should be
        frozen. Default is False
    :param bool freeze_temp_two: Whether the temperature parameter of the second APEC model should be
    frozen. Default is False
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 which should be
        equivalent to 1σ errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSerror.html)
        correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param bool spectrum_checking: Should the spectrum checking step of the XSPEC fit (where each spectrum is fit
        individually and tested to see whether it will contribute to the simultaneous fit) be activated?
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be used for this
        XSPEC spectral fit. If a stacking procedure for a particular telescope is not supported, this function will
        instead use individual spectra for an ObsID. The default is False.
    :param str/List[str] telescope: Telescope(s) to perform the XSPEC operations for. Default is None, in which
        case the XSPEC fit will be performed individually for all telescopes associated with a source.
    :return: The source object(s) passed in.
    :rtype: Union[BaseSource, BaseSample, List[BaseSource]]
    """
    # TODO ONE WAY OR ANOTHER THIS HAS TO BE RESOLVED - COULD JUST PUT THIS ERROR MESSAGE BUT AS A WARNING?
    # This is a little cheesy as we haven't decided what exactly to do yet, but the spectrum checking behaviour
    #  is super inconsistent with double_temp_apec - presumably just because the greater number of parameters
    #  makes estimating errors much less stable for individual spectral fits
    if spectrum_checking:
        raise NotImplementedError(
            "The double_temp_apec function does not currently support "
            "'spectrum_checking=True', as it exhibits very unstable behaviour."
        )

    # We need to make sure that the spectra we're going to be fitting the model to with XSPEC actually exist
    sources, inn_rad_vals, out_rad_vals, telescope, eff_stack_spec = _pregen_spectra(
        sources,
        outer_radius,
        inner_radius,
        group_spec,
        min_counts,
        min_sn,
        over_sample,
        one_rmf,
        num_cores,
        stacked_spectra,
        telescope,
    )
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # Have to check that every source has start temperature entries, if the user decided to pass a set of them - this
    #  is identical to the checks we perform at the beginning of 'single_temp_apec'
    if (not start_temp_one.isscalar and len(start_temp_one) != len(sources)) or (
        not start_temp_two.isscalar and len(start_temp_two) != len(sources)
    ):
        raise ValueError(
            f"If a non-scalar Quantity is passed for 'start_temp_one' or 'start_temp_two', there must be "
            f"one entry for each source. They currently have {len(start_temp_one)} and {len(start_temp_two)} "
            f"respectively, for {len(sources)} sources."
        )

    # TODO REPLACE THESE REPEATED CHECKS/PREPARATION OF START TEMP/MET WITH A FUNCTION - SEE ISSUE #1509
    # Want to make sure that the start_temp variable is always a non-scalar Quantity with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    if start_temp_one.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temp_ones = Quantity([start_temp_one.value.copy()] * len(sources), start_temp_one.unit)
    else:
        all_start_temp_ones = start_temp_one.copy()

    # And the same thing for start temperature the second
    if start_temp_two.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temp_twos = Quantity([start_temp_two.value.copy()] * len(sources), start_temp_two.unit)
    else:
        all_start_temp_twos = start_temp_two.copy()

    # TODO MAKE THIS CHECK SUPPORT QUANTITY AS WELL - issue #1509 is semi-relevant
    # We repeat the above, but for the starting metallicity value(s)
    if isinstance(start_met_one, (list, np.ndarray)) and len(start_met_one) != len(sources):
        raise ValueError(
            f"If a list or array is passed to 'start_met_one', it must have one entry for each "
            f"source. It currently has {len(start_met_one)} for {len(sources)} sources."
        )
    elif isinstance(start_met_one, (list, np.ndarray)):
        all_start_met_ones = deepcopy(start_met_one)
    # Want to make sure that the start_met variable is always a list with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif isinstance(start_met_one, (float, int)):
        all_start_met_ones = [start_met_one] * len(sources)
    else:
        raise ValueError(
            "'start_met_one' must be either a list/array with an entry for each source, or a float/integer."
        )

    # TODO MAKE THIS CHECK SUPPORT QUANTITY AS WELL - issue #1509 is semi-relevant
    # We repeat the above, but for the starting metallicity value(s)
    if isinstance(start_met_two, (list, np.ndarray)) and len(start_met_two) != len(sources):
        raise ValueError(
            f"If a list or array is passed to 'start_met_two', it must have one entry for each "
            f"source. It currently has {len(start_met_two)} for {len(sources)} sources."
        )
    elif isinstance(start_met_two, (list, np.ndarray)):
        all_start_met_twos = deepcopy(start_met_two)
    # Want to make sure that the start_met variable is always a list with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif isinstance(start_met_two, (float, int)):
        all_start_met_twos = [start_met_two] * len(sources)
    else:
        raise ValueError(
            "'start_met_two' must be either a list/array with an entry for each source, or a float/integer."
        )

    # This function is for a set model, absorbed double apec, so I can hard code all of this stuff.
    model = "constant*tbabs*(apec+apec)"
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be
    #  in the form of TCL lists.
    par_names = "{factor nH kT Abundanc Redshift norm kT Abundanc Redshift norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS["double_temp_apec"]

    sig = signature(double_temp_apec)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    ## This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        problem_pars = set(list(cur_args.keys())) - set(list(rel_args.keys()))
        raise XGADeveloperError(
            f"Current keyword arguments of this function and FIT_FUNC_ARGS entry do not "
            f"match - unmatched arguments are {', '.join(problem_pars)}."
        )

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for src_ind, source in enumerate(sources):
        # We do not do simultaneous fits with spectra from different telescopes, they are all fit separately - at
        #  least in this current setup
        for tel in source.telescopes:
            if tel not in telescope:
                continue

            # retrieving the spectrum objects needed for each source/ tel combo
            specs, storage_key = _spec_obj_setup(
                stacked_spectra,
                tel,
                source,
                out_rad_vals,
                src_ind,
                inn_rad_vals,
                group_spec,
                min_counts,
                min_sn,
                over_sample,
            )

            # For this model, we have to know the redshift of the source.
            if source.redshift is None:
                raise ValueError("You cannot supply a source without a redshift to this model.")

            # Whatever start temperatures are passed get converted to keV - then will be put in the template
            start_temp_one = all_start_temp_ones[src_ind].to("keV", equivalencies=u.temperature_energy())
            start_temp_two = all_start_temp_twos[src_ind].to("keV", equivalencies=u.temperature_energy())
            # Also fetch out the relevant starting metallicity values
            start_met_one = all_start_met_ones[src_ind]
            start_met_two = all_start_met_twos[src_ind]

            # Another TCL list, this time of the parameter start values for this model.
            par_values = (
                f"{{{1.0} {source.nH.to('10^22 cm^-2').value} {start_temp_one.value} {start_met_one} "
                f"{source.redshift} {1.0} {start_temp_two.value} {start_met_two} {source.redshift} {1.0}}}"
            )

            # Set up the TCL list that defines which parameters are frozen, dependent on user input - this can now
            #  include the temperature, if the user wants it fixed at the start value
            freezing = (
                f"{{F {'T' if freeze_nh else 'F'} {'T' if freeze_temp_one else 'F'} "
                f"{'T' if freeze_met_one else 'F'} T F {'T' if freeze_temp_two else 'F'} "
                f"{'T' if freeze_met_two else 'F'} T F}}"
            )

            # Set up the TCL list that defines which parameters are linked across different spectra, only the
            #  multiplicative constant that accounts for variation in normalisation over different observations is not
            #  linked
            linking = "{F T T T T T T T T T}"

            # If the user wants the spectrum cleaning step to be run, then we have to setup some acceptable
            #  limits. For this function they will be hardcoded, for simplicities sake, and we're only going to
            #  check the temperature, as its the main thing we're fitting for with constant*tbabs*apec
            if spectrum_checking:
                check_list = "{kT}"
                check_lo_lims = "{0.01}"
                check_hi_lims = "{20}"
                check_err_lims = "{15}"
            else:
                check_list = "{}"
                check_lo_lims = "{}"
                check_hi_lims = "{}"
                check_err_lims = "{}"

            # We generate the fit configuration key here, on a source by source basis, as the start temperature
            #  is allowed to be different for every source
            in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
            fit_conf = _gen_fit_conf(in_fit_conf)

            # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed
            #  luminosities. I am only specifying parameter 2 here (though there will likely be multiple models
            #  because there are likely multiple spectra) because I know that nH of tbabs is linked in this setup, so
            #  zeroing one will zero them all.
            nh_to_zero = "{2}"

            # If the fit has already been performed we do not wish to perform it again
            try:
                # We search for the norm parameter, as it is guaranteed to be there for any fit with this model
                source.get_results(
                    out_rad_vals[src_ind],
                    tel,
                    model,
                    inn_rad_vals[src_ind],
                    "norm",
                    group_spec,
                    min_counts,
                    min_sn,
                    over_sample,
                    eff_stack_spec[tel],
                    fit_conf,
                )
            except (ModelNotAssociatedError, FitConfNotAssociatedError):
                out_file, script_file, inv_ent = _write_xspec_script(
                    source,
                    storage_key,
                    model,
                    abund_table,
                    fit_method,
                    specs,
                    lo_en,
                    hi_en,
                    par_names,
                    par_values,
                    linking,
                    freezing,
                    par_fit_stat,
                    lum_low_lims,
                    lum_upp_lims,
                    lum_conf,
                    source.redshift,
                    spectrum_checking,
                    check_list,
                    check_lo_lims,
                    check_hi_lims,
                    check_err_lims,
                    True,
                    tel,
                    fit_conf,
                    nh_to_zero,
                )
                script_paths.append(script_file)
                outfile_paths.append(out_file)
                src_inds.append(src_ind)
                fit_confs.append(fit_conf)
                inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents  # type: ignore[return-value]


@xspec_call
def general_model_fit(
    sources: BaseSource | BaseSample | list[BaseSource],
    outer_radius: str | Quantity,
    model: str,
    inner_radius: str | Quantity = Quantity(0, "arcsec"),
    norm_scale: bool = True,
    start_pars: dict[str, float | int | Quantity | list | np.ndarray | None] | None = None,
    freeze_pars: dict[str, bool] | None = None,
    link_pars: dict[str, bool] | None = None,
    nh_to_zero: list[str | int] | None = None,
    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
    lo_en: Quantity = Quantity(0.3, "keV"),
    hi_en: Quantity = Quantity(7.9, "keV"),
    par_fit_stat: float = 1.0,
    lum_conf: float = 68.0,
    abund_table: str = "angr",
    fit_method: str = "leven",
    group_spec: bool = True,
    min_counts: int = 5,
    min_sn: int | float | None = None,
    over_sample: float | None = None,
    one_rmf: bool = True,
    num_cores: int = NUM_CORES,
    spectrum_checking: bool = False,
    check_pars: str | list[str] | None = None,
    check_lo_lims: float | int | Quantity | list[float | int | Quantity] | None = None,
    check_hi_lims: float | int | Quantity | list[float | int | Quantity] | None = None,
    check_err_lims: float | int | Quantity | list[float | int | Quantity] | None = None,
    timeout: Quantity = Quantity(1, "hr"),
    stacked_spectra: bool = False,
    telescope: str | list[str] | None = None,
    force_gen: bool = False,
) -> BaseSource | BaseSample | list[BaseSource]:
    """
    A general XSPEC spectral fitting function capable of fitting any valid combination of additive, multiplicative,
    and convolution models defined in the local HEASoft XSPEC installation's ``model.dat`` file.

    Default starting values, units, hard/soft parameter limits, and default freeze states are parsed directly from
    ``model.dat``. By default, ``constant*`` is prepended to the model expression (controlled by ``norm_scale``) to
    account for cross-calibration normalisation differences across simultaneous spectra, the first non-redshifted
    multiplicative absorption column density (``nH``) is populated with ``source.nH`` and frozen, and any
    non-redshifted-absorber cosmological redshift parameter is populated with ``source.redshift``.

    Users can inspect the disambiguated parameter names and obtain pre-populated configuration dictionaries for any
    model expression using :func:`~xga.xspec.fit.describe_xspec_model`, or generate a model-specific fitting function
    with explicit keyword arguments and a tailored docstring using :func:`~xga.xspec.fit.custom_xspec_fit`.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the desired spectrum
        covers (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If you are
        fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param str model: The XSPEC model expression to fit (e.g. 'tbabs*apec', 'tbabs*(apec+powerlaw)').
    :param str/Quantity inner_radius: The name or value of the inner radius of the region that the desired spectrum
        covers. By default this is zero arcseconds, resulting in a circular spectrum.
    :param bool norm_scale: Whether to prepend 'constant*' to the model expression so simultaneous fits across
        multiple observations/instruments link additive normalisations and scale via a multiplicative constant.
        Default is True.
    :param dict start_pars: Optional dictionary mapping disambiguated parameter names (e.g. 'kT', 'kT_1', 'Abundanc')
        to starting values. Values may be scalars (float, int, or Astropy Quantity), per-source lists/arrays/non-scalar
        Quantities with one entry per source, or None (which leaves default/auto-substituted values in place).
    :param dict freeze_pars: Optional dictionary mapping disambiguated parameter names to boolean freeze flags
        (True to freeze the parameter during fitting, False to thaw it).
    :param dict link_pars: Optional dictionary mapping disambiguated parameter names to boolean linking flags
        (True to link the parameter across all simultaneously fitted spectra, False to fit it independently per
        spectrum).
    :param list nh_to_zero: Optional list of disambiguated parameter names (e.g. ['nH', 'nH_1']) or 1-based parameter
        indices (e.g. [2, 6]) that should be set to zero when calculating unabsorbed X-ray luminosities. If None
        (the default), all multiplicative absorption column density nH parameters are automatically zeroed. Pass an
        empty list ``[]`` to disable zeroing any parameters.
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 (1σ).
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
    :param int/float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param bool spectrum_checking: Should the spectrum checking step of the XSPEC fit (where each spectrum is fit
        individually and tested against check_pars limits before the simultaneous fit) be activated? Default is False
        unless check_pars and limits are supplied.
    :param str/list[str] check_pars: Parameter name (or list of parameter names) to test during the individual
        spectrum checking step. Note that due to how general_xspec_fit.xcm matches parameter names via tcloutr pinfo,
        base XSPEC parameter names (e.g. 'kT') should be used here.
    :param float/int/Quantity/list check_lo_lims: Minimum acceptable value(s) for check_pars during spectrum checking.
    :param float/int/Quantity/list check_hi_lims: Maximum acceptable value(s) for check_pars during spectrum checking.
    :param float/int/Quantity/list check_err_lims: Maximum acceptable uncertainty(ies) for check_pars during spectrum
        checking.
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be used.
    :param str/List[str] telescope: Telescope(s) to perform the XSPEC operations for. Default is None.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    :return: The source object(s) passed in.
    :rtype: Union[BaseSource, BaseSample, List[BaseSource]]
    """
    # Validate and parse the model expression first so that any syntax or unknown-model errors are caught immediately
    #  before spending time generating or checking spectra
    clean_model, flat_pars, auto_nh_to_zero = validate_and_parse_model(model, norm_scale=norm_scale)

    # Ensure spectra exist and validate common fitting inputs
    sources, inn_rad_vals, out_rad_vals, telescope, eff_stack_spec = _pregen_spectra(
        sources,
        outer_radius,
        inner_radius,
        group_spec,
        min_counts,
        min_sn,
        over_sample,
        one_rmf,
        num_cores,
        stacked_spectra,
        telescope,
        force_gen,
    )
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # Build a lookup dictionary from disambiguated parameter key ('nH', 'kT', 'kT_1', ...) to its metadata entry
    par_lookup = {p["par_key"]: p for p in flat_pars}
    valid_par_keys = list(par_lookup.keys())

    # Validate user-supplied parameter override dictionaries (start_pars, freeze_pars, link_pars)
    if start_pars is not None:
        if not isinstance(start_pars, dict):
            raise TypeError("'start_pars' must be a dictionary mapping parameter names to starting values, or None.")
        unknown_start = [k for k in start_pars if k not in par_lookup]
        if unknown_start:
            raise ValueError(
                f"Unrecognised parameter key(s) in 'start_pars' for model '{clean_model}': "
                f"{', '.join(unknown_start)}. Valid parameter keys are: {', '.join(valid_par_keys)}."
            )

    if freeze_pars is not None:
        if not isinstance(freeze_pars, dict):
            raise TypeError("'freeze_pars' must be a dictionary mapping parameter names to booleans, or None.")
        unknown_freeze = [k for k in freeze_pars if k not in par_lookup]
        if unknown_freeze:
            raise ValueError(
                f"Unrecognised parameter key(s) in 'freeze_pars' for model '{clean_model}': "
                f"{', '.join(unknown_freeze)}. Valid parameter keys are: {', '.join(valid_par_keys)}."
            )
        for k, v in freeze_pars.items():
            if not isinstance(v, (bool, np.bool_)):
                raise TypeError(f"Freeze setting for parameter '{k}' in 'freeze_pars' must be a boolean.")

    if link_pars is not None:
        if not isinstance(link_pars, dict):
            raise TypeError("'link_pars' must be a dictionary mapping parameter names to booleans, or None.")
        unknown_link = [k for k in link_pars if k not in par_lookup]
        if unknown_link:
            raise ValueError(
                f"Unrecognised parameter key(s) in 'link_pars' for model '{clean_model}': "
                f"{', '.join(unknown_link)}. Valid parameter keys are: {', '.join(valid_par_keys)}."
            )
        for k, v in link_pars.items():
            if not isinstance(v, (bool, np.bool_)):
                raise TypeError(f"Link setting for parameter '{k}' in 'link_pars' must be a boolean.")

    # Resolve nh_to_zero: if None, use the automatically identified absorption nH parameter indices; otherwise
    #  allow the user to specify a list of disambiguated parameter names or 1-based parameter indices
    if nh_to_zero is None:
        resolved_nh_indices = auto_nh_to_zero
    elif isinstance(nh_to_zero, list):
        resolved_nh_indices = []
        for entry in nh_to_zero:
            if isinstance(entry, str):
                if entry not in par_lookup:
                    raise ValueError(
                        f"Parameter '{entry}' in 'nh_to_zero' is not in model '{clean_model}'. "
                        f"Valid parameter keys are: {', '.join(valid_par_keys)}."
                    )
                resolved_nh_indices.append(par_lookup[entry]["par_idx"])
            elif isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
                idx_val = int(entry)
                if idx_val < 1 or idx_val > len(flat_pars):
                    raise ValueError(
                        f"Parameter index {idx_val} in 'nh_to_zero' is out of range for model '{clean_model}' "
                        f"(valid 1-based indices are 1 to {len(flat_pars)})."
                    )
                resolved_nh_indices.append(idx_val)
            else:
                raise TypeError(
                    "Entries in 'nh_to_zero' must be parameter key strings (e.g. 'nH') or 1-based integer indices."
                )
    else:
        raise TypeError("'nh_to_zero' must be a list of parameter names or 1-based integer indices, or None.")

    if len(resolved_nh_indices) == 0:
        nh_to_zero_tcl = "None"
    else:
        nh_to_zero_tcl = "{" + " ".join(str(idx) for idx in resolved_nh_indices) + "}"

    # Validate and normalise spectrum checking parameters. If the user supplies check_pars and limits without
    #  explicitly setting spectrum_checking=True, we enable it automatically as agreed in the design.
    any_check_supplied = any(arg is not None for arg in (check_pars, check_lo_lims, check_hi_lims, check_err_lims))
    all_check_supplied = all(arg is not None for arg in (check_pars, check_lo_lims, check_hi_lims, check_err_lims))
    if any_check_supplied and not all_check_supplied:
        raise ValueError(
            "To use spectrum checking in 'general_model_fit', all four of 'check_pars', 'check_lo_lims', "
            "'check_hi_lims', and 'check_err_lims' must be supplied together."
        )
    if all_check_supplied and not spectrum_checking:
        spectrum_checking = True
    elif spectrum_checking and not all_check_supplied:
        raise ValueError(
            "When 'spectrum_checking=True' in 'general_model_fit', you must supply 'check_pars', 'check_lo_lims', "
            "'check_hi_lims', and 'check_err_lims'."
        )

    if spectrum_checking:
        norm_check_pars = [check_pars] if isinstance(check_pars, str) else list(check_pars)  # type: ignore[arg-type]
        norm_lo = (
            [check_lo_lims]
            if isinstance(check_lo_lims, (int, float, np.number))
            or (isinstance(check_lo_lims, Quantity) and check_lo_lims.isscalar)
            else list(check_lo_lims)  # type: ignore[arg-type]
        )
        norm_hi = (
            [check_hi_lims]
            if isinstance(check_hi_lims, (int, float, np.number))
            or (isinstance(check_hi_lims, Quantity) and check_hi_lims.isscalar)
            else list(check_hi_lims)  # type: ignore[arg-type]
        )
        norm_err = (
            [check_err_lims]
            if isinstance(check_err_lims, (int, float, np.number))
            or (isinstance(check_err_lims, Quantity) and check_err_lims.isscalar)
            else list(check_err_lims)  # type: ignore[arg-type]
        )

        if not (len(norm_check_pars) == len(norm_lo) == len(norm_hi) == len(norm_err)):
            raise ValueError(
                "'check_pars', 'check_lo_lims', 'check_hi_lims', and 'check_err_lims' must all have the same length."
            )

        tcl_check_names: list[str] = []
        tcl_check_lo: list[float] = []
        tcl_check_hi: list[float] = []
        tcl_check_err: list[float] = []

        for cp_name, lo_v, hi_v, err_v in zip(norm_check_pars, norm_lo, norm_hi, norm_err, strict=True):
            if cp_name in par_lookup:
                cp_meta = par_lookup[cp_name]
            else:
                # Also allow matching by base par_name in case the user passed the base XSPEC name
                base_matches = [p for p in flat_pars if p["par_name"] == cp_name]
                if not base_matches:
                    raise ValueError(
                        f"Check parameter '{cp_name}' is not present in model '{clean_model}'. "
                        f"Valid parameter keys are: {', '.join(valid_par_keys)}."
                    )
                cp_meta = base_matches[0]

            # Note: general_xspec_fit.xcm checks [tcloutr pinfo $ipar] which returns the base XSPEC parameter name
            tcl_check_names.append(cp_meta["par_name"])
            tcl_check_lo.append(_convert_par_val_to_xspec(lo_v, cp_meta, cp_name))
            tcl_check_hi.append(_convert_par_val_to_xspec(hi_v, cp_meta, cp_name))
            tcl_check_err.append(_convert_par_val_to_xspec(err_v, cp_meta, cp_name))

        check_list_str = "{" + " ".join(tcl_check_names) + "}"
        check_lo_str = "{" + " ".join(str(v) for v in tcl_check_lo) + "}"
        check_hi_str = "{" + " ".join(str(v) for v in tcl_check_hi) + "}"
        check_err_str = "{" + " ".join(str(v) for v in tcl_check_err) + "}"
        # Normalise the local variables so that _gen_fit_conf produces a canonical representation
        check_pars = tcl_check_names
        check_lo_lims = tcl_check_lo
        check_hi_lims = tcl_check_hi
        check_err_lims = tcl_check_err
    else:
        check_list_str = "{}"
        check_lo_str = "{}"
        check_hi_str = "{}"
        check_err_str = "{}"

    # Pre-process and broadcast any user-supplied start_pars across the sources in the sample (or single source).
    #  We convert Quantities to the target XSPEC unit floats and validate against hard limits upfront so errors are
    #  caught immediately.
    per_source_start_overrides: list[dict[str, float]] = [{} for _ in range(len(sources))]
    if start_pars is not None:
        for par_key, raw_val in start_pars.items():
            # A value of None (as returned by describe_xspec_model for nH/Redshift) means "keep default/auto-value"
            if raw_val is None:
                continue
            p_meta = par_lookup[par_key]
            if isinstance(raw_val, Quantity) and not raw_val.isscalar:
                if len(raw_val) != len(sources):
                    raise ValueError(
                        f"If a non-scalar Quantity is passed in 'start_pars' for '{par_key}', it must have one "
                        f"entry for each source ({len(raw_val)} supplied for {len(sources)} sources)."
                    )
                for s_idx in range(len(sources)):
                    conv_v = _convert_par_val_to_xspec(raw_val[s_idx], p_meta, par_key)
                    _validate_par_hard_limits(conv_v, p_meta, par_key)
                    per_source_start_overrides[s_idx][par_key] = conv_v
            elif isinstance(raw_val, (list, np.ndarray)):
                if len(raw_val) != len(sources):
                    raise ValueError(
                        f"If a list or array is passed in 'start_pars' for '{par_key}', it must have one "
                        f"entry for each source ({len(raw_val)} supplied for {len(sources)} sources)."
                    )
                for s_idx in range(len(sources)):
                    conv_v = _convert_par_val_to_xspec(raw_val[s_idx], p_meta, par_key)
                    _validate_par_hard_limits(conv_v, p_meta, par_key)
                    per_source_start_overrides[s_idx][par_key] = conv_v
            else:
                conv_v = _convert_par_val_to_xspec(raw_val, p_meta, par_key)
                _validate_par_hard_limits(conv_v, p_meta, par_key)
                for s_idx in range(len(sources)):
                    per_source_start_overrides[s_idx][par_key] = conv_v

    # Normalise freeze_pars and link_pars to only include non-None cleaned boolean dicts for fit_conf
    clean_freeze_overrides: dict[str, bool] | None = (
        {k: bool(v) for k, v in freeze_pars.items()} if freeze_pars else None
    )
    clean_link_overrides: dict[str, bool] | None = {k: bool(v) for k, v in link_pars.items()} if link_pars else None
    freeze_pars = clean_freeze_overrides
    link_pars = clean_link_overrides

    # Assemble the TCL parameter names list (using base XSPEC parameter names as expected by general_xspec_fit.xcm,
    #  which performs its own _1, _2 disambiguation loop matching ours)
    par_names = "{" + " ".join(p["par_name"] for p in flat_pars) + "}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Developer check that FIT_FUNC_ARGS matches the keyword arguments of general_model_fit
    rel_args = FIT_FUNC_ARGS["general_model_fit"]
    sig = signature(general_model_fit)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        problem_pars = set(list(cur_args.keys())) ^ set(list(rel_args.keys()))
        raise XGADeveloperError(
            f"Current keyword arguments of this function and FIT_FUNC_ARGS entry do not "
            f"match - unmatched arguments are {', '.join(problem_pars)}."
        )

    # Determine the final freeze and link states for every parameter in the model
    final_frozen: list[bool] = []
    final_linked: list[bool] = []
    for p in flat_pars:
        pk = p["par_key"]
        fr_val = (
            clean_freeze_overrides[pk] if (clean_freeze_overrides and pk in clean_freeze_overrides) else p["frozen"]
        )
        lk_val = clean_link_overrides[pk] if (clean_link_overrides and pk in clean_link_overrides) else p["linked"]
        final_frozen.append(fr_val)
        final_linked.append(lk_val)

    freezing = "{" + " ".join("T" if fr else "F" for fr in final_frozen) + "}"
    linking = "{" + " ".join("T" if lk else "F" for lk in final_linked) + "}"

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []

    for src_ind, source in enumerate(sources):
        src_overrides = per_source_start_overrides[src_ind]
        # Rebind start_pars in locals() for this source so _gen_fit_conf records the source-specific overrides
        #  (or None if no overrides were provided), matching how single_temp_apec handles per-source start_temp
        start_pars = dict(src_overrides) if src_overrides else None  # noqa: F841

        for tel in source.telescopes:
            if tel not in telescope:
                continue

            specs, storage_key = _spec_obj_setup(
                stacked_spectra,
                tel,
                source,
                out_rad_vals,
                src_ind,
                inn_rad_vals,
                group_spec,
                min_counts,
                min_sn,
                over_sample,
            )

            # Build the starting parameter values for this source
            cur_par_vals: list[float] = []
            for p in flat_pars:
                pk = p["par_key"]
                if pk in src_overrides:
                    cur_par_vals.append(src_overrides[pk])
                elif p["is_leading_norm_scale"]:
                    cur_par_vals.append(1.0)
                elif p["is_galactic_nh"]:
                    cur_par_vals.append(float(source.nH.to("10^22 cm^-2").value))
                elif p["is_redshift"]:
                    if source.redshift is None:
                        raise ValueError(
                            f"Source '{source.name}' does not have a redshift associated with it, which is required "
                            f"for parameter '{pk}' ({p['component']}) in model '{clean_model}' unless overridden "
                            f"via 'start_pars'."
                        )
                    cur_par_vals.append(float(source.redshift))
                else:
                    cur_par_vals.append(float(p["default"]))

            par_values = "{" + " ".join(str(v) for v in cur_par_vals) + "}"

            # Determine the redshift to pass to xga_extract for luminosity calculations (matching power_law / blackbody)
            if source.redshift is not None:
                lum_redshift = source.redshift
            else:
                # Check if any redshift parameter was overridden in start_pars that we can use, otherwise fall back to 1
                overridden_z = [
                    src_overrides[p["par_key"]] for p in flat_pars if p["is_redshift"] and p["par_key"] in src_overrides
                ]
                if overridden_z:
                    lum_redshift = overridden_z[0]
                else:
                    lum_redshift = 1
                    warn(
                        f"{source.name} has no redshift information associated, so luminosities from this fit "
                        f"will be invalid, as redshift has been set to one.",
                        stacklevel=2,
                    )

            in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
            fit_conf = _gen_fit_conf(in_fit_conf)

            # Check if this exact model and fit_conf has already been fit to these spectra
            try:
                source.get_results(
                    out_rad_vals[src_ind],
                    tel,
                    clean_model,
                    inn_rad_vals[src_ind],
                    None,
                    group_spec,
                    min_counts,
                    min_sn,
                    over_sample,
                    eff_stack_spec[tel],
                    fit_conf,
                )
            except (ModelNotAssociatedError, FitConfNotAssociatedError):
                out_file, script_file, inv_ent = _write_xspec_script(
                    source,
                    storage_key,
                    clean_model,
                    abund_table,
                    fit_method,
                    specs,
                    lo_en,
                    hi_en,
                    par_names,
                    par_values,
                    linking,
                    freezing,
                    par_fit_stat,
                    lum_low_lims,
                    lum_upp_lims,
                    lum_conf,
                    lum_redshift,
                    spectrum_checking,
                    check_list_str,
                    check_lo_str,
                    check_hi_str,
                    check_err_str,
                    norm_scale,
                    tel,
                    fit_conf,
                    nh_to_zero_tcl,
                )
                script_paths.append(script_file)
                outfile_paths.append(out_file)
                src_inds.append(src_ind)
                fit_confs.append(fit_conf)
                inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, clean_model, fit_confs, inv_ents  # type: ignore[return-value]


# This allows us to make a link between the model that was fit and the XGA function
FIT_FUNC_MODEL_NAMES = {
    "constant*tbabs*apec": single_temp_apec,
    "constant*tbabs*mekal": single_temp_mekal,
    "constant*tbabs*wdem": multi_temp_dem_apec,
    "constant*tbabs*zpowerlw": power_law,
    "constant*tbabs*powerlaw": power_law,
    "constant*tbabs*zbbody": blackbody,
    "constant*tbabs*bbody": blackbody,
    "constant*tbabs*(apec+apec)": double_temp_apec,
}
