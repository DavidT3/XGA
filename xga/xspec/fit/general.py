#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 29/03/2025, 05:01. Copyright (c) The Contributors

import warnings
from inspect import signature, Parameter
from typing import List, Union

import astropy.units as u
from astropy.units import Quantity

from ._common import _check_inputs, _write_xspec_script, _pregen_spectra
from ..fitconfgen import _gen_fit_conf, FIT_FUNC_ARGS
from ..run import xspec_call
from ... import NUM_CORES
from ...exceptions import NoProductAvailableError, ModelNotAssociatedError, XGADeveloperError, FitConfNotAssociatedError
from ...products import Spectrum
from ...samples.base import BaseSample
from ...sources import BaseSource


@xspec_call
def single_temp_apec(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                     inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                     start_temp: Quantity = Quantity(3.0, "keV"), start_met: float = 0.3,
                     lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"), freeze_nh: bool = True,
                     freeze_met: bool = True, freeze_temp: bool = False, lo_en: Quantity = Quantity(0.3, "keV"),
                     hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                     abund_table: str = "angr", fit_method: str = "leven", group_spec: bool = True, min_counts: int = 5,
                     min_sn: float = None, over_sample: float = None, one_rmf: bool = True, num_cores: int = NUM_CORES,
                     spectrum_checking: bool = True, timeout: Quantity = Quantity(1, 'hr')):
    """
    This is a convenience function for fitting an absorbed single temperature apec model(constant*tbabs*apec) to an
    object. If there are no existing spectra with the passed settings, then they will be generated automatically.

    If the spectrum checking step of the XSPEC fit is enabled (using the boolean flag spectrum_checking), then
    each individual spectrum available for a given source will be fitted, and if the measured temperature is less
    than or equal to 0.01keV, or greater than 20keV, or the temperature uncertainty is greater than 15keV, then
    that spectrum will be rejected and not included in the final fit. Spectrum checking also involves rejecting any
    spectra with fewer than 10 noticed channels.

    Freezing the temperature value of the fit is also possible, in cases where the data may not be sufficient to
    constrain it, and an external temperature constrain is used (by passing to the 'start_temp' argument).

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
    :param start_met: The initial metallicity for the fit (in ZSun).
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
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
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
    """

    sources, inn_rad_vals, out_rad_vals = _pregen_spectra(sources, outer_radius, inner_radius, group_spec, min_counts,
                                                          min_sn, over_sample, one_rmf, num_cores)
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # Have to check that every source has a start temperature entry, if the user decided to pass a set of them
    if not start_temp.isscalar and len(start_temp) != len(sources):
        raise ValueError("If a non-scalar Quantity is passed for 'start_temp', it must have one entry for each "
                         "source. It currently has {n} for {s} sources.".format(n=len(start_temp), s=len(sources)))
    # Want to make sure that the start_temp variable is always a non-scalar Quantity with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif start_temp.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temps = Quantity([start_temp.value.copy()]*len(sources), start_temp.unit)
    else:
        all_start_temps = start_temp.copy()

    # This function is for a set model, absorbed apec, so I can hard code all of this stuff.
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    model = "constant*tbabs*apec"
    par_names = "{factor nH kT Abundanc Redshift norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS['single_temp_apec']

    sig = signature(single_temp_apec)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        raise XGADeveloperError("Current keyword arguments of this function do not match the entry in FIT_FUNC_ARGS.")

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for src_ind, source in enumerate(sources):
        # Find matching spectrum objects associated with the current source
        spec_objs = source.get_spectra(out_rad_vals[src_ind], inner_radius=inn_rad_vals[src_ind],
                                       group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                                       over_sample=over_sample)
        # This is because many other parts of this function assume that spec_objs is iterable, and in the case of
        #  a cluster with only a single valid instrument for a single valid observation this may not be the case
        if isinstance(spec_objs, Spectrum):
            spec_objs = [spec_objs]

        # Obviously we can't do a fit if there are no spectra, so throw an error if that's the case
        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for {s} object, you "
                                          "need to generate them first!".format(s=source.name))

        # Turn spectra paths into TCL style list for substitution into template
        specs = "{" + " ".join([spec.path for spec in spec_objs]) + "}"
        # For this model, we have to know the redshift of the source.
        if source.redshift is None:
            raise ValueError("You cannot supply a source without a redshift to this model.")

        # Whatever start temperature is passed gets converted to keV, this will be put in the template
        start_temp = all_start_temps[src_ind].to("keV", equivalencies=u.temperature_energy())
        # Another TCL list, this time of the parameter start values for this model.
        par_values = "{{{0} {1} {2} {3} {4} {5}}}".format(1., source.nH.to("10^22 cm^-2").value, start_temp.value,
                                                          start_met, source.redshift, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependent on user input - this can now
        #  include the temperature, if the user wants it fixed at the start value
        freezing = "{{F {n} {t} {a} T F}}".format(n="T" if freeze_nh else "F",
                                                  t="T" if freeze_temp else "F",
                                                  a="T" if freeze_met else "F")

        # Set up the TCL list that defines which parameters are linked across different spectra, only the
        #  multiplicative constant that accounts for variation in normalisation over different observations is not
        #  linked
        linking = "{F T T T T T}"

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

        # We generate the fit configuration key here, on a source by source basis, as the start temperature is allowed
        #  to be different for every source
        in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
        fit_conf = _gen_fit_conf(in_fit_conf)

        # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed luminosities. I
        #  am only specifying parameter 2 here (though there will likely be multiple models because there are likely
        #  multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing one will zero
        #  them all.
        nh_to_zero = "{2}"

        # If the fit has already been performed we do not wish to perform it again
        try:
            # We search for the norm parameter, as it is guaranteed to be there for any fit with this model
            res = source.get_results(out_rad_vals[src_ind], model, inn_rad_vals[src_ind], 'norm', group_spec,
                                     min_counts, min_sn, over_sample, fit_conf)
        except (ModelNotAssociatedError, FitConfNotAssociatedError):
            out_file, script_file, inv_ent = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table,
                                                                 fit_method, specs, lo_en, hi_en, par_names, par_values,
                                                                 linking, freezing, par_fit_stat, lum_low_lims,
                                                                 lum_upp_lims, lum_conf, source.redshift,
                                                                 spectrum_checking, check_list, check_lo_lims,
                                                                 check_hi_lims, check_err_lims, True, fit_conf,
                                                                 nh_to_zero)
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)
            fit_confs.append(fit_conf)
            inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents


@xspec_call
def single_temp_mekal(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                      inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                      start_temp: Quantity = Quantity(3.0, "keV"), start_met: float = 0.3,
                      lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                      freeze_nh: bool = True, freeze_met: bool = True, freeze_temp: bool = False,
                      lo_en: Quantity = Quantity(0.3, "keV"), hi_en: Quantity = Quantity(7.9, "keV"),
                      par_fit_stat: float = 1., lum_conf: float = 68., abund_table: str = "angr",
                      fit_method: str = "leven", group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                      over_sample: float = None, one_rmf: bool = True, num_cores: int = NUM_CORES,
                      spectrum_checking: bool = True, timeout: Quantity = Quantity(1, 'hr')):
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
    :param start_met: The initial metallicity for the fit (in ZSun).
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.
    :param bool freeze_met: Whether the metallicity parameter in the fit should be frozen.
    :param bool freeze_temp: Whether the temperature parameter in the fit should be frozen. Default is False
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 which should be
        equivelant to 1σ errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSerror.html)
        correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
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
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param bool spectrum_checking: Should the spectrum checking step of the XSPEC fit (where each spectrum is fit
        individually and tested to see whether it will contribute to the simultaneous fit) be activated?
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
    """
    sources, inn_rad_vals, out_rad_vals = _pregen_spectra(sources, outer_radius, inner_radius, group_spec, min_counts,
                                                          min_sn, over_sample, one_rmf, num_cores)
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # Have to check that every source has a start temperature entry, if the user decided to pass a set of them
    if not start_temp.isscalar and len(start_temp) != len(sources):
        raise ValueError("If a non-scalar Quantity is passed for 'start_temp', it must have one entry for each "
                         "source. It currently has {n} for {s} sources.".format(n=len(start_temp), s=len(sources)))
    # Want to make sure that the start_temp variable is always a non-scalar Quantity with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    elif start_temp.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temps = Quantity([start_temp.value.copy()] * len(sources), start_temp.unit)
    else:
        all_start_temps = start_temp.copy()

    # This function is for a set model, absorbed mekal, so I can hard code all of this stuff.
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    model = "constant*tbabs*mekal"
    par_names = "{factor nH kT nH Abundanc Redshift switch norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS['single_temp_mekal']
    sig = signature(single_temp_mekal)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        raise XGADeveloperError("Current keyword arguments of this function do not match the entry in FIT_FUNC_ARGS.")

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for src_ind, source in enumerate(sources):
        # Find matching spectrum objects associated with the current source
        spec_objs = source.get_spectra(out_rad_vals[src_ind], inner_radius=inn_rad_vals[src_ind],
                                       group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                                       over_sample=over_sample)
        # This is because many other parts of this function assume that spec_objs is iterable, and in the case of
        #  a cluster with only a single valid instrument for a single valid observation this may not be the case
        if isinstance(spec_objs, Spectrum):
            spec_objs = [spec_objs]

        # Obviously we can't do a fit if there are no spectra, so throw an error if that's the case
        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for {s} object, you "
                                          "need to generate them first!".format(s=source.name))

        # Turn spectra paths into TCL style list for substitution into template
        specs = "{" + " ".join([spec.path for spec in spec_objs]) + "}"
        # For this model, we have to know the redshift of the source.
        if source.redshift is None:
            raise ValueError("You cannot supply a source without a redshift to this model.")

        # Whatever start temperature is passed gets converted to keV, this will be put in the template
        start_temp = all_start_temps[src_ind].to("keV", equivalencies=u.temperature_energy())
        # Another TCL list, this time of the parameter start values for this model.
        par_values = "{{{0} {1} {2} {3} {4} {5} {6} {7}}}".format(1., source.nH.to("10^22 cm^-2").value,
                                                                  start_temp.value, 1, start_met, source.redshift,
                                                                  1, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependent on user input
        freezing = "{{F {n} {t} T {ab} T T F}}".format(n='T' if freeze_nh else 'F',
                                                       t='T' if freeze_temp else 'F',
                                                       ab='T' if freeze_met else 'F')

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

        # We generate the fit configuration key here, on a source by source basis, as the start temperature is allowed
        #  to be different for every source
        in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
        fit_conf = _gen_fit_conf(in_fit_conf)

        # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed luminosities. I
        #  am only specifying parameter 2 here (though there will likely be multiple models because there are likely
        #  multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing one will zero
        #  them all.
        nh_to_zero = "{2}"

        # If the fit has already been performed we do not wish to perform it again
        try:
            # We search for the norm parameter, as it is guaranteed to be there for any fit with this model
            res = source.get_results(out_rad_vals[src_ind], model, inn_rad_vals[src_ind], 'norm', group_spec,
                                     min_counts, min_sn, over_sample, fit_conf)
        except (ModelNotAssociatedError, FitConfNotAssociatedError):
            out_file, script_file, inv_ent = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table,
                                                                 fit_method, specs, lo_en, hi_en, par_names,
                                                                 par_values, linking, freezing, par_fit_stat,
                                                                 lum_low_lims, lum_upp_lims, lum_conf,
                                                                 source.redshift, spectrum_checking, check_list,
                                                                 check_lo_lims, check_hi_lims, check_err_lims, True,
                                                                 fit_conf, nh_to_zero)

            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)
            fit_confs.append(fit_conf)
            inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents


@xspec_call
def multi_temp_dem_apec(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                        inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                        start_max_temp: Quantity = Quantity(5.0, "keV"), start_met: float = 0.3,
                        start_t_rat: float = 0.1, start_inv_em_slope: float = 0.25,
                        lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                        freeze_nh: bool = True, freeze_met: bool = True, lo_en: Quantity = Quantity(0.3, "keV"),
                        hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                        abund_table: str = "angr", fit_method: str = "leven", group_spec: bool = True,
                        min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                        num_cores: int = NUM_CORES, spectrum_checking: bool = True,
                        timeout: Quantity = Quantity(1, 'hr')):
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
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
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
    """
    sources, inn_rad_vals, out_rad_vals = _pregen_spectra(sources, outer_radius, inner_radius, group_spec, min_counts,
                                                          min_sn, over_sample, one_rmf, num_cores)
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
    rel_args = FIT_FUNC_ARGS['multi_temp_dem_apec']
    sig = signature(multi_temp_dem_apec)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        raise XGADeveloperError("Current keyword arguments of this function do not match the entry in FIT_FUNC_ARGS.")

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
        # Find matching spectrum objects associated with the current source
        spec_objs = source.get_spectra(out_rad_vals[src_ind], inner_radius=inn_rad_vals[src_ind],
                                       group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                                       over_sample=over_sample)
        # This is because many other parts of this function assume that spec_objs is iterable, and in the case of
        #  a cluster with only a single valid instrument for a single valid observation this may not be the case
        if isinstance(spec_objs, Spectrum):
            spec_objs = [spec_objs]

        # Obviously we can't do a fit if there are no spectra, so throw an error if that's the case
        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for {s} object, you "
                                          "need to generate them first!".format(s=source.name))

        # Turn spectra paths into TCL style list for substitution into template
        specs = "{" + " ".join([spec.path for spec in spec_objs]) + "}"
        # For this model, we have to know the redshift of the source.
        if source.redshift is None:
            raise ValueError("You cannot supply a source without a redshift to this model.")

        # Whatever start temperature is passed gets converted to keV, this will be put in the template
        t = start_max_temp.to("keV", equivalencies=u.temperature_energy()).value
        # Another TCL list, this time of the parameter start values for this model.
        par_values = "{{{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}}}".format(1., source.nH.to("10^22 cm^-2").value, t,
                                                                          start_t_rat, start_inv_em_slope,
                                                                          1, start_met, source.redshift, 2, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependant on user input
        freezing = "{{F {n} F F F T {ab} T T F}}".format(n='T' if freeze_nh else 'F',
                                                         ab='T' if freeze_met else 'F')

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
            check_hi_lims = "{20 1 10}"
            check_err_lims = "{15 5 5}"
        else:
            check_list = "{}"
            check_lo_lims = "{}"
            check_hi_lims = "{}"
            check_err_lims = "{}"

        # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed luminosities. I
        #  am only specifying parameter 2 here (though there will likely be multiple models because there are likely
        #  multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing one will zero
        #  them all.
        nh_to_zero = "{2}"

        # If the fit has already been performed we do not wish to perform it again
        try:
            res = source.get_results(out_rad_vals[src_ind], model, inn_rad_vals[src_ind], 'Tmax', group_spec,
                                     min_counts, min_sn, over_sample, fit_conf)
        except (ModelNotAssociatedError, FitConfNotAssociatedError):
            # This internal function writes out the XSPEC script with all the information we've assembled in this
            #  function - filling out the XSPEC template and writing to disk
            out_file, script_file, inv_ent = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table,
                                                                 fit_method, specs, lo_en, hi_en, par_names,
                                                                 par_values, linking, freezing, par_fit_stat,
                                                                 lum_low_lims, lum_upp_lims, lum_conf, source.redshift,
                                                                 spectrum_checking, check_list, check_lo_lims,
                                                                 check_hi_lims, check_err_lims, True, fit_conf,
                                                                 nh_to_zero)
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)
            fit_confs.append(fit_conf)
            inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents


@xspec_call
def power_law(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
              inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), redshifted: bool = False,
              lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"), start_pho_index: float = 1.,
              lo_en: Quantity = Quantity(0.3, "keV"), hi_en: Quantity = Quantity(7.9, "keV"),
              freeze_nh: bool = True, par_fit_stat: float = 1., lum_conf: float = 68., abund_table: str = "angr",
              fit_method: str = "leven", group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
              over_sample: float = None, one_rmf: bool = True, num_cores: int = NUM_CORES,
              timeout: Quantity = Quantity(1, 'hr')):
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
        should be equivelant to 1sigma errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec
        /manual/XSerror.html) correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
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
    """
    sources, inn_rad_vals, out_rad_vals = _pregen_spectra(sources, outer_radius, inner_radius, group_spec, min_counts,
                                                          min_sn, over_sample, one_rmf, num_cores)
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
    rel_args = FIT_FUNC_ARGS['power_law']
    sig = signature(power_law)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        raise XGADeveloperError(
            "Current keyword arguments of this function do not match the entry in FIT_FUNC_ARGS.")

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
        spec_objs = source.get_spectra(out_rad_vals[src_ind], inner_radius=inn_rad_vals[src_ind], group_spec=group_spec,
                                       min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)

        # This is because many other parts of this function assume that spec_objs is iterable, and in the case of
        #  a source with only a single valid instrument for a single valid observation this may not be the case
        if isinstance(spec_objs, Spectrum):
            spec_objs = [spec_objs]

        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for {s}, you "
                                          "need to generate them first!".format(s=source.name))

        # Turn spectra paths into TCL style list for substitution into template
        specs = "{" + " ".join([spec.path for spec in spec_objs]) + "}"
        # For this model, we have to know the redshift of the source.
        if redshifted and source.redshift is None:
            raise ValueError("You cannot supply a source without a redshift if you have elected to fit zpowerlw.")
        elif redshifted and source.redshift is not None:
            par_values = "{{{0} {1} {2} {3} {4}}}".format(1., source.nH.to("10^22 cm^-2").value, start_pho_index,
                                                          source.redshift, 1.)
        else:
            par_values = "{{{0} {1} {2} {3}}}".format(1., source.nH.to("10^22 cm^-2").value, start_pho_index, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependant on user input
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
        if redshifted or (not redshifted and source.redshift is not None):
            z = source.redshift
        else:
            z = 1
            warnings.warn("{s} has no redshift information associated, so luminosities from this fit"
                          " will be invalid, as redshift has been set to one.".format(s=source.name), stacklevel=2)

        # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed luminosities. I
        #  am only specifying parameter 2 here (though there will likely be multiple models because there are likely
        #  multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing one will zero
        #  them all.
        nh_to_zero = "{2}"

        # If the fit has already been performed we do not wish to perform it again
        try:
            res = source.get_results(out_rad_vals[src_ind], model, inn_rad_vals[src_ind], None, group_spec, min_counts,
                                     min_sn, over_sample, fit_conf)
        except (ModelNotAssociatedError, FitConfNotAssociatedError):
            out_file, script_file, inv_ent = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table,
                                                                 fit_method, specs, lo_en, hi_en, par_names, par_values,
                                                                 linking, freezing, par_fit_stat, lum_low_lims,
                                                                 lum_upp_lims, lum_conf, z, False, "{}", "{}", "{}",
                                                                 "{}", True, fit_conf, nh_to_zero)
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)
            fit_confs.append(fit_conf)
            inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents


@xspec_call
def blackbody(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
              inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), redshifted: bool = False,
              lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
              start_temp: Quantity = Quantity(1, "keV"), lo_en: Quantity = Quantity(0.3, "keV"),
              hi_en: Quantity = Quantity(7.9, "keV"), freeze_nh: bool = True, par_fit_stat: float = 1.,
              lum_conf: float = 68., abund_table: str = "angr", fit_method: str = "leven", group_spec: bool = True,
              min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
              num_cores: int = NUM_CORES, timeout: Quantity = Quantity(1, 'hr')):
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
    :param float start_temp: The starting value for the temperature of the blackbody.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command, default is 1.0 which
        should be equivelant to 1sigma errors if I've understood (https://heasarc.gsfc.nasa.gov/xanadu/xspec
        /manual/XSerror.html) correctly.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
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
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
    """
    sources, inn_rad_vals, out_rad_vals = _pregen_spectra(sources, outer_radius, inner_radius, group_spec, min_counts,
                                                          min_sn, over_sample, one_rmf, num_cores)
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

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS['blackbody']
    sig = signature(blackbody)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        raise XGADeveloperError(
            "Current keyword arguments of this function do not match the entry in FIT_FUNC_ARGS.")

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
        spec_objs = source.get_spectra(out_rad_vals[src_ind], inner_radius=inn_rad_vals[src_ind], group_spec=group_spec,
                                       min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)

        # This is because many other parts of this function assume that spec_objs is iterable, and in the case of
        #  a source with only a single valid instrument for a single valid observation this may not be the case
        if isinstance(spec_objs, Spectrum):
            spec_objs = [spec_objs]

        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for {s}, you "
                                          "need to generate them first!".format(s=source.name))

        # Turn spectra paths into TCL style list for substitution into template
        specs = "{" + " ".join([spec.path for spec in spec_objs]) + "}"

        # Whatever start temperature is passed gets converted to keV, this will be put in the template
        t = start_temp.to("keV", equivalencies=u.temperature_energy()).value

        # For this model, we have to know the redshift of the source.
        if redshifted and source.redshift is None:
            raise ValueError("You cannot supply a source without a redshift if you have elected to fit zbbody.")
        elif redshifted and source.redshift is not None:
            par_values = "{{{0} {1} {2} {3} {4}}}".format(1., source.nH.to("10^22 cm^-2").value, t,
                                                          source.redshift, 1.)
        else:
            par_values = "{{{0} {1} {2} {3}}}".format(1., source.nH.to("10^22 cm^-2").value, t, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependant on user input
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
        if redshifted or (not redshifted and source.redshift is not None):
            z = source.redshift
        else:
            z = 1
            warnings.warn("{s} has no redshift information associated, so luminosities from this fit"
                          " will be invalid, as redshift has been set to one.".format(s=source.name), stacklevel=2)

        # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed luminosities. I
        #  am only specifying parameter 2 here (though there will likely be multiple models because there are likely
        #  multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing one will zero
        #  them all.
        nh_to_zero = "{2}"

        # If the fit has already been performed we do not wish to perform it again
        try:
            res = source.get_results(out_rad_vals[src_ind], model, inn_rad_vals[src_ind], None, group_spec, min_counts,
                                     min_sn, over_sample, fit_conf)
        except (ModelNotAssociatedError, FitConfNotAssociatedError):
            out_file, script_file, inv_ent = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table,
                                                                 fit_method, specs, lo_en, hi_en, par_names, par_values,
                                                                 linking, freezing, par_fit_stat, lum_low_lims,
                                                                 lum_upp_lims, lum_conf, z, False, "{}", "{}", "{}",
                                                                 "{}", True, fit_conf, nh_to_zero)
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)
            fit_confs.append(fit_conf)
            inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents


@xspec_call
def double_temp_apec(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                     inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                     start_temp_one: Quantity = Quantity(3.0, "keV"), start_temp_two: Quantity = Quantity(5.0, "keV"),
                     start_met_one: float = 0.3, start_met_two: float = 0.3,
                     lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"), freeze_nh: bool = True,
                     freeze_met_one: bool = True, freeze_met_two: bool = True, freeze_temp_one: bool = False,
                     freeze_temp_two: bool = False, lo_en: Quantity = Quantity(0.3, "keV"),
                     hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                     abund_table: str = "angr", fit_method: str = "leven", group_spec: bool = True, min_counts: int = 5,
                     min_sn: float = None, over_sample: float = None, one_rmf: bool = True, num_cores: int = NUM_CORES,
                     spectrum_checking: bool = False, timeout: Quantity = Quantity(1, 'hr')):
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
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
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
    """

    # This is a little cheesy as we haven't decided what exactly to do yet, but the spectrum checking behaviour
    #  is super inconsistent with double_temp_apec - presumably just because the greater number of parameters
    #  makes estimating errors much less stable for individual spectral fits
    if spectrum_checking:
        raise NotImplementedError("The double_temp_apec function does not currently support "
                                  "'spectrum_checking=True', as it exhibits very unstable behaviour.")

    sources, inn_rad_vals, out_rad_vals = _pregen_spectra(sources, outer_radius, inner_radius, group_spec, min_counts,
                                                          min_sn, over_sample, one_rmf, num_cores)
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # Have to check that every source has start temperature entries, if the user decided to pass a set of them - this
    #  is identical to the checks we perform at the beginning of 'single_temp_apec'
    if ((not start_temp_one.isscalar and len(start_temp_one) != len(sources)) or
            (not start_temp_two.isscalar and len(start_temp_two) != len(sources))):
        raise ValueError("If a non-scalar Quantity is passed for 'start_temp_one' or 'start_temp_two', there must be "
                         "one entry for each source. They currently have {n1} and {n2} respectively, for {s} "
                         "sources.".format(n1=len(start_temp_one), n2=len(start_temp_two), s=len(sources)))

    # Want to make sure that the start_temp variable is always a non-scalar Quantity with an entry for every source
    #  after this point, it means we normalise how we deal with it.
    if start_temp_one.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temp_ones = Quantity([start_temp_one.value.copy()]*len(sources), start_temp_one.unit)
    else:
        all_start_temp_ones = start_temp_one.copy()

    # And the same thing for start temperature the second
    if start_temp_two.isscalar:
        # Doing it like this, defining a new variable and then redeclaring the start_temp in each bit of the loop
        #  below is important, as the fit_conf generation code accesses the current value of start_temp specifically
        all_start_temp_twos = Quantity([start_temp_two.value.copy()]*len(sources), start_temp_two.unit)
    else:
        all_start_temp_twos = start_temp_two.copy()

    # This function is for a set model, absorbed double apec, so I can hard code all of this stuff.
    model = "constant*tbabs*(apec+apec)"
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be
    #  in the form of TCL lists.
    par_names = "{factor nH kT Abundanc Redshift norm kT Abundanc Redshift norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    rel_args = FIT_FUNC_ARGS['double_temp_apec']

    sig = signature(double_temp_apec)
    cur_args = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

    # This is purely for developers, as a check to make sure that the FIT_FUNC_ARGS dictionary is updated if the
    #  signature of this function is altered.
    if set(list(rel_args.keys())) != set(list(cur_args.keys())):
        raise XGADeveloperError("Current keyword arguments of this function do not match the entry in FIT_FUNC_ARGS.")

    script_paths = []
    outfile_paths = []
    src_inds = []
    fit_confs = []
    inv_ents = []
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for src_ind, source in enumerate(sources):
        # Find matching spectrum objects associated with the current source
        spec_objs = source.get_spectra(out_rad_vals[src_ind], inner_radius=inn_rad_vals[src_ind],
                                       group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                                       over_sample=over_sample)
        # This is because many other parts of this function assume that spec_objs is iterable, and in the case of
        #  a cluster with only a single valid instrument for a single valid observation this may not be the case
        if isinstance(spec_objs, Spectrum):
            spec_objs = [spec_objs]

        # Obviously we can't do a fit if there are no spectra, so throw an error if that's the case
        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for {s} object, you "
                                          "need to generate them first!".format(s=source.name))

        # Turn spectra paths into TCL style list for substitution into template
        specs = "{" + " ".join([spec.path for spec in spec_objs]) + "}"
        # For this model, we have to know the redshift of the source.
        if source.redshift is None:
            raise ValueError("You cannot supply a source without a redshift to this model.")

        # Whatever start temperatures are passed get converted to keV - then will be put in the template
        start_temp_one = all_start_temp_ones[src_ind].to("keV", equivalencies=u.temperature_energy())
        start_temp_two = all_start_temp_twos[src_ind].to("keV", equivalencies=u.temperature_energy())

        # Another TCL list, this time of the parameter start values for this model.
        par_values = ("{{{0} {1} {2} {3} {4} {5} {6} {7} {8} "
                      "{9}}}").format(1., source.nH.to("10^22 cm^-2").value, start_temp_one.value, start_met_one,
                                      source.redshift, 1., start_temp_two.value, start_met_two, source.redshift, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependent on user input - this can now
        #  include the temperature, if the user wants it fixed at the start value
        freezing = "{{F {n} {t1} {a2} T F {t2} {a2} T F}}".format(n="T" if freeze_nh else "F",
                                                                  t1="T" if freeze_temp_one else "F",
                                                                  a1="T" if freeze_met_one else "F",
                                                                  t2="T" if freeze_temp_two else "F",
                                                                  a2="T" if freeze_met_two else "F")

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

        # We generate the fit configuration key here, on a source by source basis, as the start temperature is allowed
        #  to be different for every source
        in_fit_conf = {kn: locals()[kn] for kn in rel_args if rel_args[kn]}
        fit_conf = _gen_fit_conf(in_fit_conf)

        # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed luminosities. I
        #  am only specifying parameter 2 here (though there will likely be multiple models because there are likely
        #  multiple spectra) because I know that nH of tbabs is linked in this setup, so zeroing one will zero
        #  them all.
        nh_to_zero = "{2}"

        # If the fit has already been performed we do not wish to perform it again
        try:
            # We search for the norm parameter, as it is guaranteed to be there for any fit with this model
            res = source.get_results(out_rad_vals[src_ind], model, inn_rad_vals[src_ind], 'norm', group_spec,
                                     min_counts, min_sn, over_sample, fit_conf)
        except (ModelNotAssociatedError, FitConfNotAssociatedError):
            out_file, script_file, inv_ent = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table,
                                                                 fit_method, specs, lo_en, hi_en, par_names, par_values,
                                                                 linking, freezing, par_fit_stat, lum_low_lims,
                                                                 lum_upp_lims, lum_conf, source.redshift,
                                                                 spectrum_checking, check_list, check_lo_lims,
                                                                 check_hi_lims, check_err_lims, True, fit_conf,
                                                                 nh_to_zero)
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)
            fit_confs.append(fit_conf)
            inv_ents.append(inv_ent)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout, model, fit_confs, inv_ents


# This allows us to make a link between the model that was fit and the XGA function
FIT_FUNC_MODEL_NAMES = {'constant*tbabs*apec': single_temp_apec,
                        'constant*tbabs*mekal': single_temp_mekal,
                        'constant*tbabs*wdem': multi_temp_dem_apec,
                        'constant*tbabs*zpowerlw': power_law,
                        'constant*tbabs*powerlaw': power_law,
                        'constant*tbabs*zbbody': blackbody,
                        'constant*tbabs*bbody': blackbody,
                        'constant*tbabs*(apec+apec)': double_temp_apec}
