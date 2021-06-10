#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 10/06/2021, 11:19. Copyright (c) David J Turner

import warnings
from typing import List, Union

import astropy.units as u
from astropy.units import Quantity

from .common import _check_inputs, _write_xspec_script, _pregen_spectra
from ..run import xspec_call
from ... import NUM_CORES
from ...exceptions import NoProductAvailableError, ModelNotAssociatedError
from ...products import Spectrum
from ...samples.base import BaseSample
from ...sources import BaseSource


@xspec_call
def single_temp_apec(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                     inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                     start_temp: Quantity = Quantity(3.0, "keV"), start_met: float = 0.3,
                     lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                     freeze_nh: bool = True, freeze_met: bool = True, lo_en: Quantity = Quantity(0.3, "keV"),
                     hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                     abund_table: str = "angr", fit_method: str = "leven", group_spec: bool = True,
                     min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                     num_cores: int = NUM_CORES, spectrum_checking: bool = True, timeout: Quantity = Quantity(1, 'hr')):
    """
    This is a convenience function for fitting an absorbed single temperature apec model(constant*tbabs*apec) to an
    object. It would be possible to do the exact same fit using the custom_model function, but as it will
    be a very common fit a dedicated function is in order. If there are no existing spectra with the passed
    settings, then they will be generated automatically.

    If the spectrum checking step of the XSPEC fit is enabled (using the boolean flag spectrum_checking), then
    each individual spectrum available for a given source will be fitted, and if the measured temperature is less
    than or equal to 0.01keV, or greater than 20keV, or the temperature uncertainty is greater than 15keV, then
    that spectrum will be rejected and not included in the final fit. Spectrum checking also involves rejecting any
    spectra with fewer than 10 noticed channels.

    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any
        inner radius will be ignored. If you are fitting for multiple sources then you can also pass a
        Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any
        inner radius will be ignored. By default this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param Quantity start_temp: The initial temperature for the fit.
    :param start_met: The initial metallicity for the fit (in ZSun).
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.
    :param bool freeze_met: Whether the metallicity parameter in the fit should be frozen.
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

    # This function is for a set model, absorbed apec, so I can hard code all of this stuff.
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    model = "constant*tbabs*apec"
    par_names = "{factor nH kT Abundanc Redshift norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    script_paths = []
    outfile_paths = []
    src_inds = []
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
        t = start_temp.to("keV", equivalencies=u.temperature_energy()).value
        # Another TCL list, this time of the parameter start values for this model.
        par_values = "{{{0} {1} {2} {3} {4} {5}}}".format(1., source.nH.to("10^22 cm^-2").value, t, start_met,
                                                          source.redshift, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependant on user input
        if freeze_nh and freeze_met:
            freezing = "{F T F T T F}"
        elif not freeze_nh and freeze_met:
            freezing = "{F F F T T F}"
        elif freeze_nh and not freeze_met:
            freezing = "{F T F F T F}"
        elif not freeze_nh and not freeze_met:
            freezing = "{F F F F T F}"

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

        out_file, script_file = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table, fit_method,
                                                    specs, lo_en, hi_en, par_names, par_values, linking, freezing,
                                                    par_fit_stat, lum_low_lims, lum_upp_lims, lum_conf, source.redshift,
                                                    spectrum_checking, check_list, check_lo_lims, check_hi_lims,
                                                    check_err_lims, True)

        # If the fit has already been performed we do not wish to perform it again
        try:
            res = source.get_results(out_rad_vals[src_ind], model, inn_rad_vals[src_ind], 'kT', group_spec, min_counts,
                                     min_sn, over_sample)
        except ModelNotAssociatedError:
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout


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
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any
        inner radius will be ignored. If you are fitting for multiple sources then you can also pass a
        Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the outer radius of the region that the
        desired spectrum covers (for instance 'r200' would be acceptable for a GalaxyCluster,
        or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any
        inner radius will be ignored. By default this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param bool redshifted: Whether the powerlaw that includes redshift (zpowerlw) should be used.
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param float start_pho_index: The starting value for the photon index of the powerlaw.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.    :param start_pho_index:
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

    script_paths = []
    outfile_paths = []
    src_inds = []
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
                          " will be invalid, as redshift has been set to one.".format(s=source.name))

        out_file, script_file = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table, fit_method,
                                                    specs, lo_en, hi_en, par_names, par_values, linking, freezing,
                                                    par_fit_stat, lum_low_lims, lum_upp_lims, lum_conf, z, False, "{}",
                                                    "{}", "{}", "{}", True)

        # If the fit has already been performed we do not wish to perform it again
        try:
            res = source.get_results(out_rad_vals[src_ind], model, inn_rad_vals[src_ind], None, group_spec, min_counts,
                                     min_sn, over_sample)
        except ModelNotAssociatedError:
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, None, timeout


