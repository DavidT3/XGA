#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 20/01/2021, 16:31. Copyright (c) David J Turner

import os
import warnings
from typing import List, Union, Tuple

import astropy.units as u
from astropy.units import Quantity

from .run import xspec_call
from .. import OUTPUT, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT, XSPEC_FIT_METHOD, ABUND_TABLES
from ..exceptions import NoProductAvailableError, ModelNotAssociatedError
from ..samples.base import BaseSample
from ..sas import evselect_spectrum, region_setup
from ..sources import BaseSource, ExtendedSource, PointSource


def _check_inputs(sources: Union[BaseSource, BaseSample], lum_en: Quantity, lo_en: Quantity,  hi_en: Quantity,
                  fit_method: str, abund_table: str) -> Union[List[BaseSource], BaseSample]:
    """
    This performs some checks that are common to all the model fit functions.
    :param Union[BaseSource, BaseSample] sources:
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :return: Most likely just the passed in sources, but if a single source was passed
    then a list will be returned.
    :rtype: Union[List[BaseSource], BaseSample]
    """
    # This function supports passing both individual sources and samples of sources, but I do require that
    #  the sources object is iterable
    if isinstance(sources, BaseSource):
        sources = [sources]

    # Not allowed to use BaseSources for this, though they shouldn't have spectra anyway
    if not all([isinstance(src, (ExtendedSource, PointSource)) for src in sources]):
        raise TypeError("This convenience function cannot be used with BaseSource objects.")
    elif not all([src.detected for src in sources]):
        warnings.warn("Not all of these sources have been detected, you may get a poor fit.")

    # Checks that the luminosity energy bands are pairs of values
    if lum_en.shape[1] != 2:
        raise ValueError("Luminosity energy bands should be supplied in pairs, defined "
                         "like Quantity([[0.5, 2.0], [2.0, 10.0]], 'keV')")
    # Are the lower limits smaller than the upper limits? - Obviously they should be so I check
    elif not all([lum_en[pair_ind, 0] < lum_en[pair_ind, 1] for pair_ind in range(0, lum_en.shape[0])]):
        raise ValueError("Luminosity energy band first entries must be smaller than second entries.")

    # These are different energy limits to those above, these are what govern how much of the data we fit to.
    # Do the same check to make sure lower limit is less than upper limit
    if lo_en > hi_en:
        raise ValueError("lo_en cannot be greater than hi_en.")

    if fit_method not in XSPEC_FIT_METHOD:
        raise ValueError("{f} is not an XSPEC fit method, allowed fit methods are; "
                         "{a}.".format(f=fit_method, a=", ".join(XSPEC_FIT_METHOD)))

    if abund_table not in ABUND_TABLES:
        raise ValueError("{f} is not an XSPEC abundance table, allowed abundance tables are; "
                         "{a}.".format(f=fit_method, a=", ".join(ABUND_TABLES)))

    return sources


def _write_xspec_script(source: BaseSource, spec_storage_key: str, model: str, abund_table: str, fit_method: str,
                        specs: str, lo_en: Quantity, hi_en: Quantity, par_names: str, par_values: str,
                        linking: str, freezing: str, par_fit_stat: float, lum_low_lims: str, lum_upp_lims: str,
                        lum_conf: float, redshift: float) -> Tuple[str, str]:
    """
    This writes out a configured XSPEC script, and is common to all fit functions.

    :param BaseSource source: The source for which an XSPEC script is being created
    :param str spec_storage_key: The storage key that the spectra that have been included in the current fit
        are stored under.
    :param str model: The model being fitted to the data.
    :param str abund_table: The chosen abundance table for XSPEC to use.
    :param str fit_method: Which fit method should XSPEC use to fit the model to data.
    :param str specs: A string containing the paths to all spectra to be fitted.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param str par_names: A string containing the names of the model parameters.
    :param str par_values: A string containing the start values of the model parameters.
    :param str linking: A string containing the linking settings for the model.
    :param str freezing: A string containing the freezing settings for the model.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command.
    :param str lum_low_lims: A string containing the lower energy limits for the luminosity measurements.
    :param str lum_upp_lims: A string containing the upper energy limits for the luminosity measurements.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param float redshift: The redshift of the object.
    :return: The paths to the output file and the script file.
    :rtype: Tuple[str, str]
    """
    # Read in the template file for the XSPEC script.
    with open(BASE_XSPEC_SCRIPT, 'r') as x_script:
        script = x_script.read()

    # There has to be a directory to write this xspec script to, as well as somewhere for the fit output
    #  to be stored
    dest_dir = OUTPUT + "XSPEC/" + source.name + "/"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Defining where the output summary file of the fit is written
    out_file = dest_dir + source.name + "_" + spec_storage_key + "_" + model
    script_file = dest_dir + source.name + "_" + spec_storage_key + "_" + model + ".xcm"

    # The template is filled out here, taking everything we have generated and everything the user
    #  passed in. The result is an XSPEC script that can be run as is.
    script = script.format(xsp=XGA_EXTRACT, ab=abund_table, md=fit_method, H0=source.cosmo.H0.value,
                           q0=0., lamb0=source.cosmo.Ode0, sp=specs, lo_cut=lo_en.to("keV").value,
                           hi_cut=hi_en.to("keV").value, m=model, pn=par_names, pv=par_values,
                           lk=linking, fr=freezing, el=par_fit_stat, lll=lum_low_lims, lul=lum_upp_lims,
                           of=out_file, redshift=redshift, lel=lum_conf)

    # Write out the filled-in template to its destination
    with open(script_file, 'w') as xcm:
        xcm.write(script)

    return out_file, script_file


@xspec_call
def single_temp_apec(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                     inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                     start_temp: Quantity = Quantity(3.0, "keV"), start_met: float = 0.3,
                     lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                     freeze_nh: bool = True, freeze_met: bool = True,
                     link_norm: bool = False, lo_en: Quantity = Quantity(0.3, "keV"),
                     hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                     abund_table: str = "angr", fit_method: str = "leven", group_spec: bool = True,
                     min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                     num_cores: int = NUM_CORES):
    """
    This is a convenience function for fitting an absorbed single temperature apec model to an object.
    It would be possible to do the exact same fit using the custom_model function, but as it will
    be a very common fit a dedicated function is in order. If there are no existing spectra with the passed
    settings, then they will be generated automatically.

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
    :param bool link_norm: Whether the normalisations of different spectra should be linked during fitting.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command.
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
    """
    # I call the evselect_spectrum function here for two reasons; to make sure that the spectra which the user
    #  want to fit are generated, and because that function has a lot of radius parsing and checking stuff
    #  in it which will kick up a fuss if variables aren't formatted right
    evselect_spectrum(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn, over_sample, one_rmf,
                      num_cores)

    # This is the spectrum region preparation function, and I'm calling it here because it will return properly
    #  formatted arrays for the inner and outer radii
    if outer_radius != 'region':
        inn_rad_vals, out_rad_vals = region_setup(sources, outer_radius, inner_radius, True, '')[1:]
    else:
        raise NotImplementedError("I don't currently support fitting region spectra")

    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table)

    # This function is for a set model, absorbed apec, so I can hard code all of this stuff.
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    model = "tbabs*apec"
    par_names = "{nH kT Abundanc Redshift norm}"
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
        par_values = "{{{0} {1} {2} {3} {4}}}".format(source.nH.to("10^22 cm^-2").value, t,
                                                      start_met, source.redshift, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependant on user input
        if freeze_nh and freeze_met:
            freezing = "{T F T T F}"
        elif not freeze_nh and freeze_met:
            freezing = "{F F T T F}"
        elif freeze_nh and not freeze_met:
            freezing = "{T F F T F}"
        elif not freeze_nh and not freeze_met:
            freezing = "{F F F T F}"

        # Set up the TCL list that defines which parameters are linked across different spectra,
        #  dependant on user input
        if link_norm:
            linking = "{T T T T T}"
        else:
            linking = "{T T T T F}"

        out_file, script_file = _write_xspec_script(source, spec_objs[0].storage_key, model, abund_table, fit_method,
                                                    specs, lo_en, hi_en, par_names, par_values, linking, freezing,
                                                    par_fit_stat, lum_low_lims, lum_upp_lims, lum_conf, source.redshift)

        # If the fit has already been performed we do not wish to perform it again
        try:
            res = source.get_results(model, out_rad_vals[src_ind], inn_rad_vals[src_ind], 'kT', group_spec, min_counts,
                                     min_sn, over_sample)
        except ModelNotAssociatedError:
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, inn_rad_vals, out_rad_vals, run_type, src_inds


def double_temp_apec():
    raise NotImplementedError("The double temperature model for clusters is under construction.")


@xspec_call
def power_law(sources: Union[BaseSource, BaseSample], reg_type: str, redshifted: bool = False,
              lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"), start_pho_index: float = 1.,
              lo_en: Quantity = Quantity(0.3, "keV"), hi_en: Quantity = Quantity(7.9, "keV"),
              freeze_nh: bool = True, link_norm: bool = False, par_fit_stat: float = 1., lum_conf: float = 68.,
              abund_table: str = "angr", fit_method: str = "leven", num_cores: int = NUM_CORES):
    """
    This is a convenience function for fitting a tbabs absorbed powerlaw (or zpowerlw if redshifted
    is selected) to source spectra.
    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str reg_type: Tells the method what region's spectrum you want to use, for instance r500 or r200.
    :param bool redshifted: Whether the powerlaw that includes redshift (zpowerlw) should be used.
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param float start_pho_index: The starting value for the photon index of the powerlaw.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.    :param start_pho_index:
    :param bool link_norm: Whether the normalisations of different spectra should be linked during fitting.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    """
    raise NotImplementedError("Haven't updated this to work with new radii yet")
    sources = _check_inputs(sources, reg_type, lum_en, lo_en, hi_en, fit_method, abund_table)

    # This function is for a set model, either absorbed powerlaw or absorbed zpowerlw
    # These will be inserted into the general XSPEC script template, so lists of parameters need to be in the form
    #  of TCL lists.
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"
    if redshifted:
        model = "tbabs*zpowerlw"
        par_names = "{nH PhoIndex Redshift norm}"
    else:
        model = "tbabs*powerlaw"
        par_names = "{nH PhoIndex norm}"

    script_paths = []
    outfile_paths = []
    src_inds = []
    for src_ind, source in enumerate(sources):
        # Find matching spectrum objects associated with the current source, and checking if they are valid
        spec_objs = [match for match in source.get_products("spectrum", extra_key=reg_type) if match.usable]
        # Obviously we can't do a fit if there are no spectra, so throw an error if thats the case
        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for {s} object, you "
                                          "need to generate them first!".format(s=source.name))

        # Turn spectra paths into TCL style list for substitution into template
        specs = "{" + " ".join([spec.path for spec in spec_objs]) + "}"
        # For this model, we have to know the redshift of the source.
        if redshifted and source.redshift is None:
            raise ValueError("You cannot supply a source without a redshift if you have elected to fit zpowerlw.")
        elif redshifted and source.redshift is not None:
            par_values = "{{{0} {1} {2} {3}}}".format(source.nH.to("10^22 cm^-2").value, start_pho_index,
                                                      source.redshift, 1.)
        else:
            par_values = "{{{0} {1} {2}}}".format(source.nH.to("10^22 cm^-2").value, start_pho_index, 1.)

        # Set up the TCL list that defines which parameters are frozen, dependant on user input
        if redshifted and freeze_nh:
            freezing = "{T F T F}"
        elif not redshifted and freeze_nh:
            freezing = "{T F F}"
        elif redshifted and not freeze_nh:
            freezing = "{F F T F}"
        elif not redshifted and not freeze_nh:
            freezing = "{F F F}"

        # Set up the TCL list that defines which parameters are linked across different spectra,
        #  dependant on user input
        if redshifted and link_norm:
            linking = "{T T T T}"
        elif not redshifted and link_norm:
            linking = "{T T T}"
        if redshifted and not link_norm:
            linking = "{T T T F}"
        elif not redshifted and not link_norm:
            linking = "{T T F}"

        # If the powerlaw with redshift has been chosen, then we use the redshift attached to the source object
        #  If not we just pass a filler redshift and the luminosities are invalid
        if redshifted or (not redshifted and source.redshift is not None):
            z = source.redshift
        else:
            z = 0
            warnings.warn("{s} has no redshift information associated, so luminosities from this fit"
                          " will be invalid, as redshift has been set to zero.".format(s=source.name))

        out_file, script_file = _write_xspec_script(source, reg_type, model, abund_table, fit_method, specs,
                                                    lo_en, hi_en, par_names, par_values, linking, freezing,
                                                    par_fit_stat, lum_low_lims, lum_upp_lims, lum_conf, z)

        # If the fit has already been performed we do not wish to perform it again
        try:
            # TODO THIS CALL WILL NO LONGER WORK WITH THE NEW RADII STUFF
            res = source.get_results(reg_type, model)
        except ModelNotAssociatedError:
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, reg_type, run_type, src_inds


def custom():
    raise NotImplementedError("User defined model support is a little way from being implemented.")





