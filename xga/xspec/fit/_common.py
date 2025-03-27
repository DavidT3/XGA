#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/08/2024, 10:22. Copyright (c) The Contributors

import os
import warnings
from random import randint
from typing import List, Union, Tuple

from astropy.units import Quantity, UnitConversionError

from ... import OUTPUT, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT, XSPEC_FIT_METHOD, ABUND_TABLES, \
    CROSS_ARF_XSPEC_SCRIPT
from ...samples.base import BaseSample
from ...sas import evselect_spectrum, region_setup
from ...sources import ExtendedSource, PointSource, BaseSource


def _pregen_spectra(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                    inner_radius: Union[str, Quantity], group_spec: bool = True, min_counts: int = 5,
                    min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                    num_cores: int = NUM_CORES) -> Tuple[Union[List[BaseSource], BaseSample], Quantity, Quantity]:
    """
    This pre-generates the spectra necessary for the requested fit (if they do not exist), and formats the input
    radii in a more predictable way.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
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
    :return: Most likely just the passed in sources, but if a single source was passed
    then a list will be returned.
    :rtype: Union[List[BaseSource], BaseSample]
    """
    # I call the evselect_spectrum function here for two reasons; to make sure that the spectra which the user
    #  want to fit are generated, and because that function has a lot of radius parsing and checking stuff
    #  in it which will kick up a fuss if variables aren't formatted right
    sources = evselect_spectrum(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn, over_sample,
                                one_rmf, num_cores)

    # This is the spectrum region preparation function, and I'm calling it here because it will return properly
    #  formatted arrays for the inner and outer radii
    if outer_radius != 'region':
        inn_rad_vals, out_rad_vals = region_setup(sources, outer_radius, inner_radius, True, '')[1:]
    else:
        raise NotImplementedError("I don't currently support fitting region spectra")

    return sources, inn_rad_vals, out_rad_vals


def _check_inputs(sources: Union[BaseSource, BaseSample], lum_en: Quantity, lo_en: Quantity, hi_en: Quantity,
                  fit_method: str, abund_table: str, timeout: Quantity) -> Union[List[BaseSource], BaseSample]:
    """
    This performs some checks that are common to all the model fit functions, also makes sure the necessary spectra
    have been generated.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param str fit_method: The XSPEC fit method to use.
    :param str abund_table: The abundance table to use for the fit.
    :param Quantity timeout: The amount of time each individual fit is allowed to run for, the default is one hour.
        Please note that this is not a timeout for the entire fitting process, but a timeout to individual source
        fits.
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

    if not timeout.unit.is_equivalent('second'):
        raise UnitConversionError("The timeout quantity must be in units which can be converted to seconds, you have"
                                  " passed a quantity with units of {}".format(timeout.unit.to_string()))

    return sources


def _write_xspec_script(source: BaseSource, spec_storage_key: str, model: str, abund_table: str, fit_method: str,
                        specs: str, lo_en: Quantity, hi_en: Quantity, par_names: str, par_values: str, linking: str,
                        freezing: str, par_fit_stat: float, lum_low_lims: str, lum_upp_lims: str, lum_conf: float,
                        redshift: float, pre_check: bool, check_par_names: str, check_par_lo_lims: str,
                        check_par_hi_lims: str, check_par_err_lims: str, norm_scale: bool, fit_conf: str,
                        which_par_nh: str = 'None', rand_ident: str = None, cross_arf_paths: str = None,
                        cross_arf_rmf_paths: str = None) -> Tuple[str, str, list]:
    """
    This writes out a configured XSPEC script, for global fits, annular fits, and annular fits with cross-arf
    support depending on the input parameters.

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
    :param bool pre_check: Flag indicating whether a pre-check of the quality of the input spectra
        should be performed.
    :param str check_par_names: A string representing a TCL list of model parameter names that checks should be
        performed on.
    :param str check_par_lo_lims: A string representing a TCL list of allowed lower limits for the check_par_names
        parameter entries.
    :param str check_par_hi_lims: A string representing a TCL list of allowed upper limits for the check_par_names
        parameter entries.
    :param str check_par_err_lims: A string representing a TCL list of allowed upper limits for the parameter
        uncertainties.
    :param bool norm_scale: Is there an extra constant designed to account for the differences in normalisation
        you can get from different observations of a source.
    :param str fit_conf: The fit configuration key for this model fit run. This is incorporated into script and
        output file names in order to uniquely identify model fits with different configurations.
    :param str which_par_nh: The parameter IDs of the nH parameters values which should be zeroed for the calculation
        of unabsorbed luminosities.
    :param str rand_ident: A random identifier to identify the fit, and fit results file. Default is None, in which
        case this function will generate an identifier, but we also allow it to be passed in for annular spectrum
        fit methods, so we can easily identify annulus fits from the same run (the format will be the same random
        identifier with _{annulus id} appended).
    :param str cross_arf_paths: The TCL list of paths to the cross-arf files used for all the annuli
        and annuli combinations in this script.
    :param str cross_arf_rmf_paths: The TCL list of paths to the RMFs used to generate the cross-arfs.
    :param str which_par_nh: The parameter IDs of the nH parameters values which should be zeroed for the calculation
        of unabsorbed luminosities.
    :return: The paths to the output file and the script file, plus the presumptive entry for the fit inventory in the
        case that the fit is successful.
    :rtype: Tuple[str, str, list]
    """
    # There has to be a directory to write this xspec script to, as well as somewhere for the fit output
    #  to be stored
    dest_dir = OUTPUT + "XSPEC/" + source.name + "/"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # We're keeping an inventory of XSPEC fits for each source, and if it doesn't already exist we set up the file
    if not os.path.exists(dest_dir + 'inventory.csv'):
        # These are the columns we want to have in our inventory file
        inv_cols = ['results_file', 'spec_key', 'fit_conf_key', 'obs_ids', 'insts', 'src_name', 'type',
                    'set_ident']
        with open(dest_dir + 'inventory.csv', 'w') as writo:
            # Writing out the inventory column names to the new file
            inv_hdr = ",".join(inv_cols) + "\n"
            writo.write(inv_hdr)

    if rand_ident is None:
        # It is possible that some XSPEC fitting file names containing the spectrum storage key and fit
        #  configuration key will be too long to be allowed, so we're going to keep an inventory and assign them
        #  random identifiers.
        rand_ident = randint(0, int(1e+8))

    # We create the eventual final output results file from the fit - it won't necessarily get made because the fit
    #  might fail, but we're making life easier on ourselves by creating the inventory entry here so it can be put
    #  in the file by xspec_call if the fit succeeds
    # We are making it a relative path to make it more resilient to the output directory being moved
    final_res_file = source.name + '_' + str(rand_ident) + ".fits"

    # Create strings of the ObsID-instruments that represent the data currently associated with the source
    i_str = "/".join([i for o in source.instruments for i in source.instruments[o]])
    o_str = "/".join([o for o in source.instruments for i in source.instruments[o]])

    # This way we can identify if the fit is to an annular spectrum or a regular set of spectra
    if 'ident' in spec_storage_key:
        # These only occur in the annular spectra
        set_ident = spec_storage_key.split('ident')[-1].split('_')[0]
    else:
        set_ident = ''

    # Now we can define the 'type' of fit it is - this will help out when reading things in
    if set_ident == '':
        fit_type = 'global'
    elif set_ident != '' and cross_arf_paths is None:
        fit_type = 'ann'
    elif set_ident != '' and cross_arf_paths is not None:
        fit_type = 'ann_carf'

    # Now we create the presumptive entry in the inventory file, to be stored there if the fit succeeds
    inv_ent = [final_res_file, spec_storage_key, fit_conf, o_str, i_str, source.name, fit_type, str(set_ident)]

    # Defining where the output summary file of the fit is written - these file names used to contain all the
    #  information about the fit, but we're using an inventory file and randomly generated identifiers now
    # out_file = dest_dir + source.name + "_" + spec_storage_key + "_" + model + "_" + fit_conf
    # script_file = dest_dir + source.name + "_" + spec_storage_key + "_" + model + "_" + fit_conf + ".xcm"
    out_file = dest_dir + source.name + "_" + str(rand_ident)
    script_file = dest_dir + source.name + "_" + str(rand_ident) + '.xcm'

    # We can tell if the 'standard' script template or the cross-arf template is needed by checking whether the
    #  cross arf paths variable has been set
    if cross_arf_paths is None:
        # Read in the template file for the usual XSPEC script.
        with open(BASE_XSPEC_SCRIPT, 'r') as x_script:
            script = x_script.read()

        # The template is filled out here, taking everything we have generated and everything the user
        #  passed in. The result is an XSPEC script that can be run as is.
        script = script.format(xsp=XGA_EXTRACT, ab=abund_table, md=fit_method, H0=source.cosmo.H0.value,
                               q0=0., lamb0=source.cosmo.Ode0, sp=specs, lo_cut=lo_en.to("keV").value,
                               hi_cut=hi_en.to("keV").value, m=model, pn=par_names, pv=par_values,
                               lk=linking, fr=freezing, el=par_fit_stat, lll=lum_low_lims, lul=lum_upp_lims,
                               of=out_file, redshift=redshift, lel=lum_conf, check=pre_check, cps=check_par_names,
                               cpsl=check_par_lo_lims, cpsh=check_par_hi_lims, cpse=check_par_err_lims, ns=norm_scale,
                               nhmtz=which_par_nh)

    else:
        # Read in the template file for the cross-arf XSPEC script - quite different fundamentally
        with open(CROSS_ARF_XSPEC_SCRIPT, 'r') as x_script:
            script = x_script.read()

        # The template is filled out here, taking everything we have generated and everything the user
        #  passed in. The result is an XSPEC script that can be run as is.
        script = script.format(xsp=XGA_EXTRACT, ab=abund_table, md=fit_method, H0=source.cosmo.H0.value, q0=0.,
                               lamb0=source.cosmo.Ode0, sp=specs, lo_cut=lo_en.to("keV").value,
                               hi_cut=hi_en.to("keV").value, m=model, pn=par_names, pv=par_values, lk=linking,
                               fr=freezing, el=par_fit_stat, lll=lum_low_lims, lul=lum_upp_lims, of=out_file,
                               redshift=redshift, lel=lum_conf, check=pre_check, cps=check_par_names,
                               cpsl=check_par_lo_lims, cpsh=check_par_hi_lims, cpse=check_par_err_lims, ns=norm_scale,
                               nhmtz=which_par_nh, cap=cross_arf_paths, carp=cross_arf_rmf_paths)

    # Write out the filled-in template to its destination
    with open(script_file, 'w') as xcm:
        xcm.write(script)

    return out_file, script_file, inv_ent

