#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 12/03/2025, 10:50. Copyright (c) The Contributors

import os
from typing import List, Union, Tuple, Dict
from warnings import warn

from astropy.units import Quantity, UnitConversionError

from ... import OUTPUT, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT, XSPEC_FIT_METHOD, ABUND_TABLES
from ...exceptions import NoProductAvailableError
from ...generate.ciao.spec import specextract_spectrum
from ...generate.esass import srctool_spectrum
from ...generate.sas import evselect_spectrum, region_setup
from ...products import Spectrum
from ...samples.base import BaseSample
from ...sources import BaseSource, ExtendedSource, PointSource


def _pregen_spectra(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                    inner_radius: Union[str, Quantity], group_spec: bool = True, min_counts: int = 5,
                    min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                    num_cores: int = NUM_CORES, stacked_spectra: bool = False) \
        -> Tuple[Union[List[BaseSource], BaseSample], Quantity, Quantity]:
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
        inner radius will be ignored. By default, this is zero arcseconds, resulting in a circular spectrum. If
        you are fitting for multiple sources then you can also pass a Quantity with one entry per source.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be generated. If a
        stacking procedure for a particular telescope is not supported, this function will instead use individual
        spectra for an ObsID. The default is False.
    :return: Most likely just the passed in sources, but if a single source was passed then a list will be returned.
    :rtype: Union[List[BaseSource], BaseSample]
    """
    # Have to import here due to a circular import error
    from ...sourcetools._common import _get_all_telescopes
    # This returns a list of associated telescopes, for BaseSources, BaseSamples, and lists of source objects
    all_telescopes = _get_all_telescopes(sources)

    for tel in all_telescopes:
        # TODO create a function that does this sort of thing for us - as in generating spectra for each of the
        #  telescopes that are associated with a source or sample
        # Each telescope has its own methods of generating spectra
        if tel == 'xmm':
            warn("Spectrum stacking is not currently supported for XMM, and so combined spectra will not be used for"
                 " these XSPEC fits.", stacklevel=2)
            # I call the evselect_spectrum function here for two reasons; to make sure that the spectra which the user
            #  want to fit are generated, and because that function has a lot of radius parsing and checking stuff
            #  in it which will kick up a fuss if variables aren't formatted right
            sources = evselect_spectrum(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn,
                                        over_sample, one_rmf, num_cores)
        elif tel == 'chandra':
            warn("Spectrum stacking is not currently supported for Chandra, and so combined spectra will not be "
                 "used for these XSPEC fits.", stacklevel=2)
            # Make sure we have Chandra spectra generated, so that we can fit them

            # TODO DEAL WITH THIS
            # Currently have to raise an error, because the name of the spectrum generation function, and how it is
            #  called, are not currently finalised
            warn("Spectrum generation appears to be working for Chandra currently, but the actual command to call "
                 "has not been finalised, so this will need revisiting and updating later")
            sources = specextract_spectrum(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn,
                                         over_sample, False, num_cores)

        elif tel == 'erosita' or tel == 'erass':
            # This is the spectrum generation tool that is specific to eROSITA
            sources = srctool_spectrum(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn, num_cores,
                                       False, stacked_spectra)
        else:
            raise NotImplementedError("Spectrum generation functionality is not implemented "
                                      "for {t} yet!".format(t=tel))

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
        warn("Not all of these sources have been detected, you may get a poor fit.")

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
                        specs: str, lo_en: Quantity, hi_en: Quantity, par_names: str, par_values: str,
                        linking: str, freezing: str, par_fit_stat: float, lum_low_lims: str, lum_upp_lims: str,
                        lum_conf: float, redshift: float, pre_check: bool, check_par_names: str, check_par_lo_lims: str,
                        check_par_hi_lims: str, check_par_err_lims: str, norm_scale: bool,
                        which_par_nh: str = 'None', telescope: str = 'xmm') -> Tuple[str, str]:
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
        you can get from different observations of a cluster.
    :param str which_par_nh: The parameter IDs of the nH parameters values which should be zeroed for the calculation
        of unabsorbed luminosities.
    :param str telescope: The name of the telescope for which the fitting script is being generated. The default
        is 'xmm' to support the original behaviour of XGA as an XMM focused tool.
    :return: The paths to the output file and the script file.
    :rtype: Tuple[str, str]
    """
    # Read in the template file for the XSPEC script.
    with open(BASE_XSPEC_SCRIPT, 'r') as x_script:
        script = x_script.read()

    # There has to be a directory to write this xspec script to, as well as somewhere for the fit output
    #  to be stored - single telescope fits are stored in a specific XSPEC directory for that telescope
    dest_dir = OUTPUT + telescope + "/XSPEC/" + source.name + "/"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Defining where the output summary file of the fit is written - we also include the telescope in the filename,
    #  just so they can't be confused for the same fit on a different telescope data if downloaded
    out_file = dest_dir + source.name + "_" + spec_storage_key + "_" + model + "_" + telescope
    script_file = dest_dir + source.name + "_" + spec_storage_key + "_" + model + "_" + telescope + ".xcm"

    # The template is filled out here, taking everything we have generated and everything the user
    #  passed in. The result is an XSPEC script that can be run as is.
    script = script.format(xsp=XGA_EXTRACT, ab=abund_table, md=fit_method, H0=source.cosmo.H0.value,
                           q0=0., lamb0=source.cosmo.Ode0, sp=specs, lo_cut=lo_en.to("keV").value,
                           hi_cut=hi_en.to("keV").value, m=model, pn=par_names, pv=par_values,
                           lk=linking, fr=freezing, el=par_fit_stat, lll=lum_low_lims, lul=lum_upp_lims,
                           of=out_file, redshift=redshift, lel=lum_conf, check=pre_check, cps=check_par_names,
                           cpsl=check_par_lo_lims, cpsh=check_par_hi_lims, cpse=check_par_err_lims, ns=norm_scale,
                           nhmtz=which_par_nh)

    # Write out the filled-in template to its destination
    with open(script_file, 'w') as xcm:
        xcm.write(script)

    return out_file, script_file

def _spec_obj_setup(stacked_spectra: bool, tel: str, source: BaseSource, out_rad_vals: List[Quantity],
                    src_ind: float, inn_rad_vals: List[Quantity], group_spec: bool, min_counts: float,
                    min_sn: float, over_sample: float) -> str: 
    """
    Internal function that collects relevant spectrum objects per telescope. This is used in 
    each XGA xspec fitting function eg. single_temp_apec etc. 

    :param bool stacked_spectra: Whether stacked spectra (of all instruments for an ObsID) should be used for this
        XSPEC spectral fit. If a stacking procedure for a particular telescope is not supported, this function will
        instead use individual spectra for an ObsID. The default is False.
    :param str tel: The telescope to collect Spectrum objects for.
    :param source BaseSource: The source object to collect Spectrum objects for.
    :param List[Quantity] out_rad_vals: A list of outer radius quantities.
    :param int src_ind: The index of the lists of radii to be used for the source.
    :param List[Quantity] inn_rad_vals: A list of inner radius quantities.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.

    """
    # TODO This is unsustainable, but hopefully every telescope will soon (ish) have a stacking method
    if stacked_spectra and tel == 'erosita':
        search_inst = 'combined'
    else:
        search_inst = None

    if (tel == 'erosita') and (len(source.obs_ids['erosita']) > 1):
        # For erosita we need to use the spectrum generated from combined observations, so that there
        # are no duplicated events
        spec_objs = source.get_combined_spectra(out_rad_vals[src_ind], inst=search_inst, 
                                                inner_radius=inn_rad_vals[src_ind],
                                                group_spec=group_spec, min_counts=min_counts,
                                                min_sn=min_sn, telescope=tel)
    else:
        # Find matching spectrum objects associated with the current source
        spec_objs = source.get_spectra(out_rad_vals[src_ind], inner_radius=inn_rad_vals[src_ind],
                                        group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                                        over_sample=over_sample, telescope=tel, inst=search_inst)
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

    storage_key = spec_objs[0].storage_key

    return specs, storage_key

def _parse_radii_input(telescopes: List[str], radii: Union[Quantity, List[Quantity], Dict[str, Quantity], 
                       Dict[str, List[Quantity]]]):
    """
    Internal function to parse the user input of the 'radii' argument of spectral fitting methods
    into spectrum generation functions.

    :param List[str] telescopes: A list of telescopes associated to the sources.
    :param List[Quantity]/Quantity radii: A list of non-scalar quantities containing the boundary radii of the
        annuli for the sources. A single quantity containing at least three radii may be passed if one source
        is being analysed, but for multiple sources there should be a quantity (with at least three radii), PER
        source.    

    :return: A dictionary of telescope keys with values that can but input into annuluar spectrum functions.
    :rtype: Union[Dict[str, Quantity], Dict[str, List[Quantity]]]
    """
    output_dict = {}
    # If the radii is input as a Quantity, that means the user wants the same radii for all 
    # telescopes and all sources
    if isinstance(radii, Quantity):
        for telescope in telescopes:
            output_dict[telescope] = radii
    
    # If the radii is input as a List, that means the user wants the same radii for each telescope,
    # but different radii for different sources.
    elif isinstance(radii, List):
        for telescope in telescopes:
            # checking every element in the list is a Quantity
            if not all(isinstance(elem, Quantity) for elem in radii):
                raise ValueError("If 'radii' is input as a List, then every element of the List " 
                                 "must be an astropy Quantity.")
            else:
                output_dict[telescope] = radii    
    
    elif isinstance(radii, dict):
        if not all(tel in radii.keys() for tel in telescopes):
            raise KeyError("If 'radii' is input as a dictionary, this dictionary must contain a key"
                           " for each telescope associated to the source.")
        # If the radii is input as a Dictionary of lists, the user wants different radii for each 
        # source and each telescope
        if all(isinstance(value, List) for value in radii.values()):
            for list_ in radii.values():
                # checking every element in the list is a Quantity
                if not all(isinstance(elem, Quantity) for elem in list_):
                    raise ValueError("If 'radii' is input as a Dictionary of Lists, then every "
                                     "element of each List must be an astropy Quantity.")
    
            output_dict = radii

        # If the radii is input as a Dictionary of lists, the user wants the different radii for 
        # each telescope, but the same radii for each source ie. all erosita spectra have one raddi
        # all XMM spectra have another
        elif all(isinstance(value, Quantity) for value in radii.values()):
            output_dict = radii
        
        else:
            raise ValueError("The 'radii' argument must be input as either; a Quantity - which is"
                             " applied to every source in every telescope; a List of Quantities - "
                             "where every entry is applied to each source for each telescope; a "
                             "dictionary with telescope keys and Quantity keys - this is to specify"
                             " a different radii to be applied to each telescope; or a dictionary "
                             "of Lists of Quantities - which specifies radii for each source for "
                             "each telescope. In this case the radii argument has been given as a "
                             "dictionary but with the incorrect format. Please change the radii "
                             " input so that it matches one of the given options.")
    
    else:
        raise ValueError("The 'radii' argument must be input as either; a Quantity - which is"
                    " applied to every source in every telescope; a List of Quantities - "
                    "where every entry is applied to each source for each telescope; a "
                    "dictionary with telescope keys and Quantity keys - this is to specify"
                    " a different radii to be applied to each telescope; or a dictionary "
                    "of Lists of Quantities - which specifies radii for each source for "
                    "each telescope. In this case the radii argument has been given as a "
                    "dictionary but with the incorrect format. Please change the radii "
                    " input so that it matches one of the given options.")
        
    
    return output_dict