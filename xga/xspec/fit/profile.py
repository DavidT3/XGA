#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 19/08/2024, 18:06. Copyright (c) The Contributors

from typing import List, Union

import astropy.units as u
from astropy.units import Quantity

from ._common import _write_xspec_script, _check_inputs, _write_crossarf_xspec_script, _gen_fit_conf
from ..run import xspec_call
from ... import NUM_CORES
from ...exceptions import ModelNotAssociatedError, NoProductAvailableError
from ...products import Spectrum
from ...samples.base import BaseSample
from ...sas import spectrum_set, cross_arf
from ...sources import BaseSource


@xspec_call
def single_temp_apec_profile(sources: Union[BaseSource, BaseSample], radii: Union[Quantity, List[Quantity]],
                             start_temp: Quantity = Quantity(3.0, "keV"), start_met: float = 0.3,
                             lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"), freeze_nh: bool = True,
                             freeze_met: bool = True, lo_en: Quantity = Quantity(0.3, "keV"),
                             hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                             abund_table: str = "angr", fit_method: str = "leven", group_spec: bool = True,
                             min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                             num_cores: int = NUM_CORES, spectrum_checking: bool = True,
                             timeout: Quantity = Quantity(1, 'hr')):
    """
    A function that allows for the fitting of sets of annular spectra (generated from objects such as galaxy
    clusters) with an absorbed plasma emission model (tbabs*apec). This function fits the annuli completely
    independently of one another.

    If the spectrum checking step of the XSPEC fit is enabled (using the boolean flag spectrum_checking), then
    each individual spectrum available for a given source will be fitted, and if the measured temperature is less
    than or equal to 0.01keV, or greater than 20keV, or the temperature uncertainty is greater than 15keV, then
    that spectrum will be rejected and not included in the final fit. Spectrum checking also involves rejecting any
    spectra with fewer than 10 noticed channels.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param List[Quantity]/Quantity radii: A list of non-scalar quantities containing the boundary radii of the
        annuli for the sources. A single quantity containing at least three radii may be passed if one source
        is being analysed, but for multiple sources there should be a quantity (with at least three radii), PER
        source.
    :param Quantity start_temp: The initial temperature for the fit.
    :param start_met: The initial metallicity for the fit (in ZSun).
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
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
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

    # We make sure the requested sets of annular spectra have actually been generated
    spectrum_set(sources, radii, group_spec, min_counts, min_sn, over_sample, one_rmf, num_cores)
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # This should deal with instances where a single source has been passed, along with a single set of
    #  annular radii
    if isinstance(radii, Quantity):
        radii = [radii]

    # Got to try and make sure the user is passing things properly.
    if len(radii) != len(sources):
        raise ValueError("If analysing multiple sources, the radii argument must be a list containing Quantities of "
                         "annular radii. The number of annular radii sets ({ar}) does not match the number of "
                         "sources ({ns}).".format(ar=len(radii), ns=len(sources)))

    # Unfortunately, a very great deal of this function is going to be copied from the original single_temp_apec
    model = "constant*tbabs*apec"
    par_names = "{factor nH kT Abundanc Redshift norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    # Here we generate the fit configuration storage key from those arguments to this function that control the fit
    #  and how it behaves
    fit_conf = _gen_fit_conf({'start_temp': start_temp,
                              'start_met': start_met,
                              'freeze_nh': freeze_nh,
                              'freeze_met': freeze_met,
                              'lo_en': lo_en, 'hi_en': hi_en,
                              'par_fit_stat': par_fit_stat,
                              'abund_table': abund_table,
                              'fit_method': fit_method,
                              'spectrum_checking': spectrum_checking})

    script_paths = []
    outfile_paths = []
    src_inds = []

    if isinstance(sources, BaseSource):
        sources = [sources]

    deg_rad = []
    for src_ind, source in enumerate(sources):
        # Gets the set of radii for this particular source into a variable
        cur_radii = radii[src_ind]

        source: BaseSource
        # This will fetch the annular_spec, get_annular_spectra will throw an error if no matches
        #  are found, though as we have run spectrum_set that shouldn't happen
        ann_spec = source.get_annular_spectra(cur_radii, group_spec, min_counts, min_sn, over_sample)
        deg_rad.append(ann_spec.radii)

        # If source.get_annular_spectra returns a list, it means that multiple matches have been found
        if isinstance(ann_spec, list):
            # This shouldn't ever go off I don't think
            raise ValueError("Multiple annular spectra set matches have been found.")

        # We step through the annuli and make fitting scripts for them all independently
        for ann_id in range(ann_spec.num_annuli):
            # We fetch the spectrum objects for this particular annulus
            spec_objs = ann_spec.get_spectra(ann_id)
            if isinstance(spec_objs, Spectrum):
                spec_objs = [spec_objs]

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

            # Set up the TCL list that defines which parameters are linked across different spectra
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

            # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed
            #  luminosities. I am only specifying parameter 2 here (though there will likely be multiple models
            #  because there are likely multiple spectra) because I know that nH of tbabs is linked in this
            #  setup, so zeroing one will zero them all.
            nh_to_zero = "{2}"

            file_prefix = spec_objs[0].storage_key + "_ident{}_".format(spec_objs[0].set_ident) \
                          + str(spec_objs[0].annulus_ident)
            out_file, script_file = _write_xspec_script(source, file_prefix, model, abund_table, fit_method,
                                                        specs, lo_en, hi_en, par_names, par_values, linking, freezing,
                                                        par_fit_stat, lum_low_lims, lum_upp_lims, lum_conf,
                                                        source.redshift, spectrum_checking, check_list, check_lo_lims,
                                                        check_hi_lims, check_err_lims, True, nh_to_zero)

            try:
                res = ann_spec.get_results(0, model, 'kT')
            except ModelNotAssociatedError:
                script_paths.append(script_file)
                outfile_paths.append(out_file)
                src_inds.append(src_ind)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, deg_rad, timeout, model, fit_conf


@xspec_call
def single_temp_apec_crossarf_profile(sources: Union[BaseSource, BaseSample], radii: Union[Quantity, List[Quantity]],
                                      first_pass_start_pars: bool = True,
                                      default_start_temp: Quantity = Quantity(3.0, "keV"),
                                      default_start_met: float = 0.3,
                                      lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                                      freeze_nh: bool = True, freeze_met: bool = True,
                                      lo_en: Quantity = Quantity(0.3, "keV"),  hi_en: Quantity = Quantity(7.9, "keV"),
                                      par_fit_stat: float = 1., lum_conf: float = 68., abund_table: str = "angr",
                                      fit_method: str = "leven", group_spec: bool = True, min_counts: int = 5,
                                      min_sn: float = None, over_sample: float = None, one_rmf: bool = False,
                                      num_cores: int = NUM_CORES, spectrum_checking: bool = True,
                                      detmap_bin: int = 200, timeout: Quantity = Quantity(1, 'hr')):

    single_temp_apec_profile(sources, radii, default_start_temp, default_start_met, lum_en, freeze_nh, freeze_met,
                             lo_en, hi_en, par_fit_stat, lum_conf, abund_table, fit_method, group_spec, min_counts,
                             min_sn, over_sample, one_rmf, num_cores, spectrum_checking, timeout)

    # We make sure to run the XGA function that uses SAS to generate the cross-arfs necessary for
    cross_arf(sources, radii, group_spec, min_counts, min_sn, over_sample, detmap_bin=detmap_bin, num_cores=num_cores)

    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # This should deal with instances where a single source has been passed, along with a single set of
    #  annular radii
    if isinstance(radii, Quantity):
        radii = [radii]

    # Got to try and make sure the user is passing things properly.
    if len(radii) != len(sources):
        raise ValueError("If analysing multiple sources, the radii argument must be a list containing Quantities of "
                         "annular radii. The number of annular radii sets ({ar}) does not match the number of "
                         "sources ({ns}).".format(ar=len(radii), ns=len(sources)))

    model = "constant*tbabs*apec"
    par_names = "{factor nH kT Abundanc Redshift norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

    script_paths = []
    outfile_paths = []
    src_inds = []

    deg_rad = []
    for src_ind, src in enumerate(sources):

        try:
            ann_spec = src.get_annular_spectra(radii[src_ind], group_spec, min_counts, min_sn, over_sample)
        except NoProductAvailableError:
            # We make our own version of this error
            raise NoProductAvailableError("The requested AnnularSpectra cannot be located for {sn}, and this function "
                                          "will not automatically generate annular spectra.".format(sn=src.name))

        # This will now try to fetch starting values for the annuli from a previously created profile - we set the
        #  start values to None so that we can check for cases where we failed to grab the start values from a profile.
        # This won't happen if the user turns of the first_pass_start_pars option
        start_temp = None
        start_met = None
        start_norm = None
        if first_pass_start_pars:
            # This is all fairly self explanatory - fetching profiles from the source, using the annular spectrum
            #  unique identifier. If the previous annular spectra fits were a success, then a projected temperature
            #  and normalisation profile will certainly exist, the metallicity profile only exists if the metallicity
            #  was freed in the first profile fit.
            # TODO figure out how to identify the profile which belongs to the original run in the case where there
            #  have been multiple profile runs
            try:
                pt_prof = src.get_proj_temp_profiles(set_id=ann_spec.set_ident)
                start_temp = pt_prof.values.value
            except NoProductAvailableError:
                pass
            
            try:
                pn_prof = src.get_apec_norm_profiles(set_id=ann_spec.set_ident)
                start_norm = pn_prof.values.value
            except NoProductAvailableError:
                pass

            try:
                pm_prof = src.get_proj_met_profiles(set_id=ann_spec.set_ident)
                start_met = pm_prof.values.value
            except NoProductAvailableError:
                pass

        # If the start temperature, normalisation, or metallicity values have not yet been set then we revert to
        #  the default values. The default temperature and metallicity values can be set by the user, but default
        #  normalisation will currently always be one
        if start_temp is None:
            start_temp = [t.to("keV", equivalencies=u.temperature_energy()).value for t in default_start_temp]

        if start_norm is None:
            start_norm = [1] * (len(radii[src_ind]) - 1)

        if start_met is None:
            start_met = [default_start_met] * (len(radii[src_ind]) - 1)

        # The start constants will always be one
        start_con = [1]*(len(radii[src_ind])-1)
        # The nH and redshift values are known quantities that can just be set from information in the source. The
        #  nH can be allowed to vary though
        start_nh = [src.nH.to("10^22 cm^-2").value]*(len(radii[src_ind])-1)
        start_z = [src.redshift]*(len(radii[src_ind])-1)

        # This creates the string which is formatted into the XSPEC script template that is a list of lists of
        #  start values for the different annuli
        par_values = "{{{c} {nh} {t} {a} {z} {n}}}".format(c="{" + " ".join([str(c) for c in start_con]) + "}",
                                                           nh="{" + " ".join([str(nh) for nh in start_nh]) + "}",
                                                           t="{" + " ".join([str(t) for t in start_temp]) + "}",
                                                           a="{" + " ".join([str(a) for a in start_met]) + "}",
                                                           z="{" + " ".join([str(z) for z in start_z]) + "}",
                                                           n="{" + " ".join([str(n) for n in start_norm]) + "}")

        # This helps us run through the different ObsID-instrument combinations that are present in the
        #  annular spectrum we're dealing with
        oi_combos = [(o_id, inst) for o_id, insts in ann_spec.instruments.items() for inst in insts]

        # These lists will store the strings representing tcl lists of annular spectra paths, list of lists of paths
        #  to cross-arf paths for those annular spectra, and the RMF files those cross-arfs were generated with
        ann_spec_paths = []
        cross_arf_paths = []
        cross_arf_rmf_paths = []
        # Assembling spectrum list strings for the annuli, as well as cross-arf paths, and paths to the rmfs they
        #  were generated with
        for ann_id in ann_spec.annulus_ids:
            # We can just fetch the annular spectra paths from the annular spectrum object we fetched earlier,
            #  for the current source annulus defined by ann_id
            ann_spec_paths.append("{" + " ".join([ann_spec.get_spectra(ann_id, oi[0], oi[1]).path
                                                  for oi in oi_combos]) + "}")
            # Same deal with the RMFs
            cross_arf_rmf_paths.append("{" + " ".join([ann_spec.get_spectra(ann_id, oi[0], oi[1]).rmf
                                                       for oi in oi_combos]) + "}")
            # As each source annulus has a list of cross-arfs associated with it, we need another for loop here to
            #  go through them - this list will store lists of cross-arfs for each ObsID-instrument combo, as they
            #  have to be generated separately for each instrument of each observation
            cur_ann_cross_arfs = []
            for oi in oi_combos:
                rel_c_arfs = ann_spec.get_cross_arf_paths(oi[0], oi[1], ann_id)
                # This is massive overkill, as this will be setup by XGA I can almost guarantee that the keys will be
                #  integer cross-annulus identifiers, and they will be in the right order.
                cross_ids = [en for en in list(rel_c_arfs.keys())]
                cross_ids.sort()
                cur_ann_cross_arfs.append("{" + " ".join([rel_c_arfs[c_id] for c_id in cross_ids]) + "}")

            # Creates the list of lists of cross-arfs for the current source annulus
            cross_arf_paths.append("{" + " ".join(cur_ann_cross_arfs) + "}")

        # Finally, the final strings for inserting into the XSPEC script template are constructed
        ann_spec_paths = "{" + " ".join(ann_spec_paths) + "}"
        cross_arf_rmf_paths = "{" + " ".join(cross_arf_rmf_paths) + "}"
        cross_arf_paths = "{" + " ".join(cross_arf_paths) + "}"

        # Set up the TCL list that defines which parameters are frozen, dependent on user input
        freezing = "{{F {n} F {ab} T F}}".format(n='T' if freeze_nh else 'F', ab='T' if freeze_met else 'F')

        # Set up the TCL list that defines which parameters are linked across different spectra
        linking = "{F T T T T T}"

        # If the user wants the spectrum cleaning step to be run, then we have to setup some acceptable
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

        # This sets the list of parameter IDs which should be zeroed at the end to calculate unabsorbed
        #  luminosities. I am only specifying parameter 2 here (though there will likely be multiple models
        #  because there are likely multiple spectra) because I know that nH of tbabs is linked in this
        #  setup, so zeroing one will zero them all.
        nh_to_zero = "{2}"

        file_prefix = ann_spec.storage_key + "_crossarf"
        # TODO Obviously remove this
        file_prefix = "laaaaaads"
        out_file, script_file = _write_crossarf_xspec_script(src, file_prefix, model, abund_table, fit_method,
                                                             ann_spec_paths, lo_en, hi_en, par_names, par_values,
                                                             linking, freezing, par_fit_stat, lum_low_lims,
                                                             lum_upp_lims, lum_conf, src.redshift, spectrum_checking,
                                                             check_list, check_lo_lims, check_hi_lims, check_err_lims,
                                                             True, cross_arf_paths, cross_arf_rmf_paths, nh_to_zero)

        # try:
        #     # TODO REVISIT THIS WHEN IT IS ACTUALLY POSSIBLE THAT IT WORKS - RIGHT NOW THE ANNULAR SPECTRA CANNOT
        #     #  DIFFERENTIATE BETWEEN CROSS-ARF AND NORMAL PROFILE FITS
        #     # res = ann_spec.get_results(0, model, 'kT')
        #     pass
        # except ModelNotAssociatedError:
        script_paths.append(script_file)
        outfile_paths.append(out_file)
        src_inds.append(src_ind)

    run_type = "fit"
    return script_paths, outfile_paths, num_cores, run_type, src_inds, deg_rad, timeout


