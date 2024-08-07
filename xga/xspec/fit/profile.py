#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/04/2023, 12:06. Copyright (c) The Contributors

from typing import List, Union

import astropy.units as u
from astropy.units import Quantity

from ._common import _write_xspec_script, _check_inputs
from ..run import xspec_call
from ... import NUM_CORES
from ...exceptions import ModelNotAssociatedError
from ...products import Spectrum
from ...samples.base import BaseSample
from ...sas import spectrum_set
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

    # We make sure the requested sets of annular spectra have actually been generated
    spectrum_set(sources, radii, group_spec, min_counts, min_sn, over_sample, one_rmf, num_cores)
    sources = _check_inputs(sources, lum_en, lo_en, hi_en, fit_method, abund_table, timeout)

    # Unfortunately, a very great deal of this function is going to be copied from the original single_temp_apec
    model = "constant*tbabs*apec"
    par_names = "{factor nH kT Abundanc Redshift norm}"
    lum_low_lims = "{" + " ".join(lum_en[:, 0].to("keV").value.astype(str)) + "}"
    lum_upp_lims = "{" + " ".join(lum_en[:, 1].to("keV").value.astype(str)) + "}"

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
    return script_paths, outfile_paths, num_cores, run_type, src_inds, deg_rad, timeout





