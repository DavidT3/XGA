#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 22/09/2020, 13:07. Copyright (c) David J Turner

import os
import warnings
from typing import List, Union

import astropy.units as u
from astropy.units import Quantity

from xga import OUTPUT, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT
from xga.exceptions import NoProductAvailableError, ModelNotAssociatedError
from xga.samples.base import BaseSample
from xga.sources import BaseSource, ExtendedSource, GalaxyCluster, PointSource
from .run import xspec_call


@xspec_call
def single_temp_apec(sources: Union[BaseSource, BaseSample], reg_type: str,
                     start_temp: Quantity = Quantity(3.0, "keV"), start_met: float = 0.3,
                     lum_en: List[Quantity] = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                     freeze_nh: bool = True, freeze_met: bool = True,
                     link_norm: bool = False, lo_en: Quantity = Quantity(0.3, "keV"),
                     hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                     abund_table: str = "angr", fit_method: str = "leven", num_cores: int = NUM_CORES):
    """
    This is a convenience function for fitting an absorbed single temperature apec model to an object.
    It would be possible to do the exact same fit using the custom_model function, but as it will
    be a very common fit a dedicated function is in order.
    :param List[BaseSource] sources: A single source object, or a sample of sources.
    :param str reg_type: Tells the method what region's spectrum you want to use, for instance r500 or r200.
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
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    """
    allowed_bounds = ["region", "r2500", "r500", "r200", "custom"]
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    # Not allowed to use BaseSources for this, though they shouldn't have spectra anyway
    if not all([isinstance(src, (ExtendedSource, PointSource)) for src in sources]):
        raise TypeError("This convenience function can only be used with ExtendedSource and GalaxyCluster objects")
    elif not all([src.detected for src in sources]):
        warnings.warn("Not all of these sources have been detected, you will likely get a poor fit.")

    if reg_type not in allowed_bounds:
        raise ValueError("The only valid choices for reg_type are:\n {}".format(", ".join(allowed_bounds)))
    elif reg_type in ["r2500", "r500", "r200"] and not all([type(src) == GalaxyCluster for src in sources]):
        raise TypeError("You cannot use ExtendedSource classes with {}, "
                        "they have no overdensity radii.".format(reg_type))

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
        # Find matching spectrum objects associated with the current source, and checking if they are valid
        spec_objs = [match for match in source.get_products("spectrum", just_obj=False)
                     if reg_type in match and match[-1].usable]
        # Obviously we can't do a fit if there are no spectra, so throw an error if thats the case
        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for this source object, you "
                                          "need to generate them first!")

        # Turn spectra paths into TCL style list for substitution into template
        specs = "{" + " ".join([spec[-1].path for spec in spec_objs]) + "}"
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

        # Read in the template file for the XSPEC script.
        with open(BASE_XSPEC_SCRIPT, 'r') as x_script:
            script = x_script.read()

        # There has to be a directory to write this xspec script to, as well as somewhere for the fit output
        #  to be stored
        dest_dir = OUTPUT + "XSPEC/" + source.name + "/"
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # Defining where the output summary file of the fit is written
        out_file = dest_dir + source.name + "_" + reg_type + "_" + model
        script_file = dest_dir + source.name + "_" + reg_type + "_" + model + ".xcm"

        # The template is filled out here, taking everything we have generated and everything the user
        #  passed in. The result is an XSPEC script that can be run as is.
        script = script.format(xsp=XGA_EXTRACT, ab=abund_table, md=fit_method, H0=source.cosmo.H0.value,
                               q0=0., lamb0=source.cosmo.Ode0, sp=specs, lo_cut=lo_en.to("keV").value,
                               hi_cut=hi_en.to("keV").value, m=model, pn=par_names, pv=par_values,
                               lk=linking, fr=freezing, el=par_fit_stat, lll=lum_low_lims, lul=lum_upp_lims,
                               of=out_file, redshift=source.redshift, lel=lum_conf)

        # Write out the filled-in template to its destination
        with open(script_file, 'w') as xcm:
            xcm.write(script)

        # If the fit has already been performed we do not wish to perform it again
        try:
            res = source.get_results(reg_type, model)
        except ModelNotAssociatedError:
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(src_ind)
    run_type = "fit"
    return script_paths, outfile_paths, num_cores, reg_type, run_type, src_inds


def double_temp_apec():
    raise NotImplementedError("The double temperature model for clusters is under construction.")


def power_law():
    raise NotImplementedError("The power law model for point sources is under construction.")


def custom():
    raise NotImplementedError("User defined model support is a little way from being implemented.")