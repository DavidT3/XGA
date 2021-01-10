#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 09/01/2021, 19:42. Copyright (c) David J Turner

import os
from typing import List, Union

import astropy.units as u
from astropy.units import Quantity

from .run import xspec_call
from .. import OUTPUT, NUM_CORES, COUNTRATE_CONV_SCRIPT
from ..exceptions import NoProductAvailableError, ModelNotAssociatedError, ParameterNotAssociatedError
from ..samples.extended import ClusterSample
from ..sources import BaseSource, GalaxyCluster
from ..utils import ABUND_TABLES


# TODO Rename this function to something more useful
@xspec_call
def cluster_cr_conv(sources: Union[GalaxyCluster, ClusterSample], reg_type: str, sim_temp: Quantity,
                    sim_met: Union[float, List] = 0.3, conv_en: Quantity = Quantity([[0.5, 2.0]], "keV"),
                    abund_table: str = "angr", num_cores: int = NUM_CORES):
    """
    This function uses the xspec fakeit tool to calculate conversion factors between count rate and
    luminosity for ARFs and RMFs associated with spectra in the given sources. Once complete the conversion
    factors are stored within the relevant XGA spectrum object.

    :param GalaxyCluster sources: The GalaxyCluster objects to calculate conversion factors for.
    :param str reg_type: The region type of the spectra to base the conversion factors on.
    :param Quantity sim_temp: The temperature(s) to use for the apec model.
    :param float/List sim_met: The metallicity(s) (in solar met) to use for the apec model.
    :param Quantity conv_en: The energy limit pairs to calculate conversion factors for.
    :param str abund_table: The name of the XSPEC abundance table to use.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    """
    # Again these checking stages are basically copied from another function, I'm feeling lazy
    allowed_bounds = ["region", "r2500", "r500", "r200", "custom", "point"]
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    if abund_table not in ABUND_TABLES:
        ab_list = ", ".join(ABUND_TABLES)
        raise ValueError("{0} is not in the accepted list of abundance tables; {1}".format(abund_table, ab_list))

    if reg_type not in allowed_bounds:
        raise ValueError("The only valid choices for reg_type are:\n {}".format(", ".join(allowed_bounds)))
    elif reg_type in ["r2500", "r500", "r200"] and not all([type(src) == GalaxyCluster for src in sources]):
        raise TypeError("You cannot use ExtendedSource classes with {}, "
                        "they have no overdensity radii.".format(reg_type))

    # Checks that the luminosity energy bands are pairs of values
    if conv_en.shape[1] != 2:
        raise ValueError("Luminosity energy bands should be supplied in pairs, defined "
                         "like Quantity([[0.5, 2.0], [2.0, 10.0]], 'keV')")
    # Are the lower limits smaller than the upper limits? - Obviously they should be so I check
    elif not all([conv_en[pair_ind, 0] < conv_en[pair_ind, 1] for pair_ind in range(0, conv_en.shape[0])]):
        raise ValueError("Luminosity energy band first entries must be smaller than second entries.")

    # Check that the correct number of temperatures are supplied
    if not sim_temp.isscalar and len(sim_temp) != len(sources):
        raise ValueError("The sim_temp variable must either be scalar or have the "
                         "same number of entries as there are sources.")
    elif not isinstance(sim_met, float) and len(sim_met) != len(sources):
        raise ValueError("The sim_met variable must either be a float or have the "
                         "same number of entries as there are sources.")

    # Hard coding the model currently, tbabs*apec is a good simple descriptor of a cluster
    model = "tbabs*apec"
    # These are the parameter names for this model
    par_names = "{nH kT Abundanc Redshift norm}"
    convert_low_lims = "{" + " ".join(conv_en[:, 0].to("keV").value.astype(str)) + "}"
    convert_upp_lims = "{" + " ".join(conv_en[:, 1].to("keV").value.astype(str)) + "}"

    script_paths = []
    outfile_paths = []
    src_inds = []
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for s_ind, source in enumerate(sources):
        # This function can take a single temperature to simulate at, or a list of them (one for each source).
        if sim_temp.isscalar:
            the_temp = sim_temp
        else:
            the_temp = sim_temp[s_ind]
        # Equivalent of above but for metallicities
        if isinstance(sim_met, float):
            the_met = sim_met
        else:
            the_met = sim_met[s_ind]

        total_obs_inst = source.num_pn_obs + source.num_mos1_obs + source.num_mos2_obs
        # Find matching spectrum objects associated with the current source, and checking if they are valid
        spec_objs = [match for match in source.get_products("spectrum", just_obj=False, extra_key=reg_type)
                     if match[-1].usable]
        # Obviously we can't do a fit if there are no spectra, so throw an error if that's the case
        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for {} object, you "
                                          "need to generate them first!".format(source.name))
        elif len(spec_objs) != total_obs_inst:
            raise NoProductAvailableError("The number of matching spectra ({0}) is not equal to the number of "
                                          "instrument/observation combinations ({1}) for {2}.".format(len(spec_objs),
                                                                                                      total_obs_inst,
                                                                                                      source.name))

        # Turn RMF and ARF paths into TCL style list for substitution into template
        rmf_paths = "{" + " ".join([spec[-1].rmf for spec in spec_objs]) + "}"
        arf_paths = "{" + " ".join([spec[-1].arf for spec in spec_objs]) + "}"
        # Put in the ObsIDs and Instruments, to help name columns easily
        obs = "{" + " ".join([spec[-1].obs_id for spec in spec_objs]) + "}"
        inst = "{" + " ".join([spec[-1].instrument for spec in spec_objs]) + "}"

        # For this model, we have to know the redshift of the source.
        if source.redshift is None:
            raise ValueError("You cannot supply a source without a redshift to this model.")

        t = the_temp.to("keV", equivalencies=u.temperature_energy()).value
        # Another TCL list, this time of the parameter start values for this model.
        par_values = "{{{0} {1} {2} {3} {4}}}".format(source.nH.to("10^22 cm^-2").value, t,
                                                      sim_met, source.redshift, 1.)

        with open(COUNTRATE_CONV_SCRIPT, 'r') as c_script:
            script = c_script.read()

        dest_dir = OUTPUT + "XSPEC/" + source.name + "/"
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        out_file = dest_dir + source.name + "_" + reg_type + "_" + model + "_conv_factors.csv"
        script_file = dest_dir + source.name + "_" + reg_type + "_" + model + "_conv_factors" + ".xcm"

        # Populates the fakeit conversion factor template script
        script = script.format(ab=abund_table, H0=source.cosmo.H0.value, q0=0., lamb0=source.cosmo.Ode0,
                               rmf=rmf_paths, arf=arf_paths, obs=obs, inst=inst, m=model, pn=par_names,
                               pv=par_values, lll=convert_low_lims, lul=convert_upp_lims,
                               redshift=source.redshift, of=out_file)

        # Write out the filled-in template to its destination
        with open(script_file, 'w') as xcm:
            xcm.write(script)

        try:
            # Checks through the spectrum objects we retrieved earlier, and the energy limits,
            #  to look for conversion factor results, if they exist they aren't run again, otherwise an error
            #  is triggered and the scripts get added to the pile to run.
            res = [s[-1].get_conv_factor(e_pair[0], e_pair[1], "tbabs*apec") for e_pair in conv_en
                   for s in spec_objs]
        except (ModelNotAssociatedError, ParameterNotAssociatedError):
            script_paths.append(script_file)
            outfile_paths.append(out_file)
            src_inds.append(s_ind)

    # New feature of XSPEC interface, tells the xspec_call decorator what type of output from the script
    #  to expect
    run_type = "conv_factors"
    return script_paths, outfile_paths, num_cores, reg_type, run_type, src_inds






