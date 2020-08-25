#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 25/08/2020, 13:04. Copyright (c) David J Turner

import os
from typing import List, Union

import astropy.units as u
from astropy.units import Quantity

from xga import OUTPUT, NUM_CORES, COUNTRATE_CONV_SCRIPT
from xga.exceptions import NoProductAvailableError
from xga.sources import BaseSource, GalaxyCluster
from .run import xspec_call


@xspec_call
def cluster_cr_conv(sources: Union[List[BaseSource], BaseSource], reg_type: str, sim_temp: Quantity,
                    sim_met: float = 0.3, conv_en: List[Quantity] = Quantity([[0.5, 2.0]], "keV"),
                    abund_table: str = "angr", num_cores: int = NUM_CORES):
    # Again these checking stages are basically copied from another function, I'm feeling lazy
    allowed_bounds = ["region", "r2500", "r500", "r200", "custom"]
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

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

    model = "tbabs*apec"
    par_names = "{nH kT Abundanc Redshift norm}"
    convert_low_lims = "{" + " ".join(conv_en[:, 0].to("keV").value.astype(str)) + "}"
    convert_upp_lims = "{" + " ".join(conv_en[:, 1].to("keV").value.astype(str)) + "}"

    script_paths = []
    outfile_paths = []
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for s_ind, source in enumerate(sources):
        # This function can take a single temperature to simulate at, or a list of them (one for each source).
        if sim_temp.isscalar:
            the_temp = sim_temp
        else:
            the_temp = sim_temp[s_ind]

        total_obs_inst = source.num_pn_obs + source.num_mos1_obs + source.num_mos2_obs
        # Find matching spectrum objects associated with the current source, and checking if they are valid
        spec_objs = [match for match in source.get_products("spectrum", just_obj=False, extra_key=reg_type)
                     if match[-1].usable]
        # Obviously we can't do a fit if there are no spectra, so throw an error if that's the case
        if len(spec_objs) == 0:
            raise NoProductAvailableError("There are no matching spectra for this source object, you "
                                          "need to generate them first!")
        elif len(spec_objs) != total_obs_inst:
            raise NoProductAvailableError("The number of matching spectra is not equal to the number of "
                                          "instrument/observation combinations.")

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

        script = script.format(ab=abund_table, H0=source.cosmo.H0.value, q0=0., lamb0=source.cosmo.Ode0,
                               rmf=rmf_paths, arf=arf_paths, obs=obs, inst=inst, m=model, pn=par_names,
                               pv=par_values, lll=convert_low_lims, lul=convert_upp_lims,
                               redshift=source.redshift, of=out_file)

        # Write out the filled-in template to its destination
        with open(script_file, 'w') as xcm:
            xcm.write(script)

        script_paths.append(script_file)
        outfile_paths.append(out_file)

    run_type = "conv_factors"
    return script_paths, outfile_paths, num_cores, reg_type, run_type






