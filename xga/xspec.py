#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 01/07/2020, 10:45. Copyright (c) David J Turner

import os
import warnings
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE
from typing import List, Tuple

import astropy.units as u
import fitsio
import pandas as pd
from astropy.units import Quantity
from fitsio import FITS
from tqdm import tqdm

pd.set_option('display.max_columns', 500)

from xga import OUTPUT, COMPUTE_MODE, NUM_CORES, XGA_EXTRACT, BASE_XSPEC_SCRIPT
from xga.exceptions import NoProductAvailableError, XSPECFitError, ModelNotAssociatedError
from xga.sources import BaseSource, ExtendedSource, GalaxyCluster, PointSource


# TODO Make xga_extract deal with no redshift better


def execute_cmd(x_script: str, out_file: str, src: str) -> Tuple[FITS, str, bool, list, list]:
    """
    This function is called for the local compute option. It will run the supplied XSPEC script, then check
    parse the output for errors and check that the expected output file has been created
    :param str x_script: The path to an XSPEC script to be run.
    :param str out_file: The expected path for the output file of that XSPEC script.
    :param str src: A string representation of the source object that this fit is associated with.
    :return: FITS object of the results, string repr of the source associated with this fit, boolean variable
    describing if this fit can be used, list of any errors found, list of any warnings found.
    :rtype: Tuple[FITS, str, bool, list, list]
    """
    cmd = "xspec - {}".format(x_script)
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    out = out.decode("UTF-8").split("\n")
    err = err.decode("UTF-8").split("\n")

    err_out_lines = [line.split("***Error: ")[-1] for line in out if "***Error" in line]
    warn_out_lines = [line.split("***Warning: ")[-1] for line in out if "***Warning" in line]
    err_err_lines = [line.split("***Error: ")[-1] for line in err if "***Error" in line]
    warn_err_lines = [line.split("***Warning: ")[-1] for line in err if "***Warning" in line]

    if len(err_out_lines) == 0 and len(err_err_lines) == 0:
        usable = True
    else:
        usable = False

    error = err_out_lines + err_err_lines
    warn = warn_out_lines + warn_err_lines
    if os.path.exists(out_file + "_info.csv"):
        # The original version of the xga_output.tcl script output everything as one nice neat fits file
        #  but life is full of extraordinary inconveniences and for some reason it didn't work if called from
        #  a Jupyter Notebook. So now I'm going to smoosh all the csv outputs into one fits.
        results = pd.read_csv(out_file + "_results.csv", header="infer")
        # This is the csv with the fit results in, creates new fits file and adds in
        fitsio.write(out_file + ".fits", results.to_records(index=False), extname="results", clobber=True)
        del results

        # The information about individual spectra, exposure times, luminosities etc.
        spec_info = pd.read_csv(out_file + "_info.csv", header="infer")
        # Gets added into the existing file
        fitsio.write(out_file + ".fits", spec_info.to_records(index=False), extname="spec_info")
        del spec_info

        # This finds all of the matching spectrum plot csvs were generated
        rel_path = "/".join(out_file.split('/')[0:-1])
        # This is mostly just used to find how many files there are
        spec_tabs = [rel_path + "/" + sp for sp in os.listdir(rel_path)
                     if "{}_spec".format(out_file) in rel_path + "/" + sp]
        for spec_i in range(1, len(spec_tabs)+1):
            # Loop through and redefine names like this to ensure they're in the right order
            spec_plot = pd.read_csv(out_file + "_spec{}.csv".format(spec_i), header="infer")
            # Adds all the plot tables into the existing fits file in the right order
            fitsio.write(out_file + ".fits", spec_plot.to_records(index=False), extname="plot{}".format(spec_i))
            del spec_plot

        # This reads in the fits we just made
        res_tables = FITS(out_file + ".fits")
        tab_names = [tab.get_extname() for tab in res_tables]
        if "results" not in tab_names or "spec_info" not in tab_names:
            usable = False
    else:
        res_tables = None
        usable = False

    return res_tables, src, usable, error, warn


def xspec_call(sas_func):
    """
    This is used as a decorator for functions that produce XSPEC scripts. Depending on the
    system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the SAS command will change. This supports both simple multi-threading and submission
    with the Sun Grid Engine.
    :return:
    """
    def wrapper(*args, **kwargs):
        # The first argument of all of these XSPEC functions will be the source object (or a list of),
        # so rather than return them from the XSPEC model function I'll just access them like this.
        if isinstance(args[0], BaseSource):
            sources = [args[0]]
        elif isinstance(args[0], list):
            sources = args[0]
        else:
            raise TypeError("Please pass a source object, or a list of source objects.")
        src_lookup = [repr(src) for src in sources]

        # This is the output from whatever function this is a decorator for
        # First return is a list of paths of XSPEC scripts to execute, second is the expected output paths,
        #  and 3rd is the number of cores to use.
        script_list, paths, cores, reg_type = sas_func(*args, **kwargs)

        # This is what the returned information from the execute command gets stored in before being parceled out
        #  to source and spectrum objects
        results = {s: [] for s in src_lookup}
        if COMPUTE_MODE == "local" and len(script_list) > 0:
            # This mode runs the XSPEC locally in a multiprocessing pool.
            with tqdm(total=len(script_list), desc="Running XSPEC Models Fits") as fit, Pool(cores) as pool:
                def callback(results_in):
                    """
                    Callback function for the apply_async pool method, gets called when a task finishes
                    and something is returned.
                    """
                    nonlocal fit  # The progress bar will need updating
                    nonlocal results  # The dictionary the command call results are added to
                    if results_in[0] is None:
                        fit.update(1)
                        return
                    else:
                        res_fits, rel_src, successful, err_list, warn_list = results_in
                        results[rel_src] = [res_fits, successful, err_list, warn_list]
                        fit.update(1)

                for s_ind, s in enumerate(script_list):
                    pth = paths[s_ind]
                    src = src_lookup[s_ind]
                    pool.apply_async(execute_cmd, args=(s, pth, src), callback=callback)
                pool.close()  # No more tasks can be added to the pool
                pool.join()  # Joins the pool, the code will only move on once the pool is empty.

        elif COMPUTE_MODE == "sge" and len(script_list) > 0:
            # This section will run the code on an HPC that uses the Sun Grid Engine for job submission.
            raise NotImplementedError("How did you even get here?")

        elif COMPUTE_MODE == "slurm" and len(script_list) > 0:
            # This section will run the code on an HPC that uses slurm for job submission.
            raise NotImplementedError("How did you even get here?")

        elif len(script_list) == 0:
            warnings.warn("All requested XSPEC fits had already been run.")

        # Now we assign the fit results to source objects
        for entry in results:
            # Made this lookup list earlier, using string representations of source objects.
            # Finds the ind of the list of sources that we should add these results to
            ind = src_lookup.index(entry)
            s = sources[ind]
            # Is this fit usable?
            res_set = results[entry]

            if len(res_set) != 0 and res_set[1]:
                global_results = res_set[0]["RESULTS"][0]
                model = global_results["MODEL"].strip(" ")

                inst_lums = {}
                for line_ind, line in enumerate(res_set[0]["SPEC_INFO"]):
                    sp_info = line["SPEC_PATH"].strip(" ").split("/")[-1].split("_")
                    # Finds the appropriate matching spectrum object for the current table line
                    spec = [match for match in s.get_products("spectrum", sp_info[0], sp_info[1], just_obj=False)
                            if reg_type in match and match[-1].usable][0][-1]

                    # Adds information from this fit to the spectrum object.
                    spec.add_fit_data(str(model), line, res_set[0]["PLOT"+str(line_ind+1)])
                    s.update_products(spec)  # Adds the updated spectrum object back into the source

                    # The add_fit_data method formats the luminosities nicely, so we grab them back out
                    #  to help grab the luminosity needed to pass to the source object 'add_fit_data' method
                    processed_lums = spec.get_luminosities(model)
                    if spec.instrument not in inst_lums:
                        inst_lums[spec.instrument] = processed_lums

                # Ideally the luminosity reported in the source object will be a PN lum, but its not impossible
                #  that a PN value won't be available. - it shouldn't matter much, lums across the cameras are
                #  consistent
                if "pn" in inst_lums:
                    chosen_lums = inst_lums["pn"]
                # mos2 generally better than mos1, as mos1 has CCD damage after a certain point in its life
                elif "mos2" in inst_lums:
                    chosen_lums = inst_lums["mos2"]
                else:
                    chosen_lums = inst_lums["mos1"]

                # Push global fit results, luminosities etc. into the corresponding source object.
                s.add_fit_data(model, reg_type, global_results, chosen_lums)

            elif len(res_set) != 0 and not res_set[1]:
                for err in res_set[2]:
                    raise XSPECFitError(err)

            if len(res_set) != 0:
                res_set[0].close()
        # If only one source was passed, turn it back into a source object rather than a source
        # object in a list.
        if len(sources) == 1:
            sources = sources[0]
        return sources
    return wrapper


@xspec_call
def single_temp_apec(sources: List[BaseSource], reg_type: str, start_temp: Quantity = Quantity(3.0, "keV"),
                     start_met: float = 0.3, lum_en: List[Quantity] = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                     freeze_nh: bool = True, freeze_met: bool = True,
                     link_norm: bool = False, lo_en: Quantity = Quantity(0.3, "keV"),
                     hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                     abund_table: str = "angr", fit_method: str = "leven", num_cores: int = NUM_CORES):
    """
    This is a convenience function for fitting an absorbed single temperature apec model to an object.
    It would be possible to do the exact same fit using the custom_model function, but as it will
    be a very common fit a dedicated function is in order.
    :param List[BaseSource] sources: A single source object, or a list of source objects.
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
    # This function supports passing multiple sources, so we have to setup a script for all of them.
    for source in sources:
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
    return script_paths, outfile_paths, num_cores, reg_type


def double_temp_apec():
    raise NotImplementedError("The double temperature model for clusters is under construction.")


def power_law():
    raise NotImplementedError("The power law model for point sources is under construction.")


def custom():
    raise NotImplementedError("User defined model support is a little way from being implemented.")












