#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 25/08/2020, 11:49. Copyright (c) David J Turner

import os
import shutil
import warnings
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE
from typing import Tuple, Union

import fitsio
import pandas as pd
from fitsio import FITS
from tqdm import tqdm

from xga import COMPUTE_MODE
from xga.exceptions import XSPECFitError, HeasoftError
from xga.products import Spectrum
from xga.sources import BaseSource

# Got to make sure we can access command line XSPEC.
# Currently raises an error, but perhaps later on I'll relax this to a warning.
if shutil.which("xspec") is None:
    raise HeasoftError("Unable to locate an XSPEC installation.")

# TODO Make xga_extract deal with no redshift better


def execute_cmd(x_script: str, out_file: str, src: str, run_type: str) \
        -> Tuple[Union[FITS, str], str, bool, list, list]:
    """
    This function is called for the local compute option. It will run the supplied XSPEC script, then check
    parse the output for errors and check that the expected output file has been created
    :param str x_script: The path to an XSPEC script to be run.
    :param str out_file: The expected path for the output file of that XSPEC script.
    :param str src: A string representation of the source object that this fit is associated with.
    :param str run_type: A flag that tells this function what type of run this is; e.g. fit or conv_factors.
    :return: FITS object of the results, string repr of the source associated with this fit, boolean variable
    describing if this fit can be used, list of any errors found, list of any warnings found.
    :rtype: Tuple[Union[FITS, str], str, bool, list, list]
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
    if os.path.exists(out_file + "_info.csv") and run_type == "fit":
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
    elif os.path.exists(out_file) and run_type == "conv_factors":
        res_tables = out_file
        usable = True
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
        # run_type describes the type of XSPEC script being run, for instance a fit or a fakeit run to measure
        #  countrate to luminosity conversion constants
        script_list, paths, cores, reg_type, run_type = sas_func(*args, **kwargs)

        # This is what the returned information from the execute command gets stored in before being parceled out
        #  to source and spectrum objects
        results = {s: [] for s in src_lookup}
        if COMPUTE_MODE == "local" and len(script_list) > 0:
            # This mode runs the XSPEC locally in a multiprocessing pool.
            with tqdm(total=len(script_list), desc="Running XSPEC Scripts") as fit, Pool(cores) as pool:
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
                    pool.apply_async(execute_cmd, args=(s, pth, src, run_type), callback=callback)
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

            if len(res_set) != 0 and res_set[1] and run_type == "fit":
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

            elif len(res_set) != 0 and res_set[1] and run_type == "conv_factors":
                res_table = pd.read_csv(res_set[0], dtype={"lo_en": str, "hi_en": str})
                # Gets the model name from the file name of the output results table
                model = res_set[0].split("_")[-3]
                # Grabs the ObsID+instrument combinations from the headers of the csv. Makes sure they are unique
                #  by going to a set (because there will be two columns for each ObsID+Instrument, rate and Lx)
                # First two columns are skipped because they are energy limits
                combos = list(set([c.split("_")[1] for c in res_table.columns[2:]]))
                # Getting the spectra for each column, then assigning rates and lums
                for comb in combos:
                    spec: Spectrum = s.get_products("spectrum", comb[:10], comb[10:], extra_key=reg_type)[0]
                    spec.add_conv_factors(res_table["lo_en"].values, res_table["hi_en"].values,
                                          res_table["rate_{}".format(comb)].values,
                                          res_table["Lx_{}".format(comb)].values, model)

            elif len(res_set) != 0 and not res_set[1]:
                for err in res_set[2]:
                    raise XSPECFitError(err)

            if len(res_set) != 0 and run_type == "fit":
                res_set[0].close()
        # If only one source was passed, turn it back into a source object rather than a source
        # object in a list.
        if len(sources) == 1:
            sources = sources[0]
        return sources
    return wrapper



