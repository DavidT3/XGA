#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/19/26, 2:56 PM. Copyright (c) The Contributors.

import os
from functools import wraps
# from multiprocessing.dummy import Pool
from multiprocessing import Pool
from random import randint
from shutil import rmtree
from subprocess import Popen, PIPE, TimeoutExpired
from typing import Tuple, Union
from warnings import warn

import fitsio
import numpy as np
import pandas as pd
from fitsio import FITS
from tqdm import tqdm

from .. import XSPEC_VERSION, OUTPUT
from ..exceptions import XSPECNotFoundError, XGADeveloperError
from ..products import AnnularSpectra
from ..samples.base import BaseSample
from ..sources import BaseSource

XGA_XSPEC_ERRS = ["No acceptable spectra are left after the minimum channel step",
                  "No acceptable spectra are left after the cleaning step"]

def execute_cmd(x_script: str, out_file: str, src: str, run_type: str, timeout: float) \
        -> Tuple[Union[str, None], str, bool, list, list]:
    """
    This function is called for the local compute option. It will run the supplied XSPEC script, then
    parse the output for errors and check that the expected output file has been created.

    :param str x_script: The path to an XSPEC script to be run.
    :param str out_file: The expected path for the output file of that XSPEC script.
    :param str src: A string representation of the source object that this fit is associated with.
    :param str run_type: A flag that tells this function what type of run this is; e.g. fit or conv_factors.
    :param float timeout: The length of time (in seconds) which the XSPEC script is allowed to run for before being
        killed.
    :return: The path to the results file (or None if unsuccessful), string repr of the source associated with
        this fit, a boolean flag indicating if the fit is considered usable, list of any errors found, and list
        of any warnings found.
    :rtype: Tuple[Union[str, None], str, bool, list, list]
    """

    # We assume the output will be usable to start with
    usable = True

    # We're going to make a temporary pfiles directory which is a) local to the XGA directory, and thus sure to
    #  be on the same filesystem (can be a performance issue for HPCs I think); and b) is unique to a particular
    #  fit process, so there shouldn't be any clashes. The temporary file name is randomly generated
    tmp_ident = str(randint(0, int(100_000_000)))
    tmp_hea_dir = os.path.join(os.path.dirname(out_file), tmp_ident, 'pfiles/')
    os.makedirs(tmp_hea_dir)

    # I add exec to the beginning to make sure that the command inherits the same process ID as the shell, which
    #  allows the timeout to kill the XSPEC run rather than the shell process. Entirely thanks to slayton on
    #   https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    cmd = 'export PFILES="{}:$HEADAS/syspfiles";'.format(tmp_hea_dir) + "exec xspec - {}".format(x_script)
    xspec_proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    # This makes sure the process is killed if it does timeout
    try:
        out, err = xspec_proc.communicate(timeout=timeout)
    except TimeoutExpired:
        xspec_proc.kill()
        out, err = xspec_proc.communicate()
        # Need to infer the name of the source to supply it in the warning
        source_name = x_script.split('/')[-1].split("_")[0]
        warn("An XSPEC fit for {} has timed out".format(source_name), stacklevel=2)
        usable = False

    # Have to decode the output and errors as they are in byte form
    out = out.decode("UTF-8").split("\n")
    err = err.decode("UTF-8").split("\n")

    # Remove the temporary directory
    rmtree(os.path.join(os.path.dirname(out_file), tmp_ident))

    # We ignore that particular string in the errors identified from stdout because if we don't just it being
    #  present in the if statement in the executed script is enough to make XGA think that the fit failed, even if
    #  that error message was never printed at all
    err_out_lines = [line.split("***Error: ")[-1] for line in out
                     if "***Error" in line and all([xxe not in line for xxe in XGA_XSPEC_ERRS])]
    err_out_line_init = [line for line in out if "You do not have an up-to-date version of Xspec.init" in line]
    warn_out_lines = [line.split("***Warning: ")[-1] for line in out if "***Warning" in line]
    err_err_lines = [line.split("***Error: ")[-1] for line in err if "***Error" in line]
    warn_err_lines = [line.split("***Warning: ")[-1] for line in err if "***Warning" in line]

    if usable and len(err_out_lines) == 0 and len(err_err_lines) == 0:
        usable = True
    else:
        usable = False

    cur_error = err_out_lines + err_err_lines + err_out_line_init
    cur_warn = warn_out_lines + warn_err_lines

    if os.path.exists(out_file + "_info.csv") and run_type == "fit" and usable:
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

        # This finds all the matching spectrum plot csvs were generated
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
        with FITS(out_file + ".fits") as res_tables:
            tab_names = [tab.get_extname() for tab in res_tables]
            if "results" not in tab_names or "spec_info" not in tab_names:
                usable = False

        # We're also going to make sure to delete the csv files now that the information in them has been
        #  consolidated into the fits file
        part_file_now = out_file.split('/')[-1]
        to_remove = [f for f in os.listdir(rel_path) if part_file_now in f and f[-5:] != '.fits' and f[-4:] != '.xcm']
        for file_tr in to_remove:
            full_path = os.path.join(rel_path, file_tr)
            os.remove(full_path)

        # I'm going to try returning the file path as that should be pickleable
        res_tables = out_file + ".fits"
    elif os.path.exists(out_file) and run_type == "conv_factors" and usable:
        res_tables = out_file
        usable = True
    elif not os.path.exists(out_file):
        usable = False
        res_tables = out_file
        cur_error.append('Results table not written.')
    else:
        # I will pass back where the results table WOULD have been if the fit was successful, but we'll clearly mark
        #  it as unusable - the reason for this is that the expected outfile can be linked back to the original
        #  fit_conf, and that is very handy to record failed fits
        res_tables = out_file
        usable = False

    return res_tables, src, usable, cur_error, cur_warn


def xspec_call(xspec_func):
    """
    This is used as a decorator for functions that produce XSPEC scripts. Depending on the
    system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the XSPEC commands will change. This supports multi-threading.

    :param function xspec_func: The XSPEC fit/simulation function to be decorated.
    :return: The decorated function.
    :rtype: function
    """

    @wraps(xspec_func)
    def wrapper(*args, **kwargs):
        # We'll stop this process in its tracks if XGA cannot find an XSPEC installation
        if XSPEC_VERSION is None:
            raise XSPECNotFoundError("There is no XSPEC installation detectable on this machine.")

        # The first argument of all of these XSPEC functions will be the source object (or a list of),
        # so rather than return them from the XSPEC model function I'll just access them like this.
        if isinstance(args[0], BaseSource):
            sources = [args[0]]
        elif isinstance(args[0], (list, BaseSample)):
            sources = args[0]
        else:
            raise TypeError("Please pass a source object, or a list of source objects.")

        # This is the output from whatever function this is a decorator for
        # First return is a list of paths of XSPEC scripts to execute, second is the expected output paths,
        #  and 3rd is the number of cores to use.
        # run_type describes the type of XSPEC script being run, for instance a fit or a fakeit run to measure
        #  countrate to luminosity conversion constants
        (script_list, paths, cores, run_type, src_inds, radii, timeout, model_name,
            fit_conf, inv_ents) = xspec_func(*args, **kwargs)
        src_lookup = {repr(src): src_ind for src_ind, src in enumerate(sources)}
        rel_src_repr = [repr(sources[src_ind]) for src_ind in src_inds]

        # We also make lookup dictionaries that link the outfile name to the fit configuration and the inventory
        #  entry - this is necessary because annular spectrum fits may produce multiple fit files (for separate
        #  annular spectra) per source, so relying on src_ind does not work at all
        fit_conf_lookup = {o_file: fit_conf[o_file_ind] for o_file_ind, o_file in enumerate(paths)}
        inv_ent_lookup = {o_file: inv_ents[o_file_ind] for o_file_ind, o_file in enumerate(paths)}

        # Make sure the timeout is converted to seconds, then just stored as a float
        timeout = timeout.to('second').value

        # This is what the returned information from the execute command gets stored in before being parceled out
        #  to source and spectrum objects
        results = {s: [] for s in src_lookup}
        if run_type == "fit":
            desc = "Running XSPEC Fits"
        elif run_type == "conv_factors":
            desc = "Running XSPEC Simulations"

        # If an error is raised during this fit, store it here and raise it at the end after all
        # results have been dealt with
        errors_all_sources = []

        if len(script_list) > 0:
            # This mode runs the XSPEC locally in a multiprocessing pool.
            with tqdm(total=len(script_list), desc=desc) as fit, Pool(cores) as pool:
                def callback(results_in):
                    """
                    Callback function for the apply_async pool method which gets
                    called when a task finishes and something is returned.
                    """
                    nonlocal fit  # The progress bar will need updating
                    nonlocal results  # The dictionary the command call results are added to

                    res_fits, rel_src, successful, cur_err_list, cur_warn_list = results_in
                    results[rel_src].append([res_fits, successful, cur_err_list, cur_warn_list])
                    fit.update(1)

                for script_ind, cur_script in enumerate(script_list):
                    cur_outfile_pth = paths[script_ind]
                    src = rel_src_repr[script_ind]
                    pool.apply_async(execute_cmd, args=(cur_script, cur_outfile_pth, src, run_type, timeout), callback=callback)
                pool.close()  # No more tasks can be added to the pool
                pool.join()  # Joins the pool, the code will only move on once the pool is empty.

        elif len(script_list) == 0:
            warn("All XSPEC operations had already been run.", stacklevel=2)

        # Now we assign the fit results to source objects
        err_list = []
        for src_repr in results:
            # Made this lookup list earlier, using string representations of source objects.
            # Finds the ind of the list of sources that we should add these results to
            ind = src_lookup[src_repr]
            cur_src = sources[ind]

            # This flag tells this method if the current set of fits are part of an annular spectra or not
            ann_fit = False
            ann_results = {}
            ann_lums = {}
            ann_obs_order = {}
            set_ident = {}
            # And this tells this method if an annular spectrum used the cross-arf option (the results are stored
            #  quite differently).
            with_cross_arf = False

            for res_set in results[src_repr]:
                # If the fit wasn't successful (which the res_set[1] boolean tells us) then
                #  we compile the error information for later.
                if not res_set[1]:
                    for err in res_set[2]:
                        errors_all_sources.append(err + " - {s}".format(s=cur_src.name))

                # This will be useful even if the fit failed, as it gives a shortcut to the fitconf
                #  information
                o_file_lu = res_set[0].replace(".fits", "")

                # Slightly fragile way of getting the telescope out
                tel = inv_ent_lookup[o_file_lu][3]

                if len(res_set) != 0 and res_set[1] and run_type == "fit":
                    with FITS(res_set[0]) as res_table:
                        global_results = res_table["RESULTS"][0]
                        model = global_results["MODEL"].strip(" ")

                        # Just define this to check if this is an annular fit or not
                        first_key = res_table["SPEC_INFO"][0]["SPEC_PATH"].strip(" ").split("/")[-1].split('ra')[-1]
                        first_key = first_key.split('_spec.fits')[0]
                        if "_ident" in first_key:
                            ann_fit = True

                        if inv_ent_lookup[o_file_lu][7] == 'ann_carf':
                            with_cross_arf = True

                        inst_lums = {}
                        obs_order = []
                        for line_ind, line in enumerate(res_table["SPEC_INFO"]):
                            # We need to check if this spectrum is a combined_spectrum or not
                            # this can be done by seeing which directory it is stored in
                            directory = line["SPEC_PATH"].strip(" ").split("/")[-2]

                            if directory == 'combined':
                                comb_spec = True
                            else:
                                comb_spec = False

                            sp_info = line["SPEC_PATH"].strip(" ").split("/")[-1].split("_")

                            # Want to derive the spectra storage key from the file name, this strips off some
                            #  unnecessary info
                            sp_key = line["SPEC_PATH"].strip(" ").split("/")[-1].split('ra')[-1].split('_spec.fits')[0]

                            # If it's not an AnnularSpectra fit then we can just fetch the spectrum from the source
                            #  the normal way
                            if not ann_fit:
                                # This adds ra back on, and removes any ident information if it is there
                                sp_key = 'ra' + sp_key

                                if not comb_spec:
                                    # Finds the appropriate matching spectrum object for the current table line
                                    spec = cur_src.get_products("spectrum", sp_info[0], sp_info[1], extra_key=sp_key,
                                                          telescope=tel)[0]
                                else:
                                    spec = cur_src.get_products("combined_spectrum", inst=sp_info[1], extra_key=sp_key)[0]
                            elif ann_fit:
                                ann_id = int(sp_key.split("_ident")[-1].split("_")[1])

                                ann_spec = cur_src.get_annular_spectra(set_id=inv_ent_lookup[o_file_lu][8], telescope=tel)

                                # comb_spec here refers to whether the obs_id is combined
                                if not comb_spec:
                                    spec = ann_spec.get_spectra(ann_id, sp_info[0], sp_info[1])

                                else:
                                    spec = ann_spec.get_spectra(ann_id, inst=sp_info[1])

                                spec = ann_spec.get_spectra(ann_id, sp_info[0], sp_info[1])
                                ann_lums.setdefault(spec.annulus_ident, {})
                                ann_obs_order.setdefault(spec.annulus_ident, [])
                                ann_obs_order[spec.annulus_ident].append([sp_info[0], sp_info[1]])

                            # Adds information from this fit to the spectrum object.
                            spec.add_fit_data(str(model), line, res_table["PLOT" + str(line_ind + 1)],
                                              fit_conf_lookup[o_file_lu])

                            # The add_fit_data method formats the luminosities nicely, so we grab them back out
                            #  to help grab the luminosity needed to pass to the source object 'add_fit_data' method
                            processed_lums = spec.get_luminosities(model, fit_conf=fit_conf_lookup[o_file_lu])

                            if not with_cross_arf and spec.instrument not in inst_lums:
                                inst_lums[spec.instrument] = processed_lums
                            elif with_cross_arf and spec.instrument not in ann_lums[spec.annulus_ident]:
                                ann_lums[spec.annulus_ident][spec.instrument] = processed_lums

                        if tel == 'xmm':
                            # Ideally the luminosity reported in the source object will be a PN lum, but its not impossible
                            #  that a PN value won't be available. - it shouldn't matter much, lums across the cameras are
                            #  consistent
                            if not with_cross_arf and "pn" in inst_lums:
                                chosen_lums = inst_lums["pn"]
                            # mos2 generally better than mos1, as mos1 has CCD damage after a certain point in its life
                            elif not with_cross_arf and "mos2" in inst_lums:
                                chosen_lums = inst_lums["mos2"]
                            elif not with_cross_arf:
                                chosen_lums = inst_lums["mos1"]
                            else:
                                # TODO THIS IS DISGUSTING
                                chosen_lums = {}
                                for cur_ann_id in ann_lums:
                                    if "pn" in ann_lums[cur_ann_id]:
                                        cur_chos_lum = ann_lums[cur_ann_id]["pn"]
                                    elif "mos2" in ann_lums:
                                        cur_chos_lum = ann_lums[cur_ann_id]["mos2"]
                                    else:
                                        cur_chos_lum = ann_lums[cur_ann_id]["mos1"]
                                    chosen_lums[cur_ann_id] = cur_chos_lum
                        else:
                            # TODO THIS ISN'T NECESSARILY THE WAY I WANT TO DO THIS
                            # TODO IF MORE MISSIONS GET CROSS-ARF SUPPORT THIS WILL NEED TO CHANGE
                            chosen_lums = processed_lums

                        # This is your bog-standard global fit, where the results are now getting stored in the source
                        #  object - we have already added the plotting information to the individual spectra
                        if not ann_fit:
                            # Push global fit results, luminosities etc. into the corresponding source object.
                            #  There isn't a 'telescope' argument here because as far as XGA is designed currently, the
                            #  'Spectrum' class holds a single spectrum, and we haven't made any frankenstein multi-telescope
                            #  stacks yet (if ever)
                            cur_src.add_fit_data(model, global_results, chosen_lums, sp_key, tel, fit_conf_lookup[o_file_lu])

                        # If this was an annular fit and the cross-arf option was not used, the different annuli
                        #  results are completely separate in terms of their outputs, as each annuli is run separately,
                        #  however the cross-arf annular fits have all annuli results output in one file
                        elif ann_fit and not with_cross_arf:
                            ann_results[tel].setdefault(tel, {})
                            ann_lums[tel].setdefault(tel, {})
                            ann_obs_order[tel].setdefault(tel, {})

                            set_ident[tel] = spec.set_ident
                            ann_results[tel][spec.annulus_ident] = global_results
                            ann_lums[tel][spec.annulus_ident] = chosen_lums
                            ann_obs_order[tel][spec.annulus_ident] = obs_order

                        # And this is the case where the annular fit was performed with cross-arfs, and all the
                        # results for all the annuli are output into one file (because the cross-arfs require
                        #  simultaneous fitting of all annuli)
                        elif ann_fit and with_cross_arf:
                            if tel not in ['xmm']:
                                raise XGADeveloperError(f"XGA is trying to assign cross-arf XSPEC fit results for a "
                                                        f"telescope ({tel}) that does not yet have cross-arf support.")

                            ann_spec: AnnularSpectra
                            # Here our main problem is untangling the parameters in the results table for this fit, as
                            #  we need to be able to assign them to our N annuli. This starts by reading out all
                            #  the column names, and figuring out where the fit parameters (which will be relevant
                            #  to a particular annulus) start.
                            col_names = np.array(global_results.dtype.names)
                            # We know that fit parameters start after the DOF entry, because that is how we designed
                            #  the output files, so we can figure out what index to split on that will let us get
                            #  fit parameters in one array and the general parameters in the other.
                            arg_split = np.argwhere(col_names == 'DOF')[0][0]
                            # We split off the columns that aren't parameters
                            not_par_names = col_names[:arg_split+1]
                            # Then we tile them, as we're going to be reading out these values repeatedly (i.e. N times
                            #  where N is the number of annuli). Strictly speaking all the goodness of fit info is not
                            #  for individual annuli like it is when we don't cross-arf-fit, but the annular spectrum
                            #  still expects there to be an entry per annulus
                            not_par_names = np.tile(not_par_names[..., None], ann_spec.num_annuli).T
                            # We select only the column names which were fit parameters, these we need to split up
                            #  by figuring out which belong to each annulus
                            col_names = col_names[arg_split+1:]
                            # Now we figure out how many parameters per annuli there are, this approach is valid
                            #  because the model setups of each annuli are going to be identical
                            par_per_ann = len(col_names) / ann_spec.num_annuli
                            if (par_per_ann % 1) != 0:
                                raise XGADeveloperError("Assigning results to annular spectrum after cross-arf fit"
                                                        " has resulted in a non-integer number of parameters per"
                                                        " annulus. This is the fault of the developers.")
                            # Now we can split the parameter names into those that belong with each
                            par_for_ann = col_names.reshape(ann_spec.num_annuli, int(par_per_ann))
                            # Now we're adding the not-fit-parameters back on to the front of each row - that way
                            #  the not-fit-parameter info will be added into each annulus' information to be passed
                            #  to the annular spectrum
                            par_for_ann = np.concatenate([not_par_names, par_for_ann], axis=1)

                            # Then we put the results in a dictionary, the way the annulur spectrum wants it
                            ann_results = {ann_id: res_table['RESULTS'][par_for_ann[ann_id]][0]
                                           for ann_id in ann_spec.annulus_ids}

                            ann_spec.add_fit_data(model, ann_results, chosen_lums, ann_obs_order,
                                                  fit_conf_lookup[o_file_lu])

                elif len(res_set) != 0 and res_set[1] and run_type == "conv_factors":
                    res_table = pd.read_csv(res_set[0], dtype={"lo_en": str, "hi_en": str})
                    # Gets the model name from the file name of the output results table
                    model = res_set[0].split("_")[-4]

                    # We can infer the storage key from the name of the results table, just makes it easier to
                    #  grab the correct spectra
                    storage_key = res_set[0].split('/')[-1].split(cur_src.name)[-1][1:].split(model)[0][:-1]

                    # Grabs the ObsID+instrument combinations from the headers of the csv. Makes sure they are unique
                    #  by going to a set (because there will be two columns for each ObsID+Instrument, rate and Lx)
                    # First two columns are skipped because they are energy limits
                    combos = list(set([c.split("_")[1] for c in res_table.columns[2:]]))
                    # Getting the spectra for each column, then assigning rates and lums
                    # TODO this could be neater and better generalised - THIS WHOLE FUNCTION/SETUP
                    #  IS AWFUL AND NEEDS BURNING DOWN AND REPLACING
                    for comb in combos:

                        # Deal with peculiarities of eROSITA first
                        if tel in ['erosita', 'erass']:
                            # If the comb string begins and ends with combined, and is longer than just "combined" (as
                            #  if comb just equaled "combined" both startswith and endswith would return True - though
                            #  that should never happen) then we set the current ObsID and instrument accordingly.
                            if ((np.char.startswith(comb, "combined") and np.char.endswith(comb, "combined"))
                                    and len(comb) != len("combined")):
                                comb_oi = "combined"
                                comb_inst = "combined"
                            # Get this paranoid check out the way, it should never be triggered (we hope)
                            elif ((np.char.startswith(comb, "combined") and np.char.endswith(comb, "combined"))
                                  and len(comb) == len("combined")):
                                raise XGADeveloperError(f"The 'comb' string used for conversion factor assignment "
                                                        f"somehow consists just of {comb}.")
                            # Now we deal with the cases of one or the other of ObsID and instrument being combined
                            else:
                                comb_oi = "combined" if np.char.startswith(comb, "combined") else comb[:6]
                                comb_inst = "combined" if np.char.endswith(comb, "combined") else comb[6:]

                            # Now onto retrieving spectra
                            if len(cur_src.obs_ids[tel]) == 1:
                                spec = cur_src.get_products("spectrum", comb_oi, comb_inst, extra_key=storage_key,
                                                      telescope=tel)[0]
                            else:
                                spec = cur_src.get_products("combined_spectrum", comb_oi, comb_inst, extra_key=storage_key,
                                                      telescope=tel)[0]
                        # Now Chandra
                        elif tel == 'chandra':
                            # We know that only ACIS is supported by XGA currently, so we can split on it to
                            #  get the correct ObsID (Chandra ObsIDs are not necessarily all the same
                            #  length)
                            search_sp_oi = comb.lower().split('acis')[0]
                            # We will build in a safety check however, to catch us if we ever
                            #  add proper HRC support to XGA
                            if 'acis' not in comb:
                                raise XGADeveloperError("No 'ACIS' string has been found in the combined "
                                                        "ObsID-inst string ({c}) in a spectrum result row; has HRC "
                                                        "support been added?".format(c=comb))

                            spec = cur_src.get_products("spectrum", search_sp_oi, 'acis', extra_key=storage_key,
                                                  telescope=tel)[0]
                        else:
                            spec = cur_src.get_products("spectrum", comb[:10], comb[10:], extra_key=storage_key,
                                            telescope=tel)[0]
                        spec.add_conv_factors(res_table["lo_en"].values, res_table["hi_en"].values,
                                              res_table["rate_{}".format(comb)].values,
                                              res_table["Lx_{}".format(comb)].values, model, tel)

                # In this case the fit has failed, and we need to arrange for the errors to be raised
                #  at the end of the loop, as well as adding entries to the Spectrum instances that
                #  record that there _was_ an attempt at a fit, and what the configuration was.
                elif len(res_set) != 0 and not res_set[1]:
                    if not ann_fit:
                        # This uses the presumptive inventory entry to grab the spectrum storage key
                        storage_key = inv_ent_lookup[o_file_lu][1]
                        cur_src.add_fit_failure(model_name, storage_key, fit_conf_lookup[o_file_lu])

                    for err in res_set[2]:
                        errors_all_sources.append(err + " - {s}".format(s=cur_src.name))

                # If the fit succeeded then we'll put it in the inventory!
                if len(script_list) != 0 and len(res_set) != 1 and run_type == 'fit':
                    inv_ent = inv_ent_lookup[o_file_lu]
                    inv_path = os.path.join(OUTPUT, tel, "XSPEC", cur_src.name, "inventory.csv")
                    with open(inv_path, 'a') as appendo:
                        inv_ent_line = ",".join(inv_ent) + "\n"
                        appendo.write(inv_ent_line)

            # TODO Restore this with correct add_fit_failure call and after taking into account that
            #  a source can now have multiple fits (and thus results) associated with it because it could
            #  have multiple telescopes associated. See issue #1517
            # TODO ALSO THE WAY OF CHECKING FOR A FIT FAILURE USED BELOW IS NO LONGER VALID
            # This records a failure if the fit timed out - checking the length of the 0th entry of results for
            #  this source is valid in this case because there will only be one result if this isn't an annular fit
            # if (len(script_list) != 0 and len(results[src_repr]) != 0 and len(results[src_repr][0]) == 1
            #         and run_type == 'fit' and not ann_fit):
            #     # This uses the presumptive inventory entry to grab the spectrum storage key
            #     storage_key = inv_ent_lookup[o_file_lu][1]
            #     s.add_fit_failure(model_name, storage_key, fit_conf_lookup[o_file_lu])

            if ann_fit and not with_cross_arf:
                for tel in ann_results:
                    # We fetch the annular spectra object that we just fitted, searching by using the set ID of
                    #  the last spectra that was opened in the loop
                    ann_spec = cur_src.get_annular_spectra(set_id=set_ident[tel], telescope=tel)

                    try:
                        ann_spec.add_fit_data(model, ann_results[tel], ann_lums[tel],
                                              ann_obs_order[tel], fit_conf_lookup[o_file_lu])

                        # The most likely reason for running XSPEC fits to a profile is to create a temp. profile
                        #  so we check whether constant*tbabs*apec (single_temp_apec function)has been run and if so
                        #  generate a Tx profile automatically
                        if model == "constant*tbabs*apec":
                            temp_prof = ann_spec.generate_profile(model, 'kT', 'keV',
                                                                  fit_conf=fit_conf_lookup[o_file_lu])
                            cur_src.update_products(temp_prof)

                            # Normalisation profiles can be useful for many things, so we generate them too
                            norm_prof = ann_spec.generate_profile(model, 'norm', 'cm^-5',
                                                                  fit_conf=fit_conf_lookup[o_file_lu])
                            cur_src.update_products(norm_prof)

                            if 'Abundanc' in ann_spec.get_results(0, 'constant*tbabs*apec',
                                                                  fit_conf=fit_conf_lookup[o_file_lu]):
                                met_prof = ann_spec.generate_profile(model, 'Abundanc', '',
                                                                     fit_conf=fit_conf_lookup[o_file_lu])
                                cur_src.update_products(met_prof)

                        else:
                            raise XGADeveloperError(f"Attempted XSPEC result assignment of an un-implemented spectral "
                                                    f"model ({model}) for AnnularSpectra fits.")
                    except ValueError:
                        warn("{src} annular spectra profile fit was not successful for the {t} "
                                "telescope.".format(src=ann_spec.src_name, t=tel), stacklevel=2)

        if len(errors_all_sources) != 0:
            warn(f"Some XSPEC fits were not successful, the errors raised are: {errors_all_sources}")

        # If only one source was passed, turn it back into a source object rather than a source
        # object in a list.
        if len(sources) == 1:
            sources = sources[0]
        return sources
    return wrapper



