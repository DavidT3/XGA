#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 11/11/2020, 09:42. Copyright (c) David J Turner

import os
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE
from typing import Tuple

from tqdm import tqdm

from .. import COMPUTE_MODE
from ..exceptions import SASNotFoundError, SASGenerationError
from ..products import BaseProduct, Image, ExpMap, Spectrum, PSFGrid
from ..samples.base import BaseSample
from ..sources import BaseSource
from ..sources.base import NullSource

if "SAS_DIR" not in os.environ:
    raise SASNotFoundError("SAS_DIR environment variable is not set, "
                           "unable to verify SAS is present on system")
else:
    # This way, the user can just import the SAS_VERSION from this utils code
    sas_out, sas_err = Popen("sas --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
    SAS_VERSION = sas_out.decode("UTF-8").strip("]\n").split('-')[-1]


def execute_cmd(cmd: str, p_type: str, p_path: list, extra_info: dict, src: str) -> Tuple[BaseProduct, str]:
    """
    This function is called for the local compute option, and runs the passed command in a Popen shell.
    It then creates an appropriate product object, and passes it back to the callback function of the Pool
    it was called from.
    :param str cmd: SAS command to be executed on the command line.
    :param str p_type: The product type that will be produced by this command.
    :param str p_path: The final output path of the product.
    :param dict extra_info: Any extra information required to define the product object.
    :param str src: A string representation of the source object that this product is associated with.
    :return: The product object, and the string representation of the associated source object.
    :rtype: Tuple[BaseProduct, str]
    """
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    out = out.decode("UTF-8")
    err = err.decode("UTF-8")

    # The if statements also check that the source isn't a NullSource - if it is we don't want to define
    #  a product object because all NullSources are for is generating files in bulk.
    if p_type == "image" and "NullSource" not in src:
        # Maybe let the user decide not to raise errors detected in stderr
        prod = Image(p_path[0], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                     extra_info["lo_en"], extra_info["hi_en"])
        if "psf_corr" in extra_info and extra_info["psf_corr"]:
            prod.psf_corrected = True
            prod.psf_bins = extra_info["psf_bins"]
            prod.psf_model = extra_info["psf_model"]
            prod.psf_iterations = extra_info["psf_iter"]
            prod.psf_algorithm = extra_info["psf_algo"]
    elif p_type == "expmap" and "NullSource" not in src:
        prod = ExpMap(p_path[0], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                      extra_info["lo_en"], extra_info["hi_en"])
    elif p_type == "ccf" and "NullSource" not in src:
        # ccf files may not be destined to spend life as product objects, but that doesn't mean
        # I can't take momentarily advantage of the error parsing I built into the product classes
        prod = BaseProduct(p_path[0], "", "", out, err, cmd)
    elif p_type == "spectrum" and "NullSource" not in src:
        prod = Spectrum(p_path[0], extra_info["rmf_path"], extra_info["arf_path"], extra_info["b_spec_path"],
                        extra_info["b_rmf_path"], extra_info["b_arf_path"], extra_info["reg_type"],
                        extra_info["obs_id"], extra_info["instrument"], out, err, cmd)
    elif p_type == "psf" and "NullSource" not in src:
        prod = PSFGrid(extra_info["files"], extra_info["chunks_per_side"], extra_info["model"],
                       extra_info["x_bounds"], extra_info["y_bounds"], extra_info["obs_id"],
                       extra_info["instrument"], out, err, cmd)
    elif "NullSource" in src:
        prod = None
    else:
        raise NotImplementedError("Not implemented yet")

    return prod, src


def sas_call(sas_func):
    """
    This is used as a decorator for functions that produce SAS command strings. Depending on the
    system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the SAS command will change. This supports both simple multi-threading and submission
    with the Sun Grid Engine.
    :return:
    """
    def wrapper(*args, **kwargs):
        # The first argument of all of these SAS functions will be the source object (or a list of),
        # so rather than return them from the sas function I'll just access them like this.
        if isinstance(args[0], (BaseSource, NullSource)):
            sources = [args[0]]
        elif isinstance(args[0], (BaseSample, list)):
            sources = args[0]
        else:
            raise TypeError("Please pass a source, NullSource, or sample object.")

        # This is the output from whatever function this is a decorator for
        cmd_list, to_stack, to_execute, cores, p_type, paths, extra_info, disable = sas_func(*args, **kwargs)

        src_lookup = {}
        all_run = []  # Combined command list for all sources
        all_type = []  # Combined expected type list for all sources
        all_path = []  # Combined expected path list for all sources
        all_extras = []  # Combined extra information list for all sources
        source_rep = []  # For repr calls of each source object, needed for assigning products to sources
        for ind in range(len(cmd_list)):
            source = sources[ind]
            if len(cmd_list[ind]) > 0:
                src_lookup[repr(source)] = ind
                # If there are commands to add to a source queue, then do it
                source.update_queue(cmd_list[ind], p_type[ind], paths[ind], extra_info[ind], to_stack)

            # If we do want to execute the commands this time round, we read them out for all sources
            # and add them to these master lists
            if to_execute:
                to_run, expected_type, expected_path, extras = source.get_queue()
                all_run += to_run
                all_type += expected_type
                all_path += expected_path
                all_extras += extras
                source_rep += [repr(source)] * len(to_run)

        # This is what the returned products get stored in before they're assigned to sources
        results = {s: [] for s in src_lookup}
        # Any errors raised shouldn't be SAS, as they are stored within the product object.
        raised_errors = []
        if to_execute and COMPUTE_MODE == "local" and len(all_run) > 0:
            # Will run the commands locally in a pool
            prod_type_str = ", ".join(set(all_type))
            with tqdm(total=len(all_run), desc="Generating products of type(s) " + prod_type_str,
                      disable=disable) as gen, Pool(cores) as pool:
                def callback(results_in: Tuple[BaseProduct, str]):
                    """
                    Callback function for the apply_async pool method, gets called when a task finishes
                    and something is returned.
                    :param Tuple[BaseProduct, str] results_in: Results of the command call.
                    """
                    nonlocal gen  # The progress bar will need updating
                    nonlocal results  # The dictionary the command call results are added to
                    if results_in[0] is None:
                        gen.update(1)
                        return
                    else:
                        prod_obj, rel_src = results_in
                        results[rel_src].append(prod_obj)
                        gen.update(1)

                def err_callback(err):
                    """
                    The callback function for errors that occur inside a task running in the pool.
                    :param err: An error that occurred inside a task.
                    """
                    nonlocal raised_errors
                    nonlocal gen

                    if err is not None:
                        # Rather than throwing an error straight away I append them all to a list for later.
                        raised_errors.append(err)
                    gen.update(1)

                for cmd_ind, cmd in enumerate(all_run):
                    # These are just the relevant entries in all these lists for the current command
                    # Just defined like this to save on line length for apply_async call.
                    exp_type = all_type[cmd_ind]
                    exp_path = all_path[cmd_ind]
                    ext = all_extras[cmd_ind]
                    src = source_rep[cmd_ind]
                    pool.apply_async(execute_cmd, args=(str(cmd), str(exp_type), exp_path, ext, src),
                                     error_callback=err_callback, callback=callback)
                pool.close()  # No more tasks can be added to the pool
                pool.join()  # Joins the pool, the code will only move on once the pool is empty.

        elif to_execute and COMPUTE_MODE == "sge" and len(all_run) > 0:
            # This section will run the code on an HPC that uses the Sun Grid Engine for job submission.
            raise NotImplementedError("How did you even get here?")

        elif to_execute and COMPUTE_MODE == "slurm" and len(all_run) > 0:
            # This section will run the code on an HPC that uses slurm for job submission.
            raise NotImplementedError("How did you even get here?")

        elif to_execute and len(all_run) == 0:
            # It is possible to call a wrapped SAS function and find that the products already exist.
            # print("All requested products already exist")
            pass

        # Now we assign products to source objects
        for entry in results:
            # Made this lookup list earlier, using string representations of source objects.
            # Finds the ind of the list of sources that we should add this set of products to
            ind = src_lookup[entry]
            for product in results[entry]:
                product: BaseProduct
                ext_info = " {s} is the associated source, the specific data used is " \
                           "{o}-{i}.".format(s=sources[ind].name, o=product.obs_id, i=product.instrument)
                if len(product.sas_errors) == 1:
                    raise SASGenerationError(product.sas_errors[0] + ext_info)
                elif len(product.sas_errors) > 1:
                    errs = [SASGenerationError(e + ext_info) for e in product.sas_errors]
                    raise Exception(errs)
                # This is an elif because I designate SAS errors to be 'more important', and the likelihood of there
                #  being secondary errors AS WELL as SAS errors seems very low
                elif len(product.errors) == 1:
                    raise SASGenerationError(product.errors[0] + "-" + ext_info)
                elif len(product.errors) > 1:
                    errs = [SASGenerationError(e + "-" + ext_info) for e in product.errors]
                    raise Exception(errs)

                # ccfs aren't actually stored in the source product storage, but they are briefly put into
                #  BaseProducts for error parsing etc. So if the product type is None we don't store it
                if product.type is not None:
                    # For each product produced for this source, we add it to the storage hierarchy
                    sources[ind].update_products(product)

        # Errors raised here should not be to do with SAS generation problems, but other purely pythonic errors
        for error in raised_errors:
            raise error

        # If only one source was passed, turn it back into a source object rather than a source
        # object in a list.
        if len(sources) == 1:
            sources = sources[0]
        return sources
    return wrapper



