#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 01/08/2024, 12:54. Copyright (c) The Contributors

from functools import wraps
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE
from typing import Tuple
from warnings import warn

from tqdm import tqdm

from .. import SAS_AVAIL, SAS_VERSION
from ..exceptions import SASGenerationError, SASNotFoundError
from ..products import BaseProduct, Image, ExpMap, Spectrum, PSFGrid, AnnularSpectra
from ..products.lightcurve import LightCurve
from ..samples.base import BaseSample
from ..sources import BaseSource
from ..sources.base import NullSource


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
    try:
        out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
        out = out.decode("UTF-8", errors='ignore')
        err = err.decode("UTF-8", errors='ignore')

        # This part for defining an image object used to make sure that the src wasn't a NullSource, as defining product
        #  objects is wasteful considering the purpose of a NullSource, but generating exposure maps requires a
        #  pre-existing image
        if p_type == "image":
            # Maybe let the user decide not to raise errors detected in stderr
            prod = Image(p_path[0], extra_info["obs_id"], extra_info["instrument"], out, err, cmd, extra_info["lo_en"],
                         extra_info["hi_en"])
            if "psf_corr" in extra_info and extra_info["psf_corr"]:
                prod.psf_corrected = True
                prod.psf_bins = extra_info["psf_bins"]
                prod.psf_model = extra_info["psf_model"]
                prod.psf_iterations = extra_info["psf_iter"]
                prod.psf_algorithm = extra_info["psf_algo"]
        elif p_type == "expmap":
            prod = ExpMap(p_path[0], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                          extra_info["lo_en"], extra_info["hi_en"])
        elif p_type == "ccf" and "NullSource" not in src:
            # ccf files may not be destined to spend life as product objects, but that doesn't mean
            # I can't take momentarily advantage of the error parsing I built into the product classes
            prod = BaseProduct(p_path[0], "", "", out, err, cmd)
        elif (p_type == "spectrum" or p_type == "annular spectrum set components") and "NullSource" not in src:
            prod = Spectrum(p_path[0], extra_info["rmf_path"], extra_info["arf_path"], extra_info["b_spec_path"],
                            extra_info['central_coord'], extra_info["inner_radius"], extra_info["outer_radius"],
                            extra_info["obs_id"], extra_info["instrument"], extra_info["grouped"],
                            extra_info["min_counts"], extra_info["min_sn"], extra_info["over_sample"], out, err, cmd,
                            extra_info["from_region"], extra_info["b_rmf_path"], extra_info["b_arf_path"])
        elif p_type == "psf" and "NullSource" not in src:
            prod = PSFGrid(extra_info["files"], extra_info["chunks_per_side"], extra_info["model"],
                           extra_info["x_bounds"], extra_info["y_bounds"], extra_info["obs_id"],
                           extra_info["instrument"], out, err, cmd)
        elif p_type == 'light curve' and "NullSource" not in src:
            prod = LightCurve(p_path[0],  extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                              extra_info['central_coord'], extra_info["inner_radius"], extra_info["outer_radius"],
                              extra_info["lo_en"], extra_info["hi_en"], extra_info['time_bin'], extra_info['pattern'],
                              extra_info["from_region"])
        elif p_type == "cross arfs":
            prod = BaseProduct(p_path[0], extra_info['obs_id'], extra_info['inst'], out, err, cmd, extra_info)
        elif "NullSource" in src:
            prod = None
        else:
            raise NotImplementedError("Not implemented yet")

        # An extra step is required for annular spectrum set components
        if p_type == "annular spectrum set components":
            prod.annulus_ident = extra_info["ann_ident"]
            prod.set_ident = extra_info["set_ident"]

        return prod, src

    # This is deliberately an all encompassing except - as I want to modify the message of what error may get thrown
    #  and then I will re-raise it, just it will include the source, ObsID, and instrument that caused the issue
    except Exception as err:
        # Some possible errors (I'm looking at you OSError) tend to have a number as their first argument and then
        #  the actual message as the second. In most cases though, I think just the first is populated
        if len(err.args) == 1:
            err.args = (err.args[0] + "- {s} is the associated source, the specific data used is " \
                                      "{o}-{i}.".format(s=src, o=extra_info["obs_id"], i=extra_info["instrument"]), )
        # But if there are two we do want to include them both
        else:
            err.args = (err.args[0], err.args[1] + "- {s} is the associated source, the specific data used is "
                                                   "{o}-{i}.".format(s=src, o=extra_info["obs_id"],
                                                                     i=extra_info["instrument"]))
        raise err


def sas_call(sas_func):
    """
    This is used as a decorator for functions that produce SAS command strings. Depending on the
    system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the SAS command will change. This supports both simple multi-threading and submission
    with the Sun Grid Engine.
    :return:
    """
    # This is a horrible bodge to make Pycharm not remove SAS_AVAIL and SAS_VERSION from import when it cleans
    #  up prior to committing.
    new_sas_avail = SAS_AVAIL
    new_sas_version = SAS_VERSION

    @wraps(sas_func)
    def wrapper(*args, **kwargs):
        # This has to be here to let autodoc do its noble work without falling foul of these errors
        if not new_sas_avail and new_sas_version is None:
            raise SASNotFoundError("No SAS installation has been found on this machine")
        elif not new_sas_avail:
            raise SASNotFoundError(
                "A SAS installation (v{}) has been found, but the SAS_CCFPATH environment variable is"
                " not set.".format(new_sas_version))

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
        # Making sure something is defined for this variable
        prod_type_str = ""
        if to_execute and len(all_run) > 0:
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
                    nonlocal src_lookup
                    nonlocal sources

                    if err is not None:
                        # We used a memory address laden source name representation when we adjusted the error
                        #  message in execute_cmd, so we'll replace it with an actual name here
                        # Again it matters how many arguments the error has
                        if len(err.args) == 1:
                            err_src_rep = err.args[0].split(' is the associated source')[0].split('- ')[-1].strip()
                            act_src_name = sources[src_lookup[err_src_rep]].name
                            err.args = (err.args[0].replace(err_src_rep, act_src_name),)
                        else:
                            err_src_rep = err.args[1].split(' is the associated source')[0].split('- ')[-1].strip()
                            act_src_name = sources[src_lookup[err_src_rep]].name
                            err.args = (err.args[0], err.args[1].replace(err_src_rep, act_src_name))

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

        elif to_execute and len(all_run) == 0:
            # It is possible to call a wrapped SAS function and find that the products already exist.
            # print("All requested products already exist")
            pass

        # Now we assign products to source objects
        all_to_raise = []
        # This is for the special case of generating an AnnularSpectra product
        ann_spec_comps = {k: [] for k in results}
        for entry in results:
            # Made this lookup list earlier, using string representations of source objects.
            # Finds the ind of the list of sources that we should add this set of products to
            ind = src_lookup[entry]
            to_raise = []
            for product in results[entry]:
                product: BaseProduct
                ext_info = "- {s} is the associated source, the specific data used is " \
                           "{o}-{i}.".format(s=sources[ind].name, o=product.obs_id, i=product.instrument)
                if len(product.sas_errors) == 1:
                    to_raise.append(SASGenerationError(product.sas_errors[0] + ext_info))
                elif len(product.sas_errors) > 1:
                    errs = [SASGenerationError(e + ext_info) for e in product.sas_errors]
                    to_raise += errs

                if len(product.errors) == 1:
                    to_raise.append(SASGenerationError(product.errors[0] + "-" + ext_info))
                elif len(product.errors) > 1:
                    errs = [SASGenerationError(e + "-" + ext_info) for e in product.errors]
                    to_raise += errs

                # ccfs aren't actually stored in the source product storage, but they are briefly put into
                #  BaseProducts for error parsing etc. So if the product type is None we don't store it
                if product.type is not None and product.usable and prod_type_str != "annular spectrum set components":
                    # For each product produced for this source, we add it to the storage hierarchy
                    sources[ind].update_products(product)
                elif product.type is not None and product.usable and prod_type_str == "annular spectrum set components":
                    # Really we're just re-creating the results dictionary here, but I want these products
                    #  to go through the error checking stuff like everything else does
                    ann_spec_comps[entry].append(product)
                # In case they are components of an annular spectrum but they are either none or not usable
                elif prod_type_str == "annular spectrum set components":
                    warn("An annular spectrum component ({a}) for {o}{i} has not been generated properly, contact "
                         "the development team if a SAS error is not shown. The std_err entry is:\n\n "
                         "{se}\n\n The std_out entry is:\n\n "
                         "{so}".format(a=product.storage_key, o=product.obs_id, i=product.instrument,
                                       se=product.unprocessed_stderr, so=product.unprocessed_stdout), stacklevel=2)
                # Here the generated product was a cross-arf, and needs to be added to the right annular spectrum
                #  object that already exists in our source
                elif prod_type_str == "cross arfs":
                    # OH NO WE'RE USING A PROTECTED ATTRIBUTE - but don't worry, I didn't give this a property
                    #  deliberately to hopefully discourage any user from doing anything with it
                    ei = product._extra_info
                    ann_spec = sources[ind].get_annular_spectra(set_id=ei['ann_spec_set_id'])
                    ann_spec.add_cross_arf(product, ei['obs_id'], ei['inst'], ei['src_ann_id'], ei['cross_ann_id'],
                                           ei['ann_spec_set_id'])

            if len(to_raise) != 0:
                all_to_raise.append(to_raise)

        if prod_type_str == "annular spectrum set components":
            for entry in ann_spec_comps:
                # So now we pass the list of spectra to a AnnularSpectra definition - and it will sort them out
                #  itself so the order doesn't matter
                ann_spec = AnnularSpectra(ann_spec_comps[entry])
                # Refresh the value of ind so that the correct source is used for radii conversion and so that
                #  the AnnularSpectra is added to the correct source.
                ind = src_lookup[entry]
                if sources[ind].redshift is not None:
                    # If we know the redshift we will add the radii to the annular spectra in proper distance units
                    ann_spec.proper_radii = sources[ind].convert_radius(ann_spec.radii, 'kpc')
                # And adding our exciting new set of annular spectra into the storage structure
                sources[ind].update_products(ann_spec)

        # Errors raised here should not be to do with SAS generation problems, but other purely pythonic errors
        if len(raised_errors) != 0:
            raise Exception(raised_errors)

        # And here are all the errors during SAS generation, if any
        if len(all_to_raise) != 0:
            raise SASGenerationError(all_to_raise)

        # If only one source was passed, turn it back into a source object rather than a source
        # object in a list.
        if len(sources) == 1:
            sources = sources[0]
        return sources
    return wrapper


