#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 03/05/2020, 13:22. Copyright (c) David J Turner

from astropy.units import Quantity
import os
import sys
from shutil import rmtree
from numpy import array, full
from multiprocessing import Pool
from subprocess import Popen, PIPE
from tqdm import tqdm
from typing import List, Tuple

from xga.sources import BaseSource
from xga.products import Image, BaseProduct
from xga.utils import energy_to_channel
from xga import OUTPUT, COMPUTE_MODE, NUM_CORES


def execute_cmd(cmd: str, p_type: str, p_path: list, extra_info: dict, src: str) -> Tuple[BaseProduct, str]:
    """
    This function is called for the local compute option, and runs the passed command in a Popen shell.
    It then creates an appropriate product object, and passes it back to the callback function of the Pool
    it was called from.
    :param str cmd: SAS command to be executed on the command line.
    :param str p_type: The product type that will be produced by this command.
    :param str p_path: The final output path of the product.
    :param dict extra_info: Any extra information required to define the product object.
    :param str src: A string representation of the source object that this project is associated with.
    :return: The product object, and the string representation of the associated source object.
    :rtype: Tuple[BaseProduct, str]
    """
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()

    if p_type == "image":
        # Maybe let the user decide not to raise errors detected in stderr
        prod = Image(p_path[0], extra_info["obs_id"], extra_info["instrument"], out.decode("UTF-8"),
                     err.decode("UTF-8"), cmd, extra_info["lo_en"], extra_info["hi_en"])
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
        if isinstance(args[0], BaseSource):
            sources = [args[0]]
        elif isinstance(args[0], list):
            sources = args[0]
        else:
            raise TypeError("Please pass a source object, or a list of source objects.")
        src_lookup = [repr(src) for src in sources]

        # This is the output from whatever function this is a decorator for
        cmd_list, to_stack, to_execute, cores, p_type, paths, extra_info = sas_func(*args, **kwargs)

        all_run = []  # Combined command list for all sources
        all_type = []  # Combined expected type list for all sources
        all_path = []  # Combined expected path list for all sources
        all_extras = []  # Combined extra information list for all sources
        source_rep = []  # For repr calls of each source object, needed for assigning products to sources
        for ind in range(len(cmd_list)):
            source: BaseSource = sources[ind]
            source_rep.append(repr(source))
            if len(cmd_list[ind]) > 0:
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
        if to_execute and COMPUTE_MODE == "local" and len(all_run) > 0:
            # Will run the commands locally in a pool
            raised_errors = []
            with tqdm(total=len(all_run), desc="Generating Products") as gen, Pool(cores) as pool:
                def callback(results_in: Tuple[BaseProduct, str]):
                    """
                    Callback function for the apply_async pool method, gets called when a task finishes
                    and something is returned.
                    :param Tuple[BaseProduct, str] results_in: Results of the command call.
                    """
                    nonlocal gen  # The progress bar will need updating
                    nonlocal results  # The dictionary the command call results are added to
                    if results_in is None:
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

                for error in raised_errors:
                    raise error

        elif to_execute and COMPUTE_MODE == "sge" and len(all_run) > 0:
            # This section will run the code on an HPC that uses the Sun Grid Engine for job submission.
            raise NotImplementedError("How did you even get here?")

        elif to_execute and COMPUTE_MODE == "slurm" and len(all_run) > 0:
            # This section will run the code on an HPC that uses slurm for job submission.
            raise NotImplementedError("How did you even get here?")

        elif to_execute and len(all_run) == 0:
            # It is possible to call a wrapped SAS function and find that the products already exist.
            print("All requested products already exist")

        # Now we assign products to source objects
        for entry in results:
            # Made this lookup list earlier, using string representations of source objects.
            # Finds the ind of the list of sources that we should add this set of products to
            ind = src_lookup.index(entry)
            for product in results[entry]:
                # For each product produced for this source, we add it to the storage hierarchy
                sources[ind].update_products(product)

        # If only one source was passed, turn it back into a source object rather than a source
        # object in a list.
        if len(sources) == 1:
            sources = sources[0]
        return sources
    return wrapper


@sas_call
def evselect_image(sources: List[BaseSource], lo_en: Quantity, hi_en: Quantity,
                   add_expr: str = "", num_cores: int = NUM_CORES):
    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    # Don't do much value checking in this module, but this one is so fundamental that I will do it
    if lo_en > hi_en:
        raise ValueError("lo_en cannot be greater than hi_en")
    else:
        # Calls a useful little function that takes an astropy energy quantity to the XMM channels
        # required by SAS commands
        lo_chan = energy_to_channel(lo_en)
        hi_chan = energy_to_channel(hi_en)

    expr = " && ".join([e for e in ["expression='(PI in [{l}:{u}])".format(l=lo_chan, u=hi_chan),
                                    add_expr] if e != ""]) + "'"
    # These lists are to contain the lists of commands/paths/etc for each of the individual sources passed
    # to this function
    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for source in sources:
        cmds = []
        final_paths = []
        extra_info = []
        # Check which event lists are associated with each individual source
        for pack in source.get_products("events"):
            obs_id = pack[0]
            inst = pack[1]

            en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
            exists = [match for match in source.get_products("image", obs_id, inst) if en_id in match]
            if len(exists) == 1 and exists[0][-1].usable:
                continue

            evt_list = pack[-1]
            dest_dir = OUTPUT + "{o}/{i}_{l}-{u}_temp/".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)
            im = "{o}_{i}_{l}-{u}keVimg.fits".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)

            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)

            os.makedirs(dest_dir)
            cmds.append("cd {d};evselect table={e} imageset={i} xcolumn=X ycolumn=Y ximagebinsize=87 "
                        "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize "
                        "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 "
                        "withyranges=yes {ex}; mv * ../; cd ..; rm -r {d}".format(d=dest_dir, e=evt_list.path,
                                                                                  i=im, ex=expr))

            # This is the products final resting place, if it exists at the end of this command
            final_paths.append(os.path.join(OUTPUT, obs_id, im))
            extra_info.append({"lo_en": lo_en, "hi_en": hi_en, "obs_id": obs_id, "instrument": inst})
        sources_cmds.append(array(cmds))
        sources_paths.append(array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the SAS cmd has run
        sources_extras.append(array(extra_info))
        sources_types.append(full(sources_cmds[-1].shape, fill_value="image"))

    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras


def merge_images():
    raise NotImplementedError("Haven't quite got around to doing this bit yet")


def evselect_spec():
    raise NotImplementedError("Haven't quite got around to doing this bit yet")


def arfgen():
    raise NotImplementedError("Haven't quite got around to doing this bit yet")


def rmfgen():
    raise NotImplementedError("Haven't quite got around to doing this bit yet")


def backscale():
    raise NotImplementedError("Haven't quite got around to doing this bit yet")


def specgroup():
    raise NotImplementedError("Haven't quite got around to doing this bit yet")
