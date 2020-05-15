#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 03/05/2020, 13:22. Copyright (c) David J Turner

from astropy.units import Quantity
import astropy.units as u
import os
import sys
from shutil import rmtree
from numpy import array, full
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE
from tqdm import tqdm
from typing import List, Tuple

from xga.sources import BaseSource
from xga.products import BaseProduct, Image, ExpMap
from xga.utils import energy_to_channel, xmm_det, xmm_sky, xga_conf
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
    out = out.decode("UTF-8")
    err = err.decode("UTF-8")

    if p_type == "image":
        # Maybe let the user decide not to raise errors detected in stderr
        prod = Image(p_path[0], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                     extra_info["lo_en"], extra_info["hi_en"])
    elif p_type == "expmap":
        prod = ExpMap(p_path[0], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                      extra_info["lo_en"], extra_info["hi_en"])
    elif p_type == "ccf":
        # ccf files may not be destined to spend life as product objects, but that doesn't mean
        # I can't take momentarily advantage of the error parsing I built into the product classes
        prod = BaseProduct(p_path[0], "", "", out, err, cmd)
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
            prod_type_str = ", ".join(set(all_type))
            with tqdm(total=len(all_run), desc="Generating products of type(s) " + prod_type_str) as gen, \
                    Pool(cores) as pool:
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
    """
    A convenient Python wrapper for a configuration of the SAS evselect command that makes images.
    Images will be generated for every observation associated with every source passed to this function.
    If images in the requested energy band are already associated with the source,
    they will not be generated again
    :param List[BaseSource] sources: A single source object, or a list of source objects.
    :param Quantity lo_en: The lower energy limit for the image, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the image, in astropy energy units.
    :param str add_expr: A string to be added to the SAS expression keyword
    :param int num_cores: The number of cores to use (if running locally), default is set to
    90% of available.
    """
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


@sas_call
def cifbuild(sources: List[BaseSource], num_cores: int = NUM_CORES):
    """
    A wrapper for the XMM cifbuild command, which will be run before many of the more complex
    SAS commands, to check that a CIF compatible with the local version of SAS is available.
    :param List[BaseSource] sources: A single source object, or a list of source objects.
    :param int num_cores: The number of cores to use (if running locally), default is set to
    90% of available.
    """
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    # This string contains the bash code to run cifbuild
    cif_cmd = "cd {d}; export SAS_ODF={odf}; cifbuild calindexset=ccf.cif; unset SAS_ODF"

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for source in sources:
        cmds = []
        final_paths = []
        extra_info = []
        for obs_id in source.obs_ids:
            odf_path = source.get_odf_path(obs_id)

            dest_dir = "{out}{obs}/".format(out=OUTPUT, obs=obs_id)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            final_path = dest_dir + "ccf.cif"
            if not os.path.exists(final_path):
                cmds.append(cif_cmd.format(d=dest_dir, odf=odf_path))
                final_paths.append(final_path)
                extra_info.append({})  # This doesn't need any extra information

        sources_cmds.append(array(cmds))
        sources_paths.append(array(final_paths))
        sources_extras.append(array(extra_info))
        sources_types.append(full(sources_cmds[-1].shape, fill_value="ccf"))

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras


@sas_call
def eexpmap(sources: List[BaseSource], lo_en: Quantity, hi_en: Quantity, num_cores: int = NUM_CORES):
    """
    A convenient Python wrapper for the SAS eexpmap command.
    Expmaps will be generated for every observation associated with every source passed to this function.
    If expmaps in the requested energy band are already associated with the source,
    they will not be generated again
    :param List[BaseSource] sources: A single source object, or a list of source objects.
    :param Quantity lo_en: The lower energy limit for the expmap, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the expmap, in astropy energy units.
    :param int num_cores: The number of cores to use (if running locally), default is set to
    90% of available.
    """
    # I know that a lot of this code is the same as the evselect_image code, but its 1am so please don't
    # judge me too much.

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

    # These are crucial, to generate an exposure map one must have a ccf.cif calibration file, and a reference
    # image. If they do not already exist, these commands should generate them.
    cifbuild(sources)
    sources = evselect_image(sources, lo_en, hi_en)
    # This is necessary because the decorator will reduce a one element list of source objects to a single
    # source object. Useful for the user, not so much here where the code expects an iterable.
    if not isinstance(sources, list):
        sources = [sources]

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
            exists = [match for match in source.get_products("expmap", obs_id, inst) if en_id in match]
            if len(exists) == 1 and exists[0][-1].usable:
                continue
            # Generating an exposure map requires a reference image.
            ref_im = [match for match in source.get_products("image", obs_id, inst) if en_id in match][0][-1]
            # It also requires an attitude file
            att = source.get_att_file(obs_id)
            # Set up the paths and names of files
            evt_list = pack[-1]
            dest_dir = OUTPUT + "{o}/{i}_{l}-{u}_temp/".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)
            exp_map = "{o}_{i}_{l}-{u}keVexpmap.fits".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)

            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)

            os.makedirs(dest_dir)
            # TODO Maybe support det coords
            cmds.append("cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; eexpmap eventset={e} "
                        "imageset={im} expimageset={eim} withdetcoords=no withvignetting=yes "
                        "attitudeset={att} pimin={l} pimax={u}; mv * ../; cd ..; "
                        "rm -r {d}".format(e=evt_list.path, im=ref_im.path, eim=exp_map, att=att, l=lo_chan,
                                           u=hi_chan, d=dest_dir, ccf=dest_dir + "ccf.cif"))

            # This is the products final resting place, if it exists at the end of this command
            final_paths.append(os.path.join(OUTPUT, obs_id, exp_map))
            extra_info.append({"lo_en": lo_en, "hi_en": hi_en, "obs_id": obs_id, "instrument": inst})
        sources_cmds.append(array(cmds))
        sources_paths.append(array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the SAS cmd has run
        sources_extras.append(array(extra_info))
        sources_types.append(full(sources_cmds[-1].shape, fill_value="expmap"))

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
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
