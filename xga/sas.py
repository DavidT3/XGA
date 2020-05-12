#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 03/05/2020, 13:22. Copyright (c) David J Turner

from astropy.units import Quantity
import os
import sys
from shutil import rmtree
from numpy import array, full
from multiprocessing import Pool
from subprocess import Popen, call, PIPE
from tqdm import tqdm

from xga.sources import BaseSource
from xga.products import Image
from xga.utils import energy_to_channel
from xga import OUTPUT, COMPUTE_MODE, NUM_CORES


# TODO need to pass product type, final path, as well as associated source object
# TODO cmd may need to change to string if I execute stack separately one after the other,
#  would allow for path checking
def execute_cmd(cmd: str, p_type: str, p_path: list):
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    if p_type == "image":
        # Maybe let the user decide not to raise errors detected in stderr
        prod = Image(p_path[0], out.decode("UTF-8"), err.decode("UTF-8"), cmd)
    else:
        raise NotImplementedError("Not implemented yet")

    return prod


def sas_call(sas_func):
    """
    This is used as a decorator for functions that produce SAS command strings. Depending on the
    system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the SAS command will change. This supports both simple multi-threading and submission
    with the Sun Grid Engine.
    :return:
    """
    def wrapper(*args, **kwargs):
        # TODO There need to be checks that the ordered products don't already exist
        cmd_list, to_stack, to_execute, cores, p_type, paths = sas_func(*args, **kwargs)

        # The first argument of all of these SAS functions will be the source object, so rather than
        # return them from the sas function I'll just access them like this.
        source: BaseSource = args[0]
        source.update_queue(cmd_list, p_type, paths, to_stack)
        if to_execute and COMPUTE_MODE == "local":
            to_run, expected_type, expected_path = source.get_queue()

            raised_errors = []
            with tqdm(total=len(to_run), desc="Generating Products") as gen, Pool(cores) as pool:
                def callback(results_in):
                    nonlocal gen
                    if results_in is None:
                        gen.update(1)
                        return
                    else:
                        gen.update(1)

                def err_callback(err):
                    nonlocal raised_errors
                    nonlocal gen

                    if err is not None:
                        raised_errors.append(err)

                    gen.update(1)

                for cmd_ind, cmd in enumerate(to_run[:1]):
                    exp_type = expected_type[cmd_ind]
                    exp_path = expected_path[cmd_ind]
                    pool.apply_async(execute_cmd, args=(str(cmd), str(exp_type), exp_path),
                                     error_callback=err_callback, callback=callback)
                pool.close()
                pool.join()

                for error in raised_errors:
                    raise error

    return wrapper


@sas_call
def evselect_image(source: BaseSource, lo_en: Quantity, hi_en: Quantity,
                   add_expr: str = "", num_cores: int = NUM_CORES):
    stack = False
    execute = True
    # TODO support an array of source objects - this requires some changes, as jobs are added to a
    #  particular source queue - a bit of the sas_call will need to be changed, but shouldn't be too bad
    if lo_en > hi_en:
        raise ValueError("lo_en cannot be greater than hi_en")
    else:
        lo_chan = energy_to_channel(lo_en)
        hi_chan = energy_to_channel(hi_en)

    expr = " && ".join([e for e in ["expression='(PI in [{l}:{u}])".format(l=lo_chan, u=hi_chan),
                                    add_expr] if e != ""]) + "'"
    cmds = []
    final_paths = []
    for pack in source.get_products("events"):
        obs_id = pack[0]
        inst = pack[1]
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
                    "withyranges=yes {ex}; mv * ../; cd ..; rm -r {d}".format(d=dest_dir, e=evt_list,
                                                                              i=im, ex=expr))

        # This is the products final resting place, if it exists at the end of this command
        final_paths.append(os.path.join(OUTPUT, obs_id, im))
    cmds = array(cmds)
    final_paths = array(final_paths)

    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return cmds, stack, execute, num_cores, full(cmds.shape, fill_value="image"), final_paths


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
