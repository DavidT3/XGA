#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 03/05/2020, 13:22. Copyright (c) David J Turner

from astropy.units import Quantity
import os
import sys
from shutil import rmtree
from numpy import array
from multiprocessing import Pool
from subprocess import Popen, call, PIPE
from tqdm import tqdm

from xga.sources import BaseSource
from xga.utils import energy_to_channel
from xga import OUTPUT, COMPUTE_MODE, NUM_CORES


def execute_cmd(cmd):
    # TODO This should take a product type (so as to create the right type of object), and a source class
    # to add it to.
    raise NotImplementedError("While this can run commands, it can't yet instantiate product classes,"
                              "because I haven't written them yet.")
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    return None


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
        cmd_list, to_stack, to_execute, cores = sas_func(*args, **kwargs)

        # The first argument of all of these SAS functions will be the source object, so rather than
        # return them from the sas function I'll just access them like this.
        source: BaseSource = args[0]
        source.update_queue(cmd_list, to_stack)
        if to_execute and COMPUTE_MODE == "local":
            to_run = source.get_queue()

            with tqdm(total=len(to_run), desc="Generating Products") as gen, Pool(cores) as pool:
                def callback(results_in):
                    nonlocal gen
                    if results_in is None:
                        gen.update(1)
                        return
                    else:
                        # results.append(results_in)
                        gen.update(1)

                def err_callback(err):
                    # TODO Don't know if this needs fleshing out more?
                    if err is not None:
                        print(err)
                    nonlocal gen
                    gen.update(1)

                for cmd in to_run:
                    pool.apply_async(execute_cmd, args=(str(cmd),), error_callback=err_callback,
                                     callback=callback)
                pool.close()
                pool.join()

    return wrapper


@sas_call
def evselect_image(source: BaseSource, lo_en: Quantity, hi_en: Quantity, add_expr: str = "",
                   stack: bool = False, execute: bool = True, num_cores: int = NUM_CORES):
    # TODO support an array of source objects - this requires some changes, as jobs are added to a
    #  particular source queue
    if lo_en > hi_en:
        raise ValueError("lo_en cannot be greater than hi_en")
    else:
        lo_chan = energy_to_channel(lo_en)
        hi_chan = energy_to_channel(hi_en)

    expr = " && ".join([e for e in ["expression='(PI in [{l}:{u}])'".format(l=lo_chan, u=hi_chan),
                                    add_expr] if e != ""])
    cmds = []
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
    cmds = array(cmds)
    return cmds, stack, execute, num_cores


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
