#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 6/15/26, 7:44 PM. Copyright (c) The Contributors.

import os
import sys
from datetime import datetime
from random import randint
from subprocess import Popen, PIPE
from typing import Union

import numpy as np
from astropy.io import fits
from fitsio import read_header

from xga import OUTPUT, NUM_CORES
from xga.exceptions import InvalidProductError, TelescopeNotAssociatedError
from xga.samples.base import BaseSample
from xga.sources import BaseSource
from xga.sources.base import NullSource
from .run import sas_call
from ...products import BaseProduct


def mifbuild(sources):
    """
    This function generates a master index file from the SAS_CCFPATH constituents, which
    will then be used for cifbuild runs. If a MIF already exists, the creation date
    will be compared to the current date - if they do not match they will be
    regenerated.

    We use MIFs to avoid slowdowns in cifbuild caused by repeatedly traversing the
    SAS_CCFPATH constituents on slower file systems.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or a sample of sources.
    :return: Path to the MIF.
    :rtype: str
    """

    # We check to see whether there is an XMM entry in the 'telescopes' property. If sources is a Source object, then
    #  that property contains the telescopes associated with that source, and if it is a Sample object then
    #  'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if XMM isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and 'xmm' not in sources.telescopes) or
            (isinstance(sources, list) and 'xmm' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no XMM data associated with the source/sample, as such XMM "
                                          "calibration files cannot be generated.")

    dest_dir = os.path.join(OUTPUT, 'xmm', "")

    temp_name = "tempdir_{}".format(randint(0, int(100_000_000)))
    temp_dir = os.path.join(dest_dir, temp_name, "")

    final_path = os.path.join(dest_dir, 'ccf.mif')

    if os.path.exists(final_path):
        with fits.open(final_path) as miffo:
            mif_gen_date = miffo[0].header['DATE'].split('T')[0]

        gen_mif = mif_gen_date != datetime.today().strftime("%Y-%m-%d")
    else:
        gen_mif = True

    # This string contains the bash code to run cifbuild
    mif_cmd = f"cd {temp_dir}; cifbuild masterindex=yes calindexset={os.path.basename(final_path)}; mv * ../; cd ..; rm -r {temp_dir}"

    if gen_mif:
        os.makedirs(temp_dir, exist_ok=True)

        # This chunk is a fix for problems with eSASS (eROSITA package) finding the correct libraries on Apple ARM based
        #  systems, and just creates a new environment variable so it can locate them, if necessary
        sys_env = os.environ.copy()
        if sys.platform == 'darwin':
            if "LD_LIBRARY_PATH" in sys_env:
                mif_cmd = f"export LD_LIBRARY_PATH={sys_env['LD_LIBRARY_PATH']} && {mif_cmd}"
            if "DYLD_LIBRARY_PATH" in sys_env:
                mif_cmd = f"export DYLD_LIBRARY_PATH={sys_env['DYLD_LIBRARY_PATH']} && {mif_cmd}"

        # This runs the passed command - it captures the stdout and stderr as well
        out, err = Popen(mif_cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
        # Captured out/err are byte type, will decode that into str for easier use
        out = out.decode("UTF-8", errors='ignore')
        err = err.decode("UTF-8", errors='ignore')

        prod = BaseProduct(final_path, "", "", out, err, mif_cmd, telescope='xmm')
        prod.raise_errors()

    return final_path


@sas_call
def cifbuild(sources: Union[BaseSource, NullSource, BaseSample], num_cores: int = NUM_CORES,
             disable_progress: bool = False):
    """
    A wrapper for the XMM cifbuild command, which will be run before many of the more complex SAS commands, to
    check that a CIF compatible with the local version of SAS is available. The observation date is taken from an
    event list for a given ObsID, and the analysis date is set to the date which this function is run.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or a sample of sources.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    # We check to see whether there is an XMM entry in the 'telescopes' property. If sources is a Source object, then
    #  that property contains the telescopes associated with that source, and if it is a Sample object then
    #  'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if XMM isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and 'xmm' not in sources.telescopes) or
            (isinstance(sources, list) and 'xmm' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no XMM data associated with the source/sample, as such XMM "
                                          "calibration files cannot be generated.")

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    # This string contains the bash code to run cifbuild
    cif_cmd = ("cd {d}; cifbuild calindexset=ccf.cif withobservationdate=yes "
               "observationdate={od} withmasterindexset=yes masterindexset={mif}; "
               "mv * ../; cd ..; rm -r {n}")

    # This will get flipped to True if ANY cif is going to be generated, and
    #  then will be used to trigger the mifbuild function
    need_mif = False

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for source in sources:
        cmds = []
        final_paths = []
        extra_info = []

        # By this point we know that at least one of the sources has XMM data associated (we checked that at the
        #  beginning of this function), we still need to append the empty cmds, paths, extrainfo, and ptypes to
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occuring in the sas_call wrapper
        if 'xmm' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the SAS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="ccf"))

            # then we can continue with the rest of the sources
            continue

        for obs_id in source.obs_ids['xmm']:
            # Fetch an events list for this ObsID, doesn't matter which
            some_evt_lists = source.get_products("events", obs_id=obs_id, telescope='xmm')
            obs_date = None
            for evt in some_evt_lists:
                # Reads in the header of the events list file
                evt_head = read_header(evt.path)
                # Then extracts the observation date, this is what we need to give cifbuild
                if "DATE-OBS" in evt_head:
                    obs_date = evt_head['DATE-OBS']
                    del evt_head
                    break
                else:
                    del evt_head

            if obs_date is None:
                raise InvalidProductError("All event lists for {} are missing the DATE-OBS header, this is required to"
                                          " run the cifbuild function.".format(obs_id))

            if not os.path.exists(OUTPUT + 'xmm/' + obs_id):
                os.mkdir(OUTPUT + 'xmm/' + obs_id)

            dest_dir = "{out}xmm/{obs}/".format(out=OUTPUT, obs=obs_id)
            temp_name = "tempdir_{}".format(randint(0, int(100_000_000)))
            temp_dir = dest_dir + temp_name + "/"

            final_path = dest_dir + "ccf.cif"
            if not os.path.exists(final_path):
                need_mif = True

                os.makedirs(temp_dir, exist_ok=True)
                cmds.append(cif_cmd.format(d=temp_dir, od=obs_date, n=temp_name,
                                           mif=os.path.join(OUTPUT, 'xmm', 'ccf.mif')))
                final_paths.append(final_path)
                extra_info.append({})  # This doesn't need any extra information

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="ccf"))

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    # This is triggered if ANY cif is going to be built, and triggers the initial
    #  generation, or refresh, of the master index file.
    if need_mif:
        mifbuild(sources)

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress



