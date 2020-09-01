#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 01/09/2020, 17:08. Copyright (c) David J Turner

import os
from typing import List

import numpy as np

from xga import OUTPUT, NUM_CORES
from xga.sources import BaseSource
from .run import sas_call


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

            if not os.path.exists(OUTPUT + obs_id):
                os.mkdir(OUTPUT + obs_id)

            dest_dir = "{out}{obs}/".format(out=OUTPUT, obs=obs_id)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            final_path = dest_dir + "ccf.cif"
            if not os.path.exists(final_path):
                cmds.append(cif_cmd.format(d=dest_dir, odf=odf_path))
                final_paths.append(final_path)
                extra_info.append({})  # This doesn't need any extra information

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="ccf"))

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras