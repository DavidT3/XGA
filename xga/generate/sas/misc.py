#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 17/01/2024, 10:35. Copyright (c) The Contributors

import os
from random import randint
from typing import Union

import numpy as np
from fitsio import read_header

from xga import OUTPUT, NUM_CORES
from xga.exceptions import InvalidProductError, TelescopeNotAssociatedError
from xga.samples.base import BaseSample
from xga.sources import BaseSource
from xga.sources.base import NullSource
from .run import sas_call


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
    cif_cmd = "cd {d}; cifbuild calindexset=ccf.cif withobservationdate=yes " \
              "observationdate={od} ; mv * ../; cd ..; rm -r {n}"

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
            temp_name = "tempdir_{}".format(randint(0, 1e+8))
            temp_dir = dest_dir + temp_name + "/"

            final_path = dest_dir + "ccf.cif"
            if not os.path.exists(final_path):
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                cmds.append(cif_cmd.format(d=temp_dir, od=obs_date, n=temp_name))
                final_paths.append(final_path)
                extra_info.append({})  # This doesn't need any extra information

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="ccf"))

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


