#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/7/26, 10:21 AM. Copyright (c) The Contributors.

import os
from random import randint
from typing import Union

import numpy as np

from xga import NUM_CORES, OUTPUT
from xga.exceptions import TelescopeNotAssociatedError
from xga.samples import BaseSample
from xga.sources import BaseSource
from xga.sources.base import NullSource
from .run import esass_call


@esass_call
def evtool_combine_evts(sources: Union[BaseSource, NullSource, BaseSample], num_cores: int = NUM_CORES,
                        disable_progress: bool = False):
    """
    A convenient Python wrapper for the eSASS evtool command that combines event lists. In eRASS data
    releases, observations contain duplicate events. For sources that lie in this area that
    appears in multiple observations, it is best to merge these observations and create one combined
    observation that removes duplicates. This function will create this combined observation from all 
    observations associated to a source. 

    :param BaseSource/NullSource/BaseSample sources: A single source object, or a sample of sources.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    """

    # We check to see whether there is an eROSITA entry in the 'telescopes' property. If sources is a Source
    #  object, then that property contains the telescopes associated with that source, and if it is a Sample object
    #  then 'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if eROSITA isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and (
            'erosita' not in sources.telescopes and 'erass' not in sources.telescopes)) or
            (isinstance(sources, list) and (
                    'erosita' not in sources[0].telescopes and 'erass' not in sources[0].telescopes))):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA spectra cannot be generated.")

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    # Set up a template for the evtool command to be run
    evtool_cmd = 'cd {d}; evtool eventfiles="{evts}" outfile="{out}"; find . -maxdepth 1 -type f -exec mv {{}} ../ \\;; cd ..; rm -r {d}'

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
        # By this point we know that at least one of the sources has erosita data associated (we checked that at the
        #  beginning of this function). We still need to append the empty cmds, paths, extra_info, and ptypes to
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occurring in the esass_call wrapper
        if 'erosita' not in source.telescopes and 'erass' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to
            #  instantiate the Spectrum class once the eSASS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="combined events"))

            # Now we can continue with the rest of the sources
            continue

        for er_miss in ['erosita', 'erass']:
            # Skip this iteration if the current skew of eROSITA isn't associated
            #  with the current source
            if er_miss not in source.telescopes:
                continue

            # Fetch the individual skytile event lists associated with the current
            #  source - if there are more than one, we'll be combining them
            rel_skytile_evts = source.get_products("events", telescope=er_miss)

            # Then the ObsIDs (skytile IDs) - this gets passed onto the extra_info dict
            evt_obs_ids = [cur_ev.obs_id for cur_ev in rel_skytile_evts]

            # TODO if the user runs different samples with different search distances then more obs can
            #  be associated so I would need to check this
            # Checking if the combined events product already exists
            comb_evt_exists = source.get_products("combined_events", telescope=er_miss)

            # If only one skytile event list is associated, then there is nothing to
            #  combine, or if there are multiple, but we already have a combined
            #  event list, then there is nothing to do here.
            if len(rel_skytile_evts) == 1 or (len(rel_skytile_evts) != 1 and
                                              len(comb_evt_exists) == 1 and
                                              comb_evt_exists[0].usable):
                continue

            # If we've gotten this far, then there is a command to be run, so we start
            #  setting up the working and output directories.
            # The combined event list files produced by this function will be
            #  stored in the combined-ObsID directory once the process is complete.
            final_dest_dir = os.path.join(OUTPUT, er_miss, 'combined')

            # We're also going to need a temporary working directory and create a
            #  random integer to add to the directory to avoid collisions
            rand_ident = randint(0, 100_000_000)
            # Set up the path to the temporary working directory, and create it
            dest_dir = os.path.join(final_dest_dir, "temp_evtool_{}".format(rand_ident))
            os.makedirs(dest_dir, exist_ok=True)

            # eSASS command line tools (along with many others) can have issues with
            #  overlong input file paths (see closed issue #1400). So we set up
            #  symlinks to the event lists in the working directory
            for cur_ev in rel_skytile_evts:
                evt_symlink_name = os.path.basename(cur_ev.path)
                os.symlink(cur_ev.path, os.path.join(dest_dir, evt_symlink_name))

            # Combining all the symlinked skytile event list paths to make an
            #  input for evtool
            input_evts = ' '.join([os.path.basename(cur_ev.path)
                                   for cur_ev in rel_skytile_evts])

            # Create the name of the final event list file - use the random integer
            #  we've already set up to make the file name unique, as including all
            #  skytile IDs can result in file names that are just too long
            comb_evt_name = "{os}_merged_events.fits".format(os=rand_ident)

            # Add the populated command string, and the expected output path for the
            #  combined event list to the local lists for this source
            cmds.append(evtool_cmd.format(d=dest_dir, evts=input_evts, out=comb_evt_name))
            final_paths.append(os.path.join(final_dest_dir, comb_evt_name))

            # This contains any other information that will be needed to instantiate
            #  the event list class once the eSASS cmd has run.
            extra_info.append({"obs_id": "combined", "instrument": "combined", "telescope": er_miss,
                               "obs_ids": evt_obs_ids})

        # Now that we've checked both eROSITA skews, we add the results for this source to the master lists
        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="combined events"))

    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress
