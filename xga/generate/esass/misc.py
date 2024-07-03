#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 03/07/2024, 08:35. Copyright (c) The Contributors

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
    if ((not isinstance(sources, list) and 'erosita' not in sources.telescopes) or
            (isinstance(sources, list) and 'erosita' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such a "
                                          "combined eROSITA event list cannot be generated.")
    
    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]
    
    evtool_cmd = 'cd {d}; evtool eventfiles="{evts}" outfile="{out}"; mv * ../; cd ..; rm -r {d}'

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
        #  beginning of this function), we still need to append the empty cmds, paths, extrainfo, and ptypes to 
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occuring in the esass_call wrapper
        if 'erosita' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the eSASS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="events"))
            
            # then we can continue with the rest of the sources
            continue

        evt_list_paths = [match[-1].path for match in source.get_products("events", just_obj=False, telescope='erosita')]
        # This gets passed onto the extra_info dict
        obs_ids = [match[1] for match in source.get_products("events", just_obj=False, telescope='erosita')]

        # If only one event list is associated, then there is nothing to combine
        if len(evt_list_paths) == 1:
            sources_cmds.append(np.array([]))
            sources_paths.append(np.array([]))
            sources_extras.append(np.array([]))
            sources_types.append(np.array([]))
            continue

        # TODO if the user runs different samples with different search distances then more obs can
        # be associated so I would need to check this
        # Checking if the combined product already exists
        # TODO existing combined event lists arent loaded into a source
        exists = source.get_products("combined_events", just_obj=False, telescope='erosita')

        if len(exists) == 1 and exists[0][-1].usable:
            sources_cmds.append(np.array([]))
            sources_paths.append(np.array([]))
            sources_extras.append(np.array([]))
            sources_types.append(np.array([]))
            continue

        # The files produced by this function will now be stored in the combined directory.
        final_dest_dir = OUTPUT + "erosita/combined/"
        rand_ident = randint(0, 1e+8)
        # Makes absolutely sure that the random integer hasn't already been used
        while len([f for f in os.listdir(final_dest_dir)
                   if str(rand_ident) in f.split(OUTPUT+"erosita/combined/")[-1]]) != 0:
            rand_ident = randint(0, 1e+8)
        
        dest_dir = os.path.join(final_dest_dir, "temp_evtool_{}".format(rand_ident))
        os.mkdir(dest_dir)

        # Combining all the names of the paths to input into the command
        input_evts = ' '.join(evt_list_paths)

        # The name of the file used to contain all the ObsIDs that went into the combined event list. However
        #  this caused problems when too many ObsIDs were present and the filename was longer than allowed. So
        #  now I use the random identity I generated, and store the ObsID/instrument information in the inventory
        #  file
        outfile = "{os}_merged_events.fits".format(os=rand_ident)

        sources_cmds.append(np.array([evtool_cmd.format(d=dest_dir, evts=input_evts, out=outfile)]))
        sources_paths.append(np.array([os.path.join(final_dest_dir, outfile)]))
        # This contains any other information that will be needed to instantiate the class
        # once the eSASS cmd has run
        # The 'combined' values for obs and inst here are crucial, they will tell the source object that the final
        # product is assigned to that these are merged products - combinations of all available data
        sources_extras.append(np.array([{"obs_id": "combined",
                                         "instrument": "combined", 
                                         "telescope": 'erosita',
                                         "obs_ids": obs_ids}]))  
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="combined events"))

    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress
