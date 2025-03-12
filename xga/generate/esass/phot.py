#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 12/03/2025, 10:50. Copyright (c) The Contributors

import os
from random import randint
from shutil import rmtree
from typing import Union

import numpy as np
from astropy.units import Quantity, UnitConversionError

from .misc import evtool_combine_evts
from .run import esass_call
from ... import OUTPUT, NUM_CORES
from ...exceptions import TelescopeNotAssociatedError, NoProductAvailableError
from ...products import BaseProduct
from ...products.misc import EventList
from ...samples.base import BaseSample
from ...sources import BaseSource
from ...sources.base import NullSource


def _img_params_from_evtlist(evt_list: EventList):
    """
    Internal function to work out the XGA image size and centre position for eROSITA observations. This is done using 
    the minimum and maximum of the ra and dec, with a 1% buffer, as the corners of the image.
    
    :param Eventlist evt_list: An EventList product object.
    """

    # returns a dataframe of only the RA and DEC columns
    rel_df = evt_list.get_columns_from_data(['RA', 'DEC'])

    # This gives these values in degrees
    ramin = rel_df['RA'].min()
    ramax = rel_df['RA'].max()
    demin = rel_df['DEC'].min()
    demax = rel_df['DEC'].max()

    # we want the minimum separation, ie. between ra=1 and ra=359 we want a difference of 2, not 358
    if abs(ramin - ramax) > 180:
        rasep = abs(ramin - ramax + 360)
    else:
        rasep = abs(ramin - ramax)
    
    # If rasep is really tiny, then what has happened is that ramin = 0.00001 and ramax = 359.9999
    # and this means that our events go around the 0 deg. RA in the sky, so using ramin won't work
    # I could do this more thoroughly by sorting in some way, but this is the fastest way I think
    if rasep < 0.001:
        # This is the RA on the 0 - 20 deg side
        ra1 = rel_df[(rel_df['RA'] < 20) & (rel_df['RA'] > 0)]['RA'].max()
        # This is the RA on the 340 - 360
        ra2 = rel_df[(rel_df['RA'] > 340) & (rel_df['RA'] < 360)]['RA'].min()
        rasep = abs(ra1 - ra2 + 360)
    
    # deleting this to save memory
    del rel_df

    decsep = abs(demin - demax)

    # Now we have a 1% border around this to ensure all events are captured
    rasep = rasep + rasep*0.01
    decsep = decsep + decsep*0.01

    # Then we work out the x_size and y_size of the image in pixels
    rebin = 87
    # The virtual pixel size is 1.3888889095849E-5 degrees - so 87 of them is equal to 4.35 arcseconds,
    #  which is the normal binning for XCS, so that is what I'll start with
    x_size = np.ceil(rasep / 1.3888889095849e-5 / rebin).astype(int)
    y_size = np.ceil(decsep / 1.3888889095849e-5 / rebin).astype(int)

    if evt_list.obs_id != 'combined':
        # for evt lists made of one tile, the centre of the image will be at 0, 0
        centre_pos = "0 0"
    else:
        # For evt_lists that are made of multiple tiles, need to calculate the center position
        # returns a QTable of only the X and Y columns
        rel_df = evt_list.get_columns_from_data(['X', 'Y'])

        # This gives these values in degrees
        xmin = rel_df['X'].min()
        xmax = rel_df['X'].max()
        ymin = rel_df['Y'].min()
        ymax = rel_df['Y'].max()

        # deleting this to save memory
        del rel_df

        xsep = abs(xmin - xmax)
        ysep = abs(ymin - ymax)
        
        xcen = int(xmin + (xsep/2))
        ycen = int(ymin + (ysep/2))

        centre_pos = f"{xcen} {ycen}"
    
    return rebin, x_size, y_size, centre_pos


@esass_call
def evtool_image(sources: Union[BaseSource, NullSource, BaseSample], lo_en: Quantity = Quantity(0.2, 'keV'),
                 hi_en: Quantity = Quantity(10, 'keV'), combine_obs: bool = False, num_cores: int = NUM_CORES,
                 disable_progress: bool = False):
    """
    A convenient Python wrapper for a configuration of the eSASS evtool command that makes images.
    Images will be generated for every observation associated with every source passed to this function.
    If images in the requested energy band are already associated with the source,
    they will not be generated again.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or a sample of sources.
    :param Quantity lo_en: The lower energy limit for the image, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the image, in astropy energy units.
    :param bool combine_obs: Setting this to False will generate an image for each associated observation, 
        instead of for one combined observation.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    # We check to see whether there is an eROSITA entry in the 'telescopes' property. If sources is a Source
    #  object, then that property contains the telescopes associated with that source, and if it is a Sample object
    #  then 'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if eROSITA isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and 'erosita' not in sources.telescopes) or
            (isinstance(sources, list) and 'erosita' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA images cannot be generated.")

    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]
    
    # Checking user's choice of energy limit parameters
    if not isinstance(lo_en, Quantity) or not isinstance(hi_en, Quantity):
        raise TypeError("The lo_en and hi_en arguments must be astropy quantities in units "
                        "that can be converted to keV.")
    
    # Have to make sure that the energy bounds are in units that can be converted to keV (which is what evtool
    #  expects for these arguments).
    elif not lo_en.unit.is_equivalent('eV') or not hi_en.unit.is_equivalent('eV'):
        raise UnitConversionError("The lo_en and hi_en arguments must be astropy quantities in units "
                                  "that can be converted to keV.")

    # Checking that the upper energy limit is not below the lower energy limit
    elif hi_en <= lo_en:
        raise ValueError("The hi_en argument must be larger than the lo_en argument.")
    
    # Converting to the right unit
    else:
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')

    # Checking user's lo_en and hi_en inputs are in the valid energy range for eROSITA
    if ((lo_en < Quantity(200, 'eV') or lo_en > Quantity(10000, 'eV')) or
            (hi_en < Quantity(200, 'eV') or hi_en > Quantity(10000, 'eV'))):
        raise ValueError("The lo_en and hi_en value must be between 0.2 keV and 10 keV.")
    
    # Checking user's choice of combine_obs
    if not isinstance(combine_obs, bool):
        raise TypeError("The combine_obs argument must be a bool.")
    
    if combine_obs:
        # This requires combined event lists - this function will generate them
        evtool_combine_evts(sources)

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

        # By this point we know that at least one of the sources has eROSITA data associated (we checked that at the
        #  beginning of this function), we still need to append the empty cmds, paths, extrainfo, and ptypes to 
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occuring in the esass_call wrapper
        if 'erosita' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the eSASS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="image"))
            
            # then we can continue with the rest of the sources
            continue
        
        # Define this variable for each source no it doesn't get overwritten with the wrong value
        # later in the loop
        use_combine_obs = combine_obs
        
        # if the user has set combine_obs to True and there is only one observation, then we 
        # use the combine_obs = False functionality instead
        if use_combine_obs and len(source.obs_ids['erosita']) == 1:
            use_combine_obs = False
    
        if not use_combine_obs:
            # Check which event lists are associated with each individual source
            for pack in source.get_products("events", telescope='erosita', just_obj=False):
                obs_id = pack[1]
                inst = pack[2]
                if not os.path.exists(OUTPUT + 'erosita/' + obs_id):
                    os.mkdir(OUTPUT + 'erosita/' + obs_id)

                en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
                # ASSUMPTION5 source.get_products has a telescope parameter
                exists = [match for match in
                          source.get_products("image", obs_id, inst, just_obj=False, telescope='erosita')
                          if en_id in match]
                if len(exists) == 1 and exists[0][-1].usable:
                    continue

                evt_list = pack[-1]

                re_bin, x_size, y_size, centre_pos = _img_params_from_evtlist(evt_list)

                dest_dir = OUTPUT + "erosita/" + "{o}/{i}_{l}-{u}_{n}_temp/".format(o=obs_id, i=inst, l=lo_en.value,
                                                                                    u=hi_en.value, n=source.name)
                im = "{o}_{i}_{l}-{u}keVimg.fits".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)

                # If something got interrupted and the temp directory still exists, this will remove it
                if os.path.exists(dest_dir):
                    rmtree(dest_dir)

                os.makedirs(dest_dir)
                cmds.append("cd {d}; evtool eventfiles={e} outfile={i} image=yes emin={l} emax={u} events=no "
                            "size='{xs} {ys}' rebin={rb} center_position='{c}'; mv * ../; cd ..; "
                            "rm -r {d}".format(d=dest_dir, e=evt_list.path, i=im, l=lo_en.value, u=hi_en.value, rb=re_bin,
                                            xs=x_size, ys=y_size, c=centre_pos))

                # This is the products final resting place, if it exists at the end of this command
                # ASSUMPTION4 new output directory structure
                final_paths.append(os.path.join(OUTPUT, "erosita", obs_id, im))
                extra_info.append({"lo_en": lo_en, "hi_en": hi_en, "obs_id": obs_id, "instrument": inst,
                                "telescope": "erosita"})
            
        else:
            # Checking if a combined event list has be made already
            try:
                exists = source.get_combined_images(lo_en=lo_en, hi_en=hi_en, telescope='erosita')
            except NoProductAvailableError:
                exists = []

            if isinstance(exists, BaseProduct) and exists.usable:
                # we still need to append the empty cmds, paths, extrainfo, and ptypes to 
                #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
                #  bugs occuring in the esass_call wrapper
                sources_cmds.append(np.array([]))
                sources_paths.append(np.array([]))
                # This contains any other information that will be needed to instantiate the class
                # once the eSASS cmd has run
                sources_extras.append(np.array([]))
                sources_types.append(np.full(sources_cmds[-1].shape, fill_value="image"))
                continue

            en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
            # getting Eventlist product
            evt_list = source.get_products("combined_events", just_obj=True, telescope="erosita")[0]
            obs_id = evt_list.obs_id
            inst = evt_list.instrument
            re_bin, x_size, y_size, centre_pos = _img_params_from_evtlist(evt_list)

            # The files produced by this function will now be stored in the combined directory.
            final_dest_dir = OUTPUT + "erosita/combined/"
            rand_ident = randint(0, 1e+8)
            # Makes absolutely sure that the random integer hasn't already been used
            while len([f for f in os.listdir(final_dest_dir)
                    if str(rand_ident) in f.split(OUTPUT+"erosita/combined/")[-1]]) != 0:
                rand_ident = randint(0, 1e+8)

            dest_dir = os.path.join(final_dest_dir, "temp_evtool_{}".format(rand_ident))
            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)

            os.mkdir(dest_dir)

            im = "{r}_{l}-{u}keVimg.fits".format(r=rand_ident, l=lo_en.value, u=hi_en.value)

            cmds.append("cd {d}; evtool eventfiles={e} outfile={i} image=yes emin={l} emax={u} events=no "
                        "size='{xs} {ys}' rebin={rb} center_position='{c}'; mv * ../; cd ..; "
                        "rm -r {d}".format(d=dest_dir, e=evt_list.path, i=im, l=lo_en.value, u=hi_en.value, rb=re_bin,
                                        xs=x_size, ys=y_size, c=centre_pos))
            # This is the products final resting place, if it exists at the end of this command
            final_paths.append(os.path.join(final_dest_dir, im))
            extra_info.append({"lo_en": lo_en, "hi_en": hi_en, "obs_id": obs_id, "instrument": inst,
                            "telescope": "erosita"})

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the eSASS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="image"))

    # I only return num_cores here, so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@esass_call
def expmap(sources: Union[BaseSource, NullSource, BaseSample], lo_en: Quantity = Quantity(0.2, 'keV'),
           hi_en: Quantity = Quantity(10, 'keV'), combine_obs: bool = False, num_cores: int = NUM_CORES,
           disable_progress: bool = False):
    """
    A convenient Python wrapper for the eSASS expmap command.
    Expmaps will be generated for every observation associated with every source passed to this function.
    If expmaps in the requested energy band are already associated with the source,
    they will not be generated again.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or sample of sources.
    :param Quantity lo_en: The lower energy limit for the expmap, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the expmap, in astropy energy units.
    :param bool combine_obs: Setting this to False will generate an image for each associated observation,
        instead of for one combined observation.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    """
    # We check to see whether there is an eROSITA entry in the 'telescopes' property. If sources is a Source
    #  object, then that property contains the telescopes associated with that source, and if it is a Sample object
    #  then 'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if eROSITA isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and 'erosita' not in sources.telescopes) or
            (isinstance(sources, list) and 'erosita' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA exposure maps cannot be generated.")

    # TODO make sure that the same exposure map is added as a product to every source it covers
    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    # Checking user's choice of energy limit parameters
    if not isinstance(lo_en, Quantity) or not isinstance(hi_en, Quantity):
        raise TypeError("The lo_en and hi_en arguments must be astropy quantities in units "
                        "that can be converted to keV.")
    
    # Have to make sure that the energy bounds are in units that can be converted to keV (which is what evtool
    #  expects for these arguments).
    elif not lo_en.unit.is_equivalent('eV') or not hi_en.unit.is_equivalent('eV'):
        raise UnitConversionError("The lo_en and hi_en arguments must be astropy quantities in units "
                                  "that can be converted to keV.")

    # Checking that the upper energy limit is not below the lower energy limit
    elif hi_en <= lo_en:
        raise ValueError("The hi_en argument must be larger than the lo_en argument.")
    
    # Converting to the right unit
    else:
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')

    # Checking user's lo_en and hi_en inputs are in the valid energy range for eROSITA
    if (lo_en < Quantity(200, 'eV') or lo_en > Quantity(10000, 'eV')) or \
        (hi_en < Quantity(200, 'eV') or hi_en > Quantity(10000, 'eV')):
        raise ValueError("The lo_en and hi_en value must be between 0.2 keV and 10 keV.")

    # To generate an exposure map one must have a reference image. 
    # If they do not already exist, these commands should generate them.
    sources = evtool_image(sources, lo_en, hi_en, combine_obs=combine_obs)

    # This is necessary because the decorator will reduce a one element list of source objects to a single
    # source object. Useful for the user, not so much here where the code expects an iterable.
    if not isinstance(sources, (list, BaseSample)):
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

        # By this point we know that at least one of the sources has eROSITA data associated (we checked that at the
        #  beginning of this function), we still need to append the empty cmds, paths, extrainfo, and ptypes to 
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occuring in the esass_call wrapper
        if 'erosita' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the eSASS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="expmap"))
            
            # then we can continue with the rest of the sources
            continue
        
        # Define this variable for each source no it doesn't get overwritten with the wrong value
        # later in the loop
        use_combine_obs = combine_obs
        
        # if the user has set combine_obs to True and there is only one observation, then we 
        # use the combine_obs = False functionality instead
        if use_combine_obs and len(source.obs_ids['erosita']) == 1:
            use_combine_obs = False
        
        if not use_combine_obs:
            # Check which event lists are associated with each individual source
            for pack in source.get_products("events",  telescope='erosita', just_obj=False):
                obs_id = pack[1]
                inst = pack[2]
                # ASSUMPTION4 new output directory structure
                if not os.path.exists(OUTPUT + 'erosita/' + obs_id):
                    os.mkdir(OUTPUT + 'erosita/' + obs_id)

                en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
                # ASSUMPTION5 source.get_products has a telescope parameter
                exists = [match for match in source.get_products("expmap", obs_id, inst, just_obj=False,
                                                                telescope='erosita')
                        if en_id in match]
                if len(exists) == 1 and exists[0][-1].usable:
                    continue
                # Generating an exposure map requires a reference image.
                # ASSUMPTION5 source.get_products has a telescope parameter
                ref_im = [match for match in source.get_products("image", obs_id, inst, just_obj=False,
                                                                telescope='erosita')
                        if en_id in match][0][-1]
                # Set up the paths and names of files
                evt_list = pack[-1]
                # ASSUMPTION4 new output directory structure
                dest_dir = OUTPUT + "erosita/" + "{o}/{i}_{l}-{u}_{n}_temp/".format(o=obs_id, i=inst, l=lo_en.value, 
                                                                                    u=hi_en.value, n=source.name)
                exp_map = "{o}_{i}_{l}-{u}keVexpmap.fits".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)

                # If something got interrupted and the temp directory still exists, this will remove it
                if os.path.exists(dest_dir):
                    rmtree(dest_dir)

                os.makedirs(dest_dir)
                # The HEASoft environment variables set here ensure that fthedit doesn't try to access the
                #  terminal, which causes 'device not available' errors
                cmds.append("cd {d}; expmap inputdatasets={e} templateimage={im} emin={l} emax={u} mergedmaps={em}; "
                            "export HEADASNOQUERY=; export HEADASPROMPT=/dev/null; fthedit {em} REFYCRVL delete; "
                            "mv * ../; cd ..; rm -r {d}".format(e=evt_list.path, im=ref_im.path, l=lo_en.value,
                                                                u=hi_en.value, em=exp_map, d=dest_dir))

                # This is the products final resting place, if it exists at the end of this command
                # ASSUMPTION4 new output directory structure
                final_paths.append(os.path.join(OUTPUT, "erosita", obs_id, exp_map))
                extra_info.append({"lo_en": lo_en, "hi_en": hi_en, "obs_id": obs_id, "instrument": inst,
                                "telescope": "erosita"})
        else:
            # Checking if a combined event list has be made already
            try:
                exists = source.get_combined_expmaps(lo_en=lo_en, hi_en=hi_en, telescope='erosita')
            except NoProductAvailableError:
                exists = []

            if isinstance(exists, BaseProduct) and exists.usable:
                continue
            
            en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
            # getting Eventlist product
            evt_list = source.get_products("combined_events", just_obj=True, telescope="erosita")[0]
            # defining values needed for esass expmap command
            obs_id = evt_list.obs_id
            inst = evt_list.instrument
            ref_im = source.get_combined_images(lo_en=lo_en, hi_en=hi_en, telescope='erosita')

            # The files produced by this function will now be stored in the combined directory.
            final_dest_dir = OUTPUT + "erosita/combined/"
            rand_ident = randint(0, 1e+8)
            # Makes absolutely sure that the random integer hasn't already been used
            while len([f for f in os.listdir(final_dest_dir)
                    if str(rand_ident) in f.split(OUTPUT+"erosita/combined/")[-1]]) != 0:
                rand_ident = randint(0, 1e+8)
            
            dest_dir = os.path.join(final_dest_dir, "temp_evtool_{}".format(rand_ident))
            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)

            os.mkdir(dest_dir)

            exp_map = "{r}_{l}-{u}keVexpmap.fits".format(r=rand_ident, l=lo_en.value, u=hi_en.value)

            # The HEASoft environment variables set here ensure that fthedit doesn't try to access the
            #  terminal, which causes 'device not available' errors
            cmds.append("cd {d}; expmap inputdatasets={e} templateimage={im} emin={l} emax={u} mergedmaps={em}; "
                        "export HEADASNOQUERY=; export HEADASPROMPT=/dev/null; fthedit {em} REFYCRVL delete; "
                        "mv * ../; cd ..; rm -r {d}".format(e=evt_list.path, im=ref_im.path, l=lo_en.value,
                                                            u=hi_en.value, em=exp_map, d=dest_dir))

            # This is the products final resting place, if it exists at the end of this command
            # ASSUMPTION4 new output directory structure
            final_paths.append(os.path.join(final_dest_dir, exp_map))
            extra_info.append({"lo_en": lo_en, "hi_en": hi_en, "obs_id": obs_id, "instrument": inst,
                            "telescope": "erosita"})


        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the eSASS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="expmap"))

    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
    # I only return num_cores here, so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


def combine_phot_prod(sources: Union[BaseSource, BaseSample], to_combine: str, 
                      lo_en: Quantity = Quantity(0.2, 'keV'), hi_en: Quantity = Quantity(10, 'keV'),
                      num_cores: int = NUM_CORES,
                      disable_progress: bool = False):
    # We check to see whether there is an eROSITA entry in the 'telescopes' property. 
    # If sources is a Source object, then that property contains the telescopes associated with 
    # that source, and if it is a Sample object then 'telescopes' contains the list of unique 
    # telescopes that are associated with at least one member source.
    # Clearly if eROSITA isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and 'erosita' not in sources.telescopes) or
            (isinstance(sources, list) and 'erosita' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the "
                                          "source/sample, as such eROSITA"
                                          "images or exposure maps cannot be generated.")
    
    if to_combine not in ["image", "expmap"]:
        raise ValueError("The only valid choices for to_combine are image and expmap.")
    # Don't do much value checking in this module, but this one is so fundamental that I will do it
    elif lo_en > hi_en:
        raise ValueError("lo_en cannot be greater than hi_en")

    # To make a mosaic we need to have the individual products in the first place
    if to_combine == "image":
        sources = evtool_image(sources, lo_en, hi_en, combine_obs=True, 
                               disable_progress=disable_progress, num_cores=num_cores)
    elif to_combine == "expmap":
        sources = expmap(sources, lo_en, hi_en, combine_obs=True, 
                         disable_progress=disable_progress, num_cores=num_cores)

    