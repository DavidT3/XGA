#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/6/26, 9:58 AM. Copyright (c) The Contributors.

import os
from random import randint
from typing import Union

import numpy as np
from astropy.units import Quantity

from .misc import evtool_combine_evts
from .run import esass_call
from ... import OUTPUT, NUM_CORES
from ...exceptions import TelescopeNotAssociatedError, NoProductAvailableError
from ...products.misc import EventList
from ...samples.base import BaseSample
from ...sources import BaseSource
from ...sources.base import NullSource


def _img_params_from_evtlist(evt_list: EventList):
    """
    Internal function to work out the XGA image size and centre position for eROSITA observations. This is done using 
    the minimum and maximum of the ra and dec, with a 1% buffer, as the corners of the image.
    
    :param EventList evt_list: An EventList product object.
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
    if ((not isinstance(sources, list) and (
            'erosita' not in sources.telescopes and 'erass' not in sources.telescopes)) or
            (isinstance(sources, list) and (
                    'erosita' not in sources[0].telescopes and 'erass' not in sources[0].telescopes))):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA images cannot be generated.")

    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]
    
    # Checking the user's choice of energy limit parameters - first are they the
    #  right kind of object, and do they have the right units?
    if (not all([isinstance(lo_en, Quantity), isinstance(hi_en, Quantity)]) or
            not all([lo_en.unit.is_equivalent('eV'), hi_en.unit.is_equivalent('eV')])):
        raise TypeError("The 'lo_en' and 'hi_en' arguments must be astropy quantities in units "
                        "that can be converted to keV.")
    # Checking that the upper energy limit is not below the lower energy limit
    elif hi_en <= lo_en:
        raise ValueError("The hi_en argument must be larger than the lo_en argument.")
    # If we get here we know they're okay, so we make sure they're in keV
    else:
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')

    # Checking user's lo_en and hi_en inputs are in the valid energy range for eROSITA
    if ((Quantity([lo_en, hi_en]) < Quantity(0.2, 'keV')).any() or
            (Quantity([lo_en, hi_en]) > Quantity(10, 'keV')).any()):
        raise ValueError("The 'lo_en' and 'hi_en' values must be between 0.2 keV and 10 keV for eROSITA.")

    # Checking the user's choice of 'combine_obs'
    if not isinstance(combine_obs, bool):
        raise TypeError("The combine_obs argument must be a bool.")

    # Make sure that combined event lists exist
    if combine_obs:
        # This requires combined event lists - this function will generate them
        evtool_combine_evts(sources, num_cores)

    # This is the template for the evtool command that will be run to make new images
    evtool_cmd = "cd {d}; evtool eventfiles={e} outfile={i} image=yes " \
                 "emin={l} emax={u} events=no size='{xs} {ys}' rebin={rb} " \
                 "center_position='{c}'; find . -maxdepth 1 -type f -exec mv {{}} ../ \\;; cd ..; rm -r {d}"

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
        #  beginning of this function). We still need to append the empty cmds, paths, extra_info, and ptypes to
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occuring in the esass_call wrapper
        if 'erosita' not in source.telescopes and 'erass' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the eSASS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="image"))

            # We can go to the next source
            continue

        for er_miss in ['erosita', 'erass']:
            # Skip this iteration if the current skew of eROSITA isn't associated
            #  with the current source
            if er_miss not in source.telescopes:
                continue

            # Define this variable for each iteration no it doesn't get overwritten with the wrong value
            # later in the loop
            use_combine_obs = combine_obs

            # if the user has set combine_obs to True and there is only one observation, then we
            # use the combine_obs = False functionality instead
            if use_combine_obs and len(source.obs_ids[er_miss]) == 1:
                use_combine_obs = False

            if not use_combine_obs:
                # Fetch the individual skytile event lists
                rel_skytile_evts = source.get_products("events", telescope=er_miss)

                # Iterating through the event lists of separate skytiles, as we
                #  are making one image per skytile in this case
                for cur_ev in rel_skytile_evts:
                    # We try to retrieve the image we've been told to make from the
                    #  source object, if an exception occurs, then it already exists
                    try:
                        im_exists = source.get_images(cur_ev.obs_id,
                                                      cur_ev.instrument,
                                                      telescope=er_miss,
                                                      lo_en=lo_en,
                                                      hi_en=hi_en)
                        # If we get here, then the image already exists, and we can
                        #  move to the next event list
                        continue
                    except NoProductAvailableError:
                        # If there is an exception raised then we just keep going, as
                        #  an image needs to be generated for this event list in
                        #  the specified energy band
                        pass

                    # Use an internal function to figure out the centering
                    #  and boundary information
                    re_bin, x_size, y_size, centre_pos = _img_params_from_evtlist(cur_ev)

                    # Now we set up the final output directory for the image to be
                    #  generated, as well as the temporary working directory (with
                    #  a random identifier in the name to avoid collisions)
                    final_dest_dir = os.path.join(OUTPUT, er_miss, cur_ev.obs_id)

                    # Generate the random identifier for the working directory
                    rand_ident = randint(0, 100_000_000)
                    dest_dir = os.path.join(final_dest_dir,
                                            "temp_evtool_{}".format(rand_ident))
                    # Make the working directory
                    os.makedirs(dest_dir, exist_ok=True)

                    # eSASS command line tools can have issues with overlong input
                    #  file paths (see closed issue #1400). So we set up a symlink
                    #  to the event list in the working directory
                    evt_symlink_name = os.path.basename(cur_ev.path)
                    os.symlink(cur_ev.path, os.path.join(dest_dir, evt_symlink_name))

                    # The name of the output image file
                    im_name = "{o}_{i}_{l}-{u}keVimg.fits".format(o=cur_ev.obs_id,
                                                                  i=cur_ev.instrument,
                                                                  l=lo_en.value,
                                                                  u=hi_en.value)

                    # Make the command to create the image
                    cmds.append(evtool_cmd.format(d=dest_dir,
                                                  e=evt_symlink_name,
                                                  i=im_name,
                                                  l=lo_en.value,
                                                  u=hi_en.value,
                                                  rb=re_bin,
                                                  xs=x_size,
                                                  ys=y_size,
                                                  c=centre_pos))

                    # Add the final full output path for the image to the final_paths list
                    final_paths.append(os.path.join(final_dest_dir, im_name))
                    # And any necessary extra information for the class instantiation
                    extra_info.append({"lo_en": lo_en,
                                       "hi_en": hi_en,
                                       "obs_id": cur_ev.obs_id,
                                       "instrument": cur_ev.instrument,
                                       "telescope": er_miss})

            else:
                # We try to retrieve the combined image we've been asked to make, and
                #  if an exception is raised, we know we will have to generate it.
                try:
                    im_exists = source.get_combined_images(lo_en=lo_en,
                                                           hi_en=hi_en,
                                                           telescope=er_miss)
                    # If we get here, then the image already exists, and we can move
                    #  to the next source
                    continue
                except NoProductAvailableError:
                    pass

                # For a combined image, we need a combined event list. They should
                #  have already been generated by the call to 'evtool_combine_evts'
                #  in the first part of this function
                cur_comb_ev = source.get_products("combined_events", telescope=er_miss)[0]
                # Determine image centers and boundaries using an internal function
                re_bin, x_size, y_size, centre_pos = _img_params_from_evtlist(cur_comb_ev)

                # Now we set up the final output directory for the output COMBINED
                #  image to be generated
                final_dest_dir = os.path.join(OUTPUT, er_miss, 'combined')

                # Generate the random identifier for the working directory
                rand_ident = randint(0, 100_000_000)
                dest_dir = os.path.join(final_dest_dir,
                                        "temp_evtool_{}".format(rand_ident))
                # Make the working directory
                os.makedirs(dest_dir, exist_ok=True)

                # eSASS command line tools can have issues with overlong input
                #  file paths (see closed issue #1400). So we set up a symlink
                #  to the combined event list in the working directory
                comb_evt_symlink_name = os.path.basename(cur_comb_ev.path)
                os.symlink(cur_comb_ev.path, os.path.join(dest_dir, comb_evt_symlink_name))

                # Set up the name of the output combined image file
                comb_im_name = "{r}_{l}-{u}keVimg.fits".format(r=rand_ident,
                                                               l=lo_en.value,
                                                               u=hi_en.value)

                # Make the command to create the combined image
                cmds.append(evtool_cmd.format(d=dest_dir,
                                              e=comb_evt_symlink_name,
                                              i=comb_im_name,
                                              l=lo_en.value,
                                              u=hi_en.value,
                                              rb=re_bin,
                                              xs=x_size,
                                              ys=y_size,
                                              c=centre_pos))

                # Add the full path to the output combined image to the final_paths list
                final_paths.append(os.path.join(final_dest_dir, comb_im_name))
                # Then fill in the extra_info
                extra_info.append({"lo_en": lo_en,
                                   "hi_en": hi_en,
                                   "obs_id": 'combined',
                                   "instrument": 'combined',
                                   "telescope": er_miss})

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the eSASS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="image"))

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
    :param bool combine_obs: Setting this to False will generate an exposure map for each associated observation,
        instead of for one combined observation.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    """

    # To generate exposure maps, we need reference images, so we'll call the XGA
    #  function that generates from for eROSITA.
    # This has an added benefit, as evtool_image performs the validity checks on
    #  input parameters that this function would have required as well.
    sources = evtool_image(sources, lo_en, hi_en, combine_obs, num_cores)
    # This is necessary because the decorator will reduce a one element list of source objects to a single
    # source object. Useful for the user, not so much here where the code expects an iterable.
    if not isinstance(sources, (list, BaseSample)):
        sources = [sources]

    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    # This is the template for the evtool command that will be run to make new images
    # evtool_cmd = "cd {d}; evtool eventfiles={e} outfile={i} image=yes " \
    #              "emin={l} emax={u} events=no size='{xs} {ys}' rebin={rb} " \
    #              "center_position='{c}'; mv * ../; cd ..; rm -r {d}"

    # This is the template for the expmap command that will be
    #  run to make new exposure maps
    expmap_cmd = "cd {d}; expmap inputdatasets={e} templateimage={im} " \
                 "emin={l} emax={u} mergedmaps={em} withweights=yes " \
                 "withdetmaps=yes; export HEADASNOQUERY=; " \
                 "export HEADASPROMPT=/dev/null; fthedit {em} " \
                 "REFYCRVL delete; find . -maxdepth 1 -type f -exec mv {{}} ../ \\;; cd ..; rm -r {d}"

    # These lists are to contain the lists of commands/paths/etc for each of the individual sources passed
    # to this function
    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for source in sources:
        source: BaseSource
        cmds = []
        final_paths = []
        extra_info = []

        # By this point we know that at least one of the sources has eROSITA data associated (we checked that at the
        #  beginning of this function). We still need to append the empty cmds, paths, extra_info, and ptypes to
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occuring in the esass_call wrapper
        if 'erosita' not in source.telescopes and 'erass' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the eSASS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="expmap"))

            # We can go to the next source
            continue

        for er_miss in ['erosita', 'erass']:
            # Skip this iteration if the current skew of eROSITA isn't associated
            #  with the current source
            if er_miss not in source.telescopes:
                continue

            # Define this variable for each iteration no it doesn't get overwritten with the wrong value
            # later in the loop
            use_combine_obs = combine_obs

            # if the user has set combine_obs to True and there is only one observation, then we
            # use the combine_obs = False functionality instead
            if use_combine_obs and len(source.obs_ids[er_miss]) == 1:
                use_combine_obs = False

            if not use_combine_obs:
                # Fetch the individual skytile event lists
                rel_skytile_evts = source.get_products("events", telescope=er_miss)

                # Iterating through the event lists of separate skytiles, as we
                #  are making one image per skytile in this case
                for cur_ev in rel_skytile_evts:
                    # We try to retrieve the exposure map we've been told to make
                    #  from the source object, if an exception occurs, then it
                    #  already exists
                    try:
                        ex_exists = source.get_expmaps(cur_ev.obs_id,
                                                       cur_ev.instrument,
                                                       lo_en,
                                                       hi_en,
                                                       er_miss,
                                                       )
                        # If we get here, then the exposure map already
                        #  exists, and we can move to the next event list
                        continue
                    except NoProductAvailableError:
                        # If an exception was raised, then we just keep going, as
                        #  an exposure map needs to be generated for this event list in
                        #  the specified energy band
                        pass

                    # Generating an exposure map requires a reference image, so we
                    #  retrieve the image that matches the current ObsID, instrument,
                    #  and energy range.
                    # We know it exists because we called the image generation routine
                    #  at the beginning of this function
                    cur_ref_im = source.get_images(cur_ev.obs_id,
                                                   cur_ev.instrument,
                                                   lo_en,
                                                   hi_en,
                                                   telescope=er_miss)

                    # Now we set up the final output directory for the expmap to be
                    #  generated, as well as the temporary working directory (with
                    #  a random identifier in the name to avoid collisions)
                    final_dest_dir = os.path.join(OUTPUT, er_miss, cur_ev.obs_id)

                    # Generate the random identifier for the working directory
                    rand_ident = randint(0, 100_000_000)
                    dest_dir = os.path.join(final_dest_dir,
                                            "temp_expmap_{}".format(rand_ident))
                    # Make the working directory
                    os.makedirs(dest_dir, exist_ok=True)

                    # eSASS command line tools can have issues with overlong input
                    #  file paths (see closed issue #1400). So we set up a symlink
                    #  to the event list in the working directory
                    evt_symlink_name = os.path.basename(cur_ev.path)
                    os.symlink(cur_ev.path, os.path.join(dest_dir, evt_symlink_name))

                    # We also make a symlink to the reference image, for the
                    #  same reasons as the event list
                    ref_im_symlink_name = os.path.basename(cur_ref_im.path)
                    os.symlink(cur_ref_im.path, os.path.join(dest_dir,
                                                             ref_im_symlink_name))

                    # The name of the output expmap file
                    ex_name = "{o}_{i}_{l}-{u}keVexpmap.fits".format(o=cur_ev.obs_id,
                                                                     i=cur_ev.instrument,
                                                                     l=lo_en.value,
                                                                     u=hi_en.value)

                    # Make the command to create the expmap
                    cmds.append(expmap_cmd.format(e=evt_symlink_name,
                                                  im=ref_im_symlink_name,
                                                  l=lo_en.value,
                                                  u=hi_en.value,
                                                  em=ex_name,
                                                  d=dest_dir))

                    # Add the final full output path for the expmap to
                    #  the final_paths list
                    final_paths.append(os.path.join(final_dest_dir, ex_name))
                    # And any necessary extra information for the class instantiation
                    extra_info.append({"lo_en": lo_en,
                                       "hi_en": hi_en,
                                       "obs_id": cur_ev.obs_id,
                                       "instrument": cur_ev.instrument,
                                       "telescope": er_miss})

            else:
                # We try to retrieve the combined exposure map we've been asked to
                #  make, and if an exception is raised, we know we will
                #  have to generate it.
                try:
                    ex_exists = source.get_combined_expmaps(lo_en, hi_en, telescope=er_miss)
                    # If we get here, then the exposure map already exists, and
                    #  we can move to the next source
                    continue
                except NoProductAvailableError:
                    pass

                # For a combined exposure map, we need a combined event list. They should
                #  have already been generated by the call to 'evtool_combine_evts'
                #  in the first part of this function
                cur_comb_ev = source.get_products("combined_events",
                                                  telescope=er_miss)[0]

                # We need a combined image as a reference to make a new exposure map
                cur_comb_ref_im = source.get_combined_images(lo_en,
                                                             hi_en,
                                                             telescope=er_miss)

                # Now we set up the final output directory for the output COMBINED
                #  image to be generated
                final_dest_dir = os.path.join(OUTPUT, er_miss, 'combined')

                # Generate the random identifier for the working directory
                rand_ident = randint(0, 100_000_000)
                dest_dir = os.path.join(final_dest_dir,
                                        "temp_expmap_{}".format(rand_ident))
                # Make the working directory
                os.makedirs(dest_dir, exist_ok=True)

                # eSASS command line tools can have issues with overlong input
                #  file paths (see closed issue #1400). So we set up a symlink
                #  to the combined event list in the working directory
                comb_evt_symlink_name = os.path.basename(cur_comb_ev.path)
                os.symlink(cur_comb_ev.path, os.path.join(dest_dir, comb_evt_symlink_name))

                # Also make a symlink for the combined reference image
                comb_ref_im_symlink_name = os.path.basename(cur_comb_ref_im.path)
                os.symlink(cur_comb_ref_im.path, os.path.join(dest_dir,
                                                              comb_ref_im_symlink_name))

                # Set up the name of the output combined image file
                comb_ex_name = "{r}_{l}-{u}keVexpmap.fits".format(r=rand_ident,
                                                                  l=lo_en.value,
                                                                  u=hi_en.value)

                # Make the command to create the ned combined expmap
                cmds.append(expmap_cmd.format(e=comb_evt_symlink_name,
                                              im=comb_ref_im_symlink_name,
                                              l=lo_en.value,
                                              u=hi_en.value,
                                              em=comb_ex_name,
                                              d=dest_dir))

                # Add the full path to the output combined exposure map to
                #  the final_paths list
                final_paths.append(os.path.join(final_dest_dir, comb_ex_name))
                # Then fill in the extra_info
                extra_info.append({"lo_en": lo_en,
                                   "hi_en": hi_en,
                                   "obs_id": 'combined',
                                   "instrument": 'combined',
                                   "telescope": er_miss})

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the eSASS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="expmap"))

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


def combine_phot_prod(sources: Union[BaseSource, BaseSample],
                      to_combine: str,
                      lo_en: Quantity = Quantity(0.2, 'keV'),
                      hi_en: Quantity = Quantity(10, 'keV'),
                      num_cores: int = NUM_CORES,
                      disable_progress: bool = False):
    """
    A convenient Python wrapper for the eSASS evtool and expmap commands. Combined
    images or exposure maps will be generated from all skytiles associated with the
    source (duplicate events are removed).

    This only exists as an analogy to XGA-XMM's 'emosaic' function, you could
    just as easily use xga.generate.esass.phot.evtool_image/expmap directly, with
    'combine_obs=True' as an argument.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or sample of sources
    :param str to_combine: The data type to produce, can be either image or expmap.
    :param Quantity lo_en: The lower energy limit for the image or expmap, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the image or expmap, in astropy energy units.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress
        bar.
    """
    if to_combine not in ["image", "expmap"]:
        raise ValueError("The 'to_combine' argument accepts either 'image' or "
                         "'expmap' as a value.")

    if to_combine == "image":
        evtool_image(sources,
                     lo_en,
                     hi_en,
                     combine_obs=True,
                     disable_progress=disable_progress,
                     num_cores=num_cores)
    elif to_combine == "expmap":
        expmap(sources,
               lo_en,
               hi_en,
               combine_obs=True,
               disable_progress=disable_progress,
               num_cores=num_cores)

    