#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 15/02/2024, 15:06. Copyright (c) The Contributors

import os
from shutil import rmtree
from typing import Union

import numpy as np
from astropy.units import Quantity, UnitConversionError

from .run import esass_call
from ... import OUTPUT, NUM_CORES
from ...exceptions import TelescopeNotAssociatedError
from ...samples.base import BaseSample
from ...sources import BaseSource
from ...sources.base import NullSource


@esass_call
def evtool_image(sources: Union[BaseSource, NullSource, BaseSample], lo_en: Quantity = Quantity(0.2, 'keV'),
                 hi_en: Quantity = Quantity(10, 'keV'), num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A convenient Python wrapper for a configuration of the eSASS evtool command that makes images.
    Images will be generated for every observation associated with every source passed to this function.
    If images in the requested energy band are already associated with the source,
    they will not be generated again.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or a sample of sources.
    :param Quantity lo_en: The lower energy limit for the image, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the image, in astropy energy units.
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

    stack = False # This tells the esass_call routine that this command won't be part of a stack
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

        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", telescope='erosita', just_obj=False):
            obs_id = pack[1]
            inst = pack[2]
            if not os.path.exists(OUTPUT + 'erosita/' + obs_id):
                os.mkdir(OUTPUT + 'erosita/' + obs_id)

            en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
            # ASSUMPTION5 source.get_products has a telescope parameter
            exists = [match for match in source.get_products("image", obs_id, inst, just_obj=False, telescope='erosita')
                      if en_id in match]
            if len(exists) == 1 and exists[0][-1].usable:
                continue

            evt_list = pack[-1]

            # Unfortunately, specifying exactly what boundaries to generate images within is a bit difficult for
            #  eROSITA data - primarily because there is some variety in how the data are taken/presented.
            # It is pretty easy to decide for pointed observations - we know the FoV diameter of eROSITA is
            #  1.03 degrees, so we just make a standard size of image that is slightly larger
            if evt_list.header['OBS_MODE'] == "POINTING":
                # The virtual pixel size is 1.3888889095849E-5 degrees - so 87 of them is equal to 4.35 arcseconds,
                #  which is the normal binning for XCS, so that is what I'll start with
                re_bin = 87
                x_size = np.ceil(1.05 / 1.3888889095849e-5 / re_bin).astype(int)
                y_size = x_size

            # If the SKYFIELD is blank, but the OBS_MODE wasn't 'POINTING' then it (I think) can only be a calibration
            #  and performance verification field (i.e. eFEDS or some of the others). Firstly we specify the size for
            #  eFEDS fields - the x and y sizes are based on me experimenting, so take with a pinch of salt
            elif evt_list.header['SKYFIELD'] == "" and 'eFEDS' in evt_list.header["OBJECT"]:
                re_bin = 87
                x_size = np.ceil(7.5 / 1.3888889095849e-5 / re_bin).astype(int)
                y_size = np.ceil(9 / 1.3888889095849e-5 / re_bin).astype(int)

            # This one was based on the size of the eta cha survey field, but I think I am going to leave it as the
            #  catchall for the CalPV fields, at least before I can test them exhaustively
            # TODO revisit this if necessary
            elif evt_list.header['SKYFIELD'] == "":
                re_bin = 87
                x_size = np.ceil(7 / 1.3888889095849e-5 / re_bin).astype(int)
                y_size = np.ceil(7 / 1.3888889095849e-5 / re_bin).astype(int)

            # And if we're here then hopefully we're dealing with an eRASS skytile! In that case we're going to adopt
            #  the standard eRASS size of 3.6x3.6 degrees, but with our own default binning
            else:
                re_bin = 87
                x_size = np.ceil(3.6 / 1.3888889095849e-5 / re_bin).astype(int)
                y_size = x_size

            dest_dir = OUTPUT + "erosita/" + "{o}/{i}_{l}-{u}_{n}_temp/".format(o=obs_id, i=inst, l=lo_en.value,
                                                                                u=hi_en.value, n=source.name)
            im = "{o}_{i}_{l}-{u}keVimg.fits".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)

            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)

            os.makedirs(dest_dir)
            cmds.append("cd {d}; evtool eventfiles={e} outfile={i} image=yes emin={l} emax={u} events=no "
                        "size='{xs} {ys}' rebin={rb} center_position=0; mv * ../; cd ..; "
                        "rm -r {d}".format(d=dest_dir, e=evt_list.path, i=im, l=lo_en.value, u=hi_en.value, rb=re_bin,
                                           xs=x_size, ys=y_size))

            # This is the products final resting place, if it exists at the end of this command
            # ASSUMPTION4 new output directory structure
            final_paths.append(os.path.join(OUTPUT, "erosita", obs_id, im))
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
           hi_en: Quantity = Quantity(10, 'keV'), num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A convenient Python wrapper for the eSASS expmap command.
    Expmaps will be generated for every observation associated with every source passed to this function.
    If expmaps in the requested energy band are already associated with the source,
    they will not be generated again.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or sample of sources.
    :param Quantity lo_en: The lower energy limit for the expmap, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the expmap, in astropy energy units.
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
    sources = evtool_image(sources, lo_en, hi_en)

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
