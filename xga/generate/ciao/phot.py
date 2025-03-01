#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 28/02/2025, 19:19. Copyright (c) The Contributors

import os
from random import randint
from shutil import rmtree
from typing import Union

import numpy as np
from astropy.units import Quantity, deg
from tqdm import tqdm
from xga import OUTPUT, NUM_CORES, xga_conf
from xga.exceptions import NoProductAvailableError, TelescopeNotAssociatedError
from xga.imagetools import data_limits
from xga.products import BaseProduct
from xga.samples.base import BaseSample
from xga.sources.base import NullSource

from .run import ciao_call
from ...sources import BaseSource


@ciao_call
def chandra_image_expmap(sources: Union[BaseSource, NullSource, BaseSample], 
                         lo_en: Quantity = Quantity(0.5, "keV"),
                         hi_en: Quantity = Quantity(2.0, "keV"),
                         num_cores: int = NUM_CORES,
                         disable_progress: bool = False):
    """
    Generates Chandra images, exposure maps, and rate maps using CIAO's fluximage in rate mode.

    :param Union[BaseSource, NullSource, BaseSample] sources: A source object, null source, or sample of sources.
    :param Quantity lo_en: Lower energy bound (default = 0.5 keV).
    :param Quantity hi_en: Upper energy bound (default = 2.0 keV).
    :param int num_cores: Number of cores for parallel processing (default = NUM_CORES).
    :param bool disable_progress: Disable progress bar if True.
    :return: A tuple with commands, stack, execute, num_cores, product types, paths, metadata, and disable_progress flag.
    """
    stack = False  # This tells the ciao_call routine that this command won't be part of a stack.
    execute = True  # This should be executed immediately.
    # This function supports passing both individual sources and samples - but if it is a source we like to be able
    #  to iterate over it, so we put it in a list.
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    # Check if the source/sample has Chandra data.
    if not isinstance(sources, list) and "chandra" not in sources.telescopes:
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the source.")
    elif isinstance(sources, list) and "chandra" not in sources[0].telescopes:
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the sample.")

    # Validate energy bounds.
    if not isinstance(lo_en, Quantity) or not isinstance(hi_en, Quantity):
        raise TypeError("The lo_en and hi_en arguments must be astropy quantities in units "
                        "that can be converted to keV.")
    # Have to make sure that the energy bounds are in units that can be converted to keV (which is what evtool
    #  expects for these arguments).
    elif not lo_en.unit.is_equivalent('eV') or not hi_en.unit.is_equivalent('eV'):
        raise UnitConversionError("The lo_en and hi_en arguments must be astropy quantities in units "
                                  "that can be converted to keV.")
    elif lo_en >= hi_en:
        raise ValueError("The lower energy bound ('lo_en') must be less than the upper energy bound ('hi_en').")
    # Converting to the right unit
    else:
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')

    # TODO CHECK THESE ARE GOOD ENERGY RANGES AND THEN UNCOMMENT OR UPDATE THEN UNCOMMENT
    # # Checking user's lo_en and hi_en inputs are in the valid energy range for Chandra
    # if ((lo_en < Quantity(0.5, 'keV') or lo_en > Quantity(7.0, 'keV')) or
    #         (hi_en < Quantity(0.5, 'keV') or hi_en > Quantity(7.0, 'keV'))):
    #     raise ValueError("The 'lo_en' and 'hi_en' value must be between 0.5 keV and 7 keV.")

    # These lists are to contain the lists of commands/paths/etc for each of the individual sources passed
    # to this function.    
    sources_cmds = []
    sources_paths = []
    # This contains any other information that will be needed to instantiate the class
    # once the CIAO cmd has run.
    sources_extras = []
    sources_types = []
    for source in sources:
        # Explicitly states that source is at very least a BaseSource instance - useful for code completion in IDEs
        source: BaseSource

        cmds = []
        final_paths = []
        extra_info = []
        # Iterate through Chandra event lists associated with the source.
        for product in source.get_products("events", telescope="chandra", just_obj=True):
            # Getting the current ObsID, instrument, and event file path
            evt_file = product
            obs_id = evt_file.obs_id
            inst = evt_file.instrument

            # Grabbing the attitude and badpix files, which the CIAO command we aim to run will want as an input
            att_file = source.get_att_file(obs_id, 'chandra')

            # We haven't added a particular get method for bad-pixel files, as they are not as universal as attitude
            #  files - the badpix files have been stored in the source product storage framework however, as we loaded
            #  them along with all the files specified in the config file
            badpix_prod = source.get_products("badpix", telescope="chandra", obs_id=obs_id, inst=inst)
            if len(badpix_prod) > 1:
                raise ValueError("Found multiple bad pixel files for Chandra {o}-{i}; this should not be "
                                 "possible, please contact the developer.".format(o=obs_id, i=inst))
            elif len(badpix_prod) == 0:
                raise ValueError("No bad pixel has been read in for Chandra {o}-{i}, please check the bad pixel path"
                                 " entered in the XGA configuration file - "
                                 "{bpf}".format(o=obs_id, i=inst,
                                                bpf=xga_conf['CHANDRA_FILES']['{i}_badpix_file'.format(i=inst)]))

            # Now we've established that we have retrieved a single bad pixel product, we extract the path from it
            badpix_file = badpix_prod[0].path

            # Setting up the top level path for the eventual destination of the products to be generated here
            dest_dir = os.path.join(OUTPUT, "chandra", obs_id)

            # Define output filenames.
            image_file = os.path.join(dest_dir, f"{obs_id}_{inst}_{lo_en.value}-{hi_en.value}_image.fits")
            expmap_file = os.path.join(dest_dir, f"{obs_id}_{inst}_{lo_en.value}-{hi_en.value}_expmap.fits")
            ratemap_file = os.path.join(dest_dir, f"{obs_id}_{inst}_{lo_en.value}-{hi_en.value}_ratemap.fits")

            try:
                source.get_images(obs_id, inst, lo_en, hi_en, telescope='chandra')
                source.get_expmaps(obs_id, inst, lo_en, hi_en, telescope='chandra')
                source.get_ratemaps(obs_id, inst, lo_en, hi_en, telescope='chandra')
                # If the expected outputs from this function do exist for the current ObsID, we'll just
                #  move on to the next one
                continue
            except NoProductAvailableError:
                pass

            # Skip generation if files already exist.
            # if all(os.path.exists(f) for f in [image_file, expmap_file, ratemap_file]):
            #     continue

            # If something got interrupted and the temp directory still exists, this will remove it
            # if os.path.exists(dest_dir):
            #     rmtree(dest_dir)
            # os.makedirs(dest_dir)

            # Temporary directory for fluximage.
            temp_dir = os.path.join(dest_dir, f"temp_{randint(0, int(1e8))}")
            os.makedirs(temp_dir, exist_ok=True)

            # Build fluximage command - making sure to set parallel to no, seeing as we're doing our
            #  own parallelization
            flux_cmd = (
                f"cd {temp_dir}; fluximage infile={evt_file.path} outroot={obs_id}_{inst} "
                f"bands={lo_en.value}:{hi_en.value}:{(lo_en + hi_en).value / 2} binsize=4 asolfile={att_file} "
                f"badpixfile={badpix_file} units=time tmpdir={temp_dir} cleanup=yes verbose=4 parallel=no; "
                f"mv * {dest_dir}; cd ..; rm -r {temp_dir}"
            )
            cmds.append(flux_cmd)

            # This is the products final resting place, if it exists at the end of this command.
            final_paths.append([image_file, expmap_file, ratemap_file])
            extra_info.append({
                "obs_id": obs_id,
                "instrument": inst,
                "lo_en": lo_en,
                "hi_en": hi_en,
                "effective_energy": (lo_en + hi_en) / 2,
                "bin_size": 4
            })

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the CIAO cmd has run.
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(len(cmds), "image"))

    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress
