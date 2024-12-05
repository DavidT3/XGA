#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 01/08/2024, 17:36. Copyright (c) The Contributors

import os
from random import randint
from shutil import rmtree
from typing import Union

import numpy as np
from astropy.units import Quantity, deg
from tqdm import tqdm

from xga import OUTPUT, NUM_CORES
from xga.exceptions import NoProductAvailableError, TelescopeNotAssociatedError
from xga.imagetools import data_limits
from xga.samples.base import BaseSample
from xga.sources import BaseSource
from xga.sources.base import NullSource
from .misc import cifbuild
from .run import ciao_call


@ciao_call
def chandra_image_expmap(sources: Union[BaseSource, NullSource, BaseSample], 
                         lo_en: Quantity = Quantity(0.5, "keV"),
                         hi_en: Quantity = Quantity(2.0, "keV"),
                         # add_expr: str = "",
                         num_cores: int = NUM_CORES,
                         disable_progress: bool = False):
    """
    Generates Chandra images, exposure maps, and rate maps using CIAO's fluximage in rate mode.

    :param Union[BaseSource, NullSource, BaseSample] sources: A source object, null source, or sample of sources.
    :param Quantity lo_en: Lower energy bound (default = 0.5 keV).
    :param Quantity hi_en: Upper energy bound (default = 2.0 keV).
    :param str add_expr: Additional expression to filter data (default = "").
    :param int num_cores: Number of cores for parallel processing (default = NUM_CORES).
    :param bool disable_progress: Disable progress bar if True.
    :return: A tuple with commands, stack, execute, num_cores, product types, paths, metadata, and disable_progress flag.
    """
    stack = False  # This tells the sas_call routine that this command won't be part of a stack.
    execute = True  # This should be executed immediately.
    # This function supports passing both individual sources and sets of sources.
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    # Check if the source/sample has Chandra data.
    if not isinstance(sources, list) and "chandra" not in sources.telescopes:
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the source.")
    elif isinstance(sources, list) and "chandra" not in sources[0].telescopes:
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the sample.")

    # Validate energy bounds.
    if lo_en >= hi_en:
        raise ValueError("The lower energy bound (lo_en) must be less than the upper energy bound (hi_en).")

    # These lists are to contain the lists of commands/paths/etc for each of the individual sources passed
    # to this function.    
    sources_cmds = []
    sources_paths = []
    # This contains any other information that will be needed to instantiate the class
    # once the CIAO cmd has run.
    sources_extras = []
    sources_types = []
    for source in tqdm(sources, disable=disable_progress, desc="Processing sources"):
        cmds = []
        final_paths = []
        extra_info = []
        # Iterate through Chandra event lists associated with the source.
        for product in source.get_products("events", telescope="chandra", just_obj=False):
            obs_id, inst = product[1], product[2]
            evt_file, asol_file, badpix_file = product[-3], product[-2], product[-1]
            
            # Define output filenames.
            image_file = os.path.join(dest_dir, f"{obs_id}_{inst}_{lo_en.value}-{hi_en.value}_image.fits")
            expmap_file = os.path.join(dest_dir, f"{obs_id}_{inst}_{lo_en.value}-{hi_en.value}_expmap.fits")
            ratemap_file = os.path.join(dest_dir, f"{obs_id}_{inst}_{lo_en.value}-{hi_en.value}_ratemap.fits")

            # Skip generation if files already exist.
            if all(os.path.exists(f) for f in [image_file, expmap_file, ratemap_file]):
                continue

            # Check required files.
            if not all(os.path.exists(f) for f in [evt_file, asol_file, badpix_file]):
                raise NoProductAvailableError(f"Missing required files for observation {obs_id}.")
            
            dest_dir = os.path.join(OUTPUT, "chandra", obs_id)
            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)
            os.makedirs(dest_dir)

            # Temporary directory for fluximage.
            temp_dir = os.path.join(dest_dir, f"temp_{randint(0, 1e8)}")
            os.makedirs(temp_dir, exist_ok=True)

            # Build fluximage command.
            flux_cmd = (
                f"cd {temp_dir}; fluximage infile={evt_file}[EVENTS] outroot={obs_id}_{inst} "
                f"bands={lo_en.value}:{hi_en.value}:{(lo_en + hi_en).value / 2} binsize=4 asolfile={asol_file} "
                f"badpixfile={badpix_file} units=time tmpdir={temp_dir} cleanup=yes verbose=4; "
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
