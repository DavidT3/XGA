#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by Ray Wang (wangru46@msu.edu) 02/19/2025, 16:16. Copyright (c) The Contributors

import os
from copy import copy
from itertools import permutations
from random import randint
from typing import Union, List

import numpy as np
from astropy.units import Quantity

from xga import OUTPUT, NUM_CORES
from xga.exceptions import NoProductAvailableError, TelescopeNotAssociatedError
from xga.samples.base import BaseSample
from xga.sources import BaseSource, ExtendedSource, GalaxyCluster
from .run import ciao_call


@ciao_call
def _chandra_spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
               min_counts: int = 5, min_sn: float = None, over_sample: int = None, one_rmf: bool = True,
               num_cores: int = NUM_CORES, disable_progress: bool = False, force_gen: bool = False):
    """
    An internal function to generate all the commands necessary to produce a Chandra spectrum using specextract,
    but is not decorated by the ciao_call function, so the commands aren't immediately run.
    
    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default, this is zero arcseconds, resulting in a circular spectrum.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param int over_sample: The minimum energy resolution for each group, set to None to disable.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF per observation.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    """
    
    stack = False  # This tells the ciao_call routine that this command won't be part of a stack.
    execute = True  # This should be executed immediately.
    # This function supports passing both individual sources and sets of sources.
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    # Check if the source/sample has Chandra data.
    if not isinstance(sources, list) and "chandra" not in sources.telescopes:
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the source.")
    elif isinstance(sources, list) and "chandra" not in sources[0].telescopes:
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the sample.")

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

        # Skip sources that do not have Chandra data.
        if 'chandra' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(len(cmds), fill_value="spectrum"))
            continue

        # Process each observation associated with the source.
        for pack in source.get_products("events", just_obj=False, telescope='chandra'):
            obs_id = pack[1]
            inst = pack[2]
            evt_list = pack[-1]
            region_file = source.get_region_file(obs_id)

            # Define output directory and create it if needed.
            dest_dir = os.path.join(OUTPUT, 'chandra', obs_id)
            os.makedirs(dest_dir, exist_ok=True)

            # Create a temporary directory for region files.
            temp_reg_dir = os.path.join(dest_dir, "cent_reg_temp")
            os.makedirs(temp_reg_dir, exist_ok=True)
            reg_file = os.path.join(temp_reg_dir, "cent_temp.reg")
            bkg_reg_file = os.path.join(temp_reg_dir, "bkg_temp.reg")

            # Define region file contents.
            ra = source.default_coord[0].value
            dec = source.default_coord[1].value
            inner_rad = inner_radius.to('arcsec').value
            outer_rad = outer_radius.to('arcsec').value

            reg_content = (
                "# Region file format: DS9\n"
                "physical\n"
                f"annulus({ra},{dec},{inner_rad},{outer_rad})\n"
            )
            
            bkg_reg_content = (
                "# Region file format: DS9\n"
                "physical\n"
                f"annulus({ra},{dec},{outer_rad},{outer_rad + (outer_rad - inner_rad)})"
            )
            
            # Write the region file.
            with open(reg_file, "w") as f:
                f.write(reg_content)

            with open(bkg_reg_file, "w") as f:
                f.write(bkg_reg_content)
            
            # Define file paths for spectrum, response files, and background spectrum.
            spec_file = os.path.join(dest_dir, f"{obs_id}_{inst}_spectrum.pi")
            arf_file = os.path.join(dest_dir, f"{obs_id}_{inst}.arf")
            rmf_file = os.path.join(dest_dir, f"{obs_id}_{inst}.rmf")
            bkg_file = os.path.join(dest_dir, f"{obs_id}_{inst}_background.pi")
            
            # Skip generation if the files already exist.
            if os.path.exists(spec_file) and not force_gen:
                continue
            
            # Construct the specextract command for spectrum extraction.
            # !!! Maybe we need to change to 'weight=no' if we're not interested in the spatial variation of
            # the effect area or for point-source analysis
            # !!! Also, need to consider the binspec (Source spectrum grouping specification) 
            #  and grouptype (Source spectrum grouping type)
            specextract_cmd = (
                f"specextract infile="{evt_list.path}[@{reg_file}]" outroot={dest_dir}/{obs_id}_{inst} "
                f"bkgfile="{evt_list.path}[@{bkg_reg_file}]" clobber=yes weight=no grouptype=NUM_CTS binspec=5"
            )
            cmds.append(specextract_cmd)
            final_paths.append(spec_file)
            
            # Store metadata related to the extracted spectrum.
            extra_info.append({
                "obs_id": obs_id, "instrument": inst, "rmf_path": rmf_file, "arf_path": arf_file,
                "b_spec_path": bkg_file, "grouped": group_spec, "min_counts": min_counts, "min_sn": min_sn,
                "over_sample": over_sample, "telescope": 'chandra'
            })
        
        # Append processed data for each source.
        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(len(cmds), fill_value="spectrum"))
    
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress
