#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/12/26, 7:27 AM. Copyright (c) The Contributors.

import os
from copy import deepcopy, copy
from random import randint
from typing import Union, List
from warnings import warn

import numpy as np
from astropy.units import Quantity

from ._common import EROSITA_EXTMAP_LO_EN, EROSITA_EXTMAP_HI_EN, T_STEP_POINT, T_STEP_SURVEY
from .misc import evtool_combine_evts
from .phot import evtool_image
from .run import esass_call
from ..common import get_annular_esass_region
from ..sas._common import region_setup
from ... import OUTPUT, NUM_CORES, ESASS_VERSION
from ...exceptions import eSASSInputInvalid, NoProductAvailableError, \
    TelescopeNotAssociatedError
from ...products.misc import EventList
from ...samples.base import BaseSample
from ...sources import BaseSource, ExtendedSource, GalaxyCluster


def _spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True, min_counts: int = 5,
               min_sn: float = None, num_cores: int = NUM_CORES, disable_progress: bool = False,
               combine_tm: bool = True, combine_obs: bool = True, force_gen: bool = False,
               telescope: str = None):
    """
    An internal function to generate all the commands necessary to produce a srctool spectrum, but is not
    decorated by the esass_call function, so the commands aren't immediately run. This means it can be used for
    srctool functions that generate custom sets of spectra (like a set of annular spectra for instance), as well
    as for things like the standard srctool_spectrum function which produce 'global' spectra. Each spectrum
    generated is accompanied by a background spectrum, as well as the necessary ancillary files.

    NOTE: We do yet allow the user to specify their desired values for 'tstep' and 'xgrid', though this will be
    supported in a future release. We currently set default value of 'tstep=0.5' for survey observations, and
    'tstep=100.0' for pointed observations; the 'tstep' values used for background spectrum generation are four
    times larger. The default evtool value for 'xgrid' is used.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default, this is zero arcseconds, resulting in a circular spectrum.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param int min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise, set this parameter to None.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    :param bool combine_tm: Create spectra for individual ObsIDs that are a combination of the data from all the
        telescope modules used for that ObsID. This can help to offset the low signal-to-noise nature of the
        survey data eROSITA takes. Default is True.
    :param bool combine_obs: Setting this to False will generate an spectrum for each associated observation,
        instead of for one combined observation.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    :param str telescope: The telescope to make spectral generation commands for (either 'erass', 'erosita', or None). If None, then all available
        eROSITA telescopes will be used.
    """

    def _make_spec_cmd_info(cur_evt_list: EventList):
        """
        Internal function to get prepare the commands required to generate spectra
        from eROSITA observations. This functions acts on a single event list at a time.
        Output final file paths and extra information dictionaries are also
        created and returned.

        :param EventList cur_evt_list: The event list to generate spectra from.
        :return: A list of spectral generation commands, a list of final spectrum file
            paths, and a list of extra information dictionaries.
        :rtype: Tuple[List[str], List[str], List[str]]
        """

        # Then we have to account for the two different modes this function can be used in - generating spectra
        #  for individual telescope models, or generating a single stacked spectrum for all telescope modules
        if combine_tm:
            inst_names = ['combined']
            inst_nums = ['"' + ' '.join([tm[-1] for tm in list(source.num_inst_obs[cur_evt_list.telescope].keys())]) + '"']
            inst_srctool_id = ['0']
        else:
            inst_names = deepcopy(list(source.num_inst_obs[cur_evt_list.telescope].keys()))
            inst_nums = [tm[-1] for tm in list(source.num_inst_obs[cur_evt_list.telescope].keys())]
            inst_srctool_id = inst_nums

        # Mirroring some of the variables in the name space above, this is where we
        #  will store commands, final paths, etc. that originate from the
        #  currently passed event lists. They will be passed out of this function, and
        #  will always be a list even for a 'combined' generation where only one
        #  iteration of the for loop will occur
        cur_cmds = []
        cur_fin_paths = []
        cur_ex_info = []

        for inst_ind, inst in enumerate(inst_names):
            # Extracting just the instrument number for later use in eSASS commands (or indeed a list of instrument
            #  numbers if the user has requested a combined spectrum).
            inst_no = inst_nums[inst_ind]
            # Also pick out the current instrument srctool ID - this will be passed
            #  to the writeinsts argument of srctool. It will be identical to 'inst_no'
            #  for individual telescope module spectra, and will be zero (to write a
            #  combined spectrum of all specified TMs) for combined telescope
            #  module spectra.
            cur_inst_srctool_id = inst_srctool_id[inst_ind]

            try:
                if use_combine_obs and (len(source.obs_ids[cur_evt_list.telescope]) > 1):
                    check_sp = source.get_combined_spectra(outer_radii[s_ind],
                                                           inst,
                                                           inner_radii[s_ind],
                                                           group_spec,
                                                           min_counts,
                                                           min_sn,
                                                           telescope=cur_evt_list.telescope)

                else:
                    # Got to check if this spectrum already exists
                    check_sp = source.get_spectra(outer_radii[s_ind],
                                                  cur_evt_list.obs_id,
                                                  inst,
                                                  inner_radii[s_ind],
                                                  group_spec,
                                                  min_counts,
                                                  min_sn,
                                                  telescope=cur_evt_list.telescope)
                exists = True

            except NoProductAvailableError:
                exists = False

            if exists and check_sp.usable and not force_gen:
                continue

            # eROSITA observations have the potential to be in pointed or survey modes - we change the time step
            #  based on that. We suspect that the time step is almost irrelevant for pointed mode observations, as
            #  the pointing of the spacecraft won't be changing appreciably
            if cur_evt_list.header['OBS_MODE'] == 'POINTING':
                t_step = t_step_point
            elif cur_evt_list.header['OBS_MODE'] == 'SURVEY':
                t_step = t_step_survey
            else:
                warn("XGA does not recognise the eROSITA OBS_MODE '{om}' - the timestep is defaulting to the "
                     "survey mode value ({ts})".format(om=cur_evt_list.header['OBS_MODE'], ts=t_step_survey),
                     stacklevel=2)
                t_step = t_step_survey

            # The following code will set up the path and file names for output files,
            #  with the first step involving the creation of temporary directories.
            # First, we define the path to the directory where our generated products
            #  are going to live at the end of the process.
            final_dest_dir = os.path.join(OUTPUT, cur_evt_list.telescope,
                                          cur_evt_list.obs_id)

            # Generate a random number to use as a unique addition to the working
            #  directory name - ensures there won't be any clashes
            rand_ident = randint(0, 100_000_000)
            # This is the name of the working directory for the generation process
            ddir_name = "temp_srctool_{i}_{r}".format(i=inst, r=rand_ident)
            # And the full path
            dest_dir = os.path.join(final_dest_dir, ddir_name)

            # The temporary directory is made, and we also set up a symlink to the relevant event list - this is
            #  to help fix issue #1400. We found that event list paths over 204 characters long cause errors when
            #  trying to generate spectra for eROSITA (eSASS4DR1)
            os.makedirs(dest_dir, exist_ok=True)
            # The temporary directory will now have a symlink to the relevant event list, with the symlink name
            #  the same as the actual event list
            evt_symlink_name = os.path.basename(cur_evt_list.path)
            os.symlink(cur_evt_list.path, os.path.join(dest_dir, evt_symlink_name))

            # This constructs the eSASS strings/region files
            reg = get_annular_esass_region(source, src_inn_rad, src_out_rad, cur_evt_list.obs_id, er_miss,
                                           interloper_regions=interloper_regions, central_coord=source.default_coord,
                                           rand_ident=rand_ident, out_root_path=final_dest_dir)
            b_reg = get_annular_esass_region(source, bck_inn_rad, bck_out_rad, cur_evt_list.obs_id, er_miss,
                                             interloper_regions=back_inter_reg, central_coord=source.default_coord,
                                             bkg_reg=True, rand_ident=rand_ident, out_root_path=final_dest_dir)

            # Set up a string describing the central coordinate in addition to the regions
            coord_str = "icrs;{ra},{dec}".format(ra=source.default_coord[0].value,
                                                 dec=source.default_coord[1].value)

            # TODO implement the detector map
            # This creates a detection map for the source and background region
            # map_path = _det_map_creation(outer_radii[s_ind], source, cur_evt_list.obs_id, inst)

            # Setting up file names that include the extra variables
            if group_spec and min_counts is not None:
                extra_file_name = "_mincnt{c}".format(c=min_counts)
            else:
                extra_file_name = ''

            # Cannot control the naming of spectra from srctool. However, we will
            #  define our desired output file names here and move the files output by
            #  srctool to the correct file names at the end of the process.
            # The names of output files will be different depending on if they
            #  are from a combined event list.
            if use_combine_obs and (len(source.obs_ids[cur_evt_list.telescope]) > 1):
                prefix = str(rand_ident) + '_{i}_{n}_'.format(i=inst, n=source.name)
            else:
                prefix = "{o}_{i}_{n}_".format(o=cur_evt_list.obs_id,
                                               i=inst,
                                               n=source.name)

            # Format for the name of the spectrum file
            spec_str = prefix + "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_spec.fits"

            # Naming the non-grouped spectra
            no_grp_spec_str = "ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_spec_not_grouped.fits"
            no_grp_spec_str = prefix + no_grp_spec_str
            no_grp_spec = no_grp_spec_str.format(ra=source.default_coord[0].value,
                                                 dec=source.default_coord[1].value,
                                                 ri=src_inn_rad.value,
                                                 ro=src_out_rad.value,
                                                 ex=extra_file_name)

            # Creating the XGA formatted names we'll move srctool outputs to
            spec = spec_str.format(ra=source.default_coord[0].value,
                                   dec=source.default_coord[1].value,
                                   ri=src_inn_rad.value,
                                   ro=src_out_rad.value,
                                   ex=extra_file_name,
                                   gr=group_spec)

            # The RMF and ARF files can be named by replacing the '_spec.fits' part of
            #  the grouped spectrum file name with '.rmf' and '.arf' respectively.
            rmf = spec.replace("_spec.fits", '.rmf')
            arf = spec.replace("_spec.fits", '.arf')

            # The background spectrum file name can also be constructed by replacing
            #  part of the source spectrum file name. This is possible because the
            #  inner and outer radii in the background spec file name match the
            #  source region size, rather than the background region size
            b_spec = spec.replace("_spec.fits", "_backspec.fits")

            # Then the ARF/RMF file paths for the backspec are modified versions of
            #  the background spectrum file path
            b_rmf = b_spec.replace("_backspec.fits", "_back.rmf")
            b_arf = b_spec.replace("_backspec.fits", "_back.arf")

            # Defining the grouping keywords
            if group_spec and min_counts is not None:
                group_type = 'min'
                group_scale = min_counts
            elif group_spec and min_sn is not None:
                group_type = 'snmin'
                group_scale = min_sn
            else:
                group_type = ''
                group_scale = ''

            # Fills out the srctool command to make the main and background spectra
            # WE NOTE THAT, BECAUSE WE CREATE EVENT LIST SYMLINKS TO SOLVE ISSUE #1400, THE EVENT LIST PATHS
            #  ARE JUST THE BASE FILENAME OF THE EVENT LIST
            if extended_src:
                # For extended source we're going to pass an image to act as an extent map, and
                #  now have to retrieve the relevant image path.
                # We made sure that the images we need have been generated in the
                #  setup steps of the _spec_cmds function
                if use_combine_obs and (len(source.obs_ids[evt_list.telescope]) > 1):
                    # WE CURRENTLY ALWAYS USE THE COMBINED OBS COMBINED INST IMAGE AS
                    #  THE EXTENT MAP (inst="combined"), EVEN IF WE ARE MAKING SPECTRA
                    #  THAT HAVE OBS COMBINED AND TMs SEPARATE.
                    im = source.get_combined_images(lo_en=EROSITA_EXTMAP_LO_EN,
                                                    hi_en=EROSITA_EXTMAP_HI_EN,
                                                    telescope=evt_list.telescope,
                                                    inst="combined")
                else:
                    # We only need the image path for extended source generation
                    # ALSO ALWAYS USED COMBINED FOR INST HERE, SEE THE NOTE ABOVE
                    im = source.get_images(cur_evt_list.obs_id,
                                           lo_en=EROSITA_EXTMAP_LO_EN,
                                           hi_en=EROSITA_EXTMAP_HI_EN,
                                           telescope=evt_list.telescope,
                                           inst="combined")

                # As with the paths to the event lists, it is possible that the image
                #  file names will be too long for eSASS' srctool to handle. To
                #  mitigate the potential problem, we make another symlink
                im_symlink_name = os.path.basename(im.path)
                os.symlink(im.path, os.path.join(dest_dir, im_symlink_name))

                # Fill out the spectrum generation (srctool) command specific to extended sources
                s_cmd_str = ext_sp_cmd.format(d=dest_dir,
                                              ef=os.path.basename(cur_evt_list.path),
                                              sc=coord_str,
                                              reg=reg,
                                              i=inst_no,
                                              wi=cur_inst_srctool_id,
                                              ts=t_step,
                                              em=os.path.basename(im.path),
                                              et=ext_type)

            else:
                # We have a slightly different command for extended and point sources
                s_cmd_str = pnt_sp_cmd.format(d=dest_dir,
                                              ef=os.path.basename(cur_evt_list.path),
                                              sc=coord_str,
                                              reg=reg,
                                              i=inst_no,
                                              wi=cur_inst_srctool_id,
                                              ts=t_step)

            # TODO FIGURE OUT WHAT TO DO ABOUT THE TIMESTEP
            sb_cmd_str = back_sp_cmd.format(ef=os.path.basename(cur_evt_list.path),
                                            sc=coord_str,
                                            breg=b_reg,
                                            i=inst_no,
                                            wi=cur_inst_srctool_id,
                                            ts=t_step * 4)
            # Filling out the grouping command
            grp_cmd_str = grp_cmd.format(infi=no_grp_spec,
                                         of=spec,
                                         gt=group_type,
                                         gs=group_scale)

            # The instrument ID we need to rename the output files in this case
            rename_srctool_id = inst_srctool_id[inst_ind]
            # Adding rename commands for all the outputs of srctool
            #  Start with grouped spectra, because that is an optional output and has
            #  to have an if-else
            if group_spec:
                rename_spec = rename_cmd.format(i_no=rename_srctool_id,
                                                type='SourceSpec',
                                                nn=no_grp_spec)
            else:
                rename_spec = rename_cmd.format(i_no=rename_srctool_id,
                                                type='SourceSpec',
                                                nn=spec)
            # Then set up the rename commands for everything else!
            rename_rmf = rename_cmd.format(i_no=rename_srctool_id, type='RMF', nn=rmf)
            rename_arf = rename_cmd.format(i_no=rename_srctool_id, type='ARF', nn=arf)
            rename_b_spec = rename_cmd.format(i_no=rename_srctool_id,
                                              type='SourceSpec',
                                              nn=b_spec)
            rename_b_rmf = rename_cmd.format(i_no=rename_srctool_id,
                                             type='RMF',
                                             nn=b_rmf)
            rename_b_arf = rename_cmd.format(i_no=rename_srctool_id,
                                             type='ARF',
                                             nn=b_arf)

            # TODO I think all of this needs to made clearer, and each command should have a ; on the end by
            #  default perhaps - or at least it should be consistent

            # We make sure to remove the 'merged spectra' output of srctool - which is identical to the
            #  instrument one if we generate for one spectrum at a time. Though only if the user hasn't actually
            #  ASKED for the merged spectrum
            if combine_tm:
                cmd_str = ";".join([s_cmd_str, rename_spec, rename_rmf, rename_arf, remove_all_but_merged_cmd])
            else:
                cmd_str = ";".join([s_cmd_str, rename_spec, rename_rmf, rename_arf, remove_merged_cmd])

            # This currently ensures that there is a ';' divider between these two chunks of commands - hopefully
            #  we'll neaten it up at some point
            cmd_str += ';'

            # Removing the 'merged spectra' output of srctool, for the background spectra in this case
            if combine_tm:
                cmd_str += ";".join([sb_cmd_str, rename_b_spec, rename_b_rmf, rename_b_arf,
                                     remove_all_but_merged_cmd])
            else:
                cmd_str += ";".join([sb_cmd_str, rename_b_spec, rename_b_rmf, rename_b_arf, remove_merged_cmd])

            # If the user wants to group the spectrum, then this command should be added
            if group_spec:
                # This both performs the grouping and deletes the original non-grouped file. A similar effect
                #  could be ensured by turning clobber on for ftgrouppha, but I think this way is safer. That way
                #  if grouping fails, there definitely won't be a file with the name of the grouped spectrum, but
                #  no grouping applied.
                cmd_str += "; " + grp_cmd_str

            # Adds symlink-removal commands - we don't want to be moving them along
            #  with every else in the temporary working directory
            cmd_str += "; rm {esym}".format(esym=evt_symlink_name)
            # Image symlink will only be present if the source is extended
            if extended_src:
                cmd_str += "; rm {esym}".format(esym=im_symlink_name)

            # Depending on the eSASS version, we may have built up a ';;' somewhere in the command string,
            #  which is very bad - this is introduced because some versions of eSASS have a null string
            #  for remove_merged_cmd and remove_all_but_merged_cmd. This is a very lazy solution, but it will
            #  do for now (famous last words)
            cmd_str = cmd_str.replace(";;", ";")

            # Adds clean up commands to move all generated files and remove the temporary directory
            cmd_str += "; find . -maxdepth 1 -type f -exec mv {{}} ../ \\;; cd ..; rm -r {d}".format(d=dest_dir)
            # If temporary region files were made, they will be here
            if os.path.exists(os.path.join(final_dest_dir, 'temp_regs_{i}'.format(i=rand_ident))):
                # Removing this directory
                cmd_str += ";rm -r temp_regs_{i}".format(i=rand_ident)

            # Adds the full command to set of commands originating from the currently
            #  passed event list
            cur_cmds.append(cmd_str)
            # Same for the final output paths, and extra info dictionary
            cur_fin_paths.append(os.path.join(final_dest_dir, spec))
            cur_ex_info.append({"inner_radius": src_inn_rad,
                                "outer_radius": src_out_rad,
                                "rmf_path": os.path.join(final_dest_dir, rmf),
                                "arf_path": os.path.join(final_dest_dir, arf),
                                "b_spec_path": os.path.join(final_dest_dir, b_spec),
                                "b_rmf_path": os.path.join(final_dest_dir, b_rmf),
                                "b_arf_path": os.path.join(final_dest_dir, b_arf),
                                "obs_id": cur_evt_list.obs_id,
                                "instrument": inst,
                                "central_coord": source.default_coord,
                                "grouped": group_spec,
                                "min_counts": min_counts,
                                "min_sn": min_sn,
                                "over_sample": None,
                                "from_region": False,
                                "telescope": cur_evt_list.telescope})
        return cur_cmds, cur_fin_paths, cur_ex_info


    # Early XGA could generate spectra within the detection regions of the source, but
    #  that has been deprecated for quite a while, as it was a bad idea. The generation
    #  of spectra within user-specified regions will be possible, but it will be
    #  implemented differently. The original way was bad because the detection region
    #  could/almost certainly would be different for each observation
    if outer_radius == 'region':
        raise ValueError("The string 'region' is no longer a valid option for "
                         "the 'outer_radius' argument.")

    # TODO This will change in a future release, so that the user can control it - see issue #1113. The definitions
    #  are up the top of the function as a reminder
    # TODO allow user to chose tstep and xgrid
    t_step_survey = T_STEP_SURVEY
    t_step_point = T_STEP_POINT

    # We check to see whether there is an eROSITA entry in the 'telescopes' property. If sources is a Source
    #  object, then that property contains the telescopes associated with that source, and if it is a Sample object
    #  then 'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if eROSITA isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and ('erosita' not in sources.telescopes and 'erass' not in sources.telescopes)) or
            (isinstance(sources, list) and ('erosita' not in sources[0].telescopes and 'erass' not in sources[0].telescopes))):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA spectra cannot be generated.")

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    # In this case the user wants to combine separate sky tiles, so we have to make
    #  sure that combined event lists exist for each source
    if combine_obs:
        # This requires combined event lists - this function will generate them
        evtool_combine_evts(sources, num_cores)

    # TODO edit region_setup to be telescope agnostic
    sources, inner_radii, outer_radii = region_setup(sources,
                                                     outer_radius,
                                                     inner_radius,
                                                     disable_progress,
                                                     obs_id='',
                                                     num_cores=NUM_CORES)

    # Making sure this value is the expected type
    if min_counts is not None:
        min_counts = int(min_counts)
    if min_sn is not None:
        min_sn = float(min_sn)

    # Checking that the user has passed a grouping argument if group_spec is True
    if all([o is not None for o in [min_counts, min_sn]]):
        raise eSASSInputInvalid("Only one grouping option can passed, you can't group both by"
                                " minimum counts AND by minimum signal to noise.")
    # Should also check that the user has passed any sort of grouping argument if they say they want to group
    elif group_spec and all([o is None for o in [min_counts, min_sn]]):
        raise eSASSInputInvalid("If you set group_spec=True, you must supply a grouping option, either min_counts"
                                " or min_sn.")

    # Sets up the extra part of the storage key name depending on if grouping is enabled
    if group_spec and min_counts is not None:
        extra_name = "_mincnt{}".format(min_counts)
    elif group_spec and min_sn is not None:
        extra_name = "_minsn{}".format(min_sn)
    else:
        extra_name = ''

    # eSASS's srctool operates quite differently for extended sources and point sources. We are going to want a
    #  detector map for extended sources to weight the ARF calculation, so here we check the source type
    if isinstance(sources[0], (ExtendedSource, GalaxyCluster)):
        # Set up a boolean value to tell us later on if this is an extended source
        extended_src = True
        # Sets the extent model type to MAP, which means we'll be using an image to encode the emission extent
        ext_type = 'MAP'
        # Ensures that the images we intend to use as extent maps have actually been generated
        evtool_image(sources, EROSITA_EXTMAP_LO_EN, EROSITA_EXTMAP_HI_EN, combine_obs=combine_obs, num_cores=NUM_CORES)
    else:
        # Set up a boolean value to tell us later on if this is a point source
        extended_src = False
        ext_type = 'POINT'

    # TODO PATTERN AND FLAG SELECTION - REALLY NEED TO INCLUDE THAT
    # Defining the various eSASS commands that need to be populated
    # There will be a different command for extended and point sources
    ext_sp_cmd = 'cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" ' \
                 'todo="SPEC ARF RMF" srcreg="{reg}" backreg=NONE tstep={ts} ' \
                 'insts={i} psftype=NONE extmap="{em}" exttype="MAP"'

    # For extended sources, it is best to make a background spectrum with a separate command
    back_sp_cmd = 'srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF" ' \
                  'srcreg="{breg}" backreg=NONE insts={i} ' \
                  'tstep={ts} psftype=NONE'

    # TODO check the point source command in esass with some EDR obs
    pnt_sp_cmd = 'cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" ' \
                 'todo="SPEC ARF RMF" srcreg="{reg}" exttype="POINT" ' \
                 'tstep={ts} insts={i} psftype="2D_PSF"'

    # The DR1 version of eSASS has an additional argument that can be passed to specify
    #  which instruments should be written to output files - we want to be able to
    #  set that to avoid some warnings that clog up the logs
    if ESASS_VERSION == "ESASS4DR1":
        ext_sp_cmd += " writeinsts={wi}"
        back_sp_cmd += " writeinsts={wi}"
        pnt_sp_cmd += " writeinsts={wi}"

        # Null versions of the extra commands set up below for eSASS4EDR
        remove_merged_cmd = ""
        remove_all_but_merged_cmd = ""

    elif ESASS_VERSION == "ESASS4EDR":
        # Command to remove the 'merged' spectra that eSASS4EDR srctool outputs, even when you only
        #  request one instrument
        remove_merged_cmd = 'rm *srctoolout_0*'
        # We also set up a command that will remove all spectra BUT the combined one
        remove_all_but_merged_cmd = "rm *srctoolout_*"

    # You can't control the whole names of srctool outputs, so this renames it to the XGA format
    rename_cmd = 'mv srctoolout_{i_no}??_{type}* {nn}'

    # TODO SHOULD BE ABLE TO REMOVE THIS CHUNK
    # Commands to remove the merged files from eSASS4DR1. Will depend on instrument number. Insts 1, 2, 3, 4, 6 will use
    #   remove merged dr1_8; insts 5 and 7 will use remove_merged_dr1_9. Run in addition to remove_merged_cmd above.
    # Needed because in eSASS4DR1 srctool will output additional files for "TM8" and "TM9" which are the combined
    #   outputs of TMs 1, 2, 3, 4, 6 and TMs 5 & 7 respectively. This was done to supplement TM0 with a combined output
    #   that does not include the light leak affecting TMs 5 & 7.
    # remove_merged_dr1_8 = 'rm *srctoolout_8*'
    # remove_merged_dr1_9 = 'rm *srctoolout_9*'

    # Grouping the spectra will be done using the HEASoft tool 'ftgrouppha' - we make sure to remove the original
    #  ungrouped file. I think maybe that approach is safer than turning clobber on and just having the original
    #  generated file with a name that only truly applies once grouping has been done.
    # The HEASoft environment variables set here ensure that ftgrouppha doesn't try to access the terminal, which
    #  causes 'device not available' errors
    grp_cmd = ('export HEADASNOQUERY=; export HEADASPROMPT=/dev/null; '
               'ftgrouppha infile="{infi}" outfile="{of}" grouptype="{gt}" groupscale="{gs}"; rm {infi}')

    stack = False  # This tells the esass_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for s_ind, source in enumerate(sources):

        source: BaseSource
        cmds = []
        final_paths = []
        extra_info = []

        # By this point we know that at least one of the sources has eROSITA data
        #  associated (we checked that at the beginning of this function).
        #  However, for those sources that don't, we still need to append the empty
        #  cmds, paths, extra_info, and ptypes to the final output, so that the
        #  cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occurring in the esass_call wrapper
        if 'erosita' not in source.telescopes and 'erass' not in source.telescopes:
            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to
            #  instantiate the Spectrum class once the eSASS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))

            # Now we can continue with the rest of the sources
            continue

        for er_miss in (['erosita', 'erass'] if telescope is None else [telescope]):
            # Skip this iteration if the current skew of eROSITA isn't associated
            #  with the current source
            if er_miss not in source.telescopes:
                continue

            # Need to set this so the combine_obs variable doesn't get overwritten
            use_combine_obs = combine_obs
            # if the user has set combine_obs to True and there is only one observation, then we
            # use the combine_obs = False functionality instead
            if use_combine_obs and (len(source.obs_ids[er_miss]) == 1):
                use_combine_obs = False

            # For convenience, we extract the current source's src region radii
            #  from the big array and put them in some variables
            src_inn_rad = inner_radii[s_ind]
            src_out_rad = outer_radii[s_ind]
            # Can now use a source class method to find which 'interloper' (or contaminating) regions
            #  are within the source region defined by the inner and outer radii.
            # This only needs to be run once per source, because it all works in the
            #  RA-Dec system rather than sky or detector pixel coordinates
            interloper_regions = source.regions_within_radii(src_inn_rad,
                                                             src_out_rad,
                                                             er_miss,
                                                             source.default_coord)

            # We repeat the exercise for the background region
            bck_inn_rad = outer_radii[s_ind] * source.background_radius_factors[0]
            bck_out_rad = outer_radii[s_ind] * source.background_radius_factors[1]
            back_inter_reg = source.regions_within_radii(bck_inn_rad,
                                                         bck_out_rad,
                                                         er_miss,
                                                         source.default_coord)

            # The key under which these spectra will be stored
            spec_stor_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}"
            spec_stor_name = spec_stor_name.format(ra=source.default_coord[0].value,
                                                   dec=source.default_coord[1].value,
                                                   ri=src_inn_rad.value,
                                                   ro=src_out_rad.value,
                                                   gr=group_spec)

            # Adds on the extra information about grouping to the storage key
            spec_stor_name += extra_name

            # Now we start to cycle through event lists (or just the one list if there
            #  is only one skytile/observation associated with the source).
            if not use_combine_obs:
                # Cycle through the event lists for each sky tile
                for evt_list in source.get_products("events", telescope=er_miss):
                    # Mainly for code-completion purposes in IDE
                    evt_list: EventList

                    # This internal function uses the evtlist to prepare spec
                    #  commands, final paths, and extra info
                    out_cmd, out_fin_paths, out_ex_info = _make_spec_cmd_info(evt_list)

                    # Add the output commands for the current event list to the overall
                    #  set of commands we've set up
                    cmds += out_cmd
                    # Same deal for the final paths and extra information dicts
                    final_paths += out_fin_paths
                    extra_info += out_ex_info
            else:
                # In this case, the sky tiles are to be combined, so we grab
                #  the combined event list
                evt_list = source.get_products("combined_events", telescope=er_miss)[0]
                # Mainly for code-completion purposes in IDE
                evt_list: EventList

                # This function then uses the evtlist to prepare spec commands, final
                #  paths, and extra info
                out_cmd, out_fin_paths, out_ex_info = _make_spec_cmd_info(evt_list)

                # Add the output commands for the current event list to the overall set
                #  of commands we've set up and do the same for output paths and
                #  extra info
                cmds += out_cmd
                final_paths += out_fin_paths
                extra_info += out_ex_info

        # Turn the overall, final, set of commands, output paths, and extra information
        #  into numpy arrays - this is because it is easier to index and mask them
        #  than it is for lists
        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the eSASS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


# # TODO fix this function to use XGA in built function and I still need to debug
# TODO DAVID - Actually don't think that this is necessary anymore
# def _det_map_creation(outer_radius: Quantity, source: BaseSource, obs_id: str, inst: str,
#                       rot_angle: Quantity = Quantity(0, 'deg')):
#     """
#     Internal function to make detection maps for extended sources, so that they can be corrected for vignetting
#     correctly when spectra are generated in esass.
#     """
#     outer_radius = outer_radius.to('deg')
#
#     # Defining the name of the detection map
#     detmap_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ro{ro}_detmap.fits"
#     detmap_name = detmap_str.format(o=obs_id, i=inst, n=source.name, ra=source.default_coord[0].value,
#                                     dec=source.default_coord[1].value, ro=outer_radius.to_value)
#     detmap_path = OUTPUT + 'erosita/' + obs_id + '/' + detmap_name
#
#     # Checking if an image has already been made
#     en_id = "bound_{l}-{u}".format(l=0.2, u=10)
#     exists = [match for match in source.get_products("image", obs_id, inst, telescope="erosita", just_obj=False)
#                   if en_id in match]
#
#     if len(exists) == 1 and exists[0][-1].useable:
#         img = exists[0][-1]
#     # Generating an image around this region
#     else:
#         evtool_image(source)
#         exists = [match for match in source.get_products("image", obs_id, inst, telescope="erosita", just_obj=False)
#                   if en_id in match]
#         img = exists[0][-1]
#
#     # Converting that image to the detection map needed
#     with fits.open(img.path) as hdul:
#         # Getting rebinning information from the image
#         bin_key_word = 'rebin'
#         process_history = hdul[0].header['SASSHIST']
#         # This creates a pattern to search for within the string of the processing history of the image
#         # the '=(\d+)' includes an equals sign and any numbers that follow
#         pattern = re.compile(fr'\b{re.escape(bin_key_word)}=(\d+)\b')
#         matches = re.finditer(pattern, process_history) # This finds this pattern within the longer string
#
#         # returning the matches from the iter object
#         binnings = []
#         for match in matches:
#             binnings.append(match.group(1))  # using group(1) returns only the numbers, instead of the whole pattern
#
#         img_binning = int(binnings[-1])  # the results of binnings are stored as strings so need to make it an int
#
#         # Defining the WCS of the image
#         # DAVID_QUESTION, the headers are crazy!
#         hdr = hdul[0].header
#         wcs = WCS(naxis=2)
#         wcs.wcs.cdelt = [hdr["CDELT1P"], hdr["CDELT2P"]]
#         wcs.wcs.crpix = [hdr["CRPIX1P"],hdr["CRPIX2P"]]
#         wcs.wcs.crval = [hdr["CRVAL1"], hdr["CRVAL2"]]
#         wcs.wcs.ctype = [hdr["CTYPE1"], hdr["CTYPE2"]]
#
#         if outer_radius.isscalar:
#             # Defining the region first in sky coords
#             radius = outer_radius*source.background_radius_factors[1]
#             centre = SkyCoord(source.default_coord[0], source.default_coord[1], unit='deg', frame='fk5')
#
#             # Converting to image coords
#             # First converting the radius into pixels
#             arcsec_4 = Quantity(4, 'arcsec')  # For a binning of 80, the pixel scale is 4 arcsec
#             in_degrees = arcsec_4.to('deg')
#             conv_factor = in_degrees.value/80  # Defining relationship between binning and pixel scale
#             pix_scale = img_binning*conv_factor  # Finding pix_scale for this image
#             radius = radius.value/pix_scale  # Radius now in pixels
#
#             # Now converting the centre into image coords
#             x, y = wcs.world_to_pixel(centre)
#             x = int(x)
#             y = int(y)
#             centre = [x, y]
#
#             # Then creating a mask in the image over the region
#             img = hdul[0].data
#             y, x = np.ogrid[:img.shape[0], :img.shape[1]]
#             distance = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
#             region_mask = distance <= radius
#             img[region_mask & (img != 0)] = 1
#             img[~region_mask] = 0
#
#             if not os.path.exists(OUTPUT + 'erosita/' + obs_id):
#                 os.mkdir(OUTPUT + 'erosita/' + obs_id)
#
#             hdul.writeto(detmap_path)
#
#         # TODO ellipses!
#         elif not outer_radius.isscalar:
#             raise NotImplementedError("Haven't figured out how to do this with ellipses yet")
#
#     return detmap_path


@esass_call
def srctool_spectrum(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                     inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                     min_counts: int = 5, min_sn: float = None, num_cores: int = NUM_CORES,
                     disable_progress: bool = False, combine_tm: bool = True, combine_obs: bool = True, force_gen: bool = False):
    """
    A wrapper for all the eSASS and Heasoft processes necessary to generate an eROSITA spectrum that can be analysed
    in XSPEC. Every observation associated with this source, and every instrument associated with that
    observation, will have a spectrum generated using the specified outer and inner radii as a boundary. The
    default inner radius is zero, so by default this function will produce circular spectra out to the outer_radius.
    It is possible to generate both grouped and ungrouped spectra using this function, with the degree
    of grouping set by the min_counts and min_sn parameters.

    NOTE: We do yet allow the user to specify their desired values for 'tstep' and 'xgrid', though this will be
    supported in a future release. We currently set default value of 'tstep=0.5' for survey observations, and
    'tstep=100.0' for pointed observations; the 'tstep' values used for background spectrum generation are four
    times larger. The default evtool value for 'xgrid' is used.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default this is zero arcseconds, resulting in a circular spectrum.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    :param bool combine_tm: Create spectra for individual ObsIDs that are a combination of the data from all the
        telescope modules utilized for that ObsID. This can help to offset the low signal-to-noise nature of the
        survey data eROSITA takes. Default is True.
    :param bool combine_obs: Combine observations for multi-ObsID sources to create single combined spectra per
        instrument (or combined instrument if combine_tm=True). Default is True.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    """
    # All the workings of this function are in _spec_cmds so that the annular spectrum set generation function
    #  can also use them
    return _spec_cmds(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn, num_cores, disable_progress,
                      combine_tm, combine_obs, force_gen=force_gen)


# TODO I feel that I could combine this with the original SAS one, seeing as they essentially call existing
#  spectrum generation functions with particular arguments
@esass_call
def esass_spectrum_set(sources: Union[BaseSource, BaseSample], radii: Union[List[Quantity], Quantity],
                       group_spec: bool = True, min_counts: int = 5, min_sn: float = None, num_cores: int = NUM_CORES,
                       force_regen: bool = False, disable_progress: bool = False, combine_tm: bool = False):
    """
    This function can be used to produce 'sets' of XGA Spectrum objects, generated in concentric circular
    annuli, specifically using data from the eROSITA telescope.
    Such spectrum sets can be used to measure projected spectroscopic quantities, or even be de-projected to attempt
    to measure spectroscopic quantities in a three dimensional space.

    NOTE: We do yet allow the user to specify their desired values for 'tstep' and 'xgrid', though this will be
    supported in a future release. We currently set default value of 'tstep=0.5' for survey observations, and
    'tstep=100.0' for pointed observations; the 'tstep' values used for background spectrum generation are four
    times larger. The default evtool value for 'xgrid' is used.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param List[Quantity]/Quantity radii: A list of non-scalar quantities containing the boundary radii of the
        annuli for the sources. A single quantity containing at least three radii may be passed if one source
        is being analysed, but for multiple sources there should be a quantity (with at least three radii), PER
        source.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool force_regen: This will force all the constituent spectra of the set to be regenerated, use this
        if your call to this function was interrupted and an incomplete AnnularSpectrum is being read in.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    :param bool combine_tm: Create annular spectra for individual ObsIDs that are a combination of the data from
        all the telescope modules utilized for that ObsID. This can help to offset the low signal-to-noise nature
        of the survey data eROSITA takes. Default is True.
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

    # If it's a single source I put it into an iterable object (i.e. a list), just for convenience
    if isinstance(sources, BaseSource):
        sources = [sources]
    elif isinstance(sources, list) and not all([isinstance(s, BaseSource) for s in sources]):
        raise TypeError("If a list is passed, each element must be a source.")
    # And the only other option is a BaseSample instance, so if it isn't that then we get angry
    elif not isinstance(sources, (BaseSample, list)):
        raise TypeError("Please only pass source or sample objects for the 'sources' parameter of this function")

    # I just want to make sure that nobody passes anything daft for the radii
    if isinstance(radii, Quantity) and len(sources) != 1:
        raise TypeError("You may only pass a Quantity for the radii parameter if you are only analysing "
                        "one source. You are attempting to generate spectrum sets for {0} sources, so please pass "
                        "a list of {0} non-scalar quantities.".format(len(sources)))
    elif isinstance(radii, Quantity):
        pass
    elif isinstance(radii, (list, np.ndarray)) and len(sources) != len(radii):
        raise ValueError("The list of quantities passed for the radii parameter must be the same length as the "
                         "number of sources which you are analysing.")

    # If we've made it to this point then the radii type is fine, but I want to make sure that radii is a list
    #  of quantities - as expected by the rest of the function
    if isinstance(radii, Quantity):
        radii = [radii]

    # Check that all radii are passed in the units, I could convert them and make sure but I can't
    #  be bothered
    if len(set([r.unit for r in radii])) != 1:
        raise ValueError("Please pass all radii sets in the same units.")

    # I'm also going to check to make sure that every annulus N+1 is further out then annulus N. There is a check
    #  for this in the spec setup function but if I catch it here I can give a more informative error message
    for s_ind, source in enumerate(sources):
        # I'll also check that the quantity passed for the radii isn't scalar, and isn't only two long - that's not
        #  a set of annuli, they should just use evselect_spectrum for that
        cur_rad = radii[s_ind]
        src_name = source.name
        if cur_rad.isscalar:
            raise ValueError("The radii quantity you have passed for {s} only has one value in it, this function is "
                             "for generating a set of multiple annular spectra, I need at least three "
                             "entries.".format(s=src_name))
        elif len(cur_rad) < 3:
            raise ValueError("The radii quantity you have passed for {s} must have at least 3 entries, this "
                             "would generate a set of 2 annular spectra and is the minimum for this "
                             "function.".format(s=src_name))

        # This runs through the radii for this source and makes sure that annulus N+1 is larger than annulus N
        greater_check = [cur_rad[r_ind] < cur_rad[r_ind+1] for r_ind in range(0, len(cur_rad)-1)]
        if not all(greater_check):
            raise ValueError("Not all of the radii passed for {s} are larger than the annulus that "
                             "precedes them.".format(s=src_name))

    # This generates a spectra between the innermost and outmost radii for each source
    innermost_rads = Quantity([r_set[0] for r_set in radii], radii[0].unit)
    outermost_rads = Quantity([r_set[-1] for r_set in radii], radii[0].unit)
    srctool_spectrum(sources, outermost_rads, innermost_rads, group_spec, min_counts, min_sn, num_cores,
                     disable_progress, combine_tm, combine_obs=True)

    # I want to be able to generate all the individual annuli in parallel, but I need them to be associated with
    #  the correct annuli, which is why I have to iterate through the sources and radii

    # These store the final output information needed to run the commands
    all_cmds = []
    all_paths = []
    all_out_types = []
    all_extras = []
    # Iterating through the sources
    for s_ind, source in enumerate(sources):
        # This is where the commands/extra information get concatenated from the different annuli
        src_cmds = np.array([])
        src_paths = np.array([])
        src_out_types = []
        src_extras = np.array([])

        # By this point we know that at least one of the sources has eROSITA data associated (we checked that at the
        #  beginning of this function), we still need to append the empty cmds, paths, extrainfo, and ptypes to 
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occuring in the esass_call wrapper
        if 'erosita' not in source.telescopes and 'erass' not in source.telescopes:
            all_cmds.append(np.array(src_cmds))
            all_paths.append(np.array(src_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the eSASS cmd has run
            all_extras.append(np.array(src_extras))
            all_out_types.append(src_out_types)

            # then we can continue with the rest of the sources
            continue

        # This generates a random integer ID for this set of spectra
        set_id = randint(0, 100_000_000)

        # I want to be sure that this configuration doesn't already exist
        if group_spec and min_counts is not None:
            extra_name = "_mincnt{}".format(min_counts)
        elif group_spec and min_sn is not None:
            extra_name = "_minsn{}".format(min_sn)
        else:
            extra_name = ''

        # Combines the annular radii into a string
        ann_rad_str = "_".join(source.convert_radius(radii[s_ind], 'deg').value.astype(str))
        spec_storage_name = "ra{ra}_dec{dec}_ar{ar}_grp{gr}"
        spec_storage_name = spec_storage_name.format(ra=source.default_coord[0].value,
                                                     dec=source.default_coord[1].value, ar=ann_rad_str, gr=group_spec)

        spec_storage_name += extra_name

        for er_miss in ['erosita', 'erass']:
            if er_miss not in source.telescopes:
                continue

            # This generates a random integer ID for this set of spectra
            set_id = randint(0, 100_000_000)

            try:
                exists = source.get_annular_spectra(radii[s_ind], group_spec, min_counts, min_sn, telescope=er_miss)
                # If it already exists though we don't need to bother generating
                generate_spec = False
            except NoProductAvailableError:
                # If it doesn't exist then we do need to call the spectrum generation function
                generate_spec = True

            if generate_spec or force_regen:
                # Here we run through all the requested annuli for the current source
                for r_ind in range(len(radii[s_ind])-1):
                    # Generate the eSASS commands for the current annulus of the current source, for all observations
                    spec_cmd_out = _spec_cmds(source, radii[s_ind][r_ind+1], radii[s_ind][r_ind], group_spec, min_counts,
                                              min_sn, num_cores, disable_progress, combine_tm, combine_obs=True,
                                              telescope=er_miss)

                    # Read out some of the output into variables to be modified
                    interim_paths = spec_cmd_out[5][0]
                    interim_extras = spec_cmd_out[6][0]
                    interim_cmds = spec_cmd_out[0][0]

                    # Modified paths and commands will be stored in here
                    new_paths = []
                    new_cmds = []
                    for p_ind, p in enumerate(interim_paths):
                        cur_cmd = interim_cmds[p_ind]

                        # Split up the current path, so we only modify the actual file name and not any
                        #  other part of the string
                        split_p = p.split('/')
                        # We add the set and annulus identifiers
                        new_spec = (split_p[-1].replace("_spec.fits", "_ident{si}_{ai}".format(si=set_id, ai=r_ind))
                                    + "_spec.fits")
                        # Not enough just to change the name passed through XGA, it has to be changed in
                        #  the eSASS commands as well
                        cur_cmd = cur_cmd.replace(split_p[-1], new_spec)

                        # Add the new filename back into the split spec file path
                        split_p[-1] = new_spec

                        # Add an annulus identifier to the extra_info dictionary
                        interim_extras[p_ind].update({"set_ident": set_id, "ann_ident": r_ind})

                        # Only need to modify the RMF paths if the universal RMF HASN'T been used
                        if "universal" not in interim_extras[p_ind]['rmf_path']:
                            # Much the same process as with the spectrum name
                            split_r = copy(interim_extras[p_ind]['rmf_path']).split('/')
                            # split_br = copy(interim_extras[p_ind]['b_rmf_path']).split('/')
                            new_rmf = split_r[-1].replace('.rmf', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) + ".rmf"
                            # new_b_rmf = split_br[-1].replace('_back.rmf', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) \
                            #             + "_back.rmf"

                            # Replacing the names in the eSASS commands
                            cur_cmd = cur_cmd.replace(split_r[-1], new_rmf)
                            # cur_cmd = cur_cmd.replace(split_br[-1], new_b_rmf)

                            split_r[-1] = new_rmf
                            # split_br[-1] = new_b_rmf

                            # Adding the new RMF paths into the extra info dictionary
                            # interim_extras[p_ind].update({"rmf_path": "/".join(split_r),
                            # "b_rmf_path": "/".join(split_br)})
                            interim_extras[p_ind].update({"rmf_path": "/".join(split_r)})

                        # Same process as RMFs but for the ARF, background ARF, and background spec
                        split_a = copy(interim_extras[p_ind]['arf_path']).split('/')
                        # split_ba = copy(interim_extras[p_ind]['b_arf_path']).split('/')
                        split_bs = copy(interim_extras[p_ind]['b_spec_path']).split('/')

                        new_arf = split_a[-1].replace('.arf', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) + ".arf"
                        # new_b_arf = split_ba[-1].replace('_back.arf', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) \
                        #             + "_back.arf"
                        new_b_spec = (split_bs[-1].replace('_backspec.fits', "_ident{si}_{ai}".format(si=set_id, ai=r_ind))
                                      + "_backspec.fits")

                        split_brmf = copy(interim_extras[p_ind]['b_rmf_path']).split('/')
                        split_barf = copy(interim_extras[p_ind]['b_arf_path']).split('/')

                        new_b_rmf = split_brmf[-1].replace('_back.rmf', "_ident{si}_{ai}_back.rmf".format(si=set_id, ai=r_ind))
                        new_b_arf = split_barf[-1].replace('_back.arf', "_ident{si}_{ai}_back.arf".format(si=set_id, ai=r_ind))

                        # New names into the commands
                        cur_cmd = cur_cmd.replace(split_a[-1], new_arf)
                        # cur_cmd = cur_cmd.replace(split_ba[-1], new_b_arf)
                        cur_cmd = cur_cmd.replace(split_bs[-1], new_b_spec)
                        cur_cmd = cur_cmd.replace(split_brmf[-1], new_b_rmf)
                        cur_cmd = cur_cmd.replace(split_barf[-1], new_b_arf)


                        split_a[-1] = new_arf
                        # split_ba[-1] = new_b_arf
                        split_bs[-1] = new_b_spec
                        split_brmf[-1] = new_b_rmf
                        split_barf[-1] = new_b_arf

                        # Update the extra info dictionary some more
                        # interim_extras[p_ind].update({"arf_path": "/".join(split_a), "b_arf_path": "/".join(split_ba),
                        #                               "b_spec_path": "/".join(split_bs)})
                        interim_extras[p_ind].update({"arf_path": "/".join(split_a), "b_spec_path": "/".join(split_bs),
                                                      "b_rmf_path": "/".join(split_brmf), "b_arf_path": "/".join(split_barf)})

                        # Add the new paths and commands to their respective lists
                        new_paths.append("/".join(split_p))
                        new_cmds.append(cur_cmd)

                    src_paths = np.concatenate([src_paths, new_paths])
                    # Go through and concatenate things to the source lists defined above
                    src_cmds = np.concatenate([src_cmds, new_cmds])
                    src_out_types += ['annular spectrum set components'] * len(spec_cmd_out[4][0])
                    src_extras = np.concatenate([src_extras, interim_extras])

        src_out_types = np.array(src_out_types)
        # This adds the current sources final commands to the 'all sources' lists
        all_cmds.append(src_cmds)
        all_paths.append(src_paths)
        all_out_types.append(src_out_types)
        all_extras.append(src_extras)

    # This gets passed back to the esass call function and is used to run the commands
    return all_cmds, False, True, num_cores, all_out_types, all_paths, all_extras, disable_progress
