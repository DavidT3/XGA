#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 14/02/2024, 13:23. Copyright (c) The Contributors

import os
from copy import deepcopy, copy
from random import randint
from typing import Union, List
from warnings import warn
from shutil import rmtree

import numpy as np
from astropy.units import Quantity

from .misc import evtool_combine_evts
from .phot import evtool_image
from .run import esass_call
from ..common import get_annular_esass_region
from ..sas._common import region_setup
from ... import OUTPUT, NUM_CORES
from ...exceptions import eROSITAImplentationError, eSASSInputInvalid, NoProductAvailableError, \
    TelescopeNotAssociatedError
from ...samples.base import BaseSample
from ...sources import BaseSource, ExtendedSource, GalaxyCluster


def _spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True, min_counts: int = 5,
               min_sn: float = None, num_cores: int = NUM_CORES, disable_progress: bool = False,
               combine_tm: bool = True, combine_obs: bool = True, force_gen: bool = False):
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
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    :param bool combine_tm: Create spectra for individual ObsIDs that are a combination of the data from all the
        telescope modules utilized for that ObsID. This can help to offset the low signal-to-noise nature of the
        survey data eROSITA takes. Default is True.
    :param bool combine_obs: Setting this to False will generate an image for each associated observation, 
        instead of for one combined observation.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    """
    def _append_spec_info(evt_list):
        """
        Internal method to get the parameters required for the srctool spectral generation command.
        """
        # extracting the obs_id to use later
        obs_id = evt_list.obs_id

        # Then we have to account for the two different modes this function can be used in - generating spectra
        #  for individual telescope models, or generating a single stacked spectrum for all telescope modules
        if combine_tm:
            inst_names = ['combined']
            inst_nums = ['"' + ' '.join([tm[-1] for tm in list(source.num_inst_obs['erosita'].keys())]) + '"']
            inst_srctool_id = ['0']
        else:
            inst_names = deepcopy(list(source.num_inst_obs['erosita'].keys()))
            inst_nums = [tm[-1] for tm in list(source.num_inst_obs['erosita'].keys())]
            inst_srctool_id = inst_nums

        for inst_ind, inst in enumerate(inst_names):
            # Extracting just the instrument number for later use in eSASS commands (or indeed a list of instrument
            #  numbers if the user has requested a combined spectrum).
            inst_no = inst_nums[inst_ind]

            try:
                if use_combine_obs and (len(source.obs_ids['erosita']) > 1):
                    check_sp = source.get_combined_spectra(outer_radii[s_ind], inst, inner_radii[s_ind], group_spec,
                                            min_counts, min_sn, telescope='erosita')

                else:
                    # Got to check if this spectrum already exists
                    check_sp = source.get_spectra(outer_radii[s_ind], obs_id, inst, inner_radii[s_ind], group_spec,
                                            min_counts, min_sn, telescope='erosita')
                exists = True
                
            except NoProductAvailableError:
                exists = False
            
            if exists and check_sp.usable and not force_gen:
                continue

            # Getting the source name
            source_name = source.name

            # eROSITA observations have the potential to be in pointed or survey modes - we change the time step
            #  based on that. We suspect that the time step is almost irrelevant for pointed mode observations, as
            #  the pointing of the spacecraft won't be changing appreciably
            if evt_list.header['OBS_MODE'] == 'POINTING':
                t_step = t_step_point
            elif evt_list.header['OBS_MODE'] == 'SURVEY':
                t_step = t_step_survey
            else:
                warn("XGA does not recognise the eROSITA OBS_MODE '{om}' - the timestep is defaulting to the "
                    "survey mode value ({ts})".format(om=evt_list.header['OBS_MODE'], ts=t_step_survey),
                    stacklevel=2)
                t_step = t_step_survey

            # setting up file names for output files
            if use_combine_obs and (len(source.obs_ids['erosita']) > 1):
                # The files produced by this function will now be stored in the combined directory.
                final_dest_dir = OUTPUT + "erosita/combined/"
                rand_ident = randint(0, 1e+8)
                # Makes absolutely sure that the random integer hasn't already been used
                while len([f for f in os.listdir(final_dest_dir)
                        if str(rand_ident) in f.split(OUTPUT+"erosita/combined/")[-1]]) != 0:
                    rand_ident = randint(0, 1e+8)

                dest_dir = os.path.join(final_dest_dir, "temp_srctool_{}".format(rand_ident))

            else: 
                # Sets up the file names of the output files, adding a random number so that the
                #  function for generating annular spectra doesn't clash and try to use the same folder
                # The temporary region files necessary to generate eROSITA spectra (if contaminating sources are
                #  being removed) will be written to a different temporary folder using the same random identifier.
                rand_ident = randint(0, 1e+8)
                dest_dir = OUTPUT + "erosita/" + "{o}/{i}_{n}_temp_{r}/".format(o=obs_id, i=inst, n=source_name,
                                                                                r=rand_ident)
            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)

            os.mkdir(dest_dir)
                                                                            
            # If there is no match to a region, the source region returned by this method will be None,
            #  and if the user wants to generate spectra from region files, we have to ignore that observations
            # TODO ASSUMPTION6 source.source_back_regions will have a telescope parameter
            if outer_radius == "region" and source.source_back_regions("erosita", "region", obs_id)[0] is None:
                raise eROSITAImplentationError("XGA for eROSITA does not support the outer_radius='region' "
                                            "argument.")

            # Because the region will be different for each ObsID, I have to call the setup function here
            if outer_radius == 'region':
                raise eROSITAImplentationError("XGA for eROSITA does not support the outer_radius='region' "
                                            "argument.")

            else:
                # This constructs the eSASS strings/region files for any radius that isn't 'region'
                reg = get_annular_esass_region(source, inner_radii[s_ind], outer_radii[s_ind], obs_id,
                                            interloper_regions=interloper_regions,
                                            central_coord=source.default_coord, rand_ident=rand_ident)
                b_reg = get_annular_esass_region(source, outer_radii[s_ind] * source.background_radius_factors[0],
                                                outer_radii[s_ind] * source.background_radius_factors[1], obs_id,
                                                interloper_regions=back_inter_reg,
                                                central_coord=source.default_coord, bkg_reg=True,
                                                rand_ident=rand_ident)
                inn_rad_degrees = inner_radii[s_ind]
                out_rad_degrees = outer_radii[s_ind]

                # TODO implement the detector map
                # This creates a detection map for the source and background region
                # map_path = _det_map_creation(outer_radii[s_ind], source, obs_id, inst)
            
            # Setting up file names that include the extra variables
            if group_spec and min_counts is not None:
                extra_file_name = "_mincnt{c}".format(c=min_counts)
            else:
                extra_file_name = ''

            # Cannot control the naming of spectra from srctool, so need to store
            # the XGA formatting of the spectra, so that they can be renamed 
            # TODO put issue, renaming spectra
            # TODO TIDY THIS UP, THE STORAGE NAMES OF THE SPECTRA ARE ALREADY DEFINED
            # The names of spectra will be different depending on if it is from a combined eventlist
            if use_combine_obs and (len(source.obs_ids['erosita']) > 1):
                prefix = str(rand_ident) + '_' + '{n}_'.format(n=source_name)
            else:
                prefix = "{o}_{i}_{n}_".format(o=obs_id, i=inst, n=source_name)

            spec_str = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_spec.fits"
            spec_str = prefix + spec_str
            rmf_str = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}.rmf"
            rmf_str = prefix + rmf_str
            arf_str = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}.arf"
            arf_str = prefix + arf_str 
            b_spec_str = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_backspec.fits"
            b_spec_str = prefix + b_spec_str
            b_rmf_str = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_backspec.rmf"
            b_rmf_str = prefix + b_rmf_str
            b_arf_str = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_backspec.arf"
            b_arf_str = prefix + b_arf_str

            # Naming the non-grouped spectra
            no_grp_spec_str = "ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_spec_not_grouped.fits"
            no_grp_spec_str = prefix + no_grp_spec_str
            no_grp_spec = no_grp_spec_str.format(ra=source.default_coord[0].value,
                                                dec=source.default_coord[1].value, ri=src_inn_rad_str,
                                                ro=src_out_rad_str, ex=extra_file_name)

            # Making the strings of the XGA formatted names that we will rename the outputs of srctool to
            spec = spec_str.format(ra=source.default_coord[0].value,
                                dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                ex=extra_file_name, gr=group_spec)

            rmf = rmf_str.format(ra=source.default_coord[0].value,
                                dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                ex=extra_file_name, gr=group_spec)
            arf = arf_str.format(ra=source.default_coord[0].value,
                                dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                ex=extra_file_name, gr=group_spec)

            b_spec = b_spec_str.format(ra=source.default_coord[0].value,
                                    dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                    ex=extra_file_name, gr=group_spec)
            b_rmf = b_rmf_str.format(ra=source.default_coord[0].value,
                                    dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                    ex=extra_file_name, gr=group_spec)
            b_arf = b_arf_str.format(ra=source.default_coord[0].value,
                                    dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                    ex=extra_file_name, gr=group_spec)
            
            # These file names are for the debug images of the source and background images, they will not be loaded
            #  in as a XGA products, but exist purely to check by eye if necessary
            dim = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_debug.fits".format(o=obs_id, i=inst, n=source_name,
                                                                                ra=source.default_coord[0].value,
                                                                                dec=source.default_coord[1].value,
                                                                                ri=src_inn_rad_str,
                                                                                ro=src_out_rad_str)
            b_dim = ("{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_"
                    "back_debug.fits").format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                            dec=source.default_coord[1].value, ri=src_inn_rad_str,
                                            ro=src_out_rad_str)

            # TODO ADD MANY MORE COMMENTS
            coord_str = "icrs;{ra}, {dec}".format(ra=source.default_coord[0].value,
                                                dec=source.default_coord[1].value)
            src_reg_str = reg  # dealt with in get_annular_esass_region

            # TODO allow user to chose tstep and xgrid
            bsrc_reg_str = b_reg
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
            if isinstance(source, ExtendedSource):
                try:
                    if use_combine_obs and (len(source.obs_ids['erosita']) > 1):
                        im = source.get_combined_images(lo_en=Quantity(0.2, 'keV'), hi_en=Quantity(10.0, 'keV'), 
                                                        telescope='erosita')
                    else:
                        # We only need the image path for extended source generation 
                        im = source.get_images(obs_id, lo_en=Quantity(0.2, 'keV'), hi_en=Quantity(10.0, 'keV'),
                            telescope='erosita')
                    # We have a slightly different command for extended and point sources
                    s_cmd_str = ext_srctool_cmd.format(d=dest_dir, ef=evt_list.path, sc=coord_str, reg=src_reg_str,
                                                    i=inst_no, ts=t_step, em=im.path, et=et)
                except:
                    raise ValueError(f"it was this sources {source.name}")

            else:
                s_cmd_str = pnt_srctool_cmd.format(d=dest_dir, ef=evt_list.path, sc=coord_str, reg=src_reg_str,
                                                i=inst_no, ts=t_step)

            # TODO FIGURE OUT WHAT TO DO ABOUT THE TIMESTEP
            sb_cmd_str = bckgr_srctool_cmd.format(ef=evt_list.path, sc=coord_str, breg=bsrc_reg_str, 
                                                i=inst_no, ts=t_step*4)
            # Filling out the grouping command
            grp_cmd_str = grp_cmd.format(infi=no_grp_spec, of=spec, gt=group_type, gs=group_scale)

            rename_srctool_id = inst_srctool_id[inst_ind]
            # Occupying the rename command for all the outputs of srctool
            if group_spec:
                rename_spec = rename_cmd.format(i_no=rename_srctool_id, type='SourceSpec', nn=no_grp_spec)
            else:
                rename_spec = rename_cmd.format(i_no=rename_srctool_id, type='SourceSpec', nn=spec)
            rename_rmf = rename_cmd.format(i_no=rename_srctool_id, type='RMF', nn=rmf)
            rename_arf = rename_cmd.format(i_no=rename_srctool_id, type='ARF', nn=arf)
            rename_b_spec = rename_cmd.format(i_no=rename_srctool_id, type='SourceSpec', nn=b_spec)
            rename_b_rmf = rename_cmd.format(i_no=rename_srctool_id, type='RMF', nn=b_rmf)
            rename_b_arf = rename_cmd.format(i_no=rename_srctool_id, type='ARF', nn=b_arf)

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

            # Removing the 'merged spectra' output of srctool, the background in this case
            if combine_tm:
                cmd_str += ";".join([sb_cmd_str, rename_b_spec, rename_b_rmf, rename_b_arf,
                                    remove_all_but_merged_cmd])
            else:
                cmd_str += ";".join([sb_cmd_str, rename_b_spec, rename_b_rmf, rename_b_arf, remove_merged_cmd])

            # If the user wants to group the spectra then this command should be added
            if group_spec:
                # This both performs the grouping, and deletes the original non-grouped file. A similar effect
                #  could be ensured by turning clobber on for ftgrouppha, but I think this way is safer. That way
                #  if grouping fails there definitely won't be a file with the name of the grouped spectrum, but
                #  no grouping applied.
                cmd_str += "; " + grp_cmd_str

            # Adds clean up commands to move all generated files and remove temporary directory
            cmd_str += "; mv * ../; cd ..; rm -r {d}".format(d=dest_dir)
            # If temporary region files were made, they will be here
            if os.path.exists(OUTPUT + 'erosita/' + obs_id + '/temp_regs_{i}'.format(i=rand_ident)):
                # Removing this directory
                cmd_str += ";rm -r temp_regs_{i}".format(i=rand_ident)
            
            cmds.append(cmd_str)  # Adds the full command to the set
            
            final_paths.append(os.path.join(OUTPUT, "erosita", obs_id, spec))
            extra_info.append({"inner_radius": inn_rad_degrees, "outer_radius": out_rad_degrees,
                                    "rmf_path": os.path.join(OUTPUT, "erosita", obs_id, rmf),
                                    "arf_path": os.path.join(OUTPUT, "erosita", obs_id, arf),
                                    "b_spec_path": os.path.join(OUTPUT, "erosita", obs_id, b_spec),
                                    "b_rmf_path": os.path.join(OUTPUT, "erosita", obs_id, b_rmf),
                                    "b_arf_path": os.path.join(OUTPUT, "erosita", obs_id, b_arf),
                                    "obs_id": obs_id, "instrument": inst,
                                    "central_coord": source.default_coord,
                                    "grouped": group_spec,
                                    "min_counts": min_counts,
                                    "min_sn": min_sn,
                                    "over_sample": None,
                                    "from_region": from_region,
                                    "telescope": "erosita"})
            
    # TODO MORE COMMENTS

    # TODO This will change in a future release, so that the user can control it - see issue #1113. The definitions
    #  are up the top of the function as a reminder
    t_step_survey = 0.5
    t_step_point = 100

    # We check to see whether there is an eROSITA entry in the 'telescopes' property. If sources is a Source
    #  object, then that property contains the telescopes associated with that source, and if it is a Sample object
    #  then 'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if eROSITA isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and 'erosita' not in sources.telescopes) or
            (isinstance(sources, list) and 'erosita' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA spectra cannot be generated.")

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]
    
    if combine_obs:
        # This requires combined event lists - this function will generate them
        evtool_combine_evts(sources)

    if outer_radius != 'region':
        from_region = False
        # TODO edit region_setup to be telescope agnostic
        sources, inner_radii, outer_radii = region_setup(sources, outer_radius, inner_radius, disable_progress, '')
    else:
        # This is used in the extra information dictionary for when the XGA spectrum object is defined
        from_region = True

    # Making sure this value is the expected type
    if min_counts is not None:
        min_counts = int(min_counts)
    if min_sn is not None:
        min_sn = float(min_sn)
    
    # Checking user has passed a grouping argument if group spec is true
    if all([o is not None for o in [min_counts, min_sn]]):
        raise eSASSInputInvalid("Only one grouping option can passed, you can't group both by"
                                " minimum counts AND by minimum signal to noise.")
    # Should also check that the user has passed any sort of grouping argument, if they say they want to group
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

    # SASS' srctool operates quite differently for extended sources and point sources. We are going to want a
    #  detector map for extended sources to weight the ARF calculation, so here we check the source type
    if isinstance(sources[0], (ExtendedSource, GalaxyCluster)):
        ex_src = True
        if combine_obs:
            evtool_image(sources, Quantity(0.2, 'keV'), Quantity(10, 'keV'), combine_obs=True, num_cores=NUM_CORES)
        else:
            # TODO Decide if this will stay or not - it might be terribly inefficient using the whole thing for this
            evtool_image(sources, Quantity(0.2, 'keV'), Quantity(10, 'keV'), num_cores=NUM_CORES)
        # Sets the psf type to MAP, which means we'll be using an image to encode the emission extent
        et = 'MAP'
    else:
        ex_src = False
        et = 'POINT'

    # TODO implement the det map EXTTPYE, at the moment this spectrum will treat the target as a point source
    # Defining the various eSASS commands that need to be populated
    # There will be a different command for extended and point sources
    # ext_srctool_cmd = ('cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF" srcreg="{reg}" '
    #                    'backreg=NONE tstep={ts} insts={i} psftype=NONE')

    # TODO PATTERN AND FLAG SELECTION - REALLY NEED TO INCLUDE THAT

    ext_srctool_cmd = ('cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF" srcreg="{reg}" '
                       'backreg=NONE tstep={ts} insts={i} psftype=NONE extmap="{em}" exttype="MAP"')

    # For extended sources, it is best to make a background spectra with a separate command
    bckgr_srctool_cmd = 'srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF"' \
                        ' srcreg="{breg}" backreg=NONE insts={i}' \
                        ' tstep={ts} psftype=NONE'

    # TODO check the point source command in esass with some EDR obs
    pnt_srctool_cmd = 'cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF"' \
                      ' srcreg="{reg}" exttype="POINT" tstep={ts}' \
                      ' insts={i} psftype="2D_PSF"'
    
    # You can't control the whole name of the output of srctool, so this renames it to the XGA format
    rename_cmd = 'mv srctoolout_{i_no}??_{type}* {nn}'
    # Having a string to remove the 'merged' spectra that srctool outputs, even when you only request one instrument
    remove_merged_cmd = 'rm *srctoolout_0*'
    # We also set up a command that will remove all spectra BUT the combined one, for when that is all the user wants
    #  (though honestly it seems wasteful to generate them all and not use them, this might change later
    remove_all_but_merged_cmd = "rm *srctoolout_*"

    # TODO include the background file - YES
    # TODO regroup the background file too 
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
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))
            
            # then we can continue with the rest of the sources
            continue

        # need to set this so the combine_obs variable doesnt get overwritten
        use_combine_obs = combine_obs
        # if the user has set combine_obs to True and there is only one observation, then we
        # use the combine_obs = False functionality instead
        if use_combine_obs and (len(source.obs_ids['erosita']) == 1):
            use_combine_obs = False

        if outer_radius != 'region':
            # Finding interloper regions within the radii we have specified has been put here because it all works in
            #  degrees and as such only needs to be run once for all the different observations.
            # TODO ASSUMPTION8 telescope agnostic version of the regions_within_radii will have telescope argument
            interloper_regions = source.regions_within_radii(inner_radii[s_ind], outer_radii[s_ind], "erosita",
                                                             source.default_coord)
            # This finds any regions which
            # TODO ASSUMPTION8 telescope agnostic version of the regions_within_radii will have telescope argument
            back_inter_reg = source.regions_within_radii(outer_radii[s_ind] * source.background_radius_factors[0],
                                                         outer_radii[s_ind] * source.background_radius_factors[1],
                                                         "erosita", source.default_coord)
            src_inn_rad_str = inner_radii[s_ind].value
            src_out_rad_str = outer_radii[s_ind].value
            # The key under which these spectra will be stored
            spec_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}"
            spec_storage_name = spec_storage_name.format(ra=source.default_coord[0].value,
                                                         dec=source.default_coord[1].value,
                                                         ri=src_inn_rad_str, ro=src_out_rad_str, gr=group_spec)
        else:
            spec_storage_name = "region"
        
        # Adds on the extra information about grouping to the storage key
        spec_storage_name += extra_name

        if not use_combine_obs:
            # Check which event lists are associated with each individual source
            for evt_list in source.get_products("events", telescope='erosita', just_obj=True):
                # This function then uses the evtlist to generate spec commands, final paths, 
                # and extra info, it will then append them to the cmds, final_paths, and extrainfo lists
                # that are defined above
                _append_spec_info(evt_list)

            
        else:
            # getting Eventlist product
            evt_list = source.get_products("combined_events", just_obj=True, telescope="erosita")[0]

            # This function then uses the evtlist to generate spec commands, final paths, 
            # and extra info, it will then append them to the cmds, final_paths, and extrainfo lists
            # that are defined above
            _append_spec_info(evt_list)


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
                     disable_progress: bool = False, combine_tm: bool = True,  combine_obs: bool = True, force_gen: bool = False):
    
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
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    :param bool combine_tm: Create spectra for individual ObsIDs that are a combination of the data from all the
        telescope modules utilized for that ObsID. This can help to offset the low signal-to-noise nature of the
        survey data eROSITA takes. Default is True.
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
    if ((not isinstance(sources, list) and 'erosita' not in sources.telescopes) or
            (isinstance(sources, list) and 'erosita' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA spectra cannot be generated.")

    if combine_tm:
        raise NotImplementedError("We do not yet support stacking of telescope module data for annular spectra.")

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
            raise ValueError("The radii quantity have you passed for {s} must have at least 3 entries, this "
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
                     disable_progress, combine_tm, combine_obs=False)

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
        if 'erosita' not in source.telescopes:
            all_cmds.append(np.array(src_cmds))
            all_paths.append(np.array(src_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the eSASS cmd has run
            all_extras.append(np.array(src_extras))
            all_out_types.append(src_out_types)

            # then we can continue with the rest of the sources
            continue

        # This generates a random integer ID for this set of spectra
        set_id = randint(0, 1e+8)

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

        try:
            exists = source.get_annular_spectra(radii[s_ind], group_spec, min_counts, min_sn, telescope='erosita')
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
                                          min_sn, num_cores, disable_progress, combine_tm, combine_obs=False)

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

                    # New names into the commands
                    cur_cmd = cur_cmd.replace(split_a[-1], new_arf)
                    # cur_cmd = cur_cmd.replace(split_ba[-1], new_b_arf)
                    cur_cmd = cur_cmd.replace(split_bs[-1], new_b_spec)

                    split_a[-1] = new_arf
                    # split_ba[-1] = new_b_arf
                    split_bs[-1] = new_b_spec

                    # Update the extra info dictionary some more
                    # interim_extras[p_ind].update({"arf_path": "/".join(split_a), "b_arf_path": "/".join(split_ba),
                    #                               "b_spec_path": "/".join(split_bs)})
                    interim_extras[p_ind].update({"arf_path": "/".join(split_a), "b_spec_path": "/".join(split_bs)})

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
