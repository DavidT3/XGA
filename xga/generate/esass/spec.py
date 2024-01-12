#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 12/01/2024, 09:47. Copyright (c) The Contributors

import os
import re
from copy import deepcopy
from random import randint
from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS

from .phot import evtool_image
from .run import esass_call
from .._common import get_annular_esass_region
from ... import OUTPUT, NUM_CORES
from ...exceptions import eROSITAImplentationError, eSASSInputInvalid, NoProductAvailableError
from ...samples.base import BaseSample
from ...sas._common import region_setup
from ...sources import BaseSource, ExtendedSource, GalaxyCluster


def _spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True, min_counts: int = 5,
               min_sn: float = None, num_cores: int = NUM_CORES, disable_progress: bool = False,
               combine_tm: bool = True, force_gen: bool = False):
    """
    An internal function to generate all the commands necessary to produce a srctool spectrum, but is not
    decorated by the esass_call function, so the commands aren't immediately run. This means it can be used for
    srctool functions that generate custom sets of spectra (like a set of annular spectra for instance), as well
    as for things like the standard srctool_spectrum function which produce 'global' spectra. Each spectrum
    generated is accompanied by a background spectrum, as well as the necessary ancillary files.

    :param combine_tm:
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
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    """
    # TODO MORE COMMENTS

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

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
        # TODO Decide if this will stay or not - it might be terribly inefficient using the whole thing for this
        evtool_image(sources, Quantity(0.2, 'keV'), Quantity(10, 'keV'), NUM_CORES)
        # Sets the psf type to MAP, which means we'll be using an image to encode the emission extent
        et = 'MAP'
    else:
        ex_src = False
        et = 'POINT'
        raise eROSITAImplentationError("Spectral generation has not yet been implemented for point sources.")

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
    pnt_srctool_cmd = 'cd {d}; srctool eventfiles="{ef}" srcoord="{sc}" todo="SPEC ARF RMF"' \
                      ' srcreg="{reg}" backreg="{breg}" exttype="POINT" tstep={ts}' \
                      ' insts={i} psftype="2D_PSF"'
    
    # You can't control the whole name of the output of srctool, so this renames it to the XGA format
    rename_cmd = 'mv srctoolout_{i_no}??_{type}* {nn}'
    # Having a string to remove the 'merged' spectra that srctool outputs, even when you only request one instrument
    remove_merged_cmd = 'rm *srctoolout_0*'

    # TODO this command is fishy 
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

        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", telescope='erosita', just_obj=False):
            # This one is simple, just extracting the current ObsID
            obs_id = pack[1]

            # Then we have to account for the two different modes this function can be used in - generating spectra
            #  for individual telescope models, or generating a single stacked spectrum for all telescope modules
            if combine_tm:
                inst_names = ['combined']
                inst_nums = [" ".join([tm[-1] for tm in source.instruments["erosita"][obs_id]])]
            else:
                inst_names = deepcopy(source.instruments["erosita"][obs_id])
                inst_nums = [tm[-1] for tm in source.instruments["erosita"][obs_id]]

            print(inst_names)
            print(inst_nums)
            import sys
            sys.exit()

            for inst in source.instruments["erosita"][obs_id]:
                # Extracting just the instrument number for later use in eSASS commands
                inst_no = [s for s in inst if s.isdigit()][0]

                try:
                    # Got to check if this spectrum already exists
                    check_sp = source.get_spectra(outer_radii[s_ind], obs_id, inst, inner_radii[s_ind], group_spec,
                                                  min_counts, min_sn, telescope='erosita')
                    exists = True
                except NoProductAvailableError:
                    exists = False

                if exists and check_sp.usable and not force_gen:
                    continue

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
                    # This constructs the sas strings for any radius that isn't 'region'
                    reg = get_annular_esass_region(source, inner_radii[s_ind], outer_radii[s_ind], obs_id,
                                                   interloper_regions=interloper_regions,
                                                   central_coord=source.default_coord)
                    b_reg = get_annular_esass_region(source, outer_radii[s_ind] * source.background_radius_factors[0],
                                                     outer_radii[s_ind] * source.background_radius_factors[1], obs_id,
                                                     interloper_regions=back_inter_reg,
                                                     central_coord=source.default_coord, bkg_reg=True)
                    inn_rad_degrees = inner_radii[s_ind]
                    out_rad_degrees = outer_radii[s_ind]

                    # TODO implement the detector map
                    # This creates a detection map for the source and background region
                    # map_path = _det_map_creation(outer_radii[s_ind], source, obs_id, inst)
                
                # Getting the source name
                source_name = source.name
                
                # Just grabs the event list object
                evt_list = pack[-1]
                # Sets up the file names of the output files, adding a random number so that the
                #  function for generating annular spectra doesn't clash and try to use the same folder
                # ASSUMPTION4 new output directory structure
                dest_dir = OUTPUT + "erosita/" + "{o}/{i}_{n}_temp_{r}/".format(o=obs_id, i=inst, n=source_name,
                                                                                r=randint(0, 1e+8))

                # Setting up file names that include the extra variables
                if group_spec and min_counts is not None:
                    extra_file_name = "_mincnt{c}".format(c=min_counts)
                else:
                    extra_file_name = ''

                # Cannot control the naming of spectra from srctool, so need to store
                # the XGA formatting of the spectra, so that they can be renamed 
                # TODO put issue, renaming spectra
                # TODO TIDY THIS UP, THE STORAGE NAMES OF THE SPECTRA ARE ALREADY DEFINED
                spec_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_spec.fits"
                rmf_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}.rmf"
                arf_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}.arf"
                b_spec_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_backspec.fits"
                b_rmf_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_backspec.rmf"
                b_arf_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_backspec.arf"

                # Naming the non-grouped spectra
                no_grp_spec_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_spec_not_grouped.fits"
                no_grp_spec = no_grp_spec_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                                     dec=source.default_coord[1].value, ri=src_inn_rad_str,
                                                     ro=src_out_rad_str, ex=extra_file_name)

                # Making the strings of the XGA formatted names that we will rename the outputs of srctool to
                spec = spec_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                       dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                       ex=extra_file_name, gr=group_spec)

                rmf = rmf_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                     dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                     ex=extra_file_name, gr=group_spec)
                arf = arf_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                     dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                     ex=extra_file_name, gr=group_spec)

                b_spec = b_spec_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                           dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                           ex=extra_file_name, gr=group_spec)
                b_rmf = b_rmf_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                         dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                         ex=extra_file_name, gr=group_spec)
                b_arf = b_arf_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
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
                tstep = 0.5  # put it as 0.5 for now
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

                im = source.get_images(obs_id, lo_en=Quantity(0.2, 'keV'), hi_en=Quantity(10.0, 'keV'),
                                       telescope='erosita')
                # Fills out the srctool command to make the main and background spectra
                s_cmd_str = ext_srctool_cmd.format(d=dest_dir, ef=evt_list.path, sc=coord_str, reg=src_reg_str, 
                                                   i=inst_no, ts=tstep, em=im.path, et=et)
                # TODO I've changed this so the timestep for background is the same as source
                sb_cmd_str = bckgr_srctool_cmd.format(ef=evt_list.path, sc=coord_str, breg=bsrc_reg_str, 
                                                      i=inst_no, ts=tstep)
                # Filling out the grouping command
                grp_cmd_str = grp_cmd.format(infi=no_grp_spec, of=spec, gt=group_type, gs=group_scale)

                # Occupying the rename command for all the outputs of srctool
                if group_spec:
                    rename_spec = rename_cmd.format(i_no=inst_no, type='SourceSpec', nn=no_grp_spec)
                else:
                    rename_spec = rename_cmd.format(i_no=inst_no, type='SourceSpec', nn=spec)
                rename_rmf = rename_cmd.format(i_no=inst_no, type='RMF', nn=rmf)
                rename_arf = rename_cmd.format(i_no=inst_no, type='ARF', nn=arf)
                rename_b_spec = rename_cmd.format(i_no=inst_no, type='SourceSpec', nn=b_spec)
                rename_b_rmf = rename_cmd.format(i_no=inst_no, type='RMF', nn=b_rmf)
                rename_b_arf = rename_cmd.format(i_no=inst_no, type='ARF', nn=b_arf)

                cmd_str = ";".join([s_cmd_str, rename_spec, rename_rmf, rename_arf])

                # Removing the 'merged spectra' output of srctool - which is identical to the instrument one if
                #  we generate for one spectrum at a time. Though only if the user hasn't actually ASKED for the
                #  merged spectrum
                if not combine_tm:
                    cmd_str += "; " + remove_merged_cmd + "; "
                
                cmd_str += ";".join([sb_cmd_str, rename_b_spec, rename_b_rmf, rename_b_arf])

                # TODO I WAS IN THE PROCESS OF LETTING THIS FUNCTION CREATE MERGED TM SPECTRA FOR A PARTICULAR OBSID
                #  THIS WILL REQUIRE A LITTLE BIT OF MODIFICATION SO THAT A LIST OF ALL INSTRUMENTS SELECTED FOR THE
                #  CURRENT OBSID FOR THE CURRENT SOURCE IS PASSED TO SOURCETOOL, AND XGA KNOWS THAT ITS A COMBINED
                #  SPECTRUM

                # Removing the 'merged spectra' output of srctool, the background in this case
                if not combine_tm:
                    cmd_str += "; " + remove_merged_cmd

                # If the user wants to group the spectra then this command should be added
                if group_spec:
                    # This both performs the grouping, and deletes the original non-grouped file. A similar effect
                    #  could be ensured by turning clobber on for ftgrouppha, but I think this way is safer. That way
                    #  if grouping fails there definitely won't be a file with the name of the grouped spectrum, but
                    #  no grouping applied.
                    cmd_str += "; " + grp_cmd_str

                # Adds clean up commands to move all generated files and remove temporary directory
                # cmd_str += "; mv * ../; cd ..; rm -r {d}".format(d=dest_dir)
                # If temporary region files were made, they will be here
                if os.path.exists(OUTPUT + 'erosita/' + obs_id + '/temp_regs'):
                    # Removing this directory
                    cmd_str += ";rm -r temp_regs"

                cmds.append(cmd_str)  # Adds the full command to the set
                # Makes sure the whole path to the temporary directory is created
                os.makedirs(dest_dir)

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

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the eSASS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


# TODO fix this function to use XGA in built function and I still need to debug
def _det_map_creation(outer_radius: Quantity, source: BaseSource, obs_id: str, inst: str,
                      rot_angle: Quantity = Quantity(0, 'deg')):
    """
    Internal function to make detection maps for extended sources, so that they can be corrected for vignetting
    correctly when spectra are generated in esass.
    """
    outer_radius = outer_radius.to('deg')
    
    # Defining the name of the detection map
    detmap_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ro{ro}_detmap.fits"
    detmap_name = detmap_str.format(o=obs_id, i=inst, n=source.name, ra=source.default_coord[0].value,
                                    dec=source.default_coord[1].value, ro=outer_radius.to_value)
    detmap_path = OUTPUT + 'erosita/' + obs_id + '/' + detmap_name

    # Checking if an image has already been made
    en_id = "bound_{l}-{u}".format(l=0.2, u=10)
    exists = [match for match in source.get_products("image", obs_id, inst, telescope="erosita", just_obj=False)
                  if en_id in match]

    if len(exists) == 1 and exists[0][-1].useable:
        img = exists[0][-1] 
    # Generating an image around this region
    else:
        evtool_image(source)
        exists = [match for match in source.get_products("image", obs_id, inst, telescope="erosita", just_obj=False)
                  if en_id in match]
        img = exists[0][-1]
        
    # Converting that image to the detection map needed
    with fits.open(img.path) as hdul:
        # Getting rebinning information from the image
        bin_key_word = 'rebin'
        process_history = hdul[0].header['SASSHIST']
        # This creates a pattern to search for within the string of the processing history of the image
        # the '=(\d+)' includes an equals sign and any numbers that follow
        pattern = re.compile(fr'\b{re.escape(bin_key_word)}=(\d+)\b')
        matches = re.finditer(pattern, process_history) # This finds this pattern within the longer string

        # returning the matches from the iter object
        binnings = []
        for match in matches:
            binnings.append(match.group(1))  # using group(1) returns only the numbers, instead of the whole pattern

        img_binning = int(binnings[-1])  # the results of binnings are stored as strings so need to make it an int

        # Defining the WCS of the image
        # DAVID_QUESTION, the headers are crazy!
        hdr = hdul[0].header
        wcs = WCS(naxis=2)
        wcs.wcs.cdelt = [hdr["CDELT1P"], hdr["CDELT2P"]]
        wcs.wcs.crpix = [hdr["CRPIX1P"],hdr["CRPIX2P"]]
        wcs.wcs.crval = [hdr["CRVAL1"], hdr["CRVAL2"]]
        wcs.wcs.ctype = [hdr["CTYPE1"], hdr["CTYPE2"]]

        if outer_radius.isscalar:
            # Defining the region first in sky coords
            radius = outer_radius*source.background_radius_factors[1]
            centre = SkyCoord(source.default_coord[0], source.default_coord[1], unit='deg', frame='fk5')

            # Converting to image coords
            # First converting the radius into pixels
            arcsec_4 = Quantity(4, 'arcsec')  # For a binning of 80, the pixel scale is 4 arcsec
            in_degrees = arcsec_4.to('deg') 
            conv_factor = in_degrees.value/80  # Defining relationship between binning and pixel scale
            pix_scale = img_binning*conv_factor  # Finding pix_scale for this image
            radius = radius.value/pix_scale  # Radius now in pixels

            # Now converting the centre into image coords
            x, y = wcs.world_to_pixel(centre)
            x = int(x)
            y = int(y)
            centre = [x, y]

            # Then creating a mask in the image over the region
            img = hdul[0].data
            y, x = np.ogrid[:img.shape[0], :img.shape[1]]
            distance = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
            region_mask = distance <= radius
            img[region_mask & (img != 0)] = 1
            img[~region_mask] = 0

            if not os.path.exists(OUTPUT + 'erosita/' + obs_id):
                os.mkdir(OUTPUT + 'erosita/' + obs_id)

            hdul.writeto(detmap_path)
        
        # TODO ellipses!
        elif not outer_radius.isscalar:
            raise NotImplementedError("Haven't figured out how to do this with ellipses yet")
    
    return detmap_path


@esass_call
def srctool_spectrum(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                     inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                     min_counts: int = 5, min_sn: float = None, num_cores: int = NUM_CORES,
                     disable_progress: bool = False, combine_tm: bool = True, force_gen: bool = False):
    
    """
    A wrapper for all the eSASS and Heasoft processes necessary to generate an eROSITA spectrum that can be analysed
    in XSPEC. Every observation associated with this source, and every instrument associated with that
    observation, will have a spectrum generated using the specified outer and inner radii as a boundary. The
    default inner radius is zero, so by default this function will produce circular spectra out to the outer_radius.
    It is possible to generate both grouped and ungrouped spectra using this function, with the degree
    of grouping set by the min_counts and min_sn parameters.

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
                      combine_tm, force_gen=force_gen)
