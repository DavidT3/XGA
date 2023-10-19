# This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
# Last modified by Jessica Pilling (jp735@sussex.ac.uk) Wed Oct 11 2023, 13:51. Copyright (c) The Contributors

import os
from typing import Union, List
from random import randint

import numpy as np
from astropy.units import Quantity

from .._common import region_setup

from .. import OUTPUT, NUM_CORES
from ...sources import BaseSource, ExtendedSource, GalaxyCluster
from ...samples.base import BaseSample
from ...exceptions import eROSITAImplentationError

def _spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
               num_cores: int = NUM_CORES, disable_progress: bool = False, force_gen: bool = False):
    """
    An internal function to generate all the commands necessary to produce a srctool spectrum, but is not
    decorated by the esass_call function, so the commands aren't immediately run. This means it can be used for
    srctool functions that generate custom sets of spectra (like a set of annular spectra for instance), as well
    as for things like the standard srctool_spectrum function which produce relatively boring spectra. At the moment 
    each spectra will also generate a background spectra by default. 

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default this is zero arcseconds, resulting in a circular spectrum.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    """
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    if outer_radius != 'region':
        from_region = False
        #TODO the region setup for erosita --> might be the same for both telescopes?
        sources, inner_radii, outer_radii = region_setup(sources, outer_radius, inner_radius, disable_progress, '')
    else:
        # This is used in the extra information dictionary for when the XGA spectrum object is defined
        from_region = True

    # Defining the various eSASS commands that need to be populated
    # There will be a different command for extended and point sources
    ext_srctool_cmd = 'cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF"' \
                ' srcreg="{reg}" backreg={breg} insts="{i}" tstep={ts} xgrid={xg}' \
                ' psftype=NONE'

    #TODO add in separate background command
    # For extended sources, it is best to make a background spectra with a separate command
    #bckgr_srctool_cmd = 'cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF"' \
                       # ' srcreg="{breg}" backreg=NONE insts="{i}" prefix="bckgr_"' \
                       # ' tstep={ts} xgrid={xg} psftype=NONE'

    #TODO check the point source command in esass with some EDR obs
    pnt_srctool_cmd = 'cd {d}; srctool eventfiles="{ef}" srcoord="{sc}" todo="SPEC ARF RMF" insts="{i}"' \
                      ' srcreg="{reg}" backreg="{breg}" exttype="POINT" tstep={ts} xgrid={xg}' \
                      ' psftype="2D_PSF"'
    
    #You can't control the whole name of the output of srctool, so this renames it to the XGA format
    rename_cmd = 'mv srctoolout_{inst}??_{type}* {nn}'
    
    # To correct for vignetting properly, you need a detection map of the source
    #TODO how to make a detection/extent map then add into extended srctool cmd

    stack = False # This tells the esass_call routine that this command won't be part of a stack
    execute = True # This should be executed immediately

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for s_ind, source in enumerate(sources):
        # srctool operates quite differently for extended sources and point sources
        # we also need a detection map for an extended source, so here we check the source type
        if isinstance(source, (ExtendedSource, GalaxyCluster)):
            ex_src = "yes"
            # Sets the detmap type, using an image of the source is appropriate for extended sources like clusters,
            #  but not for point sources
            dt = 'dataset'
        else:
            ex_src = "no"
            dt = 'flat'
            raise eROSITAImplentationError("Spectral Generation has not yet been implemented for point sources.")

        cmds = []
        final_paths = []
        extra_info = []

        if outer_radius != 'region':
            # Finding interloper regions within the radii we have specified has been put here because it all works in
            #  degrees and as such only needs to be run once for all the different observations.
            interloper_regions = source.regions_within_radii(inner_radii[s_ind], outer_radii[s_ind],
                                                             source.default_coord)
            # This finds any regions which
            back_inter_reg = source.regions_within_radii(outer_radii[s_ind] * source.background_radius_factors[0],
                                                         outer_radii[s_ind] * source.background_radius_factors[1],
                                                         source.default_coord)
            src_inn_rad_str = inner_radii[s_ind].value
            src_out_rad_str = outer_radii[s_ind].value
            # The key under which these spectra will be stored
            spec_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}"
            spec_storage_name = spec_storage_name.format(ra=source.default_coord[0].value,
                                                         dec=source.default_coord[1].value,
                                                         ri=src_inn_rad_str, ro=src_out_rad_str)
        else:
            spec_storage_name = "region"

        # Check which event lists are associated with each individual source
         # ASSUMPTION3 output of products is now a dictionary with telescope keys #TODO remove assumption3
        for pack in source.get_products("events", telescope='erosita', just_obj=False):
            #TODO change this index in phot.py
            obs_id = pack[1]
            #ASSUMPTION7 this will be a string of the TMs that are relevant #TODO see below
            inst = pack[2]
            #TODO indent here - produce one spect product per instrument
            inst = source.instruments['erosita'][obs_id]


            # ASSUMPTION4 new output directory structure
            #TODO add slash everywhere!!!!!
            if not os.path.exists(OUTPUT + 'erosita/' + obs_id):
                os.mkdir(OUTPUT + 'erosita' + obs_id)

            # Got to check if this spectrum already exists
            # ASSUMPTION5 source.get_products has a telescope parameter #TODO will break, telescope argument at the end 
            exists = source.get_products("spectrum", "erosita", obs_id, inst, extra_key=spec_storage_name)
            if len(exists) == 1 and exists[0].usable and not force_gen:
                continue

            # If there is no match to a region, the source region returned by this method will be None,
            #  and if the user wants to generate spectra from region files, we have to ignore that observations
            # ASSUMPTION6 source.source_back_regions will have a telescope parameter
            #TODO i outer_radius == "region" throw an error, dont worry about chunks that have that statement
            if outer_radius == "region" and source.source_back_regions("erosita", "region", obs_id)[0] is None:
                continue

            # Because the region will be different for each ObsID, I have to call the setup function here
            if outer_radius == 'region':
                interim_source, inner_radii, outer_radii = region_setup([source], outer_radius, inner_radius,
                                                                        disable_progress, obs_id)
                # Need the reg for central coordinates
                #DAVID_QUESTION assuming this wouldn't need to be changed for multitelescope
                reg = source.source_back_regions('region', obs_id)[0]
                reg_cen_coords = Quantity([reg.center.ra.value, reg.center.dec.value], 'deg')
                # Pass the largest outer radius here, so we'll look for interlopers in a circle with the radius
                #  being the largest axis of the ellipse
                interloper_regions = source.regions_within_radii(inner_radii[0][0], max(outer_radii[0]), reg_cen_coords)
                back_inter_reg = source.regions_within_radii(max(outer_radii[0]) * source.background_radius_factors[0],
                                                             max(outer_radii[0]) * source.background_radius_factors[1],
                                                             reg_cen_coords)
                #TODO get_annular_esass_region
                reg = source.get_annular_sas_region(inner_radii[0], outer_radii[0], obs_id, inst,
                                                    interloper_regions=interloper_regions, central_coord=reg_cen_coords,
                                                    rot_angle=reg.angle)
                #TODO get_annular_esass_region
                b_reg = source.get_annular_sas_region(outer_radii[0] * source.background_radius_factors[0],
                                                      outer_radii[0] * source.background_radius_factors[1], obs_id,
                                                      inst, interloper_regions=back_inter_reg,
                                                      central_coord=source.default_coord)
                # Explicitly read out the current inner radius and outer radius, useful for some bits later
                src_inn_rad_str = 'and'.join(inner_radii[0].value.astype(str))
                src_out_rad_str = 'and'.join(outer_radii[0].value.astype(str)) + "_region"
                # Also explicitly read out into variables the actual radii values
                inn_rad_degrees = inner_radii[0]
                out_rad_degrees = outer_radii[0]

            else:
                # This constructs the sas strings for any radius that isn't 'region'
                #TODO get_annular_esass_region - dont put it in BaseSource
                reg = source.get_annular_sas_region(inner_radii[s_ind], outer_radii[s_ind], obs_id, inst,
                                                    interloper_regions=interloper_regions,
                                                    central_coord=source.default_coord)
                #TODO get_annular_esass_region
                b_reg = source.get_annular_sas_region(outer_radii[s_ind] * source.background_radius_factors[0],
                                                      outer_radii[s_ind] * source.background_radius_factors[1], obs_id,
                                                      inst, interloper_regions=back_inter_reg,
                                                      central_coord=source.default_coord)
                inn_rad_degrees = inner_radii[s_ind]
                out_rad_degrees = outer_radii[s_ind]
            
            # Getting the source name
            source_name = source.name
            
            # Just grabs the event list object
            evt_list = pack[-1]
            # Sets up the file names of the output files, adding a random number so that the
            #  function for generating annular spectra doesn't clash and try to use the same folder
            # ASSUMPTION4 new output directory structure
            dest_dir = OUTPUT + "erosita/" + "{o}/{i}_{n}_temp_{r}/".format(o=obs_id, i=inst, n=source_name, r=randint(0, 1e+8))

            # Cannot control the naming of spectra from srctool, so need to store
            # the XGA formatting of the spectra, so that they can be renamed 
            #TODO put issue, renaming spectra 
            spec_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_spec.fits"
            rmf_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}.rmf"
            arf_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}.arf"
            b_spec_str = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_backspec.fits"

            # Need to have different names for each different instrument so we stored them in a dictonary
            spec = {}
            rmf = {}
            arf = {}
            b_spec = {}

            # Getting the TM numbers from the list
            inst_list = list(inst)
            for idv_inst in inst_list:
                # populating the dictonary with the final names of the spectra depending on the instrument
                spec[inst] = spec_str.format(o=obs_id, i=idv_inst, n=source_name, ra=source.default_coord[0].value,
                               dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
                rmf[inst] = rmf_str.format(o=obs_id, i=idv_inst, n=source_name, ra=source.default_coord[0].value,
                               dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
                arf[inst] = arf_str.format(o=obs_id, i=idv_inst, n=source_name, ra=source.default_coord[0].value,
                               dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
                b_spec[inst] = b_spec_str.format(o=obs_id, i=idv_inst, n=source_name, ra=source.default_coord[0].value,
                               dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
            # There will also be a final spectra/rmf/arf/b_spec for the merged instruments
            spec['0'] = spec_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                            dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
            rmf['0'] = spec_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                            dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
            arf['0'] = spec_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                            dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
            b_spec['0'] = spec_str.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                            dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
            
            # These file names are for the debug images of the source and background images, they will not be loaded
            #  in as a XGA products, but exist purely to check by eye if necessary
            dim = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_debug." \
                  "fits".format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)
            b_dim = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_back_debug." \
                    "fits".format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                  dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str)

            #TODO occupy these variables 
            #TODO convert to icrs system, as per the srctool website
            coord_str = "icrs;{ra}, {dec}".format(ra=source.default_coord[0].value, dec=source.default_coord[1].value)
            src_reg_str = None # dealt with in get_annular_esass_region
            #TODO put in values that dont take too long
            #TODO allow user to chose tstep and xgrid
            tstep = None # put it as 0.5 for now
            xgrid = None # leave as default for now
            bsrc_reg_str = None

            # Fills out the srctool command to make the main and background spectra
            s_cmd_str = ext_srctool_cmd.format(d=dest_dir, ef=evt_list.path, sc=coord_str, reg=src_reg_str, 
                                               breg=bsrc_reg_str, i=inst, ts=tstep, xg=xgrid)
            #TODO might want different tstep and xgrid to the source to save processing time
            #sb_cmd_str = bckgr_srctool_cmd.format(d=dest_dir, ef=evt_list.path, sc=coord_str, breg=bsrc_reg_str, 
                                              # i=insts, ts=tstep,xg=xgrid)
            
            # Populating the rename command for all the srctool outputs, ie: SourceSpec, RMF, ARF, BackgrSpec
            # Messy I know, just wanted to have a draft done before i make the code more efficient
            spec_rename_strs = []
            for inst_key in spec:
                rename_str = rename_cmd.format(inst=inst_key, type="SourceSpec", nn=spec[inst_key])
                spec_rename_strs.append(rename_str)
            
            arf_rename_strs = []
            for inst_key in arf:
                rename_str = rename_cmd.format(inst=inst_key, type="ARF", nn=arf[inst_key])
                arf_rename_strs.append(rename_str)
            
            rmf_rename_strs = []
            for inst_key in rmf:
                rename_str = rename_cmd.format(inst=inst_key, type="RMF", nn=rmf[inst_key])
                rmf_rename_strs.append(rename_str)
            
            b_spec_rename_strs = []
            for inst_key in b_spec:
                rename_str = rename_cmd.format(inst=inst_key, type="BackgrSpec", nn=b_spec[inst_key])
                b_spec_rename_strs.append(rename_str)

            joined_spec_strs = ";".join(spec_rename_strs)
            joined_rmf_strs = ";".join(rmf_rename_strs)
            joined_arf_strs = ";".join(arf_rename_strs)
            joined_b_spec_str =  ";".join(b_spec_rename_strs)

            cmd_str = ";".join([s_cmd_str, joined_spec_strs, joined_rmf_strs, joined_arf_strs, joined_b_spec_str])
            # Adds clean up commands to move all generated files and remove temporary directory
            cmd_str += "; mv * ../; cd ..; rm -r {d}".format(d=dest_dir)
            cmds.append(cmd_str)  # Adds the full command to the set
            # Makes sure the whole path to the temporary directory is created
            os.makedirs(dest_dir)

            # ASSUMPTION4 new output directory structure
            # DAVID_QUESTION does having all the idv inst spectra in the final path make sense
            # ASSUMPTION8 The rmf, arf, b_spec paths are now lists instead of strings
            for key in spec:
                final_paths.append(os.path.join(OUTPUT, "erosita", obs_id, spec[key]))
            extra_info.append({"inner_radius": inn_rad_degrees, "outer_radius": out_rad_degrees,
                               "rmf_path": [os.path.join(OUTPUT, "erosita", obs_id, rmf[key]) for key in rmf],
                               "arf_path": [os.path.join(OUTPUT, "erosita", obs_id, arf[key]) for key in arf],
                               "b_spec_path": [os.path.join(OUTPUT, "erosita", obs_id, b_spec[key]) for key in b_spec],
                               "b_rmf_path": '',
                               "b_arf_path": '',
                               "obs_id": obs_id, "instrument": inst, 
                               "central_coord": source.default_coord,
                               "from_region": from_region})

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the eSASS cmd has run
        sources_extras.append(np.array(extra_info))
        #DAVID_QUESTION confused as to where the source_types is used
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress




            
            

            


        



