#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 15/08/2024, 10:31. Copyright (c) The Contributors

import os
from copy import copy
from itertools import permutations
from random import randint
from typing import Union, List

import numpy as np
from astropy.units import Quantity

from ._common import region_setup, _gen_detmap_cmd
from .misc import cifbuild
from .. import OUTPUT, NUM_CORES
from ..exceptions import SASInputInvalid, NotAssociatedError, NoProductAvailableError
from ..samples.base import BaseSample
from ..sas.run import sas_call
from ..sources import BaseSource, ExtendedSource, GalaxyCluster


def _spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
               min_counts: int = 5, min_sn: float = None, over_sample: int = None, one_rmf: bool = True,
               num_cores: int = NUM_CORES, disable_progress: bool = False, force_gen: bool = False):
    """
    An internal function to generate all the commands necessary to produce an evselect spectrum, but is not
    decorated by the sas_call function, so the commands aren't immediately run. This means it can be used for
    evselect functions that generate custom sets of spectra (like a set of annular spectra for instance), as well
    as for things like the standard evselect_spectrum function which produce relatively boring spectra.

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
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param int over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    """
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    if outer_radius != 'region':
        from_region = False
        sources, inner_radii, outer_radii = region_setup(sources, outer_radius, inner_radius, disable_progress,
                                                         '', num_cores)
    else:
        # This is used in the extra information dictionary for when the XGA spectrum object is defined
        from_region = True

    # Just make sure these values are the expect data type, this matters when the information is
    #  added to the storage strings and file names
    if over_sample is not None:
        over_sample = int(over_sample)
    if min_counts is not None:
        min_counts = int(min_counts)
    if min_sn is not None:
        min_sn = float(min_sn)

    # These check that the user hasn't done something silly like passing multiple grouping options, this is not
    #  allowed by SAS, will cause the generation to fail
    if all([o is not None for o in [min_counts, min_sn]]):
        raise SASInputInvalid("evselect only allows one grouping option to be passed, you can't group both by"
                              " minimum counts AND by minimum signal to noise.")
    # Should also check that the user has passed any sort of grouping argument, if they say they want to group
    elif group_spec and all([o is None for o in [min_counts, min_sn]]):
        raise SASInputInvalid("If you set group_spec=True, you must supply a grouping option, either min_counts"
                              " or min_sn.")

    # Sets up the extra part of the storage key name depending on if grouping is enabled
    if group_spec and min_counts is not None:
        extra_name = "_mincnt{}".format(min_counts)
    elif group_spec and min_sn is not None:
        extra_name = "_minsn{}".format(min_sn)
    else:
        extra_name = ''

    # Have to make sure that all observations have an up to date cif file.
    cifbuild(sources, disable_progress=disable_progress, num_cores=num_cores)

    # And if it was oversampled during generation then we need to include that as well
    if over_sample is not None:
        extra_name += "_ovsamp{ov}".format(ov=over_sample)

    # Define the various SAS commands that need to be populated, for a useful spectrum you also need ARF/RMF
    spec_cmd = "cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; evselect table={e} withspectrumset=yes " \
               "spectrumset={s} energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 " \
               "specchannelmax={u} {ex}"

    # The detmap in this context is just an image of the source distribution of observation, in detector coordinates,
    #  and is used to weight the generation of the ARF curves.
    detmap_cmd = "evselect table={e} imageset={d} xcolumn=DETX ycolumn=DETY imagebinning=binSize ximagebinsize=100 " \
                 "yimagebinsize=100 {ex}"

    # This command just makes a standard XCS image, but will be used to generate images to debug the drilling
    #  out of regions, as the spectrum expression will be supplied so we can see exactly what data has been removed.
    debug_im = "evselect table={e} imageset={i} xcolumn=X ycolumn=Y ximagebinsize=87 " \
               "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize " \
               "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 withyranges=yes {ex}"

    rmf_cmd = "rmfgen rmfset={r} spectrumset='{s}' detmaptype={dt} detmaparray={ds} extendedsource={es}"

    # Don't make use of the setbackscale functionality of arg
    arf_cmd = "arfgen spectrumset={s} arfset={a} withrmfset=yes rmfset={r} badpixlocation={e} " \
              "extendedsource={es} detmaptype={dt} detmaparray={ds} setbackscale=no badpixmaptype={dt}"

    bscal_cmd = "backscale spectrumset={s} badpixlocation={e}"

    # If the user wants to group spectra, then we'll need this template command:
    grp_cmd = "specgroup spectrumset={s} overwrite=yes backgndset={b} arfset={a} rmfset={r} addfilenames=no"

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for s_ind, source in enumerate(sources):
        # rmfgen and arfgen both take arguments that describe if something is an extended source or not,
        #  so we check the source type
        if isinstance(source, (ExtendedSource, GalaxyCluster)):
            ex_src = "yes"
            # Sets the detmap type, using an image of the source is appropriate for extended sources like clusters,
            #  but not for point sources
            dt = 'dataset'
        else:
            ex_src = "no"
            dt = 'flat'
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
            spec_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}"
            spec_storage_name = spec_storage_name.format(ra=source.default_coord[0].value,
                                                         dec=source.default_coord[1].value,
                                                         ri=src_inn_rad_str, ro=src_out_rad_str, gr=group_spec)
        else:
            spec_storage_name = "region_grp{gr}".format(gr=group_spec)

        # Adds on the extra information about grouping to the storage key
        spec_storage_name += extra_name

        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", just_obj=False):
            obs_id = pack[0]
            inst = pack[1]

            if not os.path.exists(OUTPUT + obs_id):
                os.mkdir(OUTPUT + obs_id)

            # Got to check if this spectrum already exists
            exists = source.get_products("spectrum", obs_id, inst, extra_key=spec_storage_name)
            if len(exists) == 1 and exists[0].usable and not force_gen:
                continue

            # If there is no match to a region, the source region returned by this method will be None,
            #  and if the user wants to generate spectra from region files, we have to ignore that observations
            if outer_radius == "region" and source.source_back_regions("region", obs_id)[0] is None:
                continue

            # Because the region will be different for each ObsID, I have to call the setup function here
            if outer_radius == 'region':
                interim_source, inner_radii, outer_radii = region_setup([source], outer_radius, inner_radius,
                                                                        disable_progress, obs_id, num_cores)
                # Need the reg for central coordinates
                reg = source.source_back_regions('region', obs_id)[0]
                reg_cen_coords = Quantity([reg.center.ra.value, reg.center.dec.value], 'deg')
                # Pass the largest outer radius here, so we'll look for interlopers in a circle with the radius
                #  being the largest axis of the ellipse
                interloper_regions = source.regions_within_radii(inner_radii[0][0], max(outer_radii[0]), reg_cen_coords)
                back_inter_reg = source.regions_within_radii(max(outer_radii[0]) * source.background_radius_factors[0],
                                                             max(outer_radii[0]) * source.background_radius_factors[1],
                                                             reg_cen_coords)

                reg = source.get_annular_sas_region(inner_radii[0], outer_radii[0], obs_id, inst,
                                                    interloper_regions=interloper_regions, central_coord=reg_cen_coords,
                                                    rot_angle=reg.angle)
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
                reg = source.get_annular_sas_region(inner_radii[s_ind], outer_radii[s_ind], obs_id, inst,
                                                    interloper_regions=interloper_regions,
                                                    central_coord=source.default_coord)
                b_reg = source.get_annular_sas_region(outer_radii[s_ind] * source.background_radius_factors[0],
                                                      outer_radii[s_ind] * source.background_radius_factors[1], obs_id,
                                                      inst, interloper_regions=back_inter_reg,
                                                      central_coord=source.default_coord)
                inn_rad_degrees = inner_radii[s_ind]
                out_rad_degrees = outer_radii[s_ind]

            # Some settings depend on the instrument, XCS uses different patterns for different instruments
            if "pn" in inst:
                # Also the upper channel limit is different for EPN and EMOS detectors
                spec_lim = 20479
                expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0) && {s}'".format(s=reg)
                b_expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0) && {s}'".format(s=b_reg)
                # This is an expression without region information to be used for making the detmaps
                #  required for ARF generation, we start off assuming we'll use a MOS observation as the detmap
                d_expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0)'"

                # The detmap for the arfgen call should ideally not be from the same instrument as the observation,
                #  so for PN we preferentially select MOS2 (as MOS1 was damaged). However if there isn't a MOS2
                #  events list from the same observation then we select MOS1, and failing that we use PN.
                try:
                    detmap_evts = source.get_products("events", obs_id=obs_id, inst='mos2')[0]
                except NotAssociatedError:
                    try:
                        detmap_evts = source.get_products("events", obs_id=obs_id, inst='mos1')[0]
                    except NotAssociatedError:
                        detmap_evts = source.get_products("events", obs_id=obs_id, inst='pn')[0]
                        # If all is lost and there are no MOS event lists then we must revert to the PN expression
                        d_expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0)'"

            elif "mos" in inst:
                spec_lim = 11999
                expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0) && {s}'".format(s=reg)
                b_expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0) && {s}'".format(s=b_reg)
                # This is an expression without region information to be used for making the detmaps
                #  required for ARF generation, we start off assuming we'll use the PN observation as the detmap
                d_expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0)'"

                # The detmap for the arfgen call should ideally not be from the same instrument as the observation,
                #  so for MOS observations we preferentially select PN. However if there isn't a PN events list
                #  from the same observation then for MOS2 we select MOS1, and for MOS1 we select MOS2 (as they
                #  are rotated wrt one another it is still semi-valid), and failing that MOS2 will get MOS2 and MOS1
                #  will get MOS1.
                if inst[-1] == 1:
                    cur = '1'
                    opp = '2'
                else:
                    cur = '2'
                    opp = '1'
                try:
                    detmap_evts = source.get_products("events", obs_id=obs_id, inst='pn')[0]
                except NotAssociatedError:
                    # If we must use a MOS detmap then we have to use the MOS expression
                    d_expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0)'"
                    try:
                        detmap_evts = source.get_products("events", obs_id=obs_id, inst='mos'+opp)[0]
                    except NotAssociatedError:
                        detmap_evts = source.get_products("events", obs_id=obs_id, inst='mos'+cur)[0]
            else:
                raise ValueError("You somehow have an illegal value for the instrument name...")

            # Some of the SAS tasks have issues with filenames with a '+' in them for some reason, so this
            #  replaces any + symbols that may be in the source name with another character
            source_name = source.name.replace("+", "x")

            # Just grabs the event list object
            evt_list = pack[-1]
            # Sets up the file names of the output files, adding a random number so that the
            #  function for generating annular spectra doesn't clash and try to use the same folder
            dest_dir = OUTPUT + "{o}/{i}_{n}_temp_{r}/".format(o=obs_id, i=inst, n=source_name, r=randint(0, int(1e+8)))

            # Sets up something very similar to the extra name variable above, but for the file names
            #  Stores some information about grouping in the file names
            if group_spec and min_counts is not None:
                extra_file_name = "_mincnt{c}".format(c=min_counts)
            elif group_spec and min_sn is not None:
                extra_file_name = "_minsn{s}".format(s=min_sn)
            else:
                extra_file_name = ''

            # And if it was oversampled during generation then we need to include that as well
            if over_sample is not None:
                extra_file_name += "_ovsamp{ov}".format(ov=over_sample)

            spec = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_spec.fits"
            spec = spec.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                               dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str, gr=group_spec,
                               ex=extra_file_name)
            b_spec = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_backspec.fits"
            b_spec = b_spec.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                   dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                   gr=group_spec, ex=extra_file_name)
            # Arguably checking whether this is an extended source and adjusting this is irrelevant as the file
            #  name won't be used anyway with detmaptype set to flat, but I'm doing it anyway
            if ex_src:
                det_map = "{o}_{i}_detmap.fits"
                det_map = det_map.format(o=detmap_evts.obs_id, i=detmap_evts.instrument)
            else:
                det_map = ""
            arf = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}.arf"
            arf = arf.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                             dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str, gr=group_spec,
                             ex=extra_file_name)
            # b_arf = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_back.arf"
            # b_arf = b_arf.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
            #                      dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
            #                      gr=group_spec, ex=extra_file_name)
            ccf = dest_dir + "ccf.cif"

            # These file names are for the debug images of the source and background images, they will not be loaded
            #  in as a XGA products, but exist purely to check by eye if necessary
            dim = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_debug." \
                  "fits".format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                gr=group_spec, ex=extra_file_name)
            b_dim = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_back_debug." \
                    "fits".format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                  dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                  gr=group_spec, ex=extra_file_name)

            # Fills out the evselect command to make the main and background spectra
            s_cmd_str = spec_cmd.format(d=dest_dir, ccf=ccf, e=evt_list.path, s=spec, u=spec_lim, ex=expr)
            sb_cmd_str = spec_cmd.format(d=dest_dir, ccf=ccf, e=evt_list.path, s=b_spec, u=spec_lim, ex=b_expr)

            # Does the same thing for the evselect command to make the detmap
            d_cmd_str = detmap_cmd.format(e=detmap_evts.path, d=det_map, ex=d_expr)

            # Populates the debug image commands
            dim_cmd_str = debug_im.format(e=evt_list.path, ex=expr, i=dim)
            b_dim_cmd_str = debug_im.format(e=evt_list.path, ex=b_expr, i=b_dim)

            # This chunk adds rmfgen commands depending on whether we're using a universal RMF or
            #  an individual one for each spectrum. Also adds arfgen commands on the end, as they depend on
            #  the rmf.
            if one_rmf:
                rmf = "{o}_{i}_{n}_universal.rmf".format(o=obs_id, i=inst, n=source_name)
                # b_rmf = rmf
            else:
                rmf = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}.rmf"
                rmf = rmf.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                 dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                 gr=group_spec, ex=extra_file_name)
                # b_rmf = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}{ex}_back.rmf"
                # b_rmf = b_rmf.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                #                      dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                #                      gr=group_spec, ex=extra_file_name)

            final_rmf_path = OUTPUT + obs_id + '/' + rmf
            if one_rmf and (not os.path.exists(final_rmf_path) or force_gen):
                cmd_str = ";".join([s_cmd_str, dim_cmd_str, b_dim_cmd_str, d_cmd_str,
                                    rmf_cmd.format(r=rmf, s=spec, es=ex_src, ds=det_map, dt=dt),
                                    arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path, es=ex_src, ds=det_map, dt=dt),
                                    sb_cmd_str, bscal_cmd.format(s=spec, e=evt_list.path),
                                    bscal_cmd.format(s=b_spec, e=evt_list.path)])
                #arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path, es=ex_src, ds=det_map)
            elif not one_rmf and (not os.path.exists(final_rmf_path) or force_gen):
                cmd_str = ";".join([s_cmd_str, dim_cmd_str, b_dim_cmd_str, d_cmd_str,
                                    rmf_cmd.format(r=rmf, s=spec, es=ex_src, ds=det_map, dt=dt),
                                    arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path, es=ex_src,
                                                   ds=det_map, dt=dt)]) + ";"
                cmd_str += ";".join([sb_cmd_str, bscal_cmd.format(s=spec, e=evt_list.path),
                                     bscal_cmd.format(s=b_spec, e=evt_list.path)])
                #, rmf_cmd.format(r=b_rmf, s=b_spec, es=ex_src, ds=det_map)
                # arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path, es=ex_src, ds=det_map)
            else:
                # This one just copies the existing universal rmf into the temporary generation folder
                cmd_str = "cp {f_rmf} {d};".format(f_rmf=final_rmf_path, d=dest_dir)
                cmd_str += ";".join([s_cmd_str, dim_cmd_str, b_dim_cmd_str, d_cmd_str,
                                     arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path, es=ex_src,
                                                    ds=det_map, dt=dt)]) + ";"
                cmd_str += ";".join([sb_cmd_str, bscal_cmd.format(s=spec, e=evt_list.path),
                                     bscal_cmd.format(s=b_spec, e=evt_list.path)])
                #arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path, es=ex_src, ds=det_map)

            # If the user wants to produce grouped spectra, then this if statement is triggered and adds a specgroup
            #  command at the end. The groupspec command will replace the ungrouped spectrum.
            if group_spec:
                new_grp = grp_cmd.format(s=spec, b=b_spec, r=rmf, a=arf)
                if min_counts is not None:
                    new_grp += " mincounts={mc}".format(mc=min_counts)
                if min_sn is not None:
                    new_grp += " minSN={msn}".format(msn=min_sn)
                if over_sample is not None:
                    new_grp += " oversample={os}".format(os=over_sample)
                cmd_str += "; " + new_grp

            # Adds clean up commands to move all generated files and remove temporary directory
            cmd_str += "; mv * ../; cd ..; rm -r {d}".format(d=dest_dir)
            cmds.append(cmd_str)  # Adds the full command to the set
            # Makes sure the whole path to the temporary directory is created
            os.makedirs(dest_dir)

            final_paths.append(os.path.join(OUTPUT, obs_id, spec))
            extra_info.append({"inner_radius": inn_rad_degrees, "outer_radius": out_rad_degrees,
                               "rmf_path": os.path.join(OUTPUT, obs_id, rmf),
                               "arf_path": os.path.join(OUTPUT, obs_id, arf),
                               "b_spec_path": os.path.join(OUTPUT, obs_id, b_spec),
                               "b_rmf_path": '',
                               "b_arf_path": '',
                               "obs_id": obs_id, "instrument": inst, "grouped": group_spec, "min_counts": min_counts,
                               "min_sn": min_sn, "over_sample": over_sample, "central_coord": source.default_coord,
                               "from_region": from_region})

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the SAS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@sas_call
def evselect_spectrum(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                      inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                      min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                      num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A wrapper for all of the SAS processes necessary to generate an XMM spectrum that can be analysed
    in XSPEC. Every observation associated with this source, and every instrument associated with that
    observation, will have a spectrum generated using the specified outer and inner radii as a boundary. The
    default inner radius is zero, so by default this function will produce circular spectra out to the outer_radius.
    It is possible to generate both grouped and ungrouped spectra using this function, with the degree
    of grouping set by the min_counts, min_sn, and oversample parameters.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored. If you are
        generating for multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default this is zero arcseconds, resulting in a circular spectrum. If you are
        generating for multiple sources then you can also pass a Quantity with one entry per source.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    # All the workings of this function are in _spec_cmds so that the annular spectrum set generation function
    #  can also use them
    return _spec_cmds(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn, over_sample, one_rmf,
                      num_cores, disable_progress)


@sas_call
def spectrum_set(sources: Union[BaseSource, BaseSample], radii: Union[List[Quantity], Quantity],
                 group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                 one_rmf: bool = True, num_cores: int = NUM_CORES, force_regen: bool = False,
                 disable_progress: bool = False):
    """
    This function can be used to produce 'sets' of XGA Spectrum objects, generated in concentric circular annuli.
    Such spectrum sets can be used to measure projected spectroscopic quantities, or even be de-projected to attempt
    to measure spectroscopic quantities in a three dimensional space.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param List[Quantity]/Quantity radii: A list of non-scalar quantities containing the boundary radii of the
        annuli for the sources. A single quantity containing at least three radii may be passed if one source
        is being analysed, but for multiple sources there should be a quantity (with at least three radii), PER
        source.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool force_regen: This will force all the constituent spectra of the set to be regenerated, use this
        if your call to this function was interrupted and an incomplete AnnularSpectrum is being read in.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
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

    # Just to make sure calibration files have been generated, though I don't actually think they could
    #  have gotten to this point without them
    cifbuild(sources, num_cores, disable_progress)

    # This generates a spectra between the innermost and outmost radii for each source, and a universal RMF
    if one_rmf:
        innermost_rads = Quantity([r_set[0] for r_set in radii], radii[0].unit)
        outermost_rads = Quantity([r_set[-1] for r_set in radii], radii[0].unit)
        evselect_spectrum(sources, outermost_rads, innermost_rads, group_spec, min_counts, min_sn, over_sample,
                          one_rmf, num_cores, disable_progress)

    # I want to be able to generate all the individual annuli in parallel, but I need them to be associated with
    #  the correct annuli, which is why I have to iterate through the sources and radii

    # These store the final output information needed to run the commands
    all_cmds = []
    all_paths = []
    all_out_types = []
    all_extras = []
    # Iterating through the sources
    for s_ind, source in enumerate(sources):
        # This generates a random integer ID for this set of spectra
        set_id = randint(0, int(1e+8))

        # I want to be sure that this configuration doesn't already exist
        if group_spec and min_counts is not None:
            extra_name = "_mincnt{}".format(min_counts)
        elif group_spec and min_sn is not None:
            extra_name = "_minsn{}".format(min_sn)
        else:
            extra_name = ''

        # And if it was oversampled during generation then we need to include that as well
        if over_sample is not None:
            extra_name += "_ovsamp{ov}".format(ov=over_sample)

        # Combines the annular radii into a string
        ann_rad_str = "_".join(source.convert_radius(radii[s_ind], 'deg').value.astype(str))
        spec_storage_name = "ra{ra}_dec{dec}_ar{ar}_grp{gr}"
        spec_storage_name = spec_storage_name.format(ra=source.default_coord[0].value,
                                                     dec=source.default_coord[1].value, ar=ann_rad_str, gr=group_spec)

        spec_storage_name += extra_name

        exists = source.get_products('combined_spectrum', extra_key=spec_storage_name)
        if len(exists) == 0:
            # If it doesn't exist then we do need to call evselect_spectrum
            generate_spec = True
        else:
            # If it already exists though we don't need to bother
            generate_spec = False

        # This is where the commands/extra information get concatenated from the different annuli
        src_cmds = np.array([])
        src_paths = np.array([])
        src_out_types = []
        src_extras = np.array([])
        if generate_spec or force_regen:
            # Here we run through all the requested annuli for the current source
            for r_ind in range(len(radii[s_ind])-1):
                # Generate the SAS commands for the current annulus of the current source, for all observations
                spec_cmd_out = _spec_cmds(source, radii[s_ind][r_ind+1], radii[s_ind][r_ind], group_spec, min_counts,
                                          min_sn, over_sample, one_rmf, num_cores, disable_progress, True)

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
                    new_spec = split_p[-1].replace("_spec.fits", "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) \
                               + "_spec.fits"
                    # Not enough just to change the name passed through XGA, it has to be changed in
                    #  the SAS commands as well
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

                        # Replacing the names in the SAS commands
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
                    new_b_spec = split_bs[-1].replace('_backspec.fits', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) \
                                + "_backspec.fits"

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

    # This gets passed back to the sas call function and is used to run the commands
    return all_cmds, False, True, num_cores, all_out_types, all_paths, all_extras, disable_progress


@sas_call
def cross_arf(sources: Union[BaseSource, BaseSample], radii: Union[List[Quantity], Quantity],
              group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None,
              set_id: str = None, detmap_bin: int = 200, num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    This function will generate cross-arfs for annular spectra, which describe the contribution of each annulus
    to each other annulus due to XMM's relatively sizeable PSF. The cross-arfs are generated for each instrument
    of each observation, and automatically stored in their parent AnnularSpectra instance, both for retrieval by
    fitting processes and so that the user can examine them with a plotting method and by retrieving effective
    area values from them.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param List[Quantity]/Quantity radii: A list of non-scalar quantities containing the boundary radii of the
        annuli for the sources. A single quantity containing at least three radii may be passed if one source
        is being analysed, but for multiple sources there should be a quantity (with at least three radii), PER
        source. This is used to help retrieve the correct annular spectrum.
    :param bool group_spec: A boolean flag that sets whether the spectra are grouped or not. This is used to help
        retrieve the correct annular spectrum.
    :param float min_counts: If retrieving a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None. This is used to help retrieve the correct
        annular spectrum.
    :param float min_sn: If retrieving a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None. This is used to help retrieve the correct
        annular spectrum.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.  This is
        used to help retrieve the correct annular spectrum.
    :param str/List[str] set_id: The unique annular spectrum identifier (or a list of them if analysing multiple
        sources) that specifies which annular spectrum to use.
    :param int detmap_bin: The spatial binning applied to event lists to create the detector maps used in the
        calculations of effective areas. The default is 200, smaller values will increase the resolution but will
        cause dramatically slower calculations.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :return:
    """
    # If it's a single source I put it into an iterable object (i.e. a list), just for convenience
    if isinstance(sources, BaseSource):
        sources = [sources]
    elif isinstance(sources, list) and not all([isinstance(s, BaseSource) for s in sources]):
        raise TypeError("If a list is passed, each element must be a source.")
    # And the only other option is a BaseSample instance, so if it isn't that then we get angry
    elif not isinstance(sources, (BaseSample, list)):
        raise TypeError("Please only pass source or sample objects for the 'sources' parameter of this function")

    # We want set_id to be iterable as well, so we wrap it in a list if it wasn't already a list or array
    if set_id is not None and isinstance(set_id, (list, np.ndarray)):
        set_id = [set_id]
    # We still want it to be iterable even if the user never specified it, so we make a list of Nones the same
    #  length as the sources variable
    elif set_id is None:
        set_id = [None]*len(sources)

    # This will trigger if the user passed too few or too many set ids for the number of sources there are
    if len(set_id) != len(sources):
        raise ValueError("If an XGA sample has been passed, and AnnularSpectra are being specified with the 'set_id' "
                         "argument, then a list of set_ids with the same number of entries must be passed.")

    # NOTE - There is no ';' after {dmc} because it will be included in the dmc command, or not. This is because if the
    #  requested detmap already exists then the command will just be "", and that will make bash upset if there is
    #  a ";" after it.
    arfgen_cmd = "cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; {dmc} arfgen spectrumset={s} arfset={a} " \
                 "withrmfset=yes rmfset={r} badpixlocation={e} extendedsource=yes detmaptype=dataset " \
                 "detmaparray={ds} setbackscale=no badpixmaptype=dataset crossregionarf=yes " \
                 "crossreg_spectrumset={crs}; mv * ../; cd ..; rm -r {d}"

    # These store the final output information needed to run the commands
    all_cmds = []
    all_paths = []
    all_out_types = []
    all_extras = []
    for src_ind, src in enumerate(sources):

        # This is where the commands/extra information get concatenated from the different annuli
        src_cmds = np.array([])
        src_paths = np.array([])
        src_out_types = []
        src_extras = np.array([])

        try:
            ann_spec = src.get_annular_spectra(radii, group_spec, min_counts, min_sn, over_sample, set_id[src_ind])
        except NoProductAvailableError:
            # We make our own version of this error
            raise NoProductAvailableError("The requested AnnularSpectra cannot be located for {sn}, and this function "
                                          "will not automatically generate annular spectra.".format(sn=src.name))

        oi_combos = [(o_id, inst) for o_id, insts in ann_spec.instruments.items() for inst in insts]
        for oi in oi_combos:
            rel_sp_comp = [ann_spec.get_spectra(ann_id, oi[0], oi[1]) for ann_id in ann_spec.annulus_ids]

            for sp_comb in permutations(rel_sp_comp, 2):
                obs_id = sp_comb[0].obs_id
                inst = sp_comb[0].instrument

                # This tries to retrieve an existing cross-arf generated for the current ObsID, instrument, and annuli
                #  for this annular spectra - if we succeed then we simply continue and move on to the next combo
                try:
                    ann_spec.get_cross_arf_paths(obs_id, inst, sp_comb[0].annulus_ident, sp_comb[1].annulus_ident)
                    continue
                except NotAssociatedError:
                    pass

                evt_list = src.get_products('events', obs_id, inst)[0]

                dest_dir = OUTPUT + "{o}/{i}_{n}_temp_{r}/".format(o=obs_id, i=inst, n=src.name,
                                                                   r=randint(0, int(1e+8)))

                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                ccf = dest_dir + "ccf.cif"

                det_map_cmd, det_map_cmd_path, det_map_path = _gen_detmap_cmd(src, obs_id, inst, detmap_bin)

                c_arf_name = "{o}_{i}_{n}_".format(o=obs_id, i=inst, n=src.name) + \
                             ann_spec.storage_key.split('_ar')[0] + '_grp' + \
                             ann_spec.storage_key.split('ar')[-1].split('_grp')[1] + \
                             "_ident" + str(ann_spec.set_ident) + \
                             '_cross_{inn}_{out}.arf'.format(inn=sp_comb[0].annulus_ident,
                                                             out=sp_comb[1].annulus_ident)
                c_arf_path = dest_dir + c_arf_name

                cmd = arfgen_cmd.format(d=dest_dir, ccf=ccf, s=sp_comb[0].path, a=c_arf_path, r=sp_comb[0].rmf,
                                        e=evt_list.path, crs=sp_comb[1].path, ds=det_map_cmd_path, dmc=det_map_cmd)

                extra_info = {'detmap_bin': detmap_bin,
                              'ann_spec_set_id': ann_spec.set_ident,
                              'obs_id': obs_id,
                              'inst': inst,
                              'src_ann_id': sp_comb[0].annulus_ident,
                              'cross_ann_id': sp_comb[1].annulus_ident}

                src_paths = np.concatenate([src_paths, [OUTPUT + "{o}/".format(o=obs_id) + c_arf_name]])
                # Go through and concatenate things to the source lists defined above
                src_cmds = np.concatenate([src_cmds, [cmd]])
                src_out_types += ['cross arfs'] * len(src_cmds)
                src_extras = np.concatenate([src_extras, [extra_info]])

        # This adds the current sources final commands to the 'all sources' lists
        all_cmds.append(src_cmds)
        all_paths.append(src_paths)
        all_out_types.append(src_out_types)
        all_extras.append(src_extras)

    # This gets passed back to the sas call function and is used to run the commands
    return all_cmds, False, True, num_cores, all_out_types, all_paths, all_extras, disable_progress

