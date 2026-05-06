#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/6/26, 6:28 PM. Copyright (c) The Contributors.

import os
from copy import deepcopy
from random import randint
from typing import Union
from warnings import warn

import numpy as np
from astropy.units import Quantity, UnitConversionError

from ._common import T_STEP_POINT, T_STEP_SURVEY
from .phot import evtool_combine_evts
from .run import esass_call
from ..common import get_annular_esass_region
from ..sas._common import region_setup
from ... import OUTPUT, NUM_CORES, ESASS_VERSION
from ...exceptions import TelescopeNotAssociatedError, NoProductAvailableError
from ...samples.base import BaseSample
from ...sources import BaseSource


def _lc_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
             inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = Quantity(0.5, 'keV'),
             hi_en: Quantity = Quantity(2.0, 'keV'), time_bin_size: Quantity = Quantity(100, 's'),
             patt: int = 15, num_cores: int = NUM_CORES, disable_progress: bool = False, combine_tm: bool = True,
             combine_obs: bool = True, force_gen: bool = False):
    """
    This is an internal function which sets up the commands necessary to generate light curves from eROSITA
    data - and can be used both to generate them from simple circular regions and also from annular regions. The
    light curves are corrected for background, vignetting, and PSF concerns.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the light curve (for instance 'point' would be acceptable for a Star or PointSource). If 'region' is chosen
        (to use the regions in region files), then any inner radius will be ignored. If you are generating for
        multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the light curve. By default this is zero arcseconds, resulting in a light curve from a circular region. If
        you are generating for multiple sources then you can also pass a Quantity with one entry per source.
    :param Quantity lo_en: The lower energy boundary for the light curve, in units of keV. The default is 0.5 keV.
    :param Quantity hi_en: The upper energy boundary for the light curve, in units of keV. The default is 2.0 keV.
    :param Quantity time_bin_size: The bin size to be used for the creation of the light curve, in
        seconds. The default is 100 s.
    :param int patt: An integer representation of a bitmask specifying which event patterns should be included. The
        default is 15 (i.e. all valid patterns).
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param bool combine_tm: Create lightcurves for individual ObsIDs that are a combination of the data from all the
        telescope modules utilized for that ObsID. This can help to offset the low signal-to-noise nature of the
        survey data eROSITA takes. Default is True.
    :param bool combine_obs: Setting this to False will generate an lightcurve for each associated observation, 
        instead of for one combined observation.
    """
    def _make_lc_cmd_info(cur_evt_list):
        """
        Internal function to get prepare the commands required to generate light curves
        from eROSITA observations. This functions acts on a single event list at a time.
        Output final file paths and extra information dictionaries are also
        created and returned.

        :param EventList cur_evt_list: The event list to generate spectra from.
        :return: A list of spectral generation commands, a list of final spectrum file
            paths, and a list of extra information dictionaries.
        :rtype: Tuple[List[str], List[str], List[str]]
        """
        # Then we have to account for the two different modes this function can be used in - generating light curves
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
            #  numbers if the user has requested a combined light curve).
            inst_no = inst_nums[inst_ind]
            # Also pick out the current instrument srctool ID - this will be passed
            #  to the writeinsts argument of srctool. It will be identical to 'inst_no'
            #  for individual telescope module spectra, and will be zero (to write a
            #  combined spectrum of all specified TMs) for combined telescope
            #  module spectra.
            cur_inst_srctool_id = inst_srctool_id[inst_ind]

            try:
                if use_combine_obs and (len(source.obs_ids[cur_evt_list.telescope]) > 1):
                    check_lc = source.get_combined_lightcurves(outer_radii[s_ind],
                                                               inner_radii[s_ind],
                                                               lo_en,
                                                               hi_en,
                                                               time_bin_size,
                                                               patt,
                                                               cur_evt_list.telescope)


                else:
                    # Got to check if this light curve already exists
                    check_lc = source.get_lightcurves(outer_radii[s_ind],
                                                      cur_evt_list.obs_id,
                                                      inst,
                                                      inner_radii[s_ind],
                                                      lo_en,
                                                      hi_en,
                                                      time_bin_size,
                                                      patt,
                                                      cur_evt_list.telescope)

                exists = True

            except NoProductAvailableError:
                exists = False

            if exists and check_lc.usable and not force_gen:
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

            # The name of the light curve will be different depending on if it is
            #  from a combined event list or a single skytile event list
            if use_combine_obs and (len(source.obs_ids[cur_evt_list.telescope]) > 1):
                prefix = str(rand_ident) + '_{i}_{n}_'.format(i=inst, n=source.name)
            else:
                prefix = "{o}_{i}_{n}_".format(o=cur_evt_list.obs_id,
                                               i=inst,
                                               n=source.name)
            # Set up the final name of the output file
            lc_form = prefix + "ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_lcurve.fits"
            lc_name = lc_form.format(ra=source.default_coord[0].value,
                                     dec=source.default_coord[1].value,
                                     ri=src_inn_rad.value,
                                     ro=src_out_rad.value,
                                     ex=extra_name)

            # Populate the light curve generation command for the current iteration
            cmd_str = lc_cmd.format(d=dest_dir,
                                    ef=evt_symlink_name,
                                    sc=coord_str,
                                    reg=reg,
                                    breg=b_reg,
                                    i=inst_no,
                                    ts=t_step,
                                    lct='REGULAR',
                                    lcp=str(time_bin_size.to('s').value),
                                    le=str(lo_en.value),
                                    lm=str(hi_en.value),
                                    lcg=str(lc_gamma),
                                    pat=patt,
                                    wi=cur_inst_srctool_id)

            # Command to rename the output light curve file to the file name
            #  that we want, rather than that which comes out of the srctool command
            rename_lc = rename_cmd.format(i_no=cur_inst_srctool_id,
                                          type='LightCurve',
                                          nn=lc_name)
            cmd_str += rename_lc

            # We make sure to remove the 'merged lightcurve' output of srctool - which is identical to the
            #  instrument one if we generate for one lightcurve at a time. Though only if the user hasn't actually
            #  ASKED for the merged lightcurve
            if combine_tm:
                cmd_str += remove_all_but_merged_cmd
            else:
                cmd_str += remove_merged_cmd

            # Adds symlink-removal command - we don't want to be moving them along
            #  with every else in the temporary working directory
            cmd_str += "; rm {esym}".format(esym=evt_symlink_name)

            # Adds clean up commands to move all generated files and remove temporary directory
            cmd_str += "; find . -maxdepth 1 -type f -exec mv {{}} ../ \\;; cd ..; rm -r {d}".format(d=dest_dir)

            # If temporary region files were made, they will be here
            if os.path.exists(os.path.join(final_dest_dir, '/temp_regs_{i}'.format(i=rand_ident))):
                # Removing this directory
                cmd_str += ";rm -r temp_regs_{i}".format(i=rand_ident)

            cur_cmds.append(cmd_str)  # Adds the full command to the set
            cur_fin_paths.append(os.path.join(final_dest_dir, lc_name))
            cur_ex_info.append({"inner_radius": src_inn_rad,
                                "outer_radius": src_out_rad,
                                "time_bin": time_bin_size,
                                "pattern": patt,
                                "obs_id": cur_evt_list.obs_id,
                                "instrument": inst,
                                "central_coord": source.default_coord,
                                "from_region": False,
                                "lo_en": lo_en,
                                "hi_en": hi_en,
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

    # We check to see whether there is an eROSITA entry in the 'telescopes' property. If sources is a Source
    #  object, then that property contains the telescopes associated with that source, and if it is a Sample object
    #  then 'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if eROSITA isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and ('erosita' not in sources.telescopes and 'erass' not in sources.telescopes)) or
            (isinstance(sources, list) and ('erosita' not in sources[0].telescopes and 'erass' not in sources[0].telescopes))):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA lightcurves cannot be generated.")

    # TODO This will change in a future release, so that the user can control it - see issue #1113. The definitions
    #  are up the top of the function as a reminder
    # TODO allow user to chose tstep and xgrid
    t_step_survey = T_STEP_SURVEY
    t_step_point = T_STEP_POINT

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    # At one point we allowed the 'outer_radius' argument to be 'region', but we
    #  no longer support that
    if outer_radius == 'region':
        raise ValueError("The string 'region' is no longer a valid option for "
                         "the 'outer_radius' argument.")

    # In this case the user wants to combine separate sky tiles, so we have to make
    #  sure that combined event lists exist for each source
    if combine_obs:
        evtool_combine_evts(sources, num_cores)

    # Sets up the inner and outer radii arrays for the passed sources
    sources, inner_radii, outer_radii = region_setup(sources, outer_radius, inner_radius, disable_progress,
                                                     '', num_cores)

    # Check the input time bin size type, we're going to assume that it is in
    #  seconds if the value is an integer or a float
    if not isinstance(time_bin_size, Quantity) and isinstance(time_bin_size, (float, int)):
        time_bin_size = Quantity(time_bin_size, 's')
    elif not isinstance(time_bin_size, (Quantity, float, int)):
        raise TypeError("The 'time_bin_size' argument must be either an Astropy "
                        "quantity, or an int/float (assumed to be in seconds).")

    # Make sure the time bin size can be converted to seconds and then do so
    if not time_bin_size.unit.is_equivalent('s'):
        raise UnitConversionError("The 'time_bin_size' argument must be in units convertible to seconds.")
    else:
        time_bin_size = time_bin_size.to('s')

    # Convert the integer pattern to a string
    patt = str(patt)

    # Have to make sure that the user hasn't done anything daft here, hi_en must be larger than lo_en
    if lo_en >= hi_en:
        raise ValueError("The 'lo_en' argument cannot be greater than 'hi_en'.")
    else:
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')

    extra_name = "_timebin{tb}_{l}-{u}keV".format(tb=time_bin_size.value,
                                                  l=lo_en.value,
                                                  u=hi_en.value)

    # Define a template eSASS command to generate light curves
    lc_cmd = 'cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="LC LCCORR" ' \
             'srcreg="{reg}" exttype="POINT" tstep={ts} insts={i} psftype="2D_PSF" ' \
             'lctype="{lct}" lcpars="{lcp}" lcemin="{le}" lcemax="{lm}" ' \
             'lcgamma="{lcg}" backreg="{breg}" pat_sel="{pat}";'

    # The DR1 version of eSASS has an additional argument that can be passed to specify
    #  which instruments should be written to output files - we want to be able to
    #  set that to avoid some warnings that clog up the logs
    if ESASS_VERSION == "ESASS4DR1":
        lc_cmd += " writeinsts={wi}"

    # From eSASS documentation:
    # "LC gamma - this parameter gives the photon index of the nominal power-law spectrum that will be used to
    # determine the weighting as a function of energy across the light-curve energy bands. This is necessary when
    # calculating the mean fractional response in each light-curve time bin."

    # Not really sure whether to give the user control of this, so for now we are
    #  setting the variable to the value stated in the srctool documentation, which
    #  we assume is the default - gamma=1.9
    lc_gamma = '1.9'

    # You can't control the whole name of the output of srctool, so this renames it to the XGA format
    rename_cmd = 'mv srctoolout_{i_no}??_{type}* {nn};'
    # Having a string to remove the 'merged' lightcurves that srctool outputs, even when you only
    #  request one instrument
    remove_merged_cmd = 'rm *srctoolout_0*;'
    # We also set up a command that will remove all lightcurves BUT the combined one, for when that is all the
    #  user wants
    remove_all_but_merged_cmd = "rm *srctoolout_*;"

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
            #  instantiate the LightCurve class once the eSASS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="light curve"))

            # Now we can continue with the rest of the sources
            continue

        for er_miss in ['erosita', 'erass']:
            # Skip this iteration if the current skew of eROSITA isn't associated
            #  with the current source
            if er_miss not in source.telescopes:
                continue

            # Need to set this so the combine_obs variable doesn't get overwritten
            use_combine_obs = combine_obs
            # If the user has set combine_obs to True and there is only one observation, then we
            #  use the combine_obs = False functionality instead
            if combine_obs and len(source.obs_ids[er_miss]) == 1:
                use_combine_obs = False

            # For convenience, we extract the current source's src region radii
            #  from the big array and put them in some variables
            src_inn_rad = inner_radii[s_ind]
            src_out_rad = outer_radii[s_ind]
            # Finding interloper regions within the radii we have specified has been put here because it all works in
            #  degrees and as such only needs to be run once for all the different observations.
            interloper_regions = source.regions_within_radii(src_inn_rad,
                                                             src_out_rad,
                                                             er_miss,
                                                             source.default_coord)

            # We repeat the exercise for the background region
            bck_inn_rad = outer_radii[s_ind] * source.background_radius_factors[0]
            bck_out_rad = outer_radii[s_ind] * source.background_radius_factors[1]
            # This finds any contaminating regions within the background area
            back_inter_reg = source.regions_within_radii(bck_inn_rad,
                                                         bck_out_rad,
                                                         er_miss,
                                                         source.default_coord)

            # The key under which these light curves will be stored
            lc_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}"
            lc_storage_name = lc_storage_name.format(ra=source.default_coord[0].value,
                                                     dec=source.default_coord[1].value,
                                                     ri=src_inn_rad.value,
                                                     ro=src_out_rad.value)

            # Adds on the extra information about time binning to the storage key
            lc_storage_name += extra_name

            if not use_combine_obs:
                # Check which event lists are associated with each individual source
                for evt_list in source.get_products("events", telescope=er_miss):
                    # This internal function uses the evtlist to prepare light
                    #  curve generation commands, final paths, and extra info
                    out_cmd, out_fin_paths, out_ex_info = _make_lc_cmd_info(evt_list)

                    # Add the output commands for the current event list to the overall
                    #  set of commands we've set up
                    cmds += out_cmd
                    # Same deal for the final paths and extra information dicts
                    final_paths += out_fin_paths
                    extra_info += out_ex_info

            else:
                # Getting the combined event list product
                evt_list = source.get_products("combined_events", telescope=er_miss)[0]

                # This internal function uses the combined event list to prepare light
                #  curve generation commands, final paths, and extra info
                out_cmd, out_fin_paths, out_ex_info = _make_lc_cmd_info(evt_list)

                # Add the output commands for the current event list to the overall set
                #  of commands we've set up and do the same for output paths and
                #  extra info
                cmds += out_cmd
                final_paths += out_fin_paths
                extra_info += out_ex_info

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the eSASS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="light curve"))

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@esass_call
def srctool_lightcurve(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                       inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                       lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                       time_bin_size: Quantity = Quantity(100, 's'), patt: int = 15,
                       num_cores: int = NUM_CORES, disable_progress: bool = False, combine_tm: bool = True,
                       combine_obs: bool = True, force_gen: bool = False):
    """
    A wrapper for all the SAS processes necessary to generate eROSITA light curves for a specified region.
     Every observation associated with this source, and every instrument associated with that
    observation, will have a light curve generated using the specified outer and inner radii as a boundary. The
    default inner radius is zero, so by default this function will produce light curves in a circular region out
    to the outer_radius.
    The light curves are corrected for background, vignetting, and PSF concerns using the eSASS 'srctool' tool.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the light curve (for instance 'point' would be acceptable for a Star or PointSource). If 'region' is chosen
        (to use the regions in region files), then any inner radius will be ignored. If you are generating for
        multiple sources then you can also pass a Quantity with one entry per source.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the light curve. By default this is zero arcseconds, resulting in a light curve from a circular region. If
        you are generating for multiple sources then you can also pass a Quantity with one entry per source.
    :param Quantity lo_en: The lower energy boundary for the light curve, in units of keV. The default is 0.5 keV.
    :param Quantity hi_en: The upper energy boundary for the light curve, in units of keV. The default is 2.0 keV.
    :param Quantity time_bin_size: The bin size to be used for the creation of the light curve, in
        seconds. The default is 100 s.
    :param int patt: An integer representation of a bitmask specifying which event patterns should be included. The
        default is 15 (i.e. all valid patterns).
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param bool combine_tm: Create lightcurves for individual ObsIDs that are a combination of the data from all the
        telescope modules utilized for that ObsID. This can help to offset the low signal-to-noise nature of the
        survey data eROSITA takes. Default is True.
    :param bool combine_obs: Setting this to False will generate an lightcurve for each associated observation, 
        instead of for one combined observation.
    :param bool force_gen: This boolean flag will force the regeneration of lightcurves, even if they already exist.
    """
    return _lc_cmds(sources, outer_radius, inner_radius, lo_en, hi_en, time_bin_size, patt, num_cores,
                    disable_progress, combine_tm, combine_obs, force_gen)
