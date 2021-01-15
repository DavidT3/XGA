#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 15/01/2021, 13:07. Copyright (c) David J Turner

import os
import warnings
from shutil import rmtree
from typing import Union, Tuple, List

import numpy as np
from astropy.units import Quantity

from .misc import cifbuild
from .. import OUTPUT, NUM_CORES
from ..exceptions import SASInputInvalid
from ..samples.base import BaseSample
from ..sas.run import sas_call
from ..sources import BaseSource, ExtendedSource, GalaxyCluster
from ..sources.base import NullSource
from ..utils import RAD_LABELS


def _spec_setup(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                inner_radius: Union[str, Quantity], disable_progress: bool, obs_id: str) \
        -> Tuple[Union[BaseSource, BaseSample], List[Quantity], List[Quantity]]:
    """
    The preparation and value checking stage for SAS spectrum generation.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')).
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default this is zero arcseconds, resulting in a circular spectrum.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param str obs_id: Only used if the 'region' radius name is passed, the ObsID to retrieve the region for.
    :return: The source objects, a list of inner radius quantities,
    :rtype: Tuple[Union[BaseSource, BaseSample], List[Quantity], List[Quantity]]
    """
    # NullSources are not allowed to have spectra, as they can have any observations associated and thus won't
    #  necessarily overlap
    if isinstance(sources, NullSource):
        raise TypeError("You cannot create spectra of a NullSource")

    # Checking that the user hasn't passed BaseSources
    if not all([type(src) != BaseSource for src in sources]):
        raise TypeError("You cannot generate spectra from a BaseSource object, really you shouldn't be using "
                        "them at all, they are mostly useful as a superclass.")

    # Issuing a warning to the user that one or one sources have not been detected
    if not all([src.detected for src in sources]):
        warnings.warn("Not all of these sources have been detected, the spectra generated may not be helpful.")

    # Checking that inner radii that have been passed into the spectrum generation aren't nonsense
    if isinstance(inner_radius, str) and inner_radius not in RAD_LABELS:
        raise ValueError("You have passed a radius name rather than a value for 'inner_radius', but it is "
                         "not a valid name, please use one of the following:\n {}".format(", ".join(RAD_LABELS)))

    elif isinstance(inner_radius, str) and inner_radius in ["r2500", "r500", "r200"] and \
            not all([type(src) == GalaxyCluster for src in sources]):
        raise TypeError("The {} radius is only valid for GalaxyCluster objects".format(inner_radius))

    # One radius can be passed for a whole sample, but this checks to make sure that if there are multiple sources,
    #  and multiple radii have been passed, there are the same number of sources and radii
    elif isinstance(inner_radius, Quantity) and len(sources) != 1 and not inner_radius.isscalar \
            and len(sources) != len(inner_radius):
        raise ValueError("Your sample has {s} sources, but your inner_radius variable only has {i} entries. Please "
                         "pass only one inner_radius or the same number as there are "
                         "sources".format(s=len(sources), i=len(inner_radius)))

    # Checking that outer_radius radii that have been passed into the spectrum generation aren't nonsense
    if isinstance(outer_radius, str) and outer_radius not in RAD_LABELS:
        raise ValueError("You have passed a radius name rather than a value for 'outer_radius', but it is "
                         "not a valid name, please use one of the following:\n {}".format(", ".join(RAD_LABELS)))
    elif isinstance(outer_radius, str) and outer_radius in ["r2500", "r500", "r200"] and \
            not all([type(src) == GalaxyCluster for src in sources]):
        raise TypeError("The {} radius is only valid for GalaxyCluster objects".format(outer_radius))
    elif isinstance(outer_radius, Quantity) and len(sources) != 1 and not outer_radius.isscalar \
            and len(sources) != len(outer_radius):
        raise ValueError("Your sample has {s} sources, but your outer_radius variable only has {o} entries. Please "
                         "pass only one outer_radius or the same number as there are "
                         "sources".format(s=len(sources), o=len(outer_radius)))

    # A crude way to store the radii but I'm tired and this will work fine
    final_inner = []
    final_outer = []
    # I need to convert the radii to the same units and compare them, and to make sure they
    #  are actually in distance units. The distance unit checking is done by convert_radius
    for s_ind, src in enumerate(sources):
        # Converts the inner and outer radius for this source into the same unit
        if isinstance(outer_radius, str) and outer_radius != 'region':
            cur_out_rad = src.get_radius(outer_radius, 'deg')
        elif isinstance(outer_radius, str) and outer_radius == 'region':
            reg = src.source_back_regions('region', obs_id)[0]
            cur_out_rad = Quantity([reg.width.to('deg').value/2, reg.height.to('deg').value/2], 'deg')
        elif outer_radius.isscalar:
            cur_out_rad = src.convert_radius(outer_radius, 'deg')
        else:
            cur_out_rad = src.convert_radius(outer_radius[s_ind], 'deg')

        # We need to check that the outer radius isn't region, because for region objects we ignore whatever
        #  inner radius has been passed and just set it 0
        if outer_radius == 'region':
            cur_inn_rad = Quantity([0, 0], 'deg')
        elif isinstance(inner_radius, str):
            cur_inn_rad = src.get_radius(inner_radius, 'deg')
        elif inner_radius.isscalar:
            cur_inn_rad = src.convert_radius(inner_radius, 'deg')
        else:
            cur_inn_rad = src.convert_radius(inner_radius[s_ind], 'deg')

        # Then we can check to make sure that the outer radius is larger than the inner radius
        if outer_radius != 'region' and cur_inn_rad >= cur_out_rad:
            raise ValueError("The inner_radius of {s} is greater than or equal to the outer_radius".format(s=src.name))
        else:
            final_inner.append(cur_inn_rad)
            final_outer.append(cur_out_rad)

    # Have to make sure that all observations have an up to date cif file.
    cifbuild(sources, disable_progress=disable_progress)

    return sources, final_inner, final_outer


def _spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
               min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
               num_cores: int = NUM_CORES, disable_progress: bool = False):
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
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    if outer_radius != 'region':
        sources, inner_radii, outer_radii = _spec_setup(sources, outer_radius, inner_radius, disable_progress, '')

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

    # Define the various SAS commands that need to be populated, for a useful spectrum you also need ARF/RMF
    spec_cmd = "cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; evselect table={e} withspectrumset=yes " \
               "spectrumset={s} energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 " \
               "specchannelmax={u} {ex}"

    rmf_cmd = "rmfgen rmfset={r} spectrumset='{s}' detmaptype=flat extendedsource={es}"

    # Don't need to run backscale separately, as this arfgen call will do it automatically
    arf_cmd = "arfgen spectrumset='{s}' arfset={a} withrmfset=yes rmfset='{r}' badpixlocation={e} " \
              "extendedsource={es} detmaptype=flat setbackscale=yes"

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
        else:
            ex_src = "no"
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
            spec_storage_name = "region"

        # Adds on the extra information about grouping to the storage key
        spec_storage_name += extra_name

        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", just_obj=False):
            obs_id = pack[0]
            inst = pack[1]

            if not os.path.exists(OUTPUT + obs_id):
                os.mkdir(OUTPUT + obs_id)

            # TODO restore all this when new storage system is in place

            # Got to check if this spectrum already exists
            # exists = [match for match in source.get_products("spectrum", obs_id, inst, just_obj=False)
            #           if outer_radius in match]
            # if len(exists) == 1 and exists[0][-1].usable:
            #     continue

            # If there is no match to a region, the source region returned by this method will be None,
            #  and if the user wants to generate spectra from region files, we have to ignore that observations
            # if outer_radius == "region" and source.source_back_regions("region", obs_id)[0] is None:
            #     continue

            # Because the region will be different for each ObsID, I have to call the setup function here
            if outer_radius == 'region':
                interim_source, inner_radii, outer_radii = _spec_setup([source], outer_radius, inner_radius,
                                                                       disable_progress, obs_id)
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

            else:
                # This constructs the sas strings for any radius that isn't 'region'
                reg = source.get_annular_sas_region(inner_radii[s_ind], outer_radii[s_ind], obs_id, inst,
                                                    interloper_regions=interloper_regions,
                                                    central_coord=source.default_coord)
                b_reg = source.get_annular_sas_region(outer_radii[s_ind] * source.background_radius_factors[0],
                                                      outer_radii[s_ind] * source.background_radius_factors[1], obs_id,
                                                      inst, interloper_regions=back_inter_reg,
                                                      central_coord=source.default_coord)

            # Some settings depend on the instrument, XCS uses different patterns for different instruments
            if "pn" in inst:
                # Also the upper channel limit is different for EPN and EMOS detectors
                spec_lim = 20479
                expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0) && {s}'".format(s=reg)
                b_expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0) && {s}'".format(s=b_reg)
            elif "mos" in inst:
                spec_lim = 11999
                expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0) && {s}'".format(s=reg)
                b_expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0) && {s}'".format(s=b_reg)
            else:
                raise ValueError("You somehow have an illegal value for the instrument name...")

            # Some of the SAS tasks have issues with filenames with a '+' in them for some reason, so this
            #  replaces any + symbols that may be in the source name with another character
            source_name = source.name.replace("+", "x")

            # Just grabs the event list object
            evt_list = pack[-1]
            # Sets up the file names of the output files
            dest_dir = OUTPUT + "{o}/{i}_{n}_temp/".format(o=obs_id, i=inst, n=source_name)
            spec = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_spec.fits".format(o=obs_id, i=inst, n=source_name,
                                                                                ra=source.default_coord[0].value,
                                                                                dec=source.default_coord[1].value,
                                                                                ri=src_inn_rad_str,
                                                                                ro=src_out_rad_str)
            b_spec = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_backspec.fits".format(o=obs_id, i=inst, n=source_name,
                                                                                      ra=source.default_coord[0].value,
                                                                                      dec=source.default_coord[1].value,
                                                                                      ri=src_inn_rad_str,
                                                                                      ro=src_out_rad_str)
            arf = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}.arf".format(o=obs_id, i=inst, n=source_name,
                                                                         ra=source.default_coord[0].value,
                                                                         dec=source.default_coord[1].value,
                                                                         ri=src_inn_rad_str, ro=src_out_rad_str)
            b_arf = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}_back.arf".format(o=obs_id, i=inst, n=source_name,
                                                                                ra=source.default_coord[0].value,
                                                                                dec=source.default_coord[1].value,
                                                                                ri=src_inn_rad_str,
                                                                                ro=src_out_rad_str)
            ccf = dest_dir + "ccf.cif"

            # Fills out the evselect command to make the main and background spectra
            s_cmd_str = spec_cmd.format(d=dest_dir, ccf=ccf, e=evt_list.path, s=spec, u=spec_lim, ex=expr)
            sb_cmd_str = spec_cmd.format(d=dest_dir, ccf=ccf, e=evt_list.path, s=b_spec, u=spec_lim, ex=b_expr)

            # This chunk adds rmfgen commands depending on whether we're using a universal RMF or
            #  an individual one for each spectrum. Also adds arfgen commands on the end, as they depend on
            #  the rmf.
            if one_rmf:
                rmf = "{o}_{i}_{n}_universal.rmf".format(o=obs_id, i=inst, n=source_name)
                b_rmf = rmf
            else:
                rmf = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}.rmf".format(o=obs_id, i=inst, n=source_name,
                                                                             ra=source.default_coord[0].value,
                                                                             dec=source.default_coord[1].value,
                                                                             ri=src_inn_rad_str, ro=src_out_rad_str)
                b_rmf = "{o}_{i}_{n}_ra{ra}+dec{dec}_ri{ri}_ro{ro}_back.rmf".format(o=obs_id, i=inst, n=source_name,
                                                                                    ra=source.default_coord[0].value,
                                                                                    dec=source.default_coord[1].value,
                                                                                    ri=src_inn_rad_str,
                                                                                    ro=src_out_rad_str)

            if one_rmf and not os.path.exists(dest_dir + rmf):
                cmd_str = ";".join([s_cmd_str, rmf_cmd.format(r=rmf, s=spec, es=ex_src),
                                    arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path, es=ex_src), sb_cmd_str,
                                    arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path, es=ex_src)])
            elif not one_rmf and not os.path.exists(dest_dir + rmf):
                cmd_str = ";".join([s_cmd_str, rmf_cmd.format(r=rmf, s=spec, es=ex_src),
                                    arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path, es=ex_src)]) + ";"
                cmd_str += ";".join([sb_cmd_str, rmf_cmd.format(r=b_rmf, s=b_spec, es=ex_src),
                                     arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path, es=ex_src)])
            else:
                cmd_str = ";".join([s_cmd_str, arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path,
                                                              es=ex_src)]) + ";"
                cmd_str += ";".join([sb_cmd_str, arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path,
                                                                es=ex_src)])

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
            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)
            # Makes sure the whole path to the temporary directory is created
            os.makedirs(dest_dir)

            final_paths.append(os.path.join(OUTPUT, obs_id, spec))
            extra_info.append({"reg_type": outer_radius, "rmf_path": os.path.join(OUTPUT, obs_id, rmf),
                               "arf_path": os.path.join(OUTPUT, obs_id, arf),
                               "b_spec_path": os.path.join(OUTPUT, obs_id, b_spec),
                               "b_rmf_path": os.path.join(OUTPUT, obs_id, b_rmf),
                               "b_arf_path": os.path.join(OUTPUT, obs_id, b_arf),
                               "obs_id": obs_id, "instrument": inst})

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
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default this is zero arcseconds, resulting in a circular spectrum.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    # All the workings of this function are in _spec_cmds so that the annular spectrum set generation function
    #  can also use them
    return _spec_cmds(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn, over_sample, one_rmf,
                      num_cores, disable_progress)


def evselect_annular_spectrum_set():
    raise NotImplementedError("Haven't quite got around to doing this bit yet")




