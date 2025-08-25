#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 25/08/2025, 14:05. Copyright (c) The Contributors

import os
from random import randint
from typing import Union

import numpy as np
from astropy.units import Quantity, UnitConversionError

from ._common import region_setup, check_pattern
from .run import sas_call
from .. import OUTPUT, NUM_CORES
from ..exceptions import NoProductAvailableError
from ..samples.base import BaseSample
from ..sources import BaseSource


def _lc_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
             inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = Quantity(0.5, 'keV'),
             hi_en: Quantity = Quantity(2.0, 'keV'), time_bin_size: Quantity = Quantity(100, 's'),
             pn_patt: str = '<= 4', mos_patt: str = '<= 12', num_cores: int = NUM_CORES,
             disable_progress: bool = False, force_gen: bool = False):
    """
    This is an internal function which sets up the commands necessary to generate light curves from XMM data - and
    can be used both to generate them from simple circular regions and also from annular regions. The light curves
    are corrected for background, vignetting, and PSF concerns.

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
    :param str pn_patt: The event selection pattern that should be applied for PN data. This should be a string
        containing the selection expression of a valid XMM SAS pattern definition. For instance, the default for PN
        is <= 4.
    :param str mos_patt: The event selection pattern that should be applied for MOS data. This should be a string
        containing the selection expression of a valid XMM SAS pattern definition. For instance, the default for MOS
        is <= 12.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    if outer_radius != 'region':
        from_region = False
        sources, inner_radii, outer_radii = region_setup(sources, outer_radius, inner_radius, disable_progress,
                                                         '', num_cores)
    else:
        # This is used in the extra information dictionary for when the XGA light curve object is defined
        from_region = True

    if not isinstance(time_bin_size, Quantity) and isinstance(time_bin_size, (float, int)):
        time_bin_size = Quantity(time_bin_size, 's')
    elif not isinstance(time_bin_size, (Quantity, float, int)):
        raise TypeError("The 'time_bin_size' argument must be either an Astropy quantity, or an int/float (assumed to "
                        "be in seconds).")

    if not time_bin_size.unit.is_equivalent('s'):
        raise UnitConversionError("The 'time_bin_size' argument must be in units convertible to seconds.")
    else:
        time_bin_size = time_bin_size.to('s').value

    # Have to make sure that the user hasn't done anything daft here, hi_en must be larger than lo_en
    if lo_en >= hi_en:
        raise ValueError("The 'lo_en' argument cannot be greater than 'hi_en'.")
    else:
        # Converts the energies to channels for EPIC detectors, assuming one channel per eV
        lo_chan = int(lo_en.to('eV').value)
        hi_chan = int(hi_en.to('eV').value)

    pn_patt, pn_patt_name = check_pattern(pn_patt)
    mos_patt, mos_patt_name = check_pattern(mos_patt)

    extra_name = "_timebin{tb}_{l}-{u}keV".format(tb=time_bin_size, l=lo_en.value, u=hi_en.value)

    # Define the various SAS commands that need to be populated
    lc_cmd = "cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; evselect table={e} withrateset=yes " \
             "rateset={r} energycolumn=PI timebinsize={tbs} maketimecolumn=yes makeratecolumn=yes {ex}"

    # This command just makes a standard XCS image, but will be used to generate images to debug the drilling
    #  out of regions, as the light curve expression will be supplied, so we can see exactly what data has been removed.
    debug_im = "evselect table={e} imageset={i} xcolumn=X ycolumn=Y ximagebinsize=87 " \
               "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize " \
               "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 withyranges=yes {ex}"

    lccorr_cmd = "epiclccorr srctslist={lc} eventlist={e} outset={clc} bkgtslist={blc} withbkgset=yes " \
                 "applyabsolutecorrections=yes"

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for s_ind, source in enumerate(sources):
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
            # The key under which these light curves will be stored
            lc_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}"
            lc_storage_name = lc_storage_name.format(ra=source.default_coord[0].value,
                                                     dec=source.default_coord[1].value,
                                                     ri=src_inn_rad_str, ro=src_out_rad_str)
        else:
            lc_storage_name = "region"

        # Adds on the extra information about time binning to the storage key
        lc_storage_name += extra_name

        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", just_obj=False):
            obs_id = pack[0]
            inst = pack[1]
            try:
                # If we can find an exact match then we don't need to generate this light curve, it already exists, so
                #  we would move onto the next ObsID-instrument combo
                source.get_lightcurves(outer_radii[s_ind], obs_id, inst, inner_radii[s_ind], lo_en, hi_en,
                                       Quantity(time_bin_size, 's'), {'pn': pn_patt, 'mos': mos_patt})
                continue
            except NoProductAvailableError:
                pass

            if not os.path.exists(OUTPUT + obs_id):
                os.mkdir(OUTPUT + obs_id)

            # Got to check if this light curve already exists
            exists = source.get_products("light_curve", obs_id, inst, extra_key=lc_storage_name)
            if len(exists) == 1 and exists[0].usable and not force_gen:
                continue

            # If there is no match to a region, the source region returned by this method will be None,
            #  and if the user wants to generate light curves from region files, we have to ignore that observations
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

            # Some settings depend on the instrument
            if "pn" in inst:
                lc_storage_name = "_pattern{p}".format(p=pn_patt_name) + lc_storage_name
                expr = "expression='#XMMEA_EP && (PATTERN {p}) && (FLAG .eq. 0) && (PI in [{l}:{u}]) && " \
                       "{s}'".format(s=reg, p=pn_patt, l=lo_chan, u=hi_chan)
                b_expr = "expression='#XMMEA_EP && (PATTERN {p}) && (FLAG .eq. 0) && (PI in [{l}:{u}]) && " \
                         "{s}'".format(s=b_reg, p=pn_patt, l=lo_chan, u=hi_chan)

            elif "mos" in inst:
                lc_storage_name = "_pattern{p}".format(p=mos_patt_name) + lc_storage_name

                expr = "expression='#XMMEA_EM && (PATTERN {p}) && (FLAG .eq. 0) && (PI in [{l}:{u}]) && " \
                       "{s}'".format(s=reg, p=mos_patt, l=lo_chan, u=hi_chan)
                b_expr = "expression='#XMMEA_EM && (PATTERN {p}) && (FLAG .eq. 0) && (PI in [{l}:{u}]) && " \
                         "{s}'".format(s=b_reg, p=mos_patt, l=lo_chan, u=hi_chan)

            else:
                raise ValueError("You somehow have an illegal value for the instrument name...")

            # Some SAS tasks have issues with filenames with a '+' in them for some reason, so this  replaces any
            #  + symbols that may be in the source name with another character
            source_name = source.name.replace("+", "x")

            # Just grabs the event list object
            evt_list = pack[-1]
            # Sets up the file names of the output files, adding a random number
            dest_dir = OUTPUT + "{o}/{i}_{n}_temp_{r}/".format(o=obs_id, i=inst, n=source_name, r=randint(0, int(1e+8)))

            # We know that this is where the calibration index file lives
            ccf = dest_dir + "ccf.cif"

            lc = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_lcurve.fits"
            lc = lc.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                           dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                           ex=extra_name)

            b_lc = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_backlcurve.fits"
            b_lc = b_lc.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                               dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                               ex=extra_name)

            corr_lc = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_corrlcurve.fits"
            corr_lc = corr_lc.format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                     dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                     ex=extra_name)

            # These file names are for the debug images of the source and background images, they will not be loaded
            #  in as a XGA products, but exist purely to check by eye if necessary
            dim = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_debug." \
                  "fits".format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                ex=extra_name)
            b_dim = "{o}_{i}_{n}_ra{ra}_dec{dec}_ri{ri}_ro{ro}{ex}_back_debug." \
                    "fits".format(o=obs_id, i=inst, n=source_name, ra=source.default_coord[0].value,
                                  dec=source.default_coord[1].value, ri=src_inn_rad_str, ro=src_out_rad_str,
                                  ex=extra_name)

            # Fills out the evselect command to make the source and background light curves
            lc_cmd_str = lc_cmd.format(d=dest_dir, ccf=ccf, e=evt_list.path, r=lc, tbs=time_bin_size, ex=expr)

            lcb_cmd_str = lc_cmd.format(d=dest_dir, ccf=ccf, e=evt_list.path, r=b_lc, tbs=time_bin_size, ex=b_expr)

            # Then we fill out the command which performs all the corrections (using source and background LC
            #  that actually makes them usable)
            corr_lc_str = lccorr_cmd.format(lc=lc, e=evt_list.path, clc=corr_lc, blc=b_lc,)

            # Populates the debug image commands
            dim_cmd_str = debug_im.format(e=evt_list.path, ex=expr, i=dim)
            b_dim_cmd_str = debug_im.format(e=evt_list.path, ex=b_expr, i=b_dim)

            # Adds clean up commands to move all generated files and remove temporary directory
            cmd_str = '; '.join([lc_cmd_str, lcb_cmd_str, corr_lc_str, dim_cmd_str, b_dim_cmd_str])
            # Moves up all files that don't contain 'cif' and don't contain 'spectrum' (as that would be indicative
            #  of a failed process.
            # cmd_str += "; mv !(*spectrum*|*cif*) ../; cd ..; rm -r {d}".format(d=dest_dir)
            cmd_str += "; mv {o}*lcurve* ../; mv {o}*debug* ../; cd ..; rm -r {d}".format(d=dest_dir, o=obs_id)

            cmds.append(cmd_str)  # Adds the full command to the set
            # Makes sure the whole path to the temporary directory is created
            os.makedirs(dest_dir)

            final_paths.append(os.path.join(OUTPUT, obs_id, corr_lc))
            extra_info.append({"inner_radius": inn_rad_degrees, "outer_radius": out_rad_degrees,
                               "s_lc_path": os.path.join(OUTPUT, obs_id, lc),
                               "b_lc_path": os.path.join(OUTPUT, obs_id, b_lc),
                               "time_bin": time_bin_size,
                               "pattern": pn_patt if 'pn' in inst else mos_patt,
                               "obs_id": obs_id, "instrument": inst, "central_coord": source.default_coord,
                               "from_region": from_region,
                               "lo_en": lo_en,
                               "hi_en": hi_en})

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the SAS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="light curve"))

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@sas_call
def evselect_lightcurve(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                        inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                        lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
                        time_bin_size: Quantity = Quantity(100, 's'), pn_patt: str = '<= 4',
                        mos_patt: str = '<= 12', num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A wrapper for all the SAS processes necessary to generate XMM light curves for a specified region.
     Every observation associated with this source, and every instrument associated with that
    observation, will have a light curve generated using the specified outer and inner radii as a boundary. The
    default inner radius is zero, so by default this function will produce light curves in a circular region out
    to the outer_radius.
    The light curves are corrected for background, vignetting, and PSF concerns using the SAS 'epiclccorr' tool.

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
    :param str pn_patt: The event selection pattern that should be applied for PN data. This should be a string
        containing the selection expression of a valid XMM SAS pattern definition. For instance, the default for PN
        is <= 4.
    :param str mos_patt: The event selection pattern that should be applied for MOS data. This should be a string
        containing the selection expression of a valid XMM SAS pattern definition. For instance, the default for MOS
        is <= 12.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    return _lc_cmds(sources, outer_radius, inner_radius, lo_en, hi_en, time_bin_size, pn_patt, mos_patt,
                    num_cores, disable_progress)
