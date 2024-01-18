#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 18/01/2024, 13:12. Copyright (c) The Contributors

import os
from copy import deepcopy
from random import randint
from typing import Union

from astropy.units import Quantity, UnitConversionError

from .._common import get_annular_esass_region
from ..sas._common import region_setup
from ... import OUTPUT, NUM_CORES
from ...exceptions import TelescopeNotAssociatedError
from ...samples.base import BaseSample
from ...sources import BaseSource


def _lc_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
             inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = Quantity(0.5, 'keV'),
             hi_en: Quantity = Quantity(2.0, 'keV'), time_bin_size: Quantity = Quantity(100, 's'),
             patt: str = '<= 4', num_cores: int = NUM_CORES, disable_progress: bool = False, combine_tm: bool = True,
             force_gen: bool = False):

    # We check to see whether there is an eROSITA entry in the 'telescopes' property. If sources is a Source
    #  object, then that property contains the telescopes associated with that source, and if it is a Sample object
    #  then 'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if eROSITA isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and 'erosita' not in sources.telescopes) or
            (isinstance(sources, list) and 'erosita' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no eROSITA data associated with the source/sample, as such "
                                          "eROSITA lightcurves cannot be generated.")

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    if outer_radius != 'region':
        sources, inner_radii, outer_radii = region_setup(sources, outer_radius, inner_radius, disable_progress,
                                                         '', num_cores)
    else:
        raise DeprecationWarning("Generating products from detection regions is no longer supported by XGA.")

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
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')

    # TODO PATTERNS

    extra_name = "_timebin{tb}_{l}-{u}keV".format(tb=time_bin_size, l=lo_en.value, u=hi_en.value)

    # TODO CLEARLY MUST BE REPLACED BY ESASS COMMANDS
    # Define the various SAS commands that need to be populated
    lc_cmd = "cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; evselect table={e} withrateset=yes " \
             "rateset={r} energycolumn=PI timebinsize={tbs} maketimecolumn=yes makeratecolumn=yes {ex}"

    lc_cmd = ('cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="LC LCCORR" srcreg="{reg}" exttype="POINT" '
              'tstep={ts} insts={i} psftype="2D_PSF" lctype="{lct}" lcpars="{lcp}" lcemin="{le}" lcemax="{lm}" '
              'lcgamma="{lcg}" backreg="{breg}";')

    # LC Gamma - This parameter gives the photon index of the nominal power-law spectrum that will be used to
    # determine the weighting as a function of energy across the light-curve energy bands, which is necessary when
    # calculating the mean fractional response in each light-curve time bin.
    # Not really sure whether to give the user control of this, so for now I am just setting a variable to the
    #  value stated in the srctool documentation, which I assume is the default - gamma=1.9
    lc_gamma = '1.9'

    # This command just makes a standard XCS image, but will be used to generate images to debug the drilling
    #  out of regions, as the light curve expression will be supplied, so we can see exactly what data has been removed.
    # debug_im = "evselect table={e} imageset={i} xcolumn=X ycolumn=Y ximagebinsize=87 " \
    #            "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize " \
    #            "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 withyranges=yes {ex}"
    #
    # lccorr_cmd = "epiclccorr srctslist={lc} eventlist={e} outset={clc} bkgtslist={blc} withbkgset=yes " \
    #              "applyabsolutecorrections=yes"

    # TODO These may not be valid for the lightcurve generation
    # You can't control the whole name of the output of srctool, so this renames it to the XGA format
    rename_cmd = 'mv srctoolout_{i_no}??_{type}* {nn}'
    # Having a string to remove the 'merged' spectra that srctool outputs, even when you only request one instrument
    remove_merged_cmd = 'rm *srctoolout_0*'
    # We also set up a command that will remove all lightcurves BUT the combined one, for when that is all the
    #  user wants (though honestly it seems wasteful to generate them all and not use them, this might change later
    remove_all_but_merged_cmd = "rm *srctoolout_*"

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []

    for s_ind, source in enumerate(sources):
        # By this point we know that at least one of the sources has eROSITA data associated (we checked that at the
        #  beginning of this function), so we're just skipping all the individual sources that don't have eROSITA data
        if 'erosita' not in source.telescopes:
            continue

        source: BaseSource
        cmds = []
        final_paths = []
        extra_info = []

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

        # The key under which these light curves will be stored
        lc_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}"
        lc_storage_name = lc_storage_name.format(ra=source.default_coord[0].value,
                                                 dec=source.default_coord[1].value,
                                                 ri=src_inn_rad_str, ro=src_out_rad_str)

        # Adds on the extra information about time binning to the storage key
        lc_storage_name += extra_name

        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", telescope='erosita', just_obj=False):
            # This one is simple, just extracting the current ObsID
            obs_id = pack[1]

            # Then we have to account for the two different modes this function can be used in - generating spectra
            #  for individual telescope models, or generating a single stacked spectrum for all telescope modules
            if combine_tm:
                inst_names = ['combined']
                inst_nums = ['"' + ' '.join([tm[-1] for tm in source.instruments["erosita"][obs_id]]) + '"']
                inst_srctool_id = ['0']
            else:
                inst_names = deepcopy(source.instruments["erosita"][obs_id])
                inst_nums = [tm[-1] for tm in source.instruments["erosita"][obs_id]]
                inst_srctool_id = inst_nums

            for inst_ind, inst in enumerate(inst_names):
                # Extracting just the instrument number for later use in eSASS commands (or indeed a list of instrument
                #  numbers if the user has requested a combined spectrum).
                inst_no = inst_nums[inst_ind]

                # TODO implement checking

                # try:
                #     # Got to check if this spectrum already exists
                #     check_sp = source.get_spectra(outer_radii[s_ind], obs_id, inst, inner_radii[s_ind], group_spec,
                #                                   min_counts, min_sn, telescope='erosita')
                #     exists = True
                # except NoProductAvailableError:
                #     exists = False

                # if exists and check_sp.usable and not force_gen:
                #     continue

                # Getting the source name
                source_name = source.name

                # Just grabs the event list object
                evt_list = pack[-1]
                # Sets up the file names of the output files, adding a random number so that the
                #  function for generating annular spectra doesn't clash and try to use the same folder
                # The temporary region files necessary to generate eROSITA spectra (if contaminating sources are
                #  being removed) will be written to a different temporary folder using the same random identifier.
                rand_ident = randint(0, 1e+8)
                dest_dir = OUTPUT + "erosita/" + "{o}/{i}_{n}_temp_{r}/".format(o=obs_id, i=inst, n=source_name,
                                                                                r=rand_ident)

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

                # lc_cmd = ('cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="LC LCCORR" srcreg="{reg}" exttype="POINT" '
                #               'tstep={ts} insts={i} psftype="2D_PSF" lctype="{lct}" lcpars="{lcp}" lcemin="{le}" lcemax="{lm}" '
                #               'lcgamma="{lcg}" backreg="{breg}"')

                # TODO ADD MANY MORE COMMENTS
                coord_str = "icrs;{ra}, {dec}".format(ra=source.default_coord[0].value,
                                                      dec=source.default_coord[1].value)
                src_reg_str = reg  # dealt with in get_annular_esass_region

                tstep = 0.5  # put it as 0.5 for now

                # TODO Decide on the best generation type (i.e. REGULAR OR REGULAR+/-
                cmd_str = lc_cmd.format(d=dest_dir, ef=evt_list.path, sc=coord_str, reg=src_reg_str, breg=b_reg,
                                        i=inst_no, ts=tstep, lct='REGULAR', lcp=str(time_bin_size),
                                        le=str(lo_en.value), lm=str(hi_en.value), lcg=str(lc_gamma))

                rename_srctool_id = inst_srctool_id[inst_ind]
                rename_lc = rename_cmd.format(i_no=rename_srctool_id, type='SourceSpec', nn=no_grp_spec)

                os.makedirs(dest_dir)
                print(cmd_str)
                print('')
                # print(rename_lc)

                stop


# @esass_call
# def evselect_lightcurve(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
#                         inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
#                         lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
#                         time_bin_size: Quantity = Quantity(100, 's'), pn_patt: str = '<= 4',
#                         mos_patt: str = '<= 12', num_cores: int = NUM_CORES, disable_progress: bool = False):
#     """
#     A wrapper for all the SAS processes necessary to generate XMM light curves for a specified region.
#      Every observation associated with this source, and every instrument associated with that
#     observation, will have a light curve generated using the specified outer and inner radii as a boundary. The
#     default inner radius is zero, so by default this function will produce light curves in a circular region out
#     to the outer_radius.
#     The light curves are corrected for background, vignetting, and PSF concerns using the SAS 'epiclccorr' tool.
#
#     :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
#     :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
#         the light curve (for instance 'point' would be acceptable for a Star or PointSource). If 'region' is chosen
#         (to use the regions in region files), then any inner radius will be ignored. If you are generating for
#         multiple sources then you can also pass a Quantity with one entry per source.
#     :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
#         the light curve. By default this is zero arcseconds, resulting in a light curve from a circular region. If
#         you are generating for multiple sources then you can also pass a Quantity with one entry per source.
#     :param Quantity lo_en: The lower energy boundary for the light curve, in units of keV. The default is 0.5 keV.
#     :param Quantity hi_en: The upper energy boundary for the light curve, in units of keV. The default is 2.0 keV.
#     :param Quantity time_bin_size: The bin size to be used for the creation of the light curve, in
#         seconds. The default is 100 s.
#     :param str pn_patt: The event selection pattern that should be applied for PN data. This should be a string
#         containing the selection expression of a valid XMM SAS pattern definition. For instance, the default for PN
#         is <= 4.
#     :param str mos_patt: The event selection pattern that should be applied for MOS data. This should be a string
#         containing the selection expression of a valid XMM SAS pattern definition. For instance, the default for MOS
#         is <= 12.
#     :param int num_cores: The number of cores to use (if running locally), default is set to
#         90% of available.
#     :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
#     """
#     return _lc_cmds(sources, outer_radius, inner_radius, lo_en, hi_en, time_bin_size, pn_patt, mos_patt,
#                     num_cores, disable_progress)
