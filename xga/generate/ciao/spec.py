#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 26/03/2025, 10:57. Copyright (c) The Contributors

import os
from copy import copy
from random import randint
from typing import Union, List

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from xga import OUTPUT, NUM_CORES, xga_conf
from xga.exceptions import NoProductAvailableError, TelescopeNotAssociatedError
from xga.generate.sas._common import region_setup
from xga.samples.base import BaseSample
from xga.sources import BaseSource
from xga.sources.base import NullSource
from .run import ciao_call


def _chandra_spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                       inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                       min_counts: int = 5, min_sn: float = None, over_sample: int = None, num_cores: int = NUM_CORES,
                       disable_progress: bool = False, force_gen: bool = False):
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
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    """

    stack = False  # This tells the ciao_call routine that this command won't be part of a stack.
    execute = True  # This should be executed immediately.
    # This function supports passing both individual sources and samples - but if it is a source we like to be able
    #  to iterate over it, so we put it in a list.
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    sources, inner_radii, outer_radii = region_setup(sources, outer_radius=outer_radius, inner_radius=inner_radius,
                                                     disable_progress=False, obs_id='', num_cores=num_cores)

    # Check if the source/sample has Chandra data.
    if not isinstance(sources, list) and "chandra" not in sources.telescopes:
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the source.")
    elif isinstance(sources, list) and "chandra" not in sources[0].telescopes:
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the sample.")

    # TODO CHECK THESE ARE GOOD ENERGY RANGES AND THEN UNCOMMENT OR UPDATE THEN UNCOMMENT
    # # Checking user's lo_en and hi_en inputs are in the valid energy range for Chandra
    # if ((lo_en < Quantity(0.5, 'keV') or lo_en > Quantity(7.0, 'keV')) or
    #         (hi_en < Quantity(0.5, 'keV') or hi_en > Quantity(7.0, 'keV'))):
    #     raise ValueError("The 'lo_en' and 'hi_en' value must be between 0.5 keV and 7 keV.")

    # These lists are to contain the lists of commands/paths/etc for each of the individual sources passed
    # to this function.    
    sources_cmds = []
    sources_paths = []
    # This contains any other information that will be needed to instantiate the class
    # once the CIAO cmd has run.
    sources_extras = []
    sources_types = []
    for s_ind, source in enumerate(sources):
        # Explicitly states that source is at very least a BaseSource instance - useful for code completion in IDEs
        source: BaseSource

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

        inner_r_arc = source.convert_radius(inner_radii[s_ind], 'arcmin')
        outer_r_arc = source.convert_radius(outer_radii[s_ind], 'arcmin')

        source_name = source.name.replace("+", "x")
        ra_src, dec_src = source.default_coord[0], source.default_coord[1]

        ra_src_str, dec_src_str = ra_src.value, dec_src.value
        inner_radius_str = source.convert_radius(inner_radii[s_ind], 'deg').value
        outer_radius_str = source.convert_radius(outer_radii[s_ind], 'deg').value

        # Iterate through Chandra event lists associated with the source.
        for product in source.get_products("events", telescope="chandra", just_obj=True):
            # Getting the current ObsID, instrument, and event file path
            evt_file = product
            obs_id = evt_file.obs_id
            inst = evt_file.instrument

            # Grabbing the attitude and badpix files, which the CIAO command we aim to run will want as an input
            att_file = source.get_att_file(obs_id, 'chandra')

            # We haven't added a particular get method for bad-pixel files, as they are not as universal as attitude
            #  files - the badpix files have been stored in the source product storage framework however, as we loaded
            #  them along with all the files specified in the config file
            badpix_prod = source.get_products("badpix", telescope="chandra", obs_id=obs_id, inst=inst)
            if len(badpix_prod) > 1:
                raise ValueError("Found multiple bad pixel files for Chandra {o}-{i}; this should not be "
                                 "possible, please contact the developer.".format(o=obs_id, i=inst))
            elif len(badpix_prod) == 0:
                raise ValueError("No bad pixel has been read in for Chandra {o}-{i}, please check the bad pixel path"
                                 " entered in the XGA configuration file - "
                                 "{bpf}".format(o=obs_id, i=inst,
                                                bpf=xga_conf['CHANDRA_FILES']['{i}_badpix_file'.format(i=inst)]))

            # Now we've established that we have retrieved a single bad pixel product, we extract the path from it
            badpix_file = badpix_prod[0].path

            mask_prod = source.get_products("maskfile", telescope="chandra", obs_id=obs_id, inst=inst)
            if len(mask_prod) > 1:
                raise ValueError("Found multiple mask files for Chandra {o}-{i}; this should not be "
                         "possible, please contact the developer.".format(o=obs_id, i=inst))
            elif len(mask_prod) == 0:
                raise ValueError("No mask has been read in for Chandra {o}-{i}, please check the mask path"
                                 " entered in the XGA configuration file - "
                                 "{mask}".format(o=obs_id, i=inst,
                                                 mask=xga_conf['CHANDRA_FILES']['{i}_mask_file'.format(i=inst)]))

            # Now we've established that we have retrieved a single mask file product, we extract the path from it
            mask_file = mask_prod[0].path

            # Check do we need to group the spec
            if group_spec and min_counts is not None:
                extra_file_name = "_mincnt{c}".format(c=min_counts)
                grouptype_int = 'NUM_CTS'
                binspec_int = min_counts
            elif group_spec and min_sn is not None:
                extra_file_name = "_minsn{s}".format(s=min_sn)
                grouptype_int = 'SNR'
                binspec_int = min_sn
            else:
                extra_file_name = ''
                grouptype_int = 'NONE'
                binspec_int = 'NONE'

            # Do we actually need to run this generation? If matching products already exist then we won't bother
            try:
                source.get_spectra(obs_id=obs_id, inst=inst, telescope='chandra', outer_radius=outer_radii[s_ind],
                                   inner_radius=inner_radii[s_ind], group_spec=group_spec, min_counts=min_counts,
                                   min_sn=min_sn)
                # If the expected outputs from this function do exist for the current ObsID, we'll just
                #  move on to the next one
                continue
            except NoProductAvailableError:
                pass

            # Setting up the top level path for the eventual destination of the products to be generated here
            dest_dir = os.path.join(OUTPUT, "chandra", obs_id)

            # Temporary directory for fluximage.
            temp_dir = os.path.join(dest_dir, f"temp_{randint(0, int(1e8))}")
            os.makedirs(temp_dir, exist_ok=True)

            # Just for group spec at this point, but need to add ungrouped later
            spec_name = (f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_"
                         f"ro{outer_radius_str}_grp{group_spec}{extra_file_name}_spec.fits")
            spec_file = os.path.join(dest_dir, spec_name)
            if group_spec is not None:
                spec_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}_grp.pi")
            else:
                spec_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}.pi")

            arf_name = (f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_"
                        f"ro{outer_radius_str}_grp{group_spec}{extra_file_name}.arf")
            arf_file = os.path.join(dest_dir, arf_name)
            arf_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}.arf")

            rmf_name = (f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_"
                        f"ro{outer_radius_str}_grp{group_spec}{extra_file_name}.rmf")
            rmf_file = os.path.join(dest_dir, rmf_name)
            rmf_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}.rmf")

            bkg_spec_name = (f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_"
                             f"ro{outer_radius_str}_grp{group_spec}{extra_file_name}_backspec.fits")
            bkg_spec_file = os.path.join(dest_dir, bkg_spec_name)
            bkg_spec_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}_bkg.pi")

            bkg_arf_name = (f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_"
                            f"ro{outer_radius_str}_grp{group_spec}{extra_file_name}_back.arf")
            bkg_arf_file = os.path.join(dest_dir, bkg_arf_name)
            bkg_arf_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}_bkg.arf")

            bkg_rmf_name = (f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_"
                            f"ro{outer_radius_str}_grp{group_spec}{extra_file_name}_back.rmf")
            bkg_rmf_file = os.path.join(dest_dir, bkg_rmf_name)
            bkg_rmf_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}_bkg.rmf")

            coord = SkyCoord(ra=ra_src, dec=dec_src, frame='icrs')

            ra_hms = coord.ra.to_string(unit=u.hour, sep=':', precision=5)
            dec_dms = coord.dec.to_string(unit=u.deg, sep=':', precision=5, alwayssign=True)

            bkg_inner_r_arc = outer_r_arc * source.background_radius_factors[0]
            bkg_outer_r_arc = outer_r_arc * source.background_radius_factors[1]

            # Ensure the directory exists
            temp_region_dir = os.path.join(dest_dir, f"temp_region_{randint(0, int(1e8))}") #added random numbers
            os.makedirs(temp_region_dir, exist_ok=True)
            # Define file paths
            spec_ext_reg_path = os.path.join(temp_region_dir, f"{obs_id}_{inst}_spec_ext_temp.reg")
            spec_bkg_reg_path = os.path.join(temp_region_dir, f"{obs_id}_{inst}_spec_bkg_temp.reg")

            ext_inter_reg = source.regions_within_radii(inner_r_arc,
                                                        outer_r_arc,
                                                        "chandra", source.default_coord)

            # Write the extraction region file (annulus between inner_r and outer_r)
            with open(spec_ext_reg_path, 'w') as ext_reg:
                ext_reg.write("# Region file format: DS9 version 4.1\n")
                ext_reg.write("fk5\n")
                ext_reg.write(f"annulus({ra_hms},{dec_dms},{inner_r_arc.value}',{outer_r_arc.value}')\n")
                # Add exclusion regions if provided
                for region in ext_inter_reg:
                    reg_ra = region.center.ra.to_string(unit=u.hour, sep=':', precision=5)
                    reg_dec = region.center.dec.to_string(unit=u.deg, sep=':', precision=5, alwayssign=True)

                    # Convert width and height to arcmin
                    width_arc = region.width.to(u.arcmin).value
                    height_arc = region.height.to(u.arcmin).value
                    angle = region.angle.to(u.deg).value

                    # Write the exclusion region in ellipse format
                    ext_reg.write(f"-ellipse({reg_ra},{reg_dec},{width_arc}',{height_arc}',{angle})\n")

            bkg_inter_reg = source.regions_within_radii(outer_r_arc * source.background_radius_factors[0],
                                                        outer_r_arc * source.background_radius_factors[1],
                                                        "chandra", source.default_coord)
            # Write the background region file (annulus between outer_r and bkg_r)
            with open(spec_bkg_reg_path, 'w') as bkg_reg:
                bkg_reg.write("# Region file format: DS9 version 4.1\n")
                bkg_reg.write("fk5\n")
                bkg_reg.write(f"annulus({ra_hms},{dec_dms},{bkg_inner_r_arc.value}',{bkg_outer_r_arc.value}')\n")

                # Add exclusion regions if provided
                for region in bkg_inter_reg:
                    reg_ra = region.center.ra.to_string(unit=u.hour, sep=':', precision=5)
                    reg_dec = region.center.dec.to_string(unit=u.deg, sep=':', precision=5, alwayssign=True)

                    # Convert width and height to arcmin
                    width_arc = region.width.to(u.arcmin).value
                    height_arc = region.height.to(u.arcmin).value
                    angle = region.angle.to(u.deg).value

                    # Write the exclusion region in ellipse format
                    bkg_reg.write(f"-ellipse({reg_ra},{reg_dec},{width_arc}',{height_arc}',{angle})\n")

            new_pfiles = os.path.join(temp_dir, 'pfiles/')
            os.makedirs(new_pfiles, exist_ok=True)

            # Build specextract command - making sure to set parallel to no, seeing as we're doing our
            #  own parallelization
            specextract_cmd = (
                f"export PFILES=\"{new_pfiles}:$PFILES\"; "
                f"export HEADASNOQUERY=; export HEADASPROMPT=/dev/null; "
                f"punlearn; "
                f"cd {temp_dir}; specextract infile=\"{evt_file.path}[sky=region({spec_ext_reg_path})]\" "
                f"outroot={obs_id}_{inst} bkgfile=\"{evt_file.path}[sky=region({spec_bkg_reg_path})]\" "
                f"asp={att_file} badpixfile={badpix_file} grouptype={grouptype_int} binspec={binspec_int} "
                f"weight=yes weight_rmf=no clobber=yes parallel=no mskfile={mask_file} tmpdir={temp_dir}; "
                f"mv {spec_ciao_out} {spec_file}; mv {arf_ciao_out} {arf_file}; mv {rmf_ciao_out} {rmf_file}; "
                f"mv {bkg_spec_ciao_out} {bkg_spec_file}; mv {bkg_arf_ciao_out} {bkg_arf_file}; mv {bkg_rmf_ciao_out} {bkg_rmf_file}; "
                # f"rm -r {temp_dir}; rm -r {temp_region_dir}"
            )

            cmds.append(specextract_cmd)

            final_paths.append(spec_file)
            # This is the products final resting place, if it exists at the end of this command.
            extra_info.append({"inner_radius": inner_r_arc, "outer_radius": outer_r_arc,
                               "rmf_path": rmf_file,
                               "arf_path": arf_file,
                               "b_spec_path": bkg_spec_file,
                               "b_rmf_path": bkg_rmf_file,
                               "b_arf_path": bkg_arf_file,
                               "obs_id": obs_id, "instrument": inst, "grouped": group_spec, "min_counts": min_counts,
                               "min_sn": min_sn, "over_sample": None, "central_coord": source.default_coord,
                               "from_region": False,
                               "telescope": 'chandra'})

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the CIAO cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))

    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@ciao_call
def specextract_spectrum(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                         inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                         min_counts: int = 5, min_sn: float = None, over_sample: int = None, num_cores: int = NUM_CORES,
                         disable_progress: bool = False):
    """
    A wrapper for all the CIAO processes necessary to generate Chandra spectra that can be analysed
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
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    """
    # All the workings of this function are in _chandra_spec_cmds so that the annular spectrum set generation function
    #  can also use them
    return _chandra_spec_cmds(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn,
                              over_sample, num_cores, disable_progress)


@ciao_call
def ciao_spectrum_set(sources: Union[BaseSource, BaseSample], radii: Union[List[Quantity], Quantity],
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
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal-to-noise in each channel.
        To disable minimum signal-to-noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool force_regen: This will force all the constituent spectra of the set to be regenerated, use this
        if your call to this function was interrupted and an incomplete AnnularSpectrum is being read in.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    """
    # We check to see whether there is an Chandra entry in the 'telescopes' property. If sources is a Source object, then
    #  that property contains the telescopes associated with that source, and if it is a Sample object then
    #  'telescopes' contains the list of unique telescopes that are associated with at least one member source.
    # Clearly if Chandra isn't associated at all, then continuing with this function would be pointless
    if ((not isinstance(sources, list) and 'chandra' not in sources.telescopes) or
            (isinstance(sources, list) and 'chandra' not in sources[0].telescopes)):
        raise TelescopeNotAssociatedError("There are no Chandra data associated with the source/sample, as such `Chandra` "
                                          "spectra cannot be generated.")

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

    # This generates a spectra between the innermost and outmost radii for each source, and a universal RMF
    if one_rmf:
        innermost_rads = Quantity([r_set[0] for r_set in radii], radii[0].unit)
        outermost_rads = Quantity([r_set[-1] for r_set in radii], radii[0].unit)
        _chandra_spec_cmds(sources, outermost_rads, innermost_rads, group_spec, min_counts, min_sn, over_sample,
                           num_cores, disable_progress)

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

        # By this point we know that at least one of the sources has Chandra data associated (we checked that at the
        #  beginning of this function), we still need to append the empty cmds, paths, extrainfo, and ptypes to
        #  the final output, so that the cmd_list and input argument 'sources' have the same length, which avoids
        #  bugs occuring in the ciao_call wrapper
        if 'chandra' not in source.telescopes:
            all_cmds.append(np.array(src_cmds))
            all_paths.append(np.array(src_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the CIAO cmd has run
            all_extras.append(np.array(src_extras))
            all_out_types.append(src_out_types)

            # then we can continue with the rest of the sources
            continue

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

        exists = source.get_products('combined_spectrum', extra_key=spec_storage_name, telescope='chandra')
        if len(exists) == 0:
            # If it doesn't exist then we do need to call evselect_spectrum
            generate_spec = True
        else:
            # If it already exists though we don't need to bother
            generate_spec = False

        if generate_spec or force_regen:
            # Here we run through all the requested annuli for the current source
            for r_ind in range(len(radii[s_ind])-1):
                # Generate the CIAO commands for the current annulus of the current source, for all observations
                spec_cmd_out = _chandra_spec_cmds(source, radii[s_ind][r_ind+1], radii[s_ind][r_ind], group_spec, min_counts,
                                                  min_sn, over_sample, num_cores, disable_progress)

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
                    #  the CIAO commands as well
                    cur_cmd = cur_cmd.replace(split_p[-1], new_spec)

                    # Add the new filename back into the split spec file path
                    split_p[-1] = new_spec

                    # Add an annulus identifier to the extra_info dictionary
                    interim_extras[p_ind].update({"set_ident": set_id, "ann_ident": r_ind})

                    # Only need to modify the RMF paths if the universal RMF HASN'T been used
                    if "universal" not in interim_extras[p_ind]['rmf_path']:
                        # Much the same process as with the spectrum name
                        split_r = copy(interim_extras[p_ind]['rmf_path']).split('/')
                        split_br = copy(interim_extras[p_ind]['b_rmf_path']).split('/')
                        new_rmf = split_r[-1].replace('.rmf', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) + ".rmf"
                        new_b_rmf = split_br[-1].replace('_back.rmf', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) \
                                    + "_back.rmf"

                        # Replacing the names in the CIAO commands
                        cur_cmd = cur_cmd.replace(split_r[-1], new_rmf)
                        cur_cmd = cur_cmd.replace(split_br[-1], new_b_rmf)

                        split_r[-1] = new_rmf
                        split_br[-1] = new_b_rmf

                        # Adding the new RMF paths into the extra info dictionary
                        interim_extras[p_ind].update({"rmf_path": "/".join(split_r), "b_rmf_path": "/".join(split_br)})
                        # interim_extras[p_ind].update({"rmf_path": "/".join(split_r)})

                    # Same process as RMFs but for the ARF, background ARF, and background spec
                    split_a = copy(interim_extras[p_ind]['arf_path']).split('/')
                    split_ba = copy(interim_extras[p_ind]['b_arf_path']).split('/')
                    split_bs = copy(interim_extras[p_ind]['b_spec_path']).split('/')
                    new_arf = split_a[-1].replace('.arf', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) + ".arf"
                    new_b_arf = split_ba[-1].replace('_back.arf', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) \
                                + "_back.arf"
                    new_b_spec = split_bs[-1].replace('_backspec.fits', "_ident{si}_{ai}".format(si=set_id, ai=r_ind)) \
                                + "_backspec.fits"

                    # New names into the commands
                    cur_cmd = cur_cmd.replace(split_a[-1], new_arf)
                    cur_cmd = cur_cmd.replace(split_ba[-1], new_b_arf)
                    cur_cmd = cur_cmd.replace(split_bs[-1], new_b_spec)

                    split_a[-1] = new_arf
                    split_ba[-1] = new_b_arf
                    split_bs[-1] = new_b_spec

                    # Update the extra info dictionary some more
                    interim_extras[p_ind].update({"arf_path": "/".join(split_a), "b_arf_path": "/".join(split_ba),
                                                  "b_spec_path": "/".join(split_bs)})
                    #interim_extras[p_ind].update({"arf_path": "/".join(split_a), "b_spec_path": "/".join(split_bs)})

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

    # This gets passed back to the ciao call function and is used to run the commands
    return all_cmds, False, True, num_cores, all_out_types, all_paths, all_extras, disable_progress