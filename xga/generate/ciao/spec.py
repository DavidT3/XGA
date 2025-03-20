#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by Ray Wang (wangru46@msu.edu) 02/19/2025, 16:16. Copyright (c) The Contributors

import os
from copy import copy
from itertools import permutations
from random import randint
from typing import Union, List

import numpy as np
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

from xga import OUTPUT, NUM_CORES, xga_conf
from xga.exceptions import NoProductAvailableError, TelescopeNotAssociatedError
from xga.samples.base import BaseSample
from xga.sources import BaseSource, ExtendedSource, GalaxyCluster
from xga.sources.base import NullSource
from .run import ciao_call


def _chandra_spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
               min_counts: int = 5, min_sn: float = None, over_sample: int = None, one_rmf: bool = True,
               num_cores: int = NUM_CORES, disable_progress: bool = False, force_gen: bool = False):
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
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF per observation.
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
    for source in sources:
        # Explicitly states that source is at very least a BaseSource instance - useful for code completion in IDEs
        source: BaseSource

        cmds = []
        final_paths = []
        extra_info = []
        
        # Skip sources that do not have Chandra data.
        # if 'chandra' not in source.telescopes:
        #     sources_cmds.append(np.array(cmds))
        #     sources_paths.append(np.array(final_paths))
        #     sources_extras.append(np.array(extra_info))
        #     sources_types.append(np.full(len(cmds), fill_value="spectrum"))
        #     continue
        inner_r_arc = source.convert_radius(inner_radius, 'arcmin')
        outer_r_arc = source.convert_radius(outer_radius, 'arcmin')
        print('inner/outer r:', inner_r_arc, outer_r_arc)
        source_name = source.name.replace("+", "x")
        ra_src, dec_src = source.default_coord[0], source.default_coord[1]
        
        ra_src_str, dec_src_str = ra_src.value, dec_src.value
        inner_radius_str, outer_radius_str = source.convert_radius(inner_radius, 'deg').value, source.convert_radius(outer_radius, 'deg').value
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
                source.get_spectra(obs_id=obs_id, inst=inst, telescope='chandra', outer_radius=outer_radius, inner_radius=inner_radius)
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
            spec_name = f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_ro{outer_radius_str}_grp{group_spec}{extra_file_name}_spec.fits"
            spec_file = os.path.join(dest_dir, spec_name)
            if group_spec is not None:
                spec_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}_grp.pi")
            else:
                spec_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}.pi")
            
            arf_name = f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_ro{outer_radius_str}_grp{group_spec}{extra_file_name}.arf"
            arf_file = os.path.join(dest_dir, arf_name)
            arf_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}.arf")
            
            rmf_name = f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_ro{outer_radius_str}_grp{group_spec}{extra_file_name}.rmf"
            rmf_file = os.path.join(dest_dir, rmf_name)
            rmf_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}.rmf")
            
            bkg_spec_name = f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_ro{outer_radius_str}_grp{group_spec}{extra_file_name}_backspec.fits"
            bkg_spec_file = os.path.join(dest_dir, bkg_spec_name)
            bkg_spec_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}_bkg.pi")
            
            bkg_arf_name = f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_ro{outer_radius_str}_grp{group_spec}{extra_file_name}_back.arf"
            bkg_arf_file = os.path.join(dest_dir, bkg_arf_name)
            bkg_arf_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}_bkg.arf")
            
            bkg_rmf_name = f"{obs_id}_{inst}_{source_name}_ra{ra_src_str}_dec{dec_src_str}_ri{inner_radius_str}_ro{outer_radius_str}_grp{group_spec}{extra_file_name}_back.rmf"
            bkg_rmf_file = os.path.join(dest_dir, bkg_rmf_name)
            bkg_rmf_ciao_out = os.path.join(temp_dir, f"{obs_id}_{inst}_bkg.rmf")
            
                        
            coord = SkyCoord(ra=ra_src, dec=dec_src, frame='icrs')

            ra_hms = coord.ra.to_string(unit=u.hour, sep=':', precision=5)
            dec_dms = coord.dec.to_string(unit=u.deg, sep=':', precision=5, alwayssign=True)

            bkg_inner_r_arc = outer_r_arc * source.background_radius_factors[0]
            bkg_outer_r_arc = outer_r_arc * source.background_radius_factors[1]
            
            # Ensure the directory exists
            temp_region_dir = os.path.join(dest_dir, f"temp_region_{randint(0, int(1e8))}") #added random nubmers
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


            # Build specextract command - making sure to set parallel to no, seeing as we're doing our
            #  own parallelization
            specextract_cmd = (
                f"cd {temp_dir}; punlearn specextract; specextract infile=\"{evt_file.path}[sky=region({spec_ext_reg_path})]\" "
                f"outroot={obs_id}_{inst} bkgfile=\"{evt_file.path}[sky=region({spec_bkg_reg_path})]\" "
                f"asp={att_file} badpixfile={badpix_file} grouptype={grouptype_int} binspec={binspec_int} "
                f"weight=yes weight_rmf=no clobber=yes parallel=no mskfile={mask_file}; "
                f"mv {spec_ciao_out} {spec_file}; mv {arf_ciao_out} {arf_file}; mv {rmf_ciao_out} {rmf_file}; "
                f"mv {bkg_spec_ciao_out} {bkg_spec_file}; mv {bkg_arf_ciao_out} {bkg_arf_file}; mv {bkg_rmf_ciao_out} {bkg_rmf_file}; "
                f"rm -r {temp_dir}; rm -r {temp_region_dir}"
            )            
                        
            cmds.append(specextract_cmd)

            print(specextract_cmd)
            print()
            
            final_paths.append(spec_file)
            # This is the products final resting place, if it exists at the end of this command.
            extra_info.append({"inner_radius": inner_r_arc, "outer_radius": outer_r_arc,
                               "rmf_path": rmf_file,
                               "arf_path": arf_file,
                               "b_spec_path": bkg_spec_file,
                               "b_rmf_path": bkg_rmf_file,
                               "b_arf_path": bkg_arf_file,
                               "obs_id": obs_id, "instrument": inst, "grouped": group_spec, "min_counts": min_counts,
                               "min_sn": min_sn, "over_sample": over_sample, "central_coord": source.default_coord,
                               "from_region": False,
                               "telescope": 'chandra'})

            
        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the SAS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))
        
    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@ciao_call
def specextract_spectrum(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                         inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                         min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                         num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A wrapper for all the CIAO processes necessary to generate an Chandra spectrum that can be analysed
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
    return _chandra_spec_cmds(sources, outer_radius, inner_radius, group_spec, min_counts, min_sn, over_sample, one_rmf,
                              num_cores, disable_progress)