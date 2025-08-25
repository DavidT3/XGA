#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 25/08/2025, 14:05. Copyright (c) The Contributors

import os
from random import randint
from shutil import rmtree
from typing import Union

import numpy as np
from astropy.units import Quantity, deg
from tqdm import tqdm

from .misc import cifbuild
from .run import sas_call
from .. import OUTPUT, NUM_CORES
from ..exceptions import SASInputInvalid, NoProductAvailableError
from ..imagetools import data_limits
from ..samples.base import BaseSample
from ..sources import BaseSource
from ..sources.base import NullSource


# TODO Perhaps remove the option to add to the SAS expression
@sas_call
def evselect_image(sources: Union[BaseSource, NullSource, BaseSample], lo_en: Quantity = Quantity(0.5, 'keV'),
                   hi_en: Quantity = Quantity(2.0, 'keV'), add_expr: str = "", num_cores: int = NUM_CORES,
                   disable_progress: bool = False):
    """
    A convenient Python wrapper for a configuration of the SAS evselect command that makes images.
    Images will be generated for every observation associated with every source passed to this function.
    If images in the requested energy band are already associated with the source,
    they will not be generated again.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or a sample of sources.
    :param Quantity lo_en: The lower energy limit for the image, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the image, in astropy energy units.
    :param str add_expr: A string to be added to the SAS expression keyword
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    # Don't do much value checking in this module, but this one is so fundamental that I will do it
    if lo_en > hi_en:
        raise ValueError("The 'lo_en' argument cannot be greater than 'hi_en'.")
    else:
        # Converts the energies to channels for EPIC detectors, assuming one channel per eV
        lo_chan = int(lo_en.to('eV').value)
        hi_chan = int(hi_en.to('eV').value)

    expr = " && ".join([e for e in ["expression='(PI in [{l}:{u}])".format(l=lo_chan, u=hi_chan),
                                    add_expr] if e != ""]) + "'"
    # These lists are to contain the lists of commands/paths/etc for each of the individual sources passed
    # to this function
    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for source in sources:
        cmds = []
        final_paths = []
        extra_info = []
        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", just_obj=False):
            obs_id = pack[0]
            inst = pack[1]

            if not os.path.exists(OUTPUT + obs_id):
                os.mkdir(OUTPUT + obs_id)

            en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
            exists = [match for match in source.get_products("image", obs_id, inst, just_obj=False)
                      if en_id in match]
            if len(exists) == 1 and exists[0][-1].usable:
                continue

            evt_list = pack[-1]
            dest_dir = OUTPUT + "{o}/{i}_{l}-{u}_{n}_temp/".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value,
                                                                   n=source.name)
            im = "{o}_{i}_{l}-{u}keVimg.fits".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)

            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)

            os.makedirs(dest_dir)
            cmds.append("cd {d};evselect table={e} imageset={i} xcolumn=X ycolumn=Y ximagebinsize=87 "
                        "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize "
                        "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 "
                        "withyranges=yes {ex}; mv * ../; cd ..; rm -r {d}".format(d=dest_dir, e=evt_list.path,
                                                                                  i=im, ex=expr))

            # This is the products final resting place, if it exists at the end of this command
            final_paths.append(os.path.join(OUTPUT, obs_id, im))
            extra_info.append({"lo_en": lo_en, "hi_en": hi_en, "obs_id": obs_id, "instrument": inst})
        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the SAS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="image"))

    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@sas_call
def eexpmap(sources: Union[BaseSource, NullSource, BaseSample], lo_en: Quantity = Quantity(0.5, 'keV'),
            hi_en: Quantity = Quantity(2.0, 'keV'), num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A convenient Python wrapper for the SAS eexpmap command.
    Expmaps will be generated for every observation associated with every source passed to this function.
    If expmaps in the requested energy band are already associated with the source,
    they will not be generated again.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or sample of sources.
    :param Quantity lo_en: The lower energy limit for the expmap, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the expmap, in astropy energy units.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    # I know that a lot of this code is the same as the evselect_image code, but its 1am so please don't
    #  judge me too much.

    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, (BaseSource, NullSource)):
        sources = [sources]

    # Don't do much value checking in this module, but this one is so fundamental that I will do it
    if lo_en > hi_en:
        raise ValueError("lo_en cannot be greater than hi_en")
    else:
        # Converts the energies to channels for EPIC detectors, assuming one channel per eV
        lo_chan = int(lo_en.to('eV').value)
        hi_chan = int(hi_en.to('eV').value)

    # These are crucial, to generate an exposure map one must have a ccf.cif calibration file, and a reference
    # image. If they do not already exist, these commands should generate them.
    cifbuild(sources, disable_progress=disable_progress, num_cores=num_cores)
    sources = evselect_image(sources, lo_en, hi_en)
    # This is necessary because the decorator will reduce a one element list of source objects to a single
    # source object. Useful for the user, not so much here where the code expects an iterable.
    if not isinstance(sources, (list, BaseSample)):
        sources = [sources]

    # These lists are to contain the lists of commands/paths/etc for each of the individual sources passed
    # to this function
    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for source in sources:
        cmds = []
        final_paths = []
        extra_info = []
        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", just_obj=False):
            obs_id = pack[0]
            inst = pack[1]

            if not os.path.exists(OUTPUT + obs_id):
                os.mkdir(OUTPUT + obs_id)

            en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
            exists = [match for match in source.get_products("expmap", obs_id, inst, just_obj=False)
                      if en_id in match]
            if len(exists) == 1 and exists[0][-1].usable:
                continue
            # Generating an exposure map requires a reference image.
            ref_im = [match for match in source.get_products("image", obs_id, inst, just_obj=False)
                      if en_id in match][0][-1]
            # It also requires an attitude file
            att = source.get_att_file(obs_id)
            # Set up the paths and names of files
            evt_list = pack[-1]
            dest_dir = OUTPUT + "{o}/{i}_{l}-{u}_{n}_temp/".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value,
                                                                   n=source.name)
            exp_map = "{o}_{i}_{l}-{u}keVexpmap.fits".format(o=obs_id, i=inst, l=lo_en.value, u=hi_en.value)

            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)

            os.makedirs(dest_dir)
            cmds.append("cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; eexpmap eventset={e} "
                        "imageset={im} expimageset={eim} withdetcoords=no withvignetting=yes "
                        "attitudeset={att} pimin={l} pimax={u}; mv * ../; cd ..; "
                        "rm -r {d}".format(e=evt_list.path, im=ref_im.path, eim=exp_map, att=att, l=lo_chan,
                                           u=hi_chan, d=dest_dir, ccf=dest_dir + "ccf.cif"))

            # This is the products final resting place, if it exists at the end of this command
            final_paths.append(os.path.join(OUTPUT, obs_id, exp_map))
            extra_info.append({"lo_en": lo_en, "hi_en": hi_en, "obs_id": obs_id, "instrument": inst})
        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        # once the SAS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="expmap"))

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@sas_call
def emosaic(sources: Union[BaseSource, BaseSample], to_mosaic: str, lo_en: Quantity = Quantity(0.5, 'keV'),
            hi_en: Quantity = Quantity(2.0, 'keV'), psf_corr: bool = False, psf_model: str = "ELLBETA",
            psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, num_cores: int = NUM_CORES,
            disable_progress: bool = False):
    """
    A convenient Python wrapper for the SAS emosaic command. Every image associated with the source,
    that is in the energy band specified by the user, will be added together.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str to_mosaic: The data type to produce a mosaic for, can be either image or expmap.
    :param Quantity lo_en: The lower energy limit for the combined image, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the combined image, in astropy energy units.
    :param bool psf_corr: If True, PSF corrected images will be mosaiced.
    :param str psf_model: If PSF corrected, the PSF model used.
    :param int psf_bins: If PSF corrected, the number of bins per side.
    :param str psf_algo: If PSF corrected, the algorithm used.
    :param int psf_iter: If PSF corrected, the number of algorithm iterations.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    # NullSources are not allowed to be mosaiced, as they can have any observations associated and thus won't
    #  necessarily overlap
    if isinstance(sources, NullSource):
        raise TypeError("You cannot create combined images of a NullSource")

    if to_mosaic not in ["image", "expmap"]:
        raise ValueError("The only valid choices for to_mosaic are image and expmap.")
    # Don't do much value checking in this module, but this one is so fundamental that I will do it
    elif lo_en > hi_en:
        raise ValueError("lo_en cannot be greater than hi_en")

    # To make a mosaic we need to have the individual products in the first place
    if to_mosaic == "image":
        sources = evselect_image(sources, lo_en, hi_en, disable_progress=disable_progress, num_cores=num_cores)
        for_name = "img"
    elif to_mosaic == "expmap":
        sources = eexpmap(sources, lo_en, hi_en, disable_progress=disable_progress, num_cores=num_cores)
        for_name = "expmap"

    # This is necessary because the decorator will reduce a one element list of source objects to a single
    # source object. Useful for the user, not so much here where the code expects an iterable.
    if not isinstance(sources, (list, BaseSample)):
        sources = [sources]

    # The bit on the end takes everything up out of the temporary folder and removes it
    mosaic_cmd = "cd {d}; emosaic imagesets='{ims}' mosaicedset={mim}; mv * ../; cd ..; rm -r {d}"

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []
    for source in sources:
        en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
        # If we're mosaicing PSF corrected images, we need to
        if psf_corr and to_mosaic == "expmap":
            raise ValueError("There can be no PSF corrected expmaps to mosaic, it doesn't make sense.")
        elif psf_corr:
            en_id += "_" + psf_model + "_" + str(psf_bins) + "_" + psf_algo + str(psf_iter)

        # Checking if the combined product already exists
        exists = [match for match in source.get_products("combined_{}".format(to_mosaic), just_obj=False)
                  if en_id in match]
        if len(exists) == 1 and exists[0][-1].usable:
            sources_cmds.append(np.array([]))
            sources_paths.append(np.array([]))
            sources_extras.append(np.array([]))
            sources_types.append(np.array([]))
            continue

        # This fetches all image objects with the passed energy bounds
        matches = [[match[0], match[-1]] for match in source.get_products(to_mosaic, just_obj=False)
                   if en_id in match]

        # In theory this should never be triggered, because we already ran evselect_image so the images should be
        #  there - but I am now somehow having an error where we get to this point with no errors and no images, so
        #  we're going to add this in to be absolutely sure (as otherwise emosaic fails with a very unhelpful error).
        if len(matches) == 0:
            assoc = ", ".join([cur_oi + cur_i for cur_oi in source.instruments
                               for cur_i in source.instruments[cur_oi]])
            raise NoProductAvailableError("The images required for emosaic are not available for {p} - this is not a"
                                          " usual behaviour as XGA should have generated them; the relevant "
                                          "observations are {d}.".format(p=source.name, d=assoc))

        paths = [product[1].path for product in matches if product[1].usable]
        obs_ids = [product[0] for product in matches if product[1].usable]
        obs_ids_set = []
        for obs_id in obs_ids:
            if obs_id not in obs_ids_set:
                obs_ids_set.append(obs_id)

            if not os.path.exists(OUTPUT + obs_id):
                os.mkdir(OUTPUT + obs_id)

        # The files produced by this function will now be stored in the combined directory.
        final_dest_dir = OUTPUT + "combined/"
        rand_ident = randint(0, int(1e+8))
        # Makes absolutely sure that the random integer hasn't already been used
        while len([f for f in os.listdir(final_dest_dir) if str(rand_ident) in f.split(OUTPUT+"combined/")[-1]]) != 0:
            rand_ident = randint(0, int(1e+8))

        dest_dir = os.path.join(final_dest_dir, "temp_emosaic_{}".format(rand_ident))
        os.mkdir(dest_dir)

        # The name of the file used to contain all the ObsIDs that went into the stacked image/expmap. However
        #  this caused problems when too many ObsIDs were present and the filename was longer than allowed. So
        #  now I use the random identity I generated, and store the ObsID/instrument information in the inventory
        #  file
        if not psf_corr:
            mosaic = "{os}_{l}-{u}keVmerged_{t}.fits".format(os=rand_ident, l=lo_en.value, u=hi_en.value, t=for_name)
        else:
            mosaic = "{os}_{b}bin_{it}iter_{m}mod_{a}algo_{l}-{u}keVpsfcorr_merged_img." \
                     "fits".format(os=rand_ident, l=lo_en.value, u=hi_en.value, b=psf_bins, it=psf_iter, a=psf_algo,
                                   m=psf_model)

        sources_cmds.append(np.array([mosaic_cmd.format(ims=" ".join(paths), mim=mosaic, d=dest_dir)]))
        sources_paths.append(np.array([os.path.join(final_dest_dir, mosaic)]))
        # This contains any other information that will be needed to instantiate the class
        # once the SAS cmd has run
        # The 'combined' values for obs and inst here are crucial, they will tell the source object that the final
        # product is assigned to that these are merged products - combinations of all available data
        sources_extras.append(np.array([{"lo_en": lo_en, "hi_en": hi_en, "obs_id": "combined",
                                         "instrument": "combined", "psf_corr": psf_corr, "psf_algo": psf_algo,
                                         "psf_model": psf_model, "psf_iter": psf_iter, "psf_bins": psf_bins}]))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value=to_mosaic))

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately
    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


@sas_call
def psfgen(sources: Union[BaseSource, BaseSample], bins: int = 4, psf_model: str = "ELLBETA",
           num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A wrapper for the psfgen SAS task. Used to generate XGA PSF objects, which in turn can be used to correct
    XGA images/ratemaps for optical effects. By default we use the ELLBETA model reported in Read et al. 2011
    (doi:10.1051/0004-6361/201117525), and generate a grid of binsxbins PSFs that can be used
    to correct for the PSF over an entire image. The energy dependence of the PSF is assumed to be minimal, and the
    resultant PSF object will be paired up with an image that matches it's ObsID and instrument.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param int bins: The image coordinate space will be divided into a grid of size binsxbins, PSFs will be
        generated at the central coordinates of the grid chunks.
    :param str psf_model: Which model to use when generating the PSF, default is ELLBETA, the best available.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    psf_model = psf_model.upper()
    allowed_models = ["ELLBETA", "LOW", "MEDIUM", "EXTENDED", "HIGH"]
    if psf_model not in allowed_models:
        raise SASInputInvalid("{0} is not a valid PSF model. Allowed models are "
                              "{1}".format(psf_model, ", ".join(allowed_models)))

    # Need a valid CIF for this task, so run cifbuild first
    cifbuild(sources, disable_progress=disable_progress, num_cores=num_cores)

    # This is necessary because the decorator will reduce a one element list of source objects to a single
    # source object. Useful for the user, not so much here where the code expects an iterable.
    if not isinstance(sources, (list, BaseSample)):
        sources = [sources]

    # NullSources are not allowed to be mosaiced, as they can have any observations associated and thus won't
    #  necessarily overlap
    if isinstance(sources, NullSource):
        raise NotImplementedError("You cannot currently use PSFGen with a NullSource.")

    with tqdm(desc='Preparing PSF generation commands', total=len(sources),
              disable=len(sources) == 0) as psfgen_prep_progress:
        # These lists are to contain the lists of commands/paths/etc for each of the individual sources passed
        # to this function
        sources_cmds = []
        sources_paths = []
        sources_extras = []
        sources_types = []
        for source in sources:
            cmds = []
            final_paths = []
            extra_info = []
            # Check which event lists are associated with each individual source
            for pack in source.get_products("events", just_obj=False):
                obs_id = pack[0]
                inst = pack[1]

                if not os.path.exists(OUTPUT + obs_id):
                    os.mkdir(OUTPUT + obs_id)

                # This looks for any image for this ObsID, instrument combo - it does assume that whatever
                #  it finds will be the same resolution as any images in other energy bands that XGA will
                #  create in the future.
                images = source.get_products("image", obs_id, inst, just_obj=True)

                if len(images) == 0:
                    raise NoProductAvailableError("There is no image available for {o} {i}, please generate "
                                                  "images before PSFs".format(o=obs_id, i=inst))

                # Checking if the Image products are the same shape that XGA makes
                res_match = [im for im in images if im.shape == (512, 512)]
                if len(res_match) == 0:
                    raise NoProductAvailableError("There is an image associated with {o} {i}, but it doesn't"
                                                  " appear to be at the resolution XGA uses - this is not "
                                                  "supported yet.")
                else:
                    image = res_match[0]

                # Here we try and find if this PSF configuration has already been run and has been
                #  associated with the source. If so then don't do it again.
                psfs = source.get_products("psf", obs_id, inst, extra_key=psf_model + "_" + str(bins))
                if len(psfs) != 0:
                    continue

                # This part is where we decide on the RA DEC coordinates for the centres of each
                #  PSF in our grid
                # This function gives us x and y limits for where there is data in an image, they are used as start
                #  and end coordinates for our bins so the PSFs are more focused on where there is actually data.
                x_lims, y_lims = data_limits(image)
                # Simple calculation to calculate step size in pixels, so how long each chunk will be in
                #  x and y directions
                x_step = (x_lims[1] - x_lims[0]) / bins
                y_step = (y_lims[1] - y_lims[0]) / bins

                # These are the x and y bin centre coordinates - when converted to RA and DEC this is where the
                #  PSF is generated at.
                x_cen_coords = np.arange(*x_lims, x_step) + (x_step / 2)
                y_cen_coords = np.arange(*y_lims, y_step) + (y_step / 2)

                # Get all combinations of the central coordinates using meshgrid, then turn them into
                #  an N row, 2 column numpy array of pixel coordinates for easy conversion to RA-DEC.
                pix_mesh = np.meshgrid(x_cen_coords, y_cen_coords)
                pix_coords = Quantity(np.stack([pix_mesh[0].ravel(), pix_mesh[1].ravel()]).T, 'pix')

                # But I also want to know the boundaries of the bins so I can easily select which parts of
                #  the image belong with each PSF in the grid
                x_boundaries = np.linspace(*x_lims, bins+1)
                y_boundaries = np.linspace(*y_lims, bins+1)

                # These two arrays give the x and y boundaries of the bins in the same order as the pix_coords array
                x_bound_coords = np.tile(np.stack([x_boundaries[0: -1].ravel(), x_boundaries[1:].ravel()]).T,
                                         (bins, 1))
                x_bound_coords = x_bound_coords.round(0).astype(int)

                y_bound_coords = np.repeat(np.stack([y_boundaries[0: -1].ravel(), y_boundaries[1:].ravel()]).T,
                                           bins, 0)
                y_bound_coords = y_bound_coords.round(0).astype(int)

                ra_dec_coords = image.coord_conv(pix_coords, deg)

                dest_dir = OUTPUT + "{o}/{i}_{n}_temp/".format(o=obs_id, i=inst, n=source.name)
                psf = "{o}_{i}_{b}bin_{m}mod_{ra}_{dec}_psf.fits"

                # The change directory and SAS setup commands
                init_cmd = "cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; ".format(d=dest_dir,
                                                                                    ccf=dest_dir + "ccf.cif")

                # If something got interrupted and the temp directory still exists, this will remove it
                if os.path.exists(dest_dir):
                    rmtree(dest_dir)

                os.makedirs(dest_dir)

                psf_files = []
                total_cmd = init_cmd
                for pair_ind in range(ra_dec_coords.shape[0]):
                    # The coordinates at which this PSF will be generated
                    ra, dec = ra_dec_coords[pair_ind, :].value

                    psf_file = psf.format(o=obs_id, i=inst, b=bins, ra=ra, dec=dec, m=psf_model)
                    psf_files.append(os.path.join(OUTPUT, obs_id, psf_file))
                    # Going with xsize and ysize as 400 pixels, I think its enough and quite a bit faster than 1000
                    total_cmd += "psfgen image={i} coordtype=EQPOS level={m} energy=1000 xsize=400 ysize=400 x={ra} " \
                                 "y={dec} output={p}; ".format(i=image.path, m=psf_model, ra=ra, dec=dec, p=psf_file)

                total_cmd += "mv * ../; cd ..; rm -r {d}".format(d=dest_dir)
                cmds.append(total_cmd)
                # This is the products final resting place, if it exists at the end of this command
                # In this case it just checks for the final PSF in the grid, all other files in the grid
                # get stored in extra info.
                final_paths.append(os.path.join(OUTPUT, obs_id, psf_file))
                extra_info.append({"obs_id": obs_id, "instrument": inst, "model": psf_model, "chunks_per_side": bins,
                                   "files": psf_files, "x_bounds": x_bound_coords, "y_bounds": y_bound_coords})

            sources_cmds.append(np.array(cmds))
            sources_paths.append(np.array(final_paths))
            # This contains any other information that will be needed to instantiate the class
            # once the SAS cmd has run
            sources_extras.append(np.array(extra_info))
            sources_types.append(np.full(sources_cmds[-1].shape, fill_value="psf"))

            psfgen_prep_progress.update(1)

    # I only return num_cores here so it has a reason to be passed to this function, really
    # it could just be picked up in the decorator.
    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


