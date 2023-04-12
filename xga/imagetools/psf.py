#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

import os
import warnings
from multiprocessing.dummy import Pool
from typing import Tuple, Union

import numpy as np
from astropy.units import Quantity
from fitsio import FITSHDR
from fitsio import write
from scipy.signal import convolve
from tqdm import tqdm

from ..products import PSFGrid, Image
from ..samples.base import BaseSample
from ..sas import evselect_image, psfgen, emosaic
from ..sources import BaseSource
from ..utils import NUM_CORES, OUTPUT


def rl_psf(sources: Union[BaseSource, BaseSample], iterations: int = 15, psf_model: str = "ELLBETA",
           lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
           bins: int = 4, num_cores: int = NUM_CORES):
    """
    An implementation of the Richardson-Lucy (doi:10.1364/JOSA.62.000055) PSF deconvolution algorithm that
    also takes into account the spatial variance of the XMM Newton PSF. The sources passed into this
    function will have all images matching the passed energy range deconvolved, the image objects will have the
    result stored in them alongside the original data, and a combined image will be generated. I view this
    method as quite crude, but it does seem to work, and I may implement a more intelligent way of doing
    PSF deconvolutions later.
    I initially tried convolving the PSFs generated at different spatial points with the chunks of data relevant
    to them in isolation, but the edge effects were very obvious. So I settled on convolving the whole
    original image with each PSF, and after it was finished taking the relevant chunks and patchworking
    them into a new array.

    :param BaseSource/BaseSample sources: A single source object, or list of source objects.
    :param int iterations: The number of deconvolution iterations performed by the Richardson-Lucy algorithm.
    :param str psf_model: Which model of PSF should be used for this deconvolution. The default is ELLBETA,
        the best available.
    :param Quantity lo_en: The lower energy bound of the images to be deconvolved.
    :param Quantity hi_en: The upper energy bound of the images to be deconvolved.
    :param int bins: Number of bins that the X and Y axes will be divided into when generating a PSFGrid.
    :param int num_cores: The number of cores to use (if running locally), the default is set to 90%
        of available cores in your system.
    """
    def rl_step(ind: int, cur_image: np.ndarray, last_image: np.ndarray, rel_psf: np.ndarray) \
            -> Tuple[np.ndarray, int]:
        """
        This performs one iteration of the Richardson-Lucy PSF deconvolution method. Basically copied from
        the skimage implementation, but set up so that we can multiprocess this, as well as do it in steps
        and save each step separately.

        :param int ind: The current step, passed through for the callback function.
        :param np.ndarray cur_image: The im_deconv from the last step.
        :param last_image:
        :param rel_psf: The particular spacial PSF being applied to this image.
        :return:
        """
        psf_mirror = rel_psf[::-1, ::-1]
        relative_blur = cur_image / convolve(last_image, rel_psf, mode='same')
        im_deconv = last_image * convolve(relative_blur, psf_mirror, mode='same')

        return im_deconv, ind

    def new_header(og_header: FITSHDR) -> FITSHDR:
        """
        Modifies an existing XMM Newton fits image header, removes some elements, and adds a little extra
        information. The new header is then used for PSF corrected fits image files.

        :param og_header: The header from the fits image that has been PSF corrected.
        :return: The new, modified, fits header.
        :rtype: FITSHDR
        """
        # These are nuisance entries that I want gone
        remove_list = ["CREATOR", "CONTINUE", "XPROC0", "XDAL0"]

        # FITSHDR takes a dictionary as its main content argument
        new_header_info = {}
        # We want to copy most of the original header
        for e in og_header:
            # But those that are in the remove list aren't copied over.
            if e not in remove_list:
                new_header_info[e] = og_header[e]
            # Also change the creator entry for a little self serving brand placement
            elif e == "CREATOR":
                new_header_info[e] = "XGA"

        # Setting some new headers for further information
        new_header_info["COMMENT"] = "THIS IMAGE HAS BEEN PSF CORRECTED BY XGA"
        new_header_info["PSFBins"] = bins
        new_header_info["PSFAlgorithm"] = "Richardson-Lucy"
        new_header_info["PSFAlgorithmIterations"] = iterations
        new_header_info["PSFModel"] = psf_model

        return FITSHDR(new_header_info)

    # These just make sure that the images to be deconvolved and the PSFs to deconvolve them
    #  with have actually been generated. If they are already associated with the sources then this won't
    #  do anything.
    lo_en = lo_en.to('keV')
    hi_en = hi_en.to('keV')
    # Making sure that the necessary images and PSFs are generated
    evselect_image(sources, lo_en, hi_en, num_cores=num_cores)

    # If just one source is passed in, make it a list of one, makes behaviour more consistent
    #  throughout this function.
    if not isinstance(sources, (list, BaseSample)):
        sources = [sources]

    # Only those sources that don't already have the individual PSF corrected images should be run
    sub_sources = []
    for source in sources:
        en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
        # All the image objects of the specified energy range (so every combination of ObsID and instrument)
        match_images = source.get_products("image", extra_key=en_id)

        # This is the key under which the PSF corrected image will be stored, defining it to check that
        #  it doesn't already exist.
        key = "bound_{l}-{u}_{m}_{b}_rl{i}".format(l=float(lo_en.value), u=float(hi_en.value), m=psf_model,
                                                   b=bins, i=iterations)
        # Check to see if all individual PSF corrected images are present
        psf_corr_prod = [p for p in source.get_products("image", just_obj=False) if key in p]

        # If all the PSF corrected images are present then we skip, the correction has already been performed.
        if len(psf_corr_prod) == len(match_images):
            continue
        else:
            sub_sources.append(source)

    # Should have cleaned it so that only those sources that need it will have PSFs generated
    psfgen(sub_sources, bins, psf_model, num_cores=num_cores)

    corr_prog_message = 'PSF Correcting Observations - Currently {}'
    with tqdm(desc=corr_prog_message.format(''), total=len(sub_sources),
              disable=len(sub_sources) == 0) as corr_progress:
        for source in sub_sources:
            source: BaseSource
            # Updates the source name in the message every iteration
            corr_progress.set_description(corr_prog_message.format(source.name))

            en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
            # All the image objects of the specified energy range (so every combination of ObsID and instrument)
            match_images = source.get_products("image", extra_key=en_id)

            # Just warns the user that some of the images may not be valid
            for matched in match_images:
                if "PrimeFullWindow" not in matched.header["SUBMODE"]:
                    warnings.warn("PSF corrected images for {s}-{o}-{i} may not be valid, as the data was taken"
                                  "in {m} mode".format(s=source.name, o=matched.obs_id, i=matched.instrument,
                                                       m=matched.header["SUBMODE"]))

            # For now just going to iterate through them, we'll see if I can improve it later
            for im in match_images:
                # Read these out just because they'll be useful
                obs_id = im.obs_id
                inst = im.instrument

                # Extra key we need to search for the PSFGrid we need, then fetch it.
                psf_key = "_".join([psf_model, str(bins)])
                psf_grid: PSFGrid = source.get_products("psf", obs_id, inst, psf_key)[0]

                # This uses a built in method of the PSF class to re-sample the PSF(s) to the same
                #  scale as our image
                resamp_psfs = []
                for psf in psf_grid:
                    resamp_psfs.append(psf.resample(im, Quantity(64, 'pix')))

                full_im_data = im.data.copy()
                # This adds very low level random noise to the image, getting rid of 0 values, this makes life
                #  easier with the convolution functions
                full_im_data += full_im_data.max() * 1E-8 * np.random.random(full_im_data.shape)

                # Nasty nested list to store each step for each PSF in
                # The starting array full of 0.5 values is what is used in the skimage implementation
                storage = [[np.full(full_im_data.shape, 0.5)] for i in range(len(psf_grid))]
                # Iterating over the steps
                for i in range(iterations):
                    # Sets up a multiprocessing pool
                    with Pool(num_cores) as pool:
                        # This is what is triggered when each rl_step call is done
                        def callback(results_in):
                            nonlocal storage
                            # Reading out results
                            proc_chunk, stor_ind = results_in
                            # Storing the results in the storage listss
                            storage[stor_ind].append(proc_chunk)

                        def err_callback(err):
                            raise err

                        # Starts a deconvolution function for each spatial PSF, adding them to the
                        #  multiprocessing pool
                        for psf_ind, psf in enumerate(psf_grid):
                            pool.apply_async(rl_step, callback=callback, error_callback=err_callback,
                                             args=(psf_ind, full_im_data, storage[psf_ind][-1], resamp_psfs[psf_ind]))
                        pool.close()  # No more tasks can be added to the pool
                        pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                # Because of the nature of convolutions, you can end up with a significantly
                #  different sum total for the image than you started with. To maintain physicality, we need to
                #  preserve the total number of counts, so we re-normalise each step using the original total.
                og_total = im.data.sum()
                # The 3D array that the final patch-worked images are stored in
                # Every iteration has its resultant image stored in here, earliest at index 0, final at index -1
                final_form = np.zeros((im.shape[0], im.shape[1], iterations))
                # Iterate over the PSFs generated at different points
                for psf_ind, psf in enumerate(psf_grid):
                    del storage[psf_ind][0]  # You may remember this was just the initial array of 0.5 as per skimage
                    # Change from an N, 512, 512 (for XGA images) to 512, 512, N array (makes more sense to my brain).
                    deconv_steps = np.moveaxis(np.array(storage[psf_ind]), 0, 2)

                    # Grab out the x and y boundary coordinates for the spatial chunk relevant to the current PSF
                    x_lims = psf_grid.x_bounds[psf_ind, :]
                    y_lims = psf_grid.y_bounds[psf_ind, :]

                    # For each deconvolution step we find the total value of the image and normalise by
                    #  the original total
                    step_totals = np.sum(deconv_steps, (0, 1))
                    norm_factors = og_total / step_totals
                    deconv_steps = deconv_steps * norm_factors

                    # Cut out the piece of the image array (for all iteration steps) that is relevant for the current
                    # PSF and insert it into the final image array at the same coordinates
                    # Patch-working yay, terminology definitely inspired by the new Passenger Album :)
                    final_form[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1], :] = \
                        deconv_steps[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1], :]

                # Final re-normalization, bit cheesy but oh well
                norm_factors = og_total / np.sum(final_form, (0, 1))
                final_form = final_form * norm_factors

                # Define file names for the whole datacube (all iteration steps), and the final image (just the last).
                datacube_name = "{o}_{i}_{b}bin_{it}iter_{m}mod_rlalgo_{l}-{u}keVpsfcorr_datacube." \
                                "fits".format(o=obs_id, i=inst, b=bins, it=iterations, l=lo_en.value, u=hi_en.value,
                                              m=psf_model)
                # Define this one because emosaic can't deal with fits datacubes, and thats what I use to combine images.
                im_name = "{o}_{i}_{b}bin_{it}iter_{m}mod_rlalgo_{l}-{u}keVpsfcorr_img." \
                          "fits".format(o=obs_id, i=inst, b=bins, it=iterations, l=lo_en.value, u=hi_en.value,
                                        m=psf_model)

                # Use the super handy fitsio write function to create the new fits datacube, and final image.
                write(os.path.join(OUTPUT, obs_id, datacube_name), np.moveaxis(final_form, 2, 0),
                      header=new_header(im.header))
                write(os.path.join(OUTPUT, obs_id, im_name), np.moveaxis(final_form, 2, 0)[-1, :, :],
                      header=new_header(im.header))

                # Makes an XGA product of our brand new image
                fin_im = Image(os.path.join(OUTPUT, obs_id, im_name), obs_id, inst, '', '', '', lo_en, hi_en)
                # Adds PSF correction information for XGA's internal use
                fin_im.psf_corrected = True
                fin_im.psf_algorithm = "rl"
                fin_im.psf_bins = bins
                fin_im.psf_iterations = iterations
                fin_im.psf_model = psf_model

                # Adds it into the source
                source.update_products(fin_im)

                # Removes the PSFGrid constituents from memory
                psf_grid.unload_data()

            corr_progress.update(1)
        corr_progress.set_description(corr_prog_message.format('complete'))

    # For the passed sources we now run emosaic to create combined PSF corrected images.
    emosaic(sources, "image", lo_en, hi_en, psf_corr=True, psf_model=psf_model, psf_bins=bins,
            psf_algo="rl", psf_iter=iterations)

