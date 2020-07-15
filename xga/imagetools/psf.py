#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 16/07/2020, 00:22. Copyright (c) David J Turner

import os
from multiprocessing.dummy import Pool
from typing import List

import numpy as np
from astropy.units import Quantity
from fitsio import write
from scipy.signal import convolve
from tqdm import tqdm

from xga.products import PSFGrid
from xga.sas import evselect_image, psfgen
from xga.sources import BaseSource
from xga.utils import NUM_CORES, OUTPUT


# TODO Docstrings and comments!
def rl_psf(sources: List[BaseSource], iterations: int = 15, psf_model: str = "ELLBETA",
           lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
           bins: int = 4, num_cores: int = NUM_CORES):
    """
    An implementation of the Richardson-Lucy (doi:10.1364/JOSA.62.000055) PSF deconvolution algorithm that
    also takes into account the spatial variance of the XMM Newton PSF. The sources passed into this
    function will have all images matching the passed energy range deconvolved, the image objects will have the
    result stored in them alongside the original data, and a combined image will be generated. I view this
    method as quite crude, but it does seem to work, and I may implement a more intelligent way of doing
    PSF deconvolutions later.
    :param List[BaseSource] sources: A single source object, or list of source objects.
    :param int iterations: The number of deconvolution iterations performed by the Richardson-Lucy algorithm.
    :param str psf_model: Which model of PSF should be used for this deconvolution. The default is ELLBETA,
    the best available.
    :param Quantity lo_en: The lower energy bound of the images to be deconvolved.
    :param Quantity hi_en: The upper energy bound of the images to be deconvolved.
    :param int bins: Number of bins that the X and Y axes will be divided into when generating a PSFGrid.
    :param int num_cores: The number of cores to use (if running locally), the default is set to 90%
    of available cores in your system.
    """
    def rl_step(ind, chunk, last_chunk, rel_psf):
        psf_mirror = rel_psf[::-1, ::-1]
        relative_blur = chunk / convolve(last_chunk, rel_psf, mode='same')
        im_deconv = last_chunk * convolve(relative_blur, psf_mirror, mode='same')

        return im_deconv, ind

    # These just make sure that the images to be deconvolved and the PSFs to deconvolve them
    #  with have actually been generated. If they are already associated with the sources then this won't
    #  do anything.
    lo_en = lo_en.to('keV')
    hi_en = hi_en.to('keV')
    evselect_image(sources, lo_en, hi_en, num_cores=num_cores)
    psfgen(sources, bins, psf_model, num_cores)

    # If just one source is passed in, make it a list of one, makes behaviour more consistent
    #  throughout this function.
    if not isinstance(sources, list):
        sources = [sources]

    for source in sources:
        source: BaseSource
        en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
        # All the image objects of the specified energy range (so every combination of ObsID and instrument)
        match_images = source.get_products("image", extra_key=en_id)

        onwards = tqdm(total=len(match_images), desc="PSF Correcting {n} Images".format(n=source.name))
        # For now just going to iterate through them, we'll see if I can improve it later
        for im in match_images:
            # Read these out just because they'll be useful
            obs_id = im.obs_id
            inst = im.instrument

            # Extra key we need to search for the PSFGrid we need, then fetch it.
            psf_key = "_".join([psf_model, str(bins)])
            psf_grid: PSFGrid = source.get_products("psf", obs_id, inst, psf_key)[0]

            resampled_psfs = []
            for psf in psf_grid:
                resampled_psfs.append(psf.resample(im, Quantity(64, 'pix')))

            full_im_data = im.data.copy()
            full_im_data += full_im_data.max() * 1E-5 * np.random.random(full_im_data.shape)

            storage = [[np.full(full_im_data.shape, 0.5)] for i in range(len(psf_grid))]
            for i in range(iterations):
                with Pool(num_cores) as pool:
                    def callback(results_in):
                        nonlocal storage

                        proc_chunk, stor_ind = results_in
                        storage[stor_ind].append(proc_chunk)

                    def err_callback(err):
                        raise err

                    for psf_ind, psf in enumerate(psf_grid):
                        pool.apply_async(rl_step, callback=callback, error_callback=err_callback,
                                         args=(psf_ind, full_im_data, storage[psf_ind][-1], resampled_psfs[psf_ind]))
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

            og_total = im.data.sum()
            final_form = np.zeros((im.shape[0], im.shape[1], iterations))
            for psf_ind, psf in enumerate(psf_grid):
                del storage[psf_ind][0]
                deconv_steps = np.moveaxis(np.array(storage[psf_ind]), 0, 2)

                x_lims = psf_grid.x_bounds[psf_ind, :]
                y_lims = psf_grid.y_bounds[psf_ind, :]
                
                step_totals = np.sum(deconv_steps, (0, 1))
                norm_factors = og_total / step_totals
                deconv_steps = deconv_steps * norm_factors

                final_form[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1], :] = \
                    deconv_steps[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1], :]

            new_im_name = "{o}_{i}_{b}bin_{it}iter_rl_{l}-{u}keVpsfcorr_img." \
                          "fits".format(o=obs_id, i=inst, b=bins, it=iterations, l=lo_en.value, u=hi_en.value)

            write(os.path.join(OUTPUT, obs_id, new_im_name), np.moveaxis(final_form, 2, 0))
            onwards.update(1)
        onwards.close()





