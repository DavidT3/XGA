#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 24/07/2024, 16:16. Copyright (c) The Contributors

import os
from random import randint
from typing import Union

import numpy as np
import pandas as pd
from astropy.convolution import Kernel, convolve, convolve_fft
from fitsio import FITS

from .. import OUTPUT
from ..products import Image, RateMap, ExpMap


def general_smooth(prod: Union[Image, RateMap], kernel: Kernel, mask: np.ndarray = None, fft: bool = False,
                   norm_kernel: bool = True, sm_im: bool = True) -> Union[Image, RateMap]:
    """
    Simple function to apply (in theory) any Astropy smoothing to an XGA Image/RateMap  and create a new smoothed
    XGA data product. This general function will produce XGA Image and RateMap
    objects from any instance of an Astropy Kernel, and if a RateMap is passed as the input then you may choose
    whether to smooth the image component or the image/expmap (using sm_im); if you choose the former then the final
    smoothed RateMap will be produced by dividing the smoothed Image by the original ExpMap.

    :param Image/RateMap prod: The image/ratemap to be smoothed. If a RateMap is passed please see the 'sm_im'
        parameter for extra options.
    :param Kernel kernel: The kernel with which to smooth the input data. Should be an instance of an Astropy Kernel.
    :param np.ndarray mask: A mask to apply to the data while smoothing (removing point source interlopers for
        instance). The default is None, which means no mask is applied. This function expects a mask with 1s where
        the data you wish to keep is, and 0s where the data you wish to remove is - the style of mask produced by XGA.
    :param bool fft: Should a fast fourier transform method be used for convolution, default is False.
    :param bool norm_kernel: Whether to normalize the kernel to have a sum of one.
    :param bool sm_im: If a RateMap is passed, should the image component be smoothed rather than the actual
        RateMap. Default is True, where the Image will be smoothed and divided by the original ExpMap. If set
        to False, the resulting RateMap will be bodged, with the ExpMap all 1s on the sensor.
    :return: An XGA product with the smoothed Image or RateMap.
    :rtype: Image/RateMap
    """
    raise NotImplementedError("This function is not fully implemented yet!")
    # First off we check the type of the product that has been passed in for smoothing
    if not isinstance(prod, Image) or type(prod) == ExpMap:
        raise TypeError("This function can only smooth data if input in the form of an XGA Image/RateMap.")

    # Also need to check that the kernel has the right number of dimensions
    if len(kernel.shape) != 2:
        raise ValueError("The smoothing kernel needs to be two-dimensional for application to Image/RateMap data - "
                         "Gaussian2DKernel for instance.")

    # While we ask for masks in the style XGA produces (0s where you don't want data, 1s where you do), unfortunately
    #  the smoothing functions seem to want the opposite, so I'll quickly invert the mask here
    if mask is not None:
        mask[mask == 0] = -1
        mask[mask == 1] = 0
        mask[mask == -1] = 1

    # Read in the inventory of products relevant to the input image/ratemap
    inven = pd.read_csv(OUTPUT + "{}/inventory.csv".format(prod.obs_id), dtype=str)

    lo_en, hi_en = prod.energy_bounds
    key = "bound_{l}-{u}".format(l=float(lo_en.value), u=float(hi_en.value))

    # This is what the Image storage_key method does, but I want to do it here so I can just read in an
    #  existing image if possible, and not waste time convolving over again
    if prod.psf_corrected:
        key += "_" + prod.psf_model + "_" + str(prod.psf_bins) + "_" + prod.psf_algorithm + \
               str(prod.psf_iterations)

    # I don't want to let people smooth an image that has already been smoothed, that seems daft
    if prod.smoothed:
        raise ValueError("You cannot smooth an already smoothed image")

    # Finally we add our information from the input kernel to the key, as the parse_smoothing method of Image is
    #  static we can just make use of that
    sm, sp = Image.parse_smoothing(kernel)
    sp = "_".join([str(k) + str(v) for k, v in sp.items()])
    key += "_sm{sm}_sp{sp}".format(sm=sm, sp=sp)

    # rel_inven = inven[(inven['type'] == 'image')]
    # This narrows down the inventory to images that have the exact same info key (with smoothing information in), as
    #  the one we would create for the smoothed image we're making in this function. We'd only expect anything left
    #  in rel_inven if someone has already run this exact smoothing on this exact image, in which case we'll just
    #  retrieve that file
    rel_inven = inven[(inven['info_key'] == key) & (inven['type'] == 'image')]

    # Grabs certain information from the input product, primarily about what type it is, so we can infer where it
    #  should live in the XGA storage directory structure
    if prod.instrument == 'combined' or prod.obs_id == 'combined':
        # The ObsID-Instrument string combinations in the input product
        ois = [o + i for o in prod.obs_ids for i in prod.instruments[o]]

        # Where the smoothed image will live/lives
        final_dest = OUTPUT + "combined/"

        # I want to check whether a file of this particular smoothing already exists, slightly more complicated
        #  for combined images - its an ugly solution but I can't be bothered to make it nicer right now
        # This variable describes if the file already exists
        existing = False
        # This is where a new file would live, but will be overwritten if a matching image can be found
        final_name = "{{ri}}_{l}-{u}keVmerged_img.fits".format(l=lo_en.to('keV').value, u=hi_en.to('keV').value)
        for row_ind, row in rel_inven.iterrows():
            split_insts = row['insts'].split('/')
            combo = [o + split_insts[o_ind] for o_ind, o in enumerate(row['obs_ids'].split('/'))]
            if set(combo) == set(ois):
                existing = True
                final_name = row['file_name']
                break
    else:
        final_dest = OUTPUT + prod.obs_id + "/"

        # The filename of the expected smoothed image, it will be overwritten if one already exists
        final_name = "{o}_{i}_{{ri}}_{l}-{u}keV_img.fits".format(l=lo_en.to('keV').value, u=hi_en.to('keV').value,
                                                                 o=prod.obs_id, i=prod.instrument)

        # Easier to check for existing matching files than for combined images
        f_names = rel_inven[(rel_inven['obs_id'] == prod.obs_id) & (rel_inven['inst'] == prod.instrument)]['file_name']

        if len(f_names) != 0:
            final_name = f_names.values[0]
            existing = True
        else:
            existing = False

    # Now to get into it properly, if its an XGA product then we need to retrieve the data as an array. Only check
    #  whether its an instance of Image as that is the superclass for RateMap as well.
    if not existing and type(prod) == Image:
        data = prod.data.copy()
    elif not existing and type(prod) == RateMap and not sm_im:
        raise NotImplementedError("I haven't yet made sure that the rest of XGA will like this.")
        data = prod.data.copy()
    elif not existing and type(prod) == RateMap and sm_im:
        data = prod.image.data.copy()

    # Now we see which type of convolution the user has requested - entirely down to their discretion
    if not existing and not fft:
        sm_data = convolve(data, kernel, normalize_kernel=norm_kernel, mask=mask)
    elif not existing and fft:
        sm_data = convolve_fft(data, kernel, normalize_kernel=norm_kernel, mask=mask)

    # Now that Astropy has done the heavy lifting part of the smoothing, we save the image as an actual file, then
    #  assemble the output XGA product
    if type(prod) == Image:
        new_path = final_dest + final_name
        # If the file didn't already exist, we need to actually save the smoothed array, otherwise we'll just read
        #  the file in later
        if not existing:
            rand_ident = str(randint(0, int(1e+8)))
            # Makes absolutely sure that the random integer hasn't already been used
            while any([str(rand_ident) in f for f in os.listdir(final_dest)]):
                rand_ident = str(randint(0, int(1e+8)))

            # The final name of the new image file
            final_name = final_name.format(ri=rand_ident)
            new_path = final_dest + final_name

            # Writes out the smoothed data to the fits file
            with FITS(new_path, 'rw', clobber=True) as immo:
                immo.write(sm_data, header=prod.header)

        # Sets up the XGA product, regardless of whether its just been generated or it already
        #  existed, the process is the same
        sm_prod = Image(new_path, prod.obs_id, prod.instrument, "", "", "", prod.energy_bounds[0],
                        prod.energy_bounds[1], "", smoothed=True, smoothed_info=kernel)

    elif type(prod) == RateMap and not sm_im:
        raise NotImplementedError("I haven't yet made sure that the rest of XGA will like this.")
    elif type(prod) == RateMap and sm_im:
        raise NotImplementedError("I haven't yet made sure that the rest of XGA will like this.")

    # Make sure the smoothed product source name is the same as the passed in product's
    sm_prod.src_name = prod.src_name
    # Generate a new inventory line from the smoothed product
    new_line = sm_prod.inventory_entry

    # Add the new product to the inventory, even if it already existed
    inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)
    # Drop any duplicates in the inventory, which corrects the extra added in the last step if the file
    #  already existed
    inven = inven.drop_duplicates(subset='file_name', keep='first', ignore_index=True)
    # And save it
    inven.to_csv(OUTPUT + "{}/inventory.csv".format(prod.obs_id), index=False)

    return sm_prod









