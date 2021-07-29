#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 29/07/2021, 15:59. Copyright (c) David J Turner

from typing import Union
from warnings import warn

import numpy as np
from astropy.units import Quantity, UnitConversionError
from tqdm import tqdm

from ..imagetools.misc import edge_finder
from ..products import Image, RateMap

CONT_BIN_METRICS = ['counts', 'snr']
MAX_VAL_UNITS = ['ct', '']


def contour_bin_masks(prod: Union[Image, RateMap], src_mask: np.ndarray = None, bck_mask: np.ndarray = None,
                      start_pos: Quantity = None, max_masks: int = 20, metric: str = 'counts',
                      max_val: Quantity = Quantity(1000, 'ct')) -> np.ndarray:
    """
    This method implements different spins on Jeremy Sanders' contour binning
    method (https://doi.org/10.1111/j.1365-2966.2006.10716.x) to split the 2D ratemap into bins that
    are spatially and morphologically connected (in theory). This can make some nice images, and also allows
    you to use those new regions to measure projected spectral quantities (temperature, metallicity, density)
    and make a 2D projected property map. After the first bin is defined at start_pos, further bins will be
    started at the brightest pixels left.

    Current allowable metric choices:
      * 'counts' - Stop adding to bin when total background subtracted counts are over max_val
      * 'snr' - Stop adding to bin when signal to noise is over max_val.

    :param Image/RateMap prod: The image or ratemap to apply the contour binning process to.
    :param np.ndarray src_mask: A mask that removes emission from regions not associated with the source you're
        analysing, including removing interloper sources. Default is None, in which case no mask will be applied.
    :param np.ndarray bck_mask: A mask defining the background region. Default is None in which case no background
        subtraction will be used.
    :param Quantity start_pos: The position at which to start the binning process, in units of degrees or pixels.
        This parameter is None by default, in which case the brightest pixel (after masks have been applied), will
        be used as the starting point.
    :param int max_masks: A simple cut off for the number of masks that can be produced by this
        function, default is 20.
    :param str metric: The metric by which to judge when to stop adding new pixels to a bin (see docstring).
    :param Quantity max_val: The max value for the chosen metric, above which a new bin is started.
    :return: A 3D array of bin masks, the first two dimensions are the size of the input image, the third is
        the number of regions that have been generated.
    :rtype: np.ndarray
    """
    # First checks to make sure no illegal values have been passed for the metric, and that the max value is in the
    #  right units for the chosen metric.
    if metric not in CONT_BIN_METRICS:
        cont_av = ", ".join(CONT_BIN_METRICS)
        raise ValueError("{m} is not a recognised contour binning metric, please use one of the "
                         "following; {a}".format(m=metric, a=cont_av))
    elif max_val.unit.to_string() != MAX_VAL_UNITS[CONT_BIN_METRICS.index(metric)]:
        raise UnitConversionError("The {m} metric requires a max value in units of "
                                  "{u}".format(m=metric, u=MAX_VAL_UNITS[CONT_BIN_METRICS.index(metric)]))

    # We use type here rather than isinstance because ExpMaps are also a subclass of Image, so would get through
    #  that check
    if type(prod) != Image and type(prod) != RateMap:
        raise TypeError("Only XGA Image and RateMap products can be binned with this function.")

    # I do warn the user in this case, because it seems like a strange choice
    if src_mask is None and bck_mask is None:
        warn("You have not passed a src or bck mask, the whole image will be binned and no background subtraction "
             "will be applied")

    # If the mask parameters are None, we assemble an array of ones, to allow all pixels to be considered
    if src_mask is None:
        src_mask = np.full(prod.shape, 1)
    if bck_mask is None:
        bck_mask = np.full(prod.shape, 1)

    # If the user hasn't supplied us with somewhere to start, we just select the brightest pixel
    #  remaining after masking, just using the Image or RateMap simple peak method
    if start_pos is None:
        start_pos = prod.simple_peak(src_mask, 'pix')[0].value
    else:
        # If the user has supplied a start position then we want it to be in pixels
        start_pos = prod.coord_conv(start_pos, 'pix').value

    # While start_pos is the very first start position and is immutable, each region will need its own
    #  start coordinate that is stored in this variable
    # We flip it so that it makes sense in the context of accessing numpy arrays
    cur_bin_sp = start_pos.copy()[[1, 0]]

    # This variable controls the outermost while loop, and will be set to true when all bins are complete
    all_done = False
    # This is what will be output, the final set of masks, will be made into a 3D numpy array at the
    #  end, rather than just a list of arrays
    all_masks = []
    # This list stores whether given masks reached their required value or were terminated for some other reason
    reached_max = []

    # Making a copy of the data array and masking it so that only what the user wants the algorithm to consider remains
    prod_dat = prod.data.copy() * src_mask

    # The various product type and metric combinations
    # There are many while loops in this process and they make me sad, but we don't know a priori how
    #  many bins there will be when we're done
    if type(prod) == Image and metric == 'counts':
        raise NotImplementedError("The method for images has not been implemented yet")
    elif type(prod) == Image and metric == 'snr':
        raise NotImplementedError("The method for images has not been implemented yet")
    elif type(prod) == RateMap and metric == 'counts':

        # Here we create a total mask where all previously claimed parts of the image are marked
        #  off. As bins are laid down this will be slowly blocked off, but it starts as all ones
        no_go = np.ones(prod_dat.shape)

        with tqdm(desc="Generating Contour Bins", total=max_masks) as open_ended:
            while not all_done and len(all_masks) < max_masks:
                # Setup the mask for this region
                cur_mask = np.zeros(prod.shape)
                # We already know where we're starting, so that is the first point included in the mask
                cur_mask[cur_bin_sp[0], cur_bin_sp[1]] = 1

                # For this count based metric, this is were the total counts is totted up
                tot_cnts = prod.get_count(Quantity([cur_bin_sp[1], cur_bin_sp[0]], 'pix'))
                # Did this go all the way through the loop to the max val?
                max_val_reached = True

                while tot_cnts < max_val:
                    # This grabs the pixels around the edge of the current bin mask, we also don't want values other
                    #  than 0 or 1 in here, hence why keep_corners is False
                    cand_mask = edge_finder(cur_mask, border=True, keep_corners=False)

                    # This allowed data takes into account the source mask (as prod_dat already has that applied),
                    #  the regions that have already been claimed by previous contour bins, (in no_go), and finally
                    #  the places around the edge of the current contour bin that the algorithm is allowed
                    #  to go next (in cand_mask)
                    allowed_data = prod_dat*cand_mask*no_go

                    # Obviously if all of this array is zero, then we can't continue with this contour bin, as
                    #  there is nowhere else to go
                    if np.all(allowed_data == 0):
                        # We haven't reached the maximum value and the loop has to be prematurely terminated
                        max_val_reached = False
                        break

                    # Now we know there is still data to add to the bin, we can find the point with the
                    #  maximum value at least in the parts of the image/ratemap we're allowed to look in
                    cur_bin_sp = np.unravel_index(np.argmax(allowed_data), prod_dat.shape)
                    # That position in the current contour bin mask is then set to 1, to indicate its a part
                    #  of this contour bin now
                    cur_mask[cur_bin_sp[0], cur_bin_sp[1]] = 1

                    # Then we update the total number of counts
                    cts = prod.get_count(Quantity([cur_bin_sp[1], cur_bin_sp[0]], 'pix'))
                    tot_cnts += cts

                # We've broken out of the inner while loop, for one reason or another, and now can consider
                #  that last contour bin to be finished, so we store it in the all_masks list
                all_masks.append(cur_mask)
                # And variable indicating whether we reached the maximum value is also stored
                reached_max.append(max_val_reached)

                # Here we invert the mask so it can be added to the no_go array. Elements that are 1 (indicating
                #  that they are a part of the current contour bin) are switched to zero so that future contour bins
                #  are not allowed to include them. Vice Versa for elements of 0.
                flipped = cur_mask.copy()
                flipped[flipped == 1] = -1
                flipped[flipped == 0] = 1
                flipped[flipped == -1] = 0

                # The no_go mask is updated
                no_go *= flipped

                # Now this particular bin is finished, we find the pixel at which we shall start the next one
                new_peak = prod.simple_peak(no_go*src_mask, 'pix')[0].value
                cur_bin_sp = new_peak[[1, 0]]

                open_ended.update(1)

    # This makes the list of arrays into a 3D array to be returned
    all_masks = np.dstack(all_masks)

    return all_masks

