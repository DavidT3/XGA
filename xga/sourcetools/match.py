#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 28/02/2022, 16:33. Copyright (c) The Contributors

from multiprocessing import Pool
from typing import Union, Tuple, List

import numpy as np
from astropy.units.quantity import Quantity
from pandas import DataFrame
from tqdm import tqdm

from .. import CENSUS, BLACKLIST, NUM_CORES
from ..exceptions import NoMatchFoundError


def _simple_search(ra: float, dec: float, search_rad: float) -> Tuple[float, float, DataFrame]:
    local_census = CENSUS.copy()
    local_census["dist"] = np.sqrt((local_census["RA_PNT"] - ra) ** 2
                                   + (local_census["DEC_PNT"] - dec) ** 2)
    matches = local_census[local_census["dist"] <= search_rad]
    matches = matches[~matches["ObsID"].isin(BLACKLIST["ObsID"])]

    return ra, dec, matches


def simple_xmm_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],
                     distance: Quantity = Quantity(30.0, 'arcmin'),
                     num_cores: int = NUM_CORES) -> Union[DataFrame, List[DataFrame]]:
    """
    Returns ObsIDs within a given distance from the input ra and dec values.

    :param float/np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param float/np.ndarray src_dec: DEC coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param Quantity distance: The distance to search for XMM observations within, default should be
        able to match a source on the edge of an observation to the centre of the observation.
    :param int num_cores: The number of cores to use, default is set to 90% of system cores. This is only relevant
        if multiple coordinate pairs are passed.
    :return: The ObsID, RA_PNT, and DEC_PNT of matching XMM observations.
    :rtype: Union[DataFrame, List[DataFrame]]
    """

    # Extract the search distance as a float, specifically in degrees
    rad = distance.to('deg').value

    if isinstance(src_ra, np.ndarray) and isinstance(src_dec, np.ndarray) and len(src_ra) != len(src_dec):
        raise ValueError("If passing multiple pairs of coordinates, src_ra and src_dec must be of the same length.")
    elif isinstance(src_ra, float) and isinstance(src_dec, float):
        src_ra = np.array([src_ra])
        src_dec = np.array([src_dec])
        num_cores = 1
    elif type(src_ra) != type(src_dec):
        raise TypeError("src_ra and src_dec must be the same type, either both floats or both arrays.")

    if len(src_ra) != 1:
        prog_dis = False
    else:
        prog_dis = True

    c_matches = {}
    order_list = []
    if num_cores == 1:
        with tqdm(desc='Searching for observations near source coordinates', total=len(src_ra),
                  disable=prog_dis) as onwards:
            for ra_ind, r in enumerate(src_ra):
                d = src_dec[ra_ind]
                c_matches[repr(r)+repr(d)] = _simple_search(r, d, rad)[2]
                order_list.append(repr(r)+repr(d))
                onwards.update(1)
    else:
        with tqdm(desc="Searching for observations near source coordinates", total=len(src_ra)) as onwards, \
                Pool(num_cores) as pool:
            def match_loop_callback(match_info):
                nonlocal onwards  # The progress bar will need updating
                nonlocal c_matches
                c_matches[repr(match_info[0]) + repr(match_info[1])] = match_info[2]
                onwards.update(1)

            for ra_ind, r in enumerate(src_ra):
                d = src_dec[ra_ind]
                order_list.append(repr(r)+repr(d))
                pool.apply_async(_simple_search, args=(r, d, rad), callback=match_loop_callback)

            pool.close()  # No more tasks can be added to the pool
            pool.join()  # Joins the pool, the code will only move on once the pool is empty.

    results = [c_matches[n] for n in order_list]
    del c_matches

    if len(results) == 1:
        results = results[0]

        if len(results) == 0:
            raise NoMatchFoundError("No XMM observation found within {a} of ra={r} "
                                    "dec={d}".format(r=round(src_ra[0], 4), d=round(src_dec[0], 4), a=distance))
    elif all([len(r) == 0 for r in results]):
        raise NoMatchFoundError("No XMM observation found within {a} of any input coordinate pairs".format(a=distance))

    return results
