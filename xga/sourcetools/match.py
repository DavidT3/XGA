#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/03/2022, 12:16. Copyright (c) The Contributors
import os
from multiprocessing import Pool
from typing import Union, Tuple, List

import numpy as np
from astropy.units.quantity import Quantity
from pandas import DataFrame
from tqdm import tqdm

from .. import CENSUS, BLACKLIST, NUM_CORES, OUTPUT
from ..exceptions import NoMatchFoundError, NoValidObservationsError


def _simple_search(ra: float, dec: float, search_rad: float) -> Tuple[float, float, DataFrame]:
    """
    Internal function used to multithread the simple XMM match function.

    :param float ra: The right-ascension around which to search for observations.
    :param float dec: The declination around which to search for observations.
    :param float search_rad: The radius in which to search for observations.
    :return: The input RA, input dec, and ObsID match dataframe.
    :rtype: Tuple[float, float, DataFrame]
    """
    # Making a copy of the census because I add a distance-from-coords column - don't want to do that for the
    #  original census especially when this is being multi-threaded
    local_census = CENSUS.copy()
    local_census["dist"] = np.sqrt((local_census["RA_PNT"] - ra) ** 2
                                   + (local_census["DEC_PNT"] - dec) ** 2)
    # Select any ObsIDs within (or at) the search radius input to the function
    matches = local_census[local_census["dist"] <= search_rad]
    # Remove any ObsID dataframe entries that are in the blacklist
    matches = matches[~matches["ObsID"].isin(BLACKLIST["ObsID"])]
    del local_census
    return ra, dec, matches


def _on_obs_id(ra: float, dec: float, obs_id: Union[str, list, np.ndarray]):
    if isinstance(obs_id, str):
        obs_id = [obs_id]
    from ..products import ExpMap

    local_census = CENSUS.copy()

    det = []
    for o in obs_id:
        cur_det = False
        rel_row = local_census[local_census['ObsID'] == o].iloc[0]
        for col in ['USE_PN', 'USE_MOS1', 'USE_MOS2']:
            if rel_row[col] and not cur_det:
                inst = col.split('_')[1].lower()

                epath = OUTPUT + "{o}/{o}_{i}_0.5-2.0keVexpmap.fits".format(o=o, i=inst)
                ex = ExpMap(epath, o, inst, '', '', '', Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))
                try:
                    if ex.get_exp(Quantity([ra, dec], 'deg')) != 0:
                        cur_det = True
                except ValueError:
                    pass

                del ex

        if cur_det:
            det.append(o)

    if len(det) == 0:
        det = None
    else:
        det = np.array(det)
    return ra, dec, det


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


def on_xmm_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray], num_cores: int = NUM_CORES):
    from ..sources import NullSource
    from ..sas import eexpmap

    if isinstance(src_ra, float) and isinstance(src_dec, float):
        src_ra = np.array([src_ra])
        src_dec = np.array([src_dec])
        num_cores = 1

    if len(src_ra) != 1:
        prog_dis = False
    else:
        prog_dis = True

    all_repr = [repr(src_ra[ind]) + repr(src_dec[ind]) for ind in range(0, len(src_ra))]

    init_res = np.array(simple_xmm_match(src_ra, src_dec, num_cores=num_cores), dtype=object)
    further_check = np.array([len(t) > 0 for t in init_res])
    rel_res = init_res[further_check]
    rel_ra = src_ra[further_check]
    rel_dec = src_dec[further_check]

    obs_ids = list(set([o for t in init_res for o in t['ObsID'].values]))

    epath = OUTPUT + "{o}/{o}_{i}_0.5-2.0keVexpmap.fits"
    obs_ids = [o for o in obs_ids if not os.path.exists(epath.format(o=o, i='pn'))
               and not os.path.exists(epath.format(o=o, i='mos1')) and not os.path.exists(epath.format(o=o, i='mos2'))]

    try:
        obs_src = NullSource(obs_ids)
        eexpmap(obs_src)
    except NoValidObservationsError:
        pass

    e_matches = {}
    order_list = []
    if num_cores == 1:
        raise NotImplementedError("HAVEN'T DONE THIS YET")
        with tqdm(desc='Searching for observations near source coordinates', total=len(src_ra),
                  disable=prog_dis) as onwards:
            for ra_ind, r in enumerate(src_ra):
                d = src_dec[ra_ind]
                c_matches[repr(r) + repr(d)] = _simple_search(r, d, rad)[2]
                order_list.append(repr(r) + repr(d))
                onwards.update(1)
    else:
        with tqdm(desc="Confirming coordinates fall on an observation", total=len(rel_ra)) as onwards, \
                Pool(num_cores) as pool:
            def match_loop_callback(match_info):
                nonlocal onwards  # The progress bar will need updating
                nonlocal e_matches
                e_matches[repr(match_info[0]) + repr(match_info[1])] = match_info[2]
                onwards.update(1)

            def bugger(err):
                # print(str(err))
                raise err

            for ra_ind, r in enumerate(rel_ra):
                d = rel_dec[ra_ind]
                o = rel_res[ra_ind]['ObsID'].values
                order_list.append(repr(r) + repr(d))
                pool.apply_async(_on_obs_id, args=(r, d, o), callback=match_loop_callback, error_callback=bugger)

            pool.close()  # No more tasks can be added to the pool
            pool.join()  # Joins the pool, the code will only move on once the pool is empty.

    results = []
    for rpr in all_repr:
        if rpr in e_matches:
            results.append(e_matches[rpr])
        else:
            results.append(None)
    del e_matches

    if len(results) == 1:
        results = results[0]

        if len(results) == 0:
            raise NoMatchFoundError("No XMM observation found within {a} of ra={r} "
                                    "dec={d}".format(r=round(src_ra[0], 4), d=round(src_dec[0], 4), a=distance))
    elif all([r is None or len(r) == 0 for r in results]):
        raise NoMatchFoundError("No XMM observation found within {a} of any input coordinate pairs".format(a=distance))

    results = np.array(results, dtype=object)
    return results





