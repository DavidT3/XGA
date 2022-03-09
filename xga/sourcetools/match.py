#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 09/03/2022, 10:39. Copyright (c) The Contributors
import os
from multiprocessing import Pool
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
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


def _on_obs_id(ra: float, dec: float, obs_id: Union[str, list, np.ndarray]) -> Tuple[float, float, np.ndarray]:
    """
    Internal function used by the on_xmm_match function to check whether a passed coordinate falls directly on a
    camera for a single (or set of) ObsID(s). Checks whether exposure time is 0 at the coordinate. It cycles through
    cameras (PN, then MOS1, then MOS2), so if exposure time is 0 on PN it'll go to MOS1, etc. to try and
    account for chip gaps in different cameras.

    :param float ra: The right-ascension of the coordinate that may fall on the ObsID.
    :param float dec: The declination of the coordinate that may fall on the ObsID.
    :param str/list/np.ndarray obs_id: The ObsID(s) which we want to check whether the passed coordinate falls on.
    :return: The input RA, input dec, and ObsID match array.
    :rtype: Tuple[float, float, np.ndarray]
    """
    # Insert my standard complaint about not wanting to do an import here
    from ..products import ExpMap

    # Makes sure that the obs_id variable is iterable, whether there is just one ObsID or a set, makes it easier
    #  to write just one piece of code that deals with either type of input
    if isinstance(obs_id, str):
        obs_id = [obs_id]

    # Less convinced I actually need to do this here, as I don't modify the census dataframe
    local_census = CENSUS.copy()

    # Set oup a list to store detection information
    det = []
    # We loop through the ObsID(s) - if just one was passed it'll only loop once (I made sure that ObsID was a list
    #  a few lines above this)
    for o in obs_id:
        # This variable describes whether the RA-Dec has a non-zero exposure for this current ObsID, starts off False
        cur_det = False
        # Get the relevant census row for this ObsID, we specifically want to know which instruments XGA thinks
        #  that it is allowed to use
        rel_row = local_census[local_census['ObsID'] == o].iloc[0]
        # Loops through the census column names describing whether instruments can be used or not
        for col in ['USE_PN', 'USE_MOS1', 'USE_MOS2']:
            # If the current instrument is allowed to be used (in the census), and ONLY if we haven't already
            #  found a non-zero exposure for this ObsID, then we proceed
            if rel_row[col] and not cur_det:
                # Get the actual instrument name by splitting the column
                inst = col.split('_')[1].lower()

                # Define an XGA exposure map - we can assume that it already exists because this internal function
                #  will only be called from other functions that have already made sure the exposure maps are generated
                epath = OUTPUT + "{o}/{o}_{i}_0.5-2.0keVexpmap.fits".format(o=o, i=inst)
                ex = ExpMap(epath, o, inst, '', '', '', Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))
                # Then check to see if the exposure time is non-zero, if so then the coordinate lies on the current
                #  XMM camera. The try-except is there to catch instances where the requested coordinate is outside
                #  the data array, which is expected to happen sometimes.
                try:
                    if ex.get_exp(Quantity([ra, dec], 'deg')) != 0:
                        cur_det = True
                except ValueError:
                    pass

                # Don't know if this is necessary, but I delete the exposure map to try and minimise the memory usage
                del ex

        # When we've looped through all possible instruments for an ObsID, and if the coordinate falls on a
        #  camera, then we add the ObsID to the list of ObsIDs 'det' - meaning that there is at least some data
        #  in that observation for the input coordinates
        if cur_det:
            det.append(o)

    # If the list of ObsIDs with data is empty then its set to None, otherwise turned into a numpy array
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

    # Here we perform a check to see whether a set of coordinates is being passed, and if so are the two
    #  arrays the same length
    if isinstance(src_ra, np.ndarray) and isinstance(src_dec, np.ndarray) and len(src_ra) != len(src_dec):
        raise ValueError("If passing multiple pairs of coordinates, src_ra and src_dec must be of the same length.")
    # Just one coordinate is also allowed, but still want it to be iterable so put it in an array
    elif isinstance(src_ra, float) and isinstance(src_dec, float):
        src_ra = np.array([src_ra])
        src_dec = np.array([src_dec])
        num_cores = 1
    # Don't want one input being a single number and one being an array
    elif type(src_ra) != type(src_dec):
        raise TypeError("src_ra and src_dec must be the same type, either both floats or both arrays.")

    # The prog_dis variable controls whether the tqdm progress bar is displayed or not, don't want it to be there
    #  for single coordinate pairs
    if len(src_ra) != 1:
        prog_dis = False
    else:
        prog_dis = True

    # The dictionary stores match dataframe information, with the keys comprised of the str(ra)+str(dec)
    c_matches = {}
    # This helps keep track of the original coordinate order, so we can return information in the same order it
    #  was passed in
    order_list = []
    # If we only want to use one core, we don't set up a pool as it could be that a pool is open where
    #  this function is being called from
    if num_cores == 1:
        # Set up the tqdm instance in a with environment
        with tqdm(desc='Searching for observations near source coordinates', total=len(src_ra),
                  disable=prog_dis) as onwards:
            # Simple enough, just iterates through the RAs and Decs calling the search function and stores the
            #  results in the dictionary
            for ra_ind, r in enumerate(src_ra):
                d = src_dec[ra_ind]
                c_matches[repr(r)+repr(d)] = _simple_search(r, d, rad)[2]
                order_list.append(repr(r)+repr(d))
                onwards.update(1)
    else:
        # This is all equivalent to what's above, but with function calls added to the multiprocessing pool
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

    # Changes the order of the results to the original pass in order and stores them in a list
    results = [c_matches[n] for n in order_list]
    del c_matches

    # Result length of one means one coordinate was passed in, so we should pass back out a single dataframe
    #  rather than a single dataframe in a list
    if len(results) == 1:
        results = results[0]

        # Checks whether the dataframe inside the single result is length zero, if so then there are no relevant ObsIDs
        if len(results) == 0:
            raise NoMatchFoundError("No XMM observation found within {a} of ra={r} "
                                    "dec={d}".format(r=round(src_ra[0], 4), d=round(src_dec[0], 4), a=distance))
    # If all the dataframes in the results list are length zero, then none of the coordinates has a
    #  valid ObsID
    elif all([len(r) == 0 for r in results]):
        raise NoMatchFoundError("No XMM observation found within {a} of any input coordinate pairs".format(a=distance))

    return results


def on_xmm_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray], num_cores: int = NUM_CORES):
    """
    An extension to the simple_xmm_match function, this first finds ObsIDs close to the input coordinate(s), then it
    generates exposure maps for those observations, and finally checks to see whether the value of the exposure maps
    at an input coordinate is zero. If the value is zero for all the instruments of an observation, then that
    coordinate does not fall on the observation, otherwise if even one of the instruments has a non-zero exposure, the
    coordinate does fall on the observation.

    :param float/np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param float/np.ndarray src_dec: Dec coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param int num_cores: The number of cores to use, default is set to 90% of system cores. This is only relevant
        if multiple coordinate pairs are passed.
    :return: For a single input coordinate, a numpy array of ObsID(s) will be returned. For multiple input coordinates
        an array of arrays of ObsID(s) and None values will be returned. Each entry corresponds to the input coordinate
        array, a None value indicates that the coordinate did not fall on an XMM observation at all.
    :rtype: np.ndarray
    """
    # Boohoo local imports very sad very sad, but stops circular import errors. NullSource is a basic Source class
    #  that allows for a list of ObsIDs to be passed rather than coordinates
    from ..sources import NullSource
    from ..sas import eexpmap

    # Checks whether there are multiple input coordinates or just one. If one then the floats are turned into
    #  an array of length one to make later code easier to write (i.e. everything is iterable regardless)
    if isinstance(src_ra, float) and isinstance(src_dec, float):
        src_ra = np.array([src_ra])
        src_dec = np.array([src_dec])
        num_cores = 1

    # Again if there's just one source I don't really care about a progress bar, so I turn it off
    if len(src_ra) != 1:
        prog_dis = False
    else:
        prog_dis = True

    # This is the initial call to the simple_xmm_match function. This gives us knowledge of which coordinates are
    #  worth checking further, and which ObsIDs should be checked for those coordinates.
    init_res = simple_xmm_match(src_ra, src_dec, num_cores=num_cores)
    # If only one coordinate was passed, the return from simple_xmm_match will just be a dataframe, and I want
    #  a list of dataframes because its iterable and easier to deal with more generally
    if isinstance(init_res, pd.DataFrame):
        init_res = [init_res]
    # This is a horrible bodge. I add an empty DataFrame to the end of the list of DataFrames returned by
    #  simple_xmm_match. This is because (when I turn init_res into a numpy array), if all of the DataFrames have
    #  data (so if all the input coordinates fall near an observation) then numpy tries to be clever and just turns
    #  it into a 3D array - this throws off the rest of the code. As such I add a DataFrame at the end with no data
    #  to makes sure numpy doesn't screw me over like that.
    init_res += [DataFrame(columns=['ObsID', 'RA_PNT', 'DEC_PNT', 'USE_PN', 'USE_MOS1', 'USE_MOS2', 'dist'])]

    # These reprs are what I use as dictionary keys to store matching information in a dictionary during
    #  the multithreading approach, I construct a list of them for ALL of the input coordinates, regardless of
    #  whether they passed the initial call to simple_xmm_match or not
    all_repr = [repr(src_ra[ind]) + repr(src_dec[ind]) for ind in range(0, len(src_ra))]

    # This constructs a masking array that tells us which of the coordinates had any sort of return from
    #  simple_xmm_match - if further check is True then a coordinate fell near an observation and the exposure
    #  map should be investigated.
    further_check = np.array([len(t) > 0 for t in init_res])
    # Incidentally we don't need to check whether these are all False, because simple_xmm_match will already have
    #  errored if that was the case
    # Now construct cut down initial matching result, ra, and dec arrays
    rel_res = np.array(init_res, dtype=object)[further_check]
    # The further_check masking array is ignoring the last entry here because that corresponds to the empty DataFrame
    #  that I artificially added to bodge numpy slightly further up in the code
    rel_ra = src_ra[further_check[:-1]]
    rel_dec = src_dec[further_check[:-1]]

    # Nested list comprehension that extracts all the ObsIDs that are mentioned in the initial matching
    #  information, turning that into a set removes any duplicates, and then it gets turned back into a list.
    # This is just used to know what exposure maps need to be generated
    obs_ids = list(set([o for t in init_res for o in t['ObsID'].values]))

    # I don't super like this way of doing it, but this is where exposure maps generated by XGA will be stored, so
    #  we check and remove any ObsID that already has had exposure maps generated for that ObsID. Normally XGA sources
    #  do this automatically, but NullSource is not as clever as that
    epath = OUTPUT + "{o}/{o}_{i}_0.5-2.0keVexpmap.fits"
    obs_ids = [o for o in obs_ids if not os.path.exists(epath.format(o=o, i='pn'))
               and not os.path.exists(epath.format(o=o, i='mos1')) and not os.path.exists(epath.format(o=o, i='mos2'))]

    try:
        # Declaring the NullSource with all the ObsIDs that; a) we need to use to check whether coordinates fall on
        #  an XMM camera, and b) don't already have exposure maps generated
        obs_src = NullSource(obs_ids)
        # Run exposure map generation for those ObsIDs
        eexpmap(obs_src)
    except NoValidObservationsError:
        pass

    # This is all the same deal as in simple_xmm_match, but calls the _on_obs_id internal function
    e_matches = {}
    order_list = []
    if num_cores == 1:
        with tqdm(desc='Confirming coordinates fall on an observation', total=len(rel_ra),
                  disable=prog_dis) as onwards:
            for ra_ind, r in enumerate(rel_ra):
                d = src_dec[ra_ind]
                o = rel_res[ra_ind]['ObsID'].values
                e_matches[repr(r) + repr(d)] = _on_obs_id(r, d, o)[2]
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

            for ra_ind, r in enumerate(rel_ra):
                d = rel_dec[ra_ind]
                o = rel_res[ra_ind]['ObsID'].values
                order_list.append(repr(r) + repr(d))
                pool.apply_async(_on_obs_id, args=(r, d, o), callback=match_loop_callback)

            pool.close()  # No more tasks can be added to the pool
            pool.join()  # Joins the pool, the code will only move on once the pool is empty.

    # Makes sure that the results list contains entries for ALL the input coordinates, not just those ones
    #  that we investigated further with exposure maps
    results = []
    for rpr in all_repr:
        if rpr in e_matches:
            results.append(e_matches[rpr])
        else:
            results.append(None)
    del e_matches

    # Again it's all the same deal as in simple_xmm_match
    if len(results) == 1:
        results = results[0]

        if results is None:
            raise NoMatchFoundError("The coordinates ra={r} dec={d} do not fall on the camera of an XMM "
                                    "observation".format(r=round(src_ra[0], 4), d=round(src_dec[0], 4)))
    elif all([r is None or len(r) == 0 for r in results]):
        raise NoMatchFoundError("None of the input coordinates fall on the camera of an XMM observation.")

    results = np.array(results, dtype=object)
    return results





