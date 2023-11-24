#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 24/11/2023, 13:22. Copyright (c) The Contributors
import gc
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.units.quantity import Quantity
from exceptiongroup import ExceptionGroup
from pandas import DataFrame
from regions import read_ds9, PixelRegion
from tqdm import tqdm

from .. import CENSUS, BLACKLIST, NUM_CORES, OUTPUT, xga_conf
from ..exceptions import NoMatchFoundError, NoValidObservationsError, NoRegionsError, XGAConfigError
from ..utils import SRC_REGION_COLOURS


def _dist_from_source(search_ra: float, search_dec: float, cur_reg):
    """
    Calculates the euclidean distance between the centre of a supplied region, and the
    position of the source.

    :param reg: A region object.
    :return: Distance between region centre and source position.
    """
    r_ra = cur_reg.center.ra.value
    r_dec = cur_reg.center.dec.value
    return np.sqrt(abs(r_ra - search_ra) ** 2 + abs(r_dec - search_dec) ** 2)


def _process_init_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],
                        initial_results: Union[DataFrame, List[DataFrame]]):
    """
    An internal function that takes the results of a simple match and assembles a list of unique ObsIDs which are of
    interest to the coordinate(s) we're searching for data for.  Sets of RA and Decs that were found to be near XMM data by
    the initial simple match are also created and returned.

    :param float/np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param float/np.ndarray src_dec: Dec coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param DataFrame/List[DataFrame] initial_results: The result of a simple_xmm_match run.
    :return: The simple match initial results (normalised so that they are a list of dataframe, even if only one
        source is being searched for), a list of  unique ObsIDs, unique string representations generated from RA and
        Dec for the positions  we're looking at, an array of dataframes for those coordinates that are near an
        XMM observation according to the initial match, and the RA and Decs that are near an XMM observation
        according to the initial simple match. The final output is a dictionary with ObsIDs as keys, and arrays of
        source coordinates that are an initial match with them.
    """
    # If only one coordinate was passed, the return from simple_xmm_match will just be a dataframe, and I want
    #  a list of dataframes because its iterable and easier to deal with more generally
    if isinstance(initial_results, pd.DataFrame):
        initial_results = [initial_results]
    # This is a horrible bodge. I add an empty DataFrame to the end of the list of DataFrames returned by
    #  simple_xmm_match. This is because (when I turn init_res into a numpy array), if all of the DataFrames have
    #  data (so if all the input coordinates fall near an observation) then numpy tries to be clever and just turns
    #  it into a 3D array - this throws off the rest of the code. As such I add a DataFrame at the end with no data
    #  to makes sure numpy doesn't screw me over like that.
    initial_results += [DataFrame(columns=['ObsID', 'RA_PNT', 'DEC_PNT', 'USE_PN', 'USE_MOS1', 'USE_MOS2', 'dist'])]

    # These reprs are what I use as dictionary keys to store matching information in a dictionary during
    #  the multithreading approach, I construct a list of them for ALL of the input coordinates, regardless of
    #  whether they passed the initial call to simple_xmm_match or not
    all_repr = [repr(src_ra[ind]) + repr(src_dec[ind]) for ind in range(0, len(src_ra))]

    # This constructs a masking array that tells us which of the coordinates had any sort of return from
    #  simple_xmm_match - if further check is True then a coordinate fell near an observation and the exposure
    #  map should be investigated.
    further_check = np.array([len(t) > 0 for t in initial_results])
    # Incidentally we don't need to check whether these are all False, because simple_xmm_match will already have
    #  errored if that was the case
    # Now construct cut down initial matching result, ra, and dec arrays
    rel_res = np.array(initial_results, dtype=object)[further_check]
    # The further_check masking array is ignoring the last entry here because that corresponds to the empty DataFrame
    #  that I artificially added to bodge numpy slightly further up in the code
    rel_ra = src_ra[further_check[:-1]]
    rel_dec = src_dec[further_check[:-1]]

    # Nested list comprehension that extracts all the ObsIDs that are mentioned in the initial matching
    #  information, turning that into a set removes any duplicates, and then it gets turned back into a list.
    # This is just used to know what exposure maps need to be generated
    obs_ids = list(set([o for t in initial_results for o in t['ObsID'].values]))

    # This assembles a big numpy array with every source coordinate and its ObsIDs (source coordinate will have one
    #  entry per ObsID associated with them).
    repeats = [len(cur_res) for cur_res in rel_res]
    full_info = np.vstack([np.concatenate([cur_res['ObsID'].to_numpy() for cur_res in rel_res]),
                           np.repeat(rel_ra, repeats), np.repeat(rel_dec, repeats)]).T

    # This assembles a dictionary that links source coordinates to ObsIDs (ObsIDs are the keys)
    obs_id_srcs = {o: full_info[np.where(full_info[:, 0] == o)[0], :][:, 1:] for o in obs_ids}

    return initial_results, obs_ids, all_repr, rel_res, rel_ra, rel_dec, obs_id_srcs


def _simple_search(ra: float, dec: float, search_rad: float) -> Tuple[float, float, DataFrame, DataFrame]:
    """
    Internal function used to multithread the simple XMM match function.

    :param float ra: The right-ascension around which to search for observations, as a float in units of degrees.
    :param float dec: The declination around which to search for observations, as a float in units of degrees.
    :param float search_rad: The radius in which to search for observations, as a float in units of degrees.
    :return: The input RA, input dec, ObsID match dataframe, and the completely blacklisted array (ObsIDs that
        were relevant but have ALL instruments blacklisted).
    :rtype: Tuple[float, float, DataFrame, DataFrame]
    """
    # Making a copy of the census because I add a distance-from-coords column - don't want to do that for the
    #  original census especially when this is being multi-threaded
    local_census = CENSUS.copy()
    local_blacklist = BLACKLIST.copy()
    local_census["dist"] = np.sqrt((local_census["RA_PNT"] - ra) ** 2
                                   + (local_census["DEC_PNT"] - dec) ** 2)
    # Select any ObsIDs within (or at) the search radius input to the function
    matches = local_census[local_census["dist"] <= search_rad]
    # Locate any ObsIDs that are in the blacklist, then test to see whether ALL the instruments are to be excluded
    in_bl = local_blacklist[
        local_blacklist['ObsID'].isin(matches[matches["ObsID"].isin(local_blacklist["ObsID"])]['ObsID'])]
    # This will find relevant blacklist entries that have specifically ALL instruments excluded. In that case
    #  the ObsID shouldn't be returned
    all_excl = in_bl[(in_bl['EXCLUDE_PN'] == 'T') & (in_bl['EXCLUDE_MOS1'] == 'T') & (in_bl['EXCLUDE_MOS2'] == 'T')]

    # These are the observations that a) match (within our criteria) to the supplied coordinates, and b) have at
    #  least some usable data.
    matches = matches[~matches["ObsID"].isin(all_excl["ObsID"])]

    del local_census
    del local_blacklist
    return ra, dec, matches, all_excl


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

    # Set up a list to store detection information
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


def _in_region(ra: Union[float, List[float], np.ndarray], dec: Union[float, List[float], np.ndarray], obs_id: str,
               allowed_colours: List[str]) -> Tuple[str, dict]:
    """
    Internal function to search a particular ObsID's region files for matches to the sources defined in the RA
    and Dec arguments. This is achieved using the Regions module, and a region is a 'match' to a source if the
    source coordinates fall somewhere within the region, and the region is of an acceptable coloru (defined in
    allowed_colours). This requires that both images and region files are properly setup in the XGA config file.

    :param float/List[float]/np.ndarray ra: The set of source RA coords to match with the obs_id's regions.
    :param float/List[float]/np.ndarray dec: The set of source DEC coords to match with the obs_id's regions.
    :param str obs_id: The ObsID whose regions we are matching to.
    :param List[str] allowed_colours: The colours of region that should be accepted as a match.
    :return: The ObsID that was being searched, and a dictionary of matched regions (the keys are unique
        representations of the sources passed in), and the values are lists of region objects.
    :rtype: Tuple[str, dict]
    """
    from ..products import Image

    if isinstance(ra, float):
        ra = [ra]
        dec = [dec]

    # From that ObsID construct a path to the relevant region file using the XGA config
    reg_path = xga_conf["XMM_FILES"]["region_file"].format(obs_id=obs_id)
    im_path = None
    # We need to check whether any of the images in the config file exist for this ObsID - have to use the
    #  pre-configured images in case the region files are in pixel coordinates
    for key in ['pn_image', 'mos1_image', 'mos2_image']:
        for en_comb in zip(xga_conf["XMM_FILES"]["lo_en"], xga_conf["XMM_FILES"]["hi_en"]):
            cur_path = xga_conf["XMM_FILES"][key].format(obs_id=obs_id, lo_en=en_comb[0], hi_en=en_comb[1])
            if os.path.exists(cur_path):
                im_path = cur_path

    # This dictionary stores the match regions for each coordinate
    matched = {}

    # If there is a region file to search then we can proceed
    if os.path.exists(reg_path) and im_path is not None:
        # onwards.write("None of the specified image files for {} can be located - skipping region match "
        #               "search.".format(obs_id))

        # Reading in the region file using the Regions module
        og_ds9_regs = read_ds9(reg_path)

        # Bodged declaration, the instrument and energy bounds don't matter - all I need this for is the
        #  nice way it extracts the WCS information that I need
        im = Image(im_path, obs_id, '', '', '', '', Quantity(0, 'keV'), Quantity(1, 'keV'), )

        # There's nothing for us to do if there are no regions in the region file, so we continue onto the next
        #  possible ObsID match (if there is one) - same deal if there is no WCS information in the image
        if len(og_ds9_regs) != 0 and im.radec_wcs is not None:

            # Make sure to convert the regions to sky coordinates if they in pixels (just on principle in case
            #  any DO match and are returned to the user, I would much give them image agnostic regions).
            if any([isinstance(r, PixelRegion) for r in og_ds9_regs]):
                og_ds9_regs = np.array([reg.to_sky(im.radec_wcs) for reg in og_ds9_regs])

            # This cycles through every ObsID in the possible matches for the current object
            for r_ind, cur_ra in enumerate(ra):
                cur_dec = dec[r_ind]
                cur_repr = repr(cur_ra) + repr(cur_dec)

                # Make a local (to this iteration) copy as this array is modified during the checking process
                ds9_regs = deepcopy(og_ds9_regs)

                # Hopefully this bodge doesn't have any unforeseen consequences
                if ds9_regs[0] is not None and len(ds9_regs) > 1:
                    # Quickly calculating distance between source and center of regions, then sorting
                    # and getting indices. Thus I only match to the closest 5 regions.
                    diff_sort = np.array([_dist_from_source(cur_ra, cur_dec, r) for r in ds9_regs]).argsort()
                    # Unfortunately due to a limitation of the regions module I think you need images
                    #  to do this contains match...
                    within = np.array([reg.contains(SkyCoord(cur_ra, cur_dec, unit='deg'), im.radec_wcs)
                                       for reg in ds9_regs[diff_sort[0:5]]])

                    # Make sure to re-order the region list to match the sorted within array
                    ds9_regs = ds9_regs[diff_sort]

                    # Expands it so it can be used as a mask on the whole set of regions for this observation
                    within = np.pad(within, [0, len(diff_sort) - len(within)])
                    match_within = ds9_regs[within]
                # In the case of only one region being in the list, we simplify the above expression
                elif ds9_regs[0] is not None and len(ds9_regs) == 1 and \
                        ds9_regs[0].contains(SkyCoord(cur_ra, cur_dec, unit='deg'), im.radec_wcs):
                    match_within = ds9_regs
                else:
                    match_within = np.array([])

                match_within = [r for r in match_within if r.visual['color'] in allowed_colours]
                if len(match_within) != 0:
                    matched[cur_repr] = match_within

        del im

    gc.collect()
    return obs_id, matched


def simple_xmm_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],
                     distance: Quantity = Quantity(30.0, 'arcmin'), num_cores: int = NUM_CORES) \
        -> Tuple[Union[DataFrame, List[DataFrame]], Union[DataFrame, List[DataFrame]]]:
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
    :return: A dataframe containing ObsID, RA_PNT, and DEC_PNT of matching XMM observations, and a dataframe
        containing information on observations that would have been a match, but that are in the blacklist.
    :rtype: Tuple[Union[DataFrame, List[DataFrame]], Union[DataFrame, List[DataFrame]]]
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
    # This dictionary stores any ObsIDs that were COMPLETELY blacklisted (i.e. all instruments were excluded) for
    #  a given coordinate. So they were initially found as being nearby, but then completely removed
    fully_blacklisted = {}

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
                search_results = _simple_search(r, d, rad)
                c_matches[repr(r) + repr(d)] = search_results[2]
                fully_blacklisted[repr(r) + repr(d)] = search_results[3]
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
                fully_blacklisted[repr(match_info[0]) + repr(match_info[1])] = match_info[3]

                onwards.update(1)

            for ra_ind, r in enumerate(src_ra):
                d = src_dec[ra_ind]
                order_list.append(repr(r)+repr(d))
                pool.apply_async(_simple_search, args=(r, d, rad), callback=match_loop_callback)

            pool.close()  # No more tasks can be added to the pool
            pool.join()  # Joins the pool, the code will only move on once the pool is empty.

    # Changes the order of the results to the original pass in order and stores them in a list
    results = [c_matches[n] for n in order_list]
    bl_results = [fully_blacklisted[n] for n in order_list]
    del c_matches
    del fully_blacklisted

    # Result length of one means one coordinate was passed in, so we should pass back out a single dataframe
    #  rather than a single dataframe in a list
    if len(results) == 1:
        results = results[0]
        bl_results = bl_results[0]

        # Checks whether the dataframe inside the single result is length zero, if so then there are no relevant ObsIDs
        if len(results) == 0:
            raise NoMatchFoundError("No XMM observation found within {a} of ra={r} "
                                    "dec={d}".format(r=round(src_ra[0], 4), d=round(src_dec[0], 4), a=distance))
    # If all the dataframes in the results list are length zero, then none of the coordinates has a
    #  valid ObsID
    elif all([len(r) == 0 for r in results]):
        raise NoMatchFoundError("No XMM observation found within {a} of any input coordinate pairs".format(a=distance))

    return results, bl_results


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
    init_res, init_bl = simple_xmm_match(src_ra, src_dec, num_cores=num_cores)

    # This function constructs a list of unique ObsIDs that we have determined are of interest to us using the simple
    #  match function, then sets up a NullSource and generate exposure maps that will be used to figure out if the
    #  sources are actually on an XMM pointing. Also returns cut down lists of RA, Decs etc. that are the sources
    #  which the simple match found to be near XMM data
    init_res, obs_ids, all_repr, rel_res, rel_ra, rel_dec, obs_id_srcs = _process_init_match(src_ra, src_dec, init_res)

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
        eexpmap(obs_src, num_cores=num_cores)
    except NoValidObservationsError:
        pass

    # This is all the same deal as in simple_xmm_match, but calls the _on_obs_id internal function
    e_matches = {}
    order_list = []
    if num_cores == 1:
        with tqdm(desc='Confirming coordinates fall on an observation', total=len(rel_ra),
                  disable=prog_dis) as onwards:
            for ra_ind, r in enumerate(rel_ra):
                d = rel_dec[ra_ind]
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


def xmm_region_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],
                     src_type: Union[str, List[str]], num_cores: int = NUM_CORES) -> np.ndarray:
    """
    A function which, if XGA has been configured with access to pre-generated region files, will search for region
    matches for a set of source coordinates passed in by the user. A region match is defined as when a source
    coordinate falls within a source region with a particular colour (largely used to represent point vs
    extended) - the type of region that should be matched to can be defined using the src_type argument.

    The simple_xmm_match function will be run before the source matching process, to narrow down the sources which
    need to have the more expensive region matching performed, as well as to identify which ObsID(s) should be
    examined for each source.

    :param float/np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param float/np.ndarray src_dec: Dec coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param str/List[str] src_type: The type(s) of region that should be matched to. Pass either 'ext' or 'pnt' or
        a list containing both.
    :param int num_cores: The number of cores that can be used for the matching process.
    :return: An array the same length as the sets of input coordinates (ordering is the same). If there are no
        matches for a source then the element will be None, if there are matches then the element will be a
        dictionary, with the key(s) being ObsID(s) and the values being a list of region objects (or more
        likely just one object).
    :rtype: np.ndarray
    """
    # Checks the input src_type argument, and makes it a list even if it is just a single string - easier
    #  to deal with it like that!
    if isinstance(src_type, str):
        src_type = [src_type]

    # Also checks to make sure that no illegal values for src_type have been passed (SRC_REGION_COLOURS basically
    #  maps from region colours to source types).
    if any([st not in SRC_REGION_COLOURS for st in src_type]):
        raise ValueError("The values supported for 'src_type' are "
                         "{}".format(', '.join(list(SRC_REGION_COLOURS.keys()))))

    # Ugly but oh well, constructs the list of region colours that we can match to from the source types
    # that the user chose
    allowed_colours = []
    for st in src_type:
        allowed_colours += SRC_REGION_COLOURS[st]

    # Checks to make sure that the user has actually pointed XGA at a set of region files (and images they were
    #  generated from, in case said region files are in pixel coordinates).
    if xga_conf["XMM_FILES"]["region_file"] == "/this/is/optional/xmm_obs/regions/{obs_id}/regions.reg":
        raise NoRegionsError("The configuration file does not contain information on region files, so this function "
                             "cannot continue.")
    elif xga_conf["XMM_FILES"]['pn_image'] == "/this/is/optional/xmm_obs/regions/{obs_id}/regions.reg" and \
            xga_conf["XMM_FILES"]['mos1_image'] == "/this/is/optional/xmm_obs/regions/{obs_id}/regions.reg" and \
            xga_conf["XMM_FILES"]['mos2_image'] == "/this/is/optional/xmm_obs/regions/{obs_id}/regions.reg":
        raise XGAConfigError("This function requires at least one set of images (PN, MOS1, or MOS2) be referenced in "
                             "the XGA configuration file.")

    # This runs the simple xmm match and gathers the results.
    s_match, s_match_bl = simple_xmm_match(src_ra, src_dec, num_cores=num_cores)
    # The initial results are then processed into some more useful formats.
    s_match, uniq_obs_ids, all_repr, rel_res, rel_ra, rel_dec, \
        obs_id_srcs = _process_init_match(src_ra, src_dec, s_match)

    # This is the dictionary in which matching information is stored
    reg_match_info = {rp: {} for rp in all_repr}
    # If the user only wants us to use one core, then we don't make a Pool because that would just add overhead
    if num_cores == 1:
        with tqdm(desc="Searching for ObsID region matches", total=len(uniq_obs_ids)) as onwards:
            # Here we iterate through the ObsIDs that the initial match found to possibly have sources on - I
            #  considered this more efficient than iterating through the sources and possibly reading in WCS
            #  information for the same ObsID in many different processes (the non-parallelised version just calls
            #  the same internal function so its setup the same).
            for cur_obs_id in obs_id_srcs:
                cur_ra_arr = obs_id_srcs[cur_obs_id][:, 0]
                cur_dec_arr = obs_id_srcs[cur_obs_id][:, 1]
                # Runs the matching function
                match_inf = _in_region(cur_ra_arr, cur_dec_arr, cur_obs_id, allowed_colours)
                # Adds to the match storage dictionary, but so that the top keys are source representations, and
                #  the lower level keys are ObsIDs
                for cur_repr in match_inf[1]:
                    reg_match_info[cur_repr][match_inf[0]] = match_inf[1][cur_repr]
                onwards.update(1)

    else:
        # This is to store exceptions that are raised in separate processes, so they can all be raised at the end.
        search_errors = []
        # We setup a Pool with the number of cores the user specified (or the default).
        with tqdm(desc="Searching for ObsID region matches", total=len(uniq_obs_ids)) as onwards, Pool(
                num_cores) as pool:
            # This is called when a match process finished successfully, and the results need storing
            def match_loop_callback(match_info):
                nonlocal onwards  # The progress bar will need updating
                nonlocal reg_match_info
                # Adds to the match storage dictionary, but so that the top keys are source representations, and
                #  the lower level keys are ObsIDs
                for cur_repr in match_info[1]:
                    reg_match_info[cur_repr][match_info[0]] = match_info[1][cur_repr]

                onwards.update(1)

            # This is called when a process errors out.
            def error_callback(err):
                nonlocal onwards
                nonlocal search_errors
                # Stores the exception object in a list for later.
                search_errors.append(err)
                onwards.update(1)

            for cur_obs_id in obs_id_srcs:
                # Here we iterate through the ObsIDs that the initial match found to possibly have sources on - I
                #  considered this more efficient than iterating through the sources and possibly reading in WCS
                #  information for the same ObsID in many different processes.
                cur_ra_arr = obs_id_srcs[cur_obs_id][:, 0]
                cur_dec_arr = obs_id_srcs[cur_obs_id][:, 1]
                pool.apply_async(_in_region, args=(cur_ra_arr, cur_dec_arr, cur_obs_id, allowed_colours),
                                 callback=match_loop_callback, error_callback=error_callback)

            pool.close()  # No more tasks can be added to the pool
            pool.join()  # Joins the pool, the code will only move on once the pool is empty.

        # If any errors occurred during the matching process, they are all raised here as a grouped exception
        if len(search_errors) != 0:
            ExceptionGroup("The following exceptions were raised in the multi-threaded region finder", search_errors)

    # This formats the match and no-match information so that the output is the same length and order as the input
    #  source lists
    to_return = []
    for cur_repr in all_repr:
        if len(reg_match_info[cur_repr]) != 0:
            to_return.append(reg_match_info[cur_repr])
        else:
            to_return.append(None)

    # Makes it into an array rather than a list
    to_return = np.array(to_return)

    return to_return
