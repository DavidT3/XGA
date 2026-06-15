#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 6/14/26, 8:32 PM. Copyright (c) The Contributors.
from __future__ import annotations

import gc
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Union, Tuple, List
from warnings import warn

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.units.quantity import Quantity
from exceptiongroup import ExceptionGroup
from pandas import DataFrame
from regions import PixelRegion, Regions, SkyRegion
from tqdm import tqdm

from .. import (CENSUS, BLACKLIST, NUM_CORES, xga_conf, DEFAULT_TELE_SEARCH_DIST, SRC_REGION_COLOURS,
                check_telescope_choices, PRETTY_TELESCOPE_NAMES)
from ..exceptions import NoMatchFoundError, NoRegionsError, NoProductAvailableError

# Global variables for worker processes to store pre-calculated coordinates and products
#  This ensures the indexing and serialization costs are paid only once per worker, not once per chunk.
WORKER_TEL_COORDS = {}
WORKER_EXP_MAPS = {}


def _worker_init(census_coords: dict = None, exp_map_dict: dict = None):
    """
    Initializer for worker processes to store pre-calculated data in global scope.
    """
    global WORKER_TEL_COORDS
    global WORKER_EXP_MAPS
    if census_coords is not None:
        WORKER_TEL_COORDS = census_coords
    if exp_map_dict is not None:
        WORKER_EXP_MAPS = exp_map_dict


def _dist_from_source(search_ra: float, search_dec: float, cur_reg: SkyRegion):
    """
    Calculates the distance between the centre of a supplied region, and the position of the source. We use the
    Haversine formula to determine the separation on the surface of a sphere.

    :param SkyRegion cur_reg: A region object.
    :rtype: float
    :return: Distance between region centre and source position, in degrees.
    """
    r_ra = cur_reg.center.ra.to('radian').value
    r_dec = cur_reg.center.dec.to('radian').value

    # The numpy trig functions want everything in radians, so we make sure that is the case
    search_ra = search_ra * (np.pi / 180)
    search_dec = search_dec * (np.pi / 180)

    # Then just use the Haversine formula to calculate the separation
    hav_sep = 2 * np.arcsin(np.sqrt((np.sin((search_dec - r_dec) / 2) ** 2)
                                    + np.cos(r_dec) * np.cos(search_dec) * np.sin((search_ra - r_ra) / 2) ** 2))
    # Converted from radians to degrees - not using quantities in this internal function
    return hav_sep / (np.pi / 180)


def _vectorized_separation_match(src_ra: np.ndarray, src_dec: np.ndarray, src_indices: np.ndarray,
                                telescope: List[str], distance: dict,
                                census_coords: dict = None, return_flat: bool = False) -> Tuple[dict, dict]:
    """
    Internal function that performs vectorized coordinate matching for a set of sources against a set of telescopes.
    Used by both the serial and parallel paths of separation_match.

    :param np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees.
    :param np.ndarray src_dec: Dec coordinate(s) of the source(s), in degrees.
    :param np.ndarray src_indices: Global indices of the sources being processed.
    :param List[str] telescope: List of telescopes to search.
    :param dict distance: Dictionary of search distances for each telescope.
    :param dict census_coords: Dictionary of pre-calculated SkyCoord objects for each telescope census. If None,
        then worker processes will check the global WORKER_TEL_COORDS.
    :param bool return_flat: If True, then a flat DataFrame will be returned instead of a sparse dictionary.
    :return: A tuple containing two sparse dictionaries (matches and blacklisted matches).
        Format: {telescope: {global_index: DataFrame}} (if return_flat is False)
        OR (DataFrame, DataFrame) (if return_flat is True)
    :rtype: Tuple[dict, dict]
    """
    # Sparse results: {telescope: {global_index: matches_df}}
    if not return_flat:
        tel_matches = {tel: {} for tel in telescope}
        tel_bl = {tel: {} for tel in telescope}
    else:
        tel_matches = []
        tel_bl = []

    # We also need an array of SkyCoords for the sources
    src_coords = SkyCoord(ra=src_ra, dec=src_dec, unit='deg')

    for tel in telescope:
        # We grab the census for the current telescope, and drop any rows that have a NaN coordinate
        rel_census = CENSUS[tel].dropna(subset=['RA_PNT', 'DEC_PNT'])
        if len(rel_census) == 0:
            continue

        # And the blacklist for the current telescope
        rel_bl = BLACKLIST[tel]
        # We identify all ObsIDs that are completely blacklisted for this telescope
        excl_cols = [col for col in rel_bl.columns if 'EXCLUDE' in col]
        if len(excl_cols) > 0:
            is_fully_bl = np.logical_and.reduce([rel_bl[col].values == True for col in excl_cols])
            fully_bl_df = rel_bl[is_fully_bl]
            fully_bl_obsids = set(fully_bl_df['ObsID'].values)
        else:
            fully_bl_df = DataFrame(columns=rel_bl.columns)
            fully_bl_obsids = set()

        # We determine the coordinates for the census entries - either from passed argument, worker global, or manual calc
        if census_coords is not None and tel in census_coords:
            tel_coords = census_coords[tel]
        elif tel in WORKER_TEL_COORDS:
            tel_coords = WORKER_TEL_COORDS[tel]
        else:
            tel_coords = SkyCoord(ra=rel_census['RA_PNT'].values, dec=rel_census['DEC_PNT'].values, unit='deg')

        # We need to make sure the search distance is a scalar if it only has one entry
        search_dist = distance[tel]
        if isinstance(search_dist, Quantity) and search_dist.shape == (1,):
            search_dist = search_dist[0]

        # And then we perform a vectorized search for all sources at once
        # idx_src is the index of the source within this chunk, idx_tel is the index of the census entry
        idx_tel, idx_src, d2d, _ = src_coords.search_around_sky(tel_coords, search_dist)

        if len(idx_src) > 0:
            # We create a dataframe of all matches
            matched_census = rel_census.iloc[idx_tel].copy()
            # We add the source index so we can group the results later
            matched_census['src_idx'] = src_indices[idx_src]
            # We also add the distance to the source
            matched_census['dist'] = d2d.to('deg').value
            # We also mark which ones are fully blacklisted
            matched_census['is_bl'] = matched_census['ObsID'].isin(fully_bl_obsids)
            # Add telescope name
            matched_census['telescope'] = tel

            if not return_flat:
                # We group by source index to process the results for each source
                matched_groups = matched_census.groupby('src_idx')

                # We iterate through the sources that had at least one match
                for s_idx, group in matched_groups:
                    # These are the observations that are not completely blacklisted
                    matches_df = group.loc[~group['is_bl']].drop(columns=['src_idx', 'is_bl', 'telescope'])
                    # And these are the ones that were matching but are completely blacklisted
                    bl_obs = group.loc[group['is_bl'], 'ObsID'].values
                    all_excl_df = fully_bl_df[fully_bl_df['ObsID'].isin(bl_obs)]

                    tel_matches[tel][s_idx] = matches_df
                    tel_bl[tel][s_idx] = all_excl_df
            else:
                # For flat results we just split into matches and blacklisted
                tel_matches.append(matched_census.loc[~matched_census['is_bl']].drop(columns=['is_bl']))

                bl_obs = matched_census.loc[matched_census['is_bl'], 'ObsID'].unique()
                if len(bl_obs) > 0:
                    # We need to map src_idx back to the blacklist entries
                    # This is slightly complex because fully_bl_df doesn't have src_idx
                    bl_matches = matched_census.loc[matched_census['is_bl'], ['src_idx', 'ObsID', 'dist', 'telescope']]
                    bl_df = fully_bl_df.merge(bl_matches, on='ObsID')
                    tel_bl.append(bl_df)

    if return_flat:
        if len(tel_matches) > 0:
            tel_matches = pd.concat(tel_matches, ignore_index=True)
        else:
            # Create an empty dataframe with correct columns if no matches
            cols = list(CENSUS[telescope[0]].columns) + ['src_idx', 'dist', 'telescope']
            tel_matches = DataFrame(columns=cols)

        if len(tel_bl) > 0:
            tel_bl = pd.concat(tel_bl, ignore_index=True)
        else:
            cols = list(BLACKLIST[telescope[0]].columns) + ['src_idx', 'dist', 'telescope']
            tel_bl = DataFrame(columns=cols)

    return tel_matches, tel_bl



def _on_obs_id(ra: float, dec: float, exp_maps: Union[ExpMap, List[ExpMap]], s_idx: int = None) \
        -> Tuple[float, float, np.ndarray, int]:
    """
    Internal function used by the on_detector_match function to check whether a passed coordinate falls directly on a
    camera for a single (or set of) ObsID(s). Checks whether exposure time is 0 at the coordinate.
    It cycles through cameras so (using XMM as an example) if exposure time is 0 on PN it'll go to MOS1, etc. to try
    and account for chip gaps in different cameras.

    :param float ra: The right-ascension of the coordinate that may fall on the ObsID.
    :param float dec: The declination of the coordinate that may fall on the ObsID.
    :param ExpMap/List[ExpMap] exp_maps: The exposure maps which we will use to check whether the RA-Dec lie on
        a detector.
    :param int s_idx: The index of the source in the input coordinate array.
    :return: The input RA, input dec, ObsID match array, and source index.
    :rtype: Tuple[float, float, np.ndarray, int]
    """

    # Makes sure that the exp_maps variable is iterable, whether there is just one ExpMap or a set, makes it easier
    #  to write just one piece of code that deals with either type of input
    if isinstance(exp_maps, str):
        obs_id = [exp_maps]

    # Possible that multiple exposure maps per ObsID-inst have been passed - we shall just make sure that we only
    #  use one per ObsID-Instrument
    exp_for_use = {}
    for ex in exp_maps:
        if ex.obs_id not in exp_for_use:
            exp_for_use[ex.obs_id] = {ex.instrument: ex}
        elif ex.obs_id in exp_for_use and ex.instrument not in exp_for_use[ex.obs_id]:
            exp_for_use[ex.obs_id][ex.instrument] = ex

    # Set up a list to store detection information
    det = []
    # We loop through the ObsID(s) - if just one was passed it'll only loop once (I made sure that ObsID was a list
    #  a few lines above this)
    for o in exp_for_use:
        # This variable describes whether the RA-Dec has a non-zero exposure for this current ObsID, starts off False
        cur_det = False
        # Loops through the instruments for the current ObsID in the exposure map dictionary we constructed
        for inst in exp_for_use[o]:
            ex = exp_for_use[o][inst]
            try:
                if ex.get_exp(Quantity([ra, dec], 'deg')) != 0:
                    cur_det = True
                    # We unload the exposure map data from memory
                    ex.unload()
                    # Break out of this loop, as we know that the coordinate falls on at least one instrument
                    #  of this observation
                    break
            except ValueError:
                pass

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

    return ra, dec, det, s_idx


def _in_region(ra: Union[float, List[float], np.ndarray], dec: Union[float, List[float], np.ndarray],
               s_indices: Union[int, List[int], np.ndarray], obs_id: str,
               telescope: str, im: 'Image', allowed_colours: List[str]) -> Tuple[str, dict]:
    """
    Internal function to search a particular ObsID's region files for matches to the sources defined in the RA
    and Dec arguments. This is achieved using the Regions module, and a region is a 'match' to a source if the
    source coordinates fall somewhere within the region, and the region is of an acceptable colour (defined in
    allowed_colours). This requires that both images and region files are properly setup in the XGA config file.

    :param float/List[float]/np.ndarray ra: The set of source RA coords to match with the obs_id's regions.
    :param float/List[float]/np.ndarray dec: The set of source DEC coords to match with the obs_id's regions.
    :param int/List[int]/np.ndarray s_indices: The set of source indices to match with the obs_id's regions.
    :param str obs_id: The ObsID whose regions we are matching to.
    :param str telescope: The telescope whose regions we're checking for a match.
    :param Image im: The image (as an XGA Image product) that goes with the regions being checked.
    :param List[str] allowed_colours: The colours of region that should be accepted as a match.
    :return: The ObsID that was being searched, and a dictionary of matched regions (the keys are the integer
        indices of the sources passed in), and the values are lists of region objects.
    :rtype: Tuple[str, dict]
    """

    if isinstance(ra, float):
        ra = [ra]
        dec = [dec]
        s_indices = [s_indices]

    # From that ObsID construct a path to the relevant region file using the XGA config
    reg_path = xga_conf["{}_FILES".format(telescope.upper())]["region_file"].format(obs_id=obs_id)

    # This dictionary stores the match regions for each coordinate
    matched = {}

    # If there is a region file to search then we can proceed
    if os.path.exists(reg_path):
        # onwards.write("None of the specified image files for {} can be located - skipping region match "
        #               "search.".format(obs_id))

        # Reading in the region file using the Regions module
        og_ds9_regs = np.array(Regions.read(reg_path, format='ds9').regions)

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
                s_idx = s_indices[r_ind]

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

                match_within = [r for r in match_within if r.visual['edgecolor'] in allowed_colours]
                if len(match_within) != 0:
                    matched[s_idx] = match_within

        im.unload()

    gc.collect()
    return obs_id, matched


def _process_flat_init_match(src_ra: np.ndarray, src_dec: np.ndarray, res_df: DataFrame):
    """
    An internal function that takes the flat results of a separation match and assembles the mapping of ObsIDs to
    source coordinates/indices.

    :param np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees.
    :param np.ndarray src_dec: Dec coordinate(s) of the source(s), in degrees.
    :param DataFrame res_df: The flat results of a separation_match run.
    :return: A dictionary of unique ObsIDs for each telescope, a list of all indices, and a dictionary that links
        source coordinates/indices to ObsIDs (ObsIDs are the keys).
    :rtype: Tuple[dict, list, dict]
    """
    telescopes = res_df['telescope'].unique().tolist()
    all_indices = list(range(len(src_ra)))

    final_obs_ids = {tel: [] for tel in telescopes}
    final_obs_id_srcs = {tel: {} for tel in telescopes}

    # Group by telescope first to simplify processing
    for tel, tel_group in res_df.groupby('telescope'):
        final_obs_ids[tel] = tel_group['ObsID'].unique().tolist()

        # We need RA, Dec, and global index for each match
        # We can get RA and Dec from the original input using src_idx
        # We add them to the group to make the dictionary construction easier
        # This is vectorized and fast
        tel_group = tel_group.copy()
        tel_group['ra'] = src_ra[tel_group['src_idx'].values]
        tel_group['dec'] = src_dec[tel_group['src_idx'].values]

        # Construct the mapping: {obs_id: [[ra, dec, global_idx], ...]}
        # We use groupby on ObsID
        obs_groups = tel_group.groupby('ObsID')
        final_obs_id_srcs[tel] = {o: group[['ra', 'dec', 'src_idx']].values
                                  for o, group in obs_groups}

    return final_obs_ids, all_indices, final_obs_id_srcs


def _process_init_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],
                        initial_results: dict):
    """
    An internal function that takes the results of a separation match and assembles the lists of unique ObsIDs for
    the requested telescopes which are of interest to the coordinate(s) we're searching for data for. Sets of RA
    and Decs that were found to be near an observation by the initial separation match are also created and returned.

    :param float/np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param float/np.ndarray src_dec: Dec coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param dict initial_results: The result of a separation_match run.
    :return: The simple match initial results (normalised so that they are a list of dataframe, even if only one
        source is being searched for), a list of  unique ObsIDs, unique string representations generated from RA and
        Dec for the positions  we're looking at, an array of dataframes for those coordinates that are near an
        observation according to the initial match, and the RA and Decs that are near an observation
        according to the initial simple match. The final output is a dictionary with ObsIDs as keys, and arrays of
        source coordinates that are an initial match with them.
    """
    # If only one coordinate was passed, the return from separation_match will be a dictionary of dataframes, and
    #  I want a list of dataframes because its iterable and easier to deal with more generally
    if isinstance(initial_results, dict):
        initial_results = [initial_results]

    # TODO Honestly I think I should just rewrite this - at the moment I just want to convert it so it acts the same
    #  as it did but for multi-mission stuff though

    # We use integer indices as identifiers instead of string representations, as it is much faster for large samples
    all_indices = list(range(len(src_ra)))

    final_obs_ids = {}
    final_res = {}
    final_ra = {}
    final_dec = {}
    final_obs_id_srcs = {}

    telescopes = list(initial_results[0].keys())

    # This is a horrible bodge. I add an empty DataFrame to the end of the list of DataFrames returned by
    #  simple_xmm_match. This is because (when I turn init_res into a numpy array), if all of the DataFrames have
    #  data (so if all the input coordinates fall near an observation) then numpy tries to be clever and just turns
    #  it into a 3D array - this throws off the rest of the code. As such I add a DataFrame at the end with no data
    #  to makes sure numpy doesn't screw me over like that.
    initial_results += [{tel:  DataFrame(columns=initial_results[0][tel].columns) for tel in telescopes}]
    for tel in telescopes:
        # This constructs a masking array that tells us which of the coordinates had any sort of return from
        #  separation_match for the current telescope - if further check is True then a coordinate fell near an
        #  observation and the exposure map should be investigated.
        further_check = np.array([len(t[tel]) > 0 for t in initial_results])

        # Incidentally we don't need to check whether these are all False, because simple_xmm_match will already have
        #  errored if that was the case
        # Now construct cut down initial matching result, ra, and dec arrays
        rel_res = np.array([ir[tel] for ir in initial_results], dtype=object)[further_check]
        final_res[tel] = rel_res

        # The further_check masking array is ignoring the last entry here because that corresponds to the
        #  empty DataFrame that I artificially added to bodge numpy slightly further up in the code
        rel_ra = np.array(src_ra)[further_check[:-1]]
        rel_dec = np.array(src_dec)[further_check[:-1]]
        final_ra[tel] = rel_ra
        final_dec[tel] = rel_dec

        # Nested list comprehension that extracts all the ObsIDs that are mentioned in the initial matching
        #  information, turning that into a set removes any duplicates, and then it gets turned back into a list.
        # This is just used to know what exposure maps need to be generated
        obs_ids = list(set([o for t in rel_res for o in t['ObsID'].values]))
        final_obs_ids[tel] = obs_ids

        # This assembles a big numpy array with every source coordinate and its ObsIDs (source coordinate will have one
        #  entry per ObsID associated with them).
        repeats = [len(cur_res) for cur_res in rel_res]

        if len(rel_res) != 0:
            full_info = np.vstack([np.concatenate([cur_res['ObsID'].to_numpy() for cur_res in rel_res]),
                                   np.repeat(rel_ra, repeats), np.repeat(rel_dec, repeats),
                                   np.repeat(np.array(all_indices)[further_check[:-1]], repeats)]).T
        else:
            full_info = np.array([])

        # This assembles a dictionary that links source coordinates to ObsIDs (ObsIDs are the keys)
        final_obs_id_srcs[tel] = {o: full_info[np.where(full_info[:, 0] == o)[0], :][:, 1:] for o in obs_ids}

    return initial_results, final_obs_ids, all_indices, final_res, final_ra, final_dec, final_obs_id_srcs


def census_match(telescope: Union[str, list] = None, obs_ids: Union[List[str], dict] = None) -> Tuple[dict, dict]:
    """
    Returns XGA census entries (with ObsID, ra, and dec) that are not completely blacklisted, for the specified
    telescope(s). This is an extremely simple function, and could be largely replicated by just working with the
    CENSUS directly - however this does check against the blacklist, and will return things in the same style as
    the 'proper' matching functions.

    The user can also pass a list of strings (or a dictionary of lists of strings in the case of multiple telescopes
    being considered) to limit the ObsIDs from the census that are to be considered.

    :param str/List[str] telescope: The telescope censuses that should be searched for matches, the default is None, in
        which case all telescopes that have been set up with this installation of XGA will be used. The user may pass
        a single telescope name, or a list of telescope names, to control which are used.
    :param List[str]/dict obs_ids: ObsIDs that are to be considered
    :return: A dictionary of dataframes of matching ObsIDs, where the dictionary keys correspond to
        different telescopes. The second return is structured exactly the same, but represents observations that were
        completely excluded in the blacklist.
    :rtype: Tuple[dict, dict]
    """

    # This function checks the choices of telescopes, raising errors if there are problems, and returning a list of
    #  validated telescope names (even if there is only one).
    telescope = check_telescope_choices(telescope)

    # Here we parse the ObsID information (that the user can give us to limit the ObsIDs that should be considered
    #  for this particular 'match'), to try to account for the different formats of information that can be passed
    if obs_ids is not None and not isinstance(obs_ids, (dict, list)):
        raise TypeError("The 'obs_ids' argument must either be None, a list of ObsIDs (for a single telescope), or a "
                        "dictionary of lists of ObsIDs (for multiple telescopes).")
    # In this case all observations are valid for all telescopes
    elif obs_ids is None:
        obs_ids = {tel: None for tel in telescope}
    # If a list of ObsIDs is passed, but multiple telescopes are under consideration, then we can't really know what
    #  to do so we have to throw an error
    elif obs_ids is not None and isinstance(obs_ids, list) and len(telescope) > 1:
        raise TypeError("It is not possible to pass a list of ObsIDs when multiple telescopes have been passed, please "
                        "pass a dictionary of lists of ObsIDs.")
    # However, if dictionary is passed for ObsIDs and doesn't relate to the telescopes we're looking at, we will
    #  be forgiving and just use all observations
    elif obs_ids is not None and isinstance(obs_ids, dict) and all([tel not in obs_ids for tel in telescope]):
        warn("None of the telescopes specified were contained in the obs_ids dictionary; defaulting to using all "
             "viable observations.", stacklevel=2)
        obs_ids = {tel: None for tel in telescope}
    # Here ObsIDs for some telescopes are passed - so for the others we'll use all observations
    elif obs_ids is not None and isinstance(obs_ids, dict) and any([tel not in obs_ids for tel in telescope]):
        obs_ids = {tel: None if tel not in obs_ids else obs_ids[tel] for tel in telescope}
    # This just makes sure that obs_ids is a dictionary regardless
    elif obs_ids is not None and isinstance(obs_ids, list) and len(telescope) == 1:
        obs_ids = {telescope[0]: obs_ids}

    # This dictionary stores any ObsIDs that were COMPLETELY blacklisted (i.e. all instruments were excluded)
    bl_results = {}
    # This is what the census entries are stored in
    results = {}
    for tel in telescope:
        rel_obs_ids = obs_ids[tel]
        # We grab the observation census for the current telescope, and drop any rows that have a NaN coordinate (this
        #  can happen for calibration pointings, and pointings where X-ray telescopes weren't used, for telescopes that
        #  have non X-ray telescopes, like the OM on XMM).
        rel_census = CENSUS[tel].dropna(subset=['RA_PNT', 'DEC_PNT']).copy()
        # To make the output entirely the same as the matches on position we add a NaN distance column
        rel_census['dist'] = np.nan
        if rel_obs_ids is not None:
            # If the ObsIDs under consideration for a particular telescope have been limited, then we include only
            #  those ObsIDs that the user wanted us to include
            rel_census = rel_census[rel_census['ObsID'].isin(rel_obs_ids)]
        rel_blacklist = BLACKLIST[tel]

        # Locate any ObsIDs that are in the blacklist, then test to see whether ALL the instruments are to be excluded
        in_bl = rel_blacklist[rel_blacklist['ObsID'].isin(rel_census[rel_census["ObsID"].isin(rel_blacklist["ObsID"]
                                                                                              )]['ObsID'])]
        # Firstly we locate the 'exclude_{INST NAME}' columns for this telescope's blacklist
        excl_col = [col for col in in_bl.columns if 'EXCLUDE' in col]
        all_excl = in_bl[np.logical_and.reduce([in_bl[excl] == True for excl in excl_col])]

        # These are the observations that have at  least some usable data.
        all_incl = rel_census[~rel_census["ObsID"].isin(all_excl["ObsID"])]

        results[tel] = all_incl
        # And we store the fully blacklisted observations in another dictionary
        bl_results[tel] = all_excl

    return results, bl_results


def _vectorized_separation_match_wrapper(args):
    """
    Wrapper for _vectorized_separation_match to be used with multiprocessing.imap.
    """
    return _vectorized_separation_match(*args)


def separation_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],
                     distance: Union[Quantity, dict] = None, telescope: Union[str, List[str]] = None,
                     num_cores: int = NUM_CORES, show_warnings: bool = True,
                     return_flat: bool = False) \
        -> Tuple[Union[dict, List[dict], pd.DataFrame], Union[dict, List[dict], pd.DataFrame]]:
    """
    Returns XGA census entries (with ObsID, ra, and dec) that match to the input coordinates (either a single
    coordinate or a set). This is done for a set of telescopes (or a single telescope), and a match is made by the
    source coordinates being within specified search distances for the different telescopes.

    :param float/np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param float/np.ndarray src_dec: DEC coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param Quantity/dict distance: The radius to search for observations within, the default is None in which case
        standard search distances for different telescopes are used. The user may pass a single Quantity to use for
        all telescopes, a dictionary with keys corresponding to ALL or SOME of the telescopes specified by the
        'telescope' argument. In the case where only SOME of the telescopes are specified in a distance dictionary,
        the default XGA values will be used for any that are missing.
    :param str/List[str] telescope: The telescope censuses that should be searched for matches, the default is None, in
        which case all telescopes that have been set up with this installation of XGA will be used. The user may pass
        a single telescope name, or a list of telescope names, to control which are used.
    :param int num_cores: The number of cores to use, default is set to 90% of system cores. This is only relevant
        if multiple coordinate pairs are passed.
    :param bool show_warnings: If False, then any warnings that occur will not be displayed. Default is True.
    :param bool return_flat: If True, then a flat DataFrame will be returned instead of a sparse dictionary or list of
        dictionaries. Default is False.
    :return: A list of dictionaries (or single dictionary for one coordinate) of dataframes of matching ObsIDs, where
        each dictionary corresponds to an input RA-Dec (in the same order), and the dictionary keys correspond to
        different telescopes. The second return is structured exactly the same, but represents observations that were
        completely excluded in the blacklist.
    :rtype: Tuple[Union[dict, List[dict], pd.DataFrame], Union[dict, List[dict], pd.DataFrame]]
    """

    # This function checks the choices of telescopes, raising errors if there are problems, and returning a list of
    #  validated telescope names (even if there is only one).
    telescope = check_telescope_choices(telescope)

    # Set up the search distance, making sure the output at the end if the same format of dictionary.
    # If the distance is not set by the user, then we have to set it ourselves using the default values for each
    #  telescope/mission
    if distance is None:
        distance = {t: DEFAULT_TELE_SEARCH_DIST[t] for t in telescope}
    # Whereas if the user has passed a dictionary of values and NONE of the keys are for the requested telescopes,
    #  then I think they're probably confused, and we throw an error
    elif type(distance) == dict and all([t not in distance for t in telescope]):
        raise KeyError("When it is a dictionary, the 'distance' argument must contain an entry for every mission "
                       "specified by 'telescope'.")
    # However if the passed dictionary contains SOME of the requested telescopes, then we fill in the rest with the
    #  default values - I think it is probably more convenient
    elif type(distance) == dict and any([t not in distance for t in telescope]):
        if show_warnings:
            warn("A dictionary of search distances that did not contain all requested telescopes has been "
                 "passed, default values have been used for the missing telescopes.", stacklevel=2)
        distance = {t: distance[t] if t in distance else DEFAULT_TELE_SEARCH_DIST[t] for t in telescope}
    elif isinstance(distance, Quantity):
        # Just make sure that distance is a dictionary whatever, to simplify the code later
        distance = {t: distance for t in telescope}

    # Here we perform a check to see whether a set of coordinates is being passed, and if so are the two
    #  arrays the same length
    if isinstance(src_ra, np.ndarray) and isinstance(src_dec, np.ndarray) and len(src_ra) != len(src_dec):
        raise ValueError("If passing multiple pairs of coordinates, src_ra and src_dec must be of the same length.")
    # Just one coordinate is also allowed, but still want it to be iterable so put it in an array
    elif isinstance(src_ra, (float, int)) and isinstance(src_dec, (float, int)):
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

    # If we only want to use one core, we don't set up a pool as it could be that a pool is open where
    #  this function is being called from
    if num_cores == 1:
        # The tqdm progress bar is handled here for the serial path
        with tqdm(desc='Searching for telescope observations near source coordinates',
                  total=1, disable=prog_dis) as onwards:
            # Serial vectorized matching
            # Format: {telescope: {global_index: DataFrame}}
            all_indices = np.arange(len(src_ra))
            sparse_results, sparse_bl = _vectorized_separation_match(src_ra, src_dec, all_indices, telescope, distance,
                                                                    return_flat=return_flat)
            onwards.update(1)
    else:
        # We pre-calculate the census coordinates for each telescope to avoid redundant indexing in workers
        pre_calc_census = {}
        for tel in telescope:
            rel_census = CENSUS[tel].dropna(subset=['RA_PNT', 'DEC_PNT'])
            if len(rel_census) > 0:
                pre_calc_census[tel] = SkyCoord(ra=rel_census['RA_PNT'].values, dec=rel_census['DEC_PNT'].values,
                                                unit='deg')

        # Determine chunk size - aiming for ~4 chunks per core
        chunk_size = max(1, len(src_ra) // (num_cores * 4))
        num_chunks = int(np.ceil(len(src_ra) / chunk_size))

        # Split inputs and indices into chunks
        ra_chunks = np.array_split(src_ra, num_chunks)
        dec_chunks = np.array_split(src_dec, num_chunks)
        idx_chunks = np.array_split(np.arange(len(src_ra)), num_chunks)

        # Prepare arguments for the map function
        map_args = [(ra_chunks[i], dec_chunks[i], idx_chunks[i], telescope, distance, None, return_flat)
                    for i in range(num_chunks)]

        # We set up the pool with an initializer to store the census coordinates in worker global scope
        if not return_flat:
            sparse_results = {tel: {} for tel in telescope}
            sparse_bl = {tel: {} for tel in telescope}
        else:
            sparse_results = []
            sparse_bl = []

        with tqdm(desc="Searching for telescope observations near source coordinates",
                  total=num_chunks) as onwards, Pool(num_cores, initializer=_worker_init,
                                                     initargs=(pre_calc_census,)) as pool:
            # imap to process chunks and preserve order while supporting a progress bar
            for chunk_res, chunk_bl in pool.imap(_vectorized_separation_match_wrapper, map_args):
                # Merge the sparse results from each chunk
                if not return_flat:
                    for tel in telescope:
                        sparse_results[tel].update(chunk_res[tel])
                        sparse_bl[tel].update(chunk_bl[tel])
                else:
                    sparse_results.append(chunk_res)
                    sparse_bl.append(chunk_bl)
                onwards.update(1)

        if return_flat:
            if len(sparse_results) > 0:
                sparse_results = pd.concat(sparse_results, ignore_index=True)
                sparse_bl = pd.concat(sparse_bl, ignore_index=True)
            else:
                # Should have been handled in _vectorized_separation_match but just in case
                cols = list(CENSUS[telescope[0]].columns) + ['src_idx', 'dist', 'telescope']
                sparse_results = DataFrame(columns=cols)
                cols = list(BLACKLIST[telescope[0]].columns) + ['src_idx', 'dist', 'telescope']
                sparse_bl = DataFrame(columns=cols)

    if return_flat:
        # We can just return the flat DataFrames now
        # But we still need to check for matches to raise the NoMatchFoundError
        if len(sparse_results) == 0:
            if len(src_ra) == 1:
                raise NoMatchFoundError("No {t} observations found within {a} of ra={r} "
                                        "dec={d}.".format(r=round(src_ra[0], 4),
                                                          d=round(src_dec[0], 4),
                                                          a='/'.join([str(distance[t].to('deg')) for t in telescope]),
                                                          t='/'.join(telescope)))
            else:
                raise NoMatchFoundError("No {t} observation found within {a} of any input coordinate "
                                        "pairs.".format(a='/'.join([str(distance[t].to('deg')) for t in telescope]),
                                                        t='/'.join(telescope)))
        return sparse_results, sparse_bl

    # --- Assembly Pass ---

    # We now assemble the final result list of dictionaries from the sparse maps.
    # We use a single empty template for sources with no matches to save time/memory.
    empty_res = {tel: DataFrame(columns=list(CENSUS[tel].columns) + ['dist']) for tel in telescope}
    empty_bl = {tel: DataFrame(columns=BLACKLIST[tel].columns) for tel in telescope}

    # Final lists to return
    results = []
    bl_results = []

    for i in range(len(src_ra)):
        # Check if this source had any matches for any telescope
        has_matches = any(i in sparse_results[tel] for tel in telescope)
        if not has_matches:
            results.append(empty_res)
            bl_results.append(empty_bl)
        else:
            # Build the result dict for this specific source
            src_res = {}
            src_bl = {}
            for tel in telescope:
                src_res[tel] = sparse_results[tel].get(i, empty_res[tel])
                src_bl[tel] = sparse_bl[tel].get(i, empty_bl[tel])
            results.append(src_res)
            bl_results.append(src_bl)

    # Result length of one means one coordinate was passed in, so we should pass back out a single dataframe
    #  rather than a single dataframe in a list
    if len(results) == 1:
        results = results[0]
        bl_results = bl_results[0]

        # This calculates the total number of results across all requested telescopes
        tot_res = sum([len(results[tel]) for tel in results])

        # As in this case there is only one source, we can include more information in our error message telling
        #  the user that there are no matches
        if tot_res == 0:
            raise NoMatchFoundError("No {t} observations found within {a} of ra={r} "
                                    "dec={d}.".format(r=round(src_ra[0], 4),
                                                      d=round(src_dec[0], 4),
                                                      a='/'.join([str(distance[t].to('deg')) for t in telescope]),
                                                      t='/'.join(telescope)))
    # If all the dataframes in the results list are length zero, then none of the coordinates has a
    #  valid ObsID
    elif all([sum([len(r[tel]) for tel in r]) == 0 for r in results]):
        raise NoMatchFoundError("No {t} observation found within {a} of any input coordinate "
                                "pairs.".format(a='/'.join([str(distance[t].to('deg')) for t in telescope]),
                                                t='/'.join(telescope)))

    return results, bl_results


def _vectorized_on_detector_match(src_ra: np.ndarray, src_dec: np.ndarray,
                                 telescope: List[str],
                                 obs_id_chunks: dict,
                                 obs_to_src_indices: dict,
                                 exp_map_dict: dict = None) -> list:
    """
    Internal function that performs vectorized detector matching for a set of ObsIDs.
    Used by both the serial and parallel paths of on_detector_match.

    :param np.ndarray src_ra: FULL RA coordinate array of the entire source sample.
    :param np.ndarray src_dec: FULL Dec coordinate array of the entire source sample.
    :param List[str] telescope: List of telescopes to search.
    :param dict obs_id_chunks: The ObsIDs to be processed by this worker {tel: [obs_ids]}.
    :param dict obs_to_src_indices: Mapping of {tel: {obs_id: [source_indices]}} for all sources near data.
    :param dict exp_map_dict: Dictionary of pre-loaded (but unloaded data) ExpMap objects {tel: {obs_id: [ExpMaps]}}.
        If None, then worker processes will check the global WORKER_EXP_MAPS.
    :return: A list of matches. Format: [(source_index, telescope, obsid), ...]
    :rtype: list
    """
    # Flat results: [(source_index, telescope, obs_id), ...]
    matches = []

    # If exp_map_dict is None, we use the global variable
    if exp_map_dict is None:
        exp_map_dict = WORKER_EXP_MAPS

    # We iterate through the telescopes and ObsIDs assigned to this worker
    for tel in obs_id_chunks:
        for o in obs_id_chunks[tel]:
            # Get the indices of sources that separation_match said were near this ObsID
            if tel not in obs_to_src_indices or o not in obs_to_src_indices[tel]:
                continue

            # This is now a list of [ra, dec, global_idx]
            info = np.array(obs_to_src_indices[tel][o])
            if len(info) == 0:
                continue

            src_indices = info[:, 2].astype(int)

            # Get exposure maps for this ObsID (passed in from the main process)
            if tel not in exp_map_dict or o not in exp_map_dict[tel]:
                continue
            e_maps = exp_map_dict[tel][o]

            # Coordinates to check for this ObsID
            ra_to_check = src_ra[src_indices]
            dec_to_check = src_dec[src_indices]
            coords = Quantity(np.stack([ra_to_check, dec_to_check], axis=1), 'deg')

            # We use a set to keep track of which sources in the sample fall on the detector for this ObsID
            detected_indices = set()
            for ex in e_maps:
                # We convert to pixel coordinates, but ignore any that fall outside the image
                # This avoids the ValueError mentioned by the user
                # We use atleast_2d to ensure that we can index it consistently even for a single source
                pix_coords = np.atleast_2d(ex.coord_conv(coords, 'pix', ignore_bad_pix_coord=True).value)

                # We manually check for valid pixel coordinates
                # pix_coords is (N, 2) array where [:, 0] is x and [:, 1] is y
                x = pix_coords[:, 0]
                y = pix_coords[:, 1]
                valid_mask = (x >= 0) & (y >= 0) & (x < ex.shape[1]) & (y < ex.shape[0])

                # Only check non-zero exposure for valid coordinates
                if np.any(valid_mask):
                    v_idx = np.where(valid_mask)[0]
                    valid_x = x[v_idx].astype(int)
                    valid_y = y[v_idx].astype(int)

                    # Query the exposure map data directly - this triggers the lazy load if not already in memory
                    ex_times = ex.data[valid_y, valid_x]
                    on_det = np.where(ex_times != 0)[0]
                    for idx in on_det:
                        detected_indices.add(src_indices[v_idx[idx]])

                # We unload the data to keep worker memory usage low
                ex.unload()

            # Record the detections
            for s_idx in detected_indices:
                matches.append((s_idx, tel, o))

    return matches



def _vectorized_on_detector_match_wrapper(args):
    """
    Wrapper for _vectorized_on_detector_match to be used with multiprocessing.imap.
    """
    return _vectorized_on_detector_match(*args)


def on_detector_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],
                      distance: Union[Quantity, dict] = None, telescope: Union[str, List[str]] = None,
                      num_cores: int = NUM_CORES) -> Union[dict, np.ndarray]:
    """
    A matching function that checks whether supplied coordinates lie on a detector by using exposure maps to determine
    the exposure time at the supplied coordinates. Of course, this means that we need an idea of which observations
    should be checked, so we first run the 'separation_match' function.

    :param float/np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param float/np.ndarray src_dec: Dec coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param Quantity/dict distance: As this function calls 'separation_match', we have to supply the distance to
        search for observations within, the default is None in which case standard search distances for different
        telescopes are used. The user may pass a single Quantity to use for all telescopes, a dictionary with keys
        corresponding to ALL or SOME of the telescopes specified by the 'telescope' argument. In the case where
        only SOME of the telescopes are specified in a distance dictionary, the default XGA values will be used for
        any that are missing.
    :param str/List[str] telescope: The telescope censuses that should be searched for matches, the default is None, in
        which case all telescopes that have been set up with this installation of XGA will be used. The user may pass
        a single telescope name, or a list of telescope names, to control which are used.
    :param int num_cores: The number of cores to use, default is set to 90% of system cores. This is only relevant
        if multiple coordinate pairs are passed.
    :return: For a single input coordinate, a dictionary (with telescope names as keys) of numpy arrays of ObsID(s)
        will be returned. For multiple input coordinates an array of dictionaries (with telescope names as keys) of
        arrays of ObsID(s) and None values will be returned. Each entry corresponds to the input coordinate
        array, a None value indicates that the coordinate did not fall on an telescope observation at all.
    :rtype: Union[dict, np.ndarray]
    """
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

    # This function checks the choices of telescopes, raising errors if there are problems, and returning a list of
    #  validated telescope names (even if there is only one).
    telescope = check_telescope_choices(telescope)

    # This is the initial call to the separation_match function. This gives us knowledge of which coordinates are
    #  worth checking further, and which ObsIDs should be checked for those coordinates.
    init_res, init_bl = separation_match(src_ra, src_dec, distance, telescope, num_cores=num_cores, return_flat=True)

    # Use the process function to get unique ObsIDs across the entire sample
    # This identifies what products need to be generated/checked
    full_obs_ids, all_repr, obs_to_src_indices = _process_flat_init_match(src_ra, src_dec, init_res)

    # Boohoo local imports very sad very sad, but stops circular import errors. NullSource is a basic Source class
    #  that allows for a list of ObsIDs to be passed rather than coordinates
    from ..sources import NullSource

    # Declaring the SINGLE NullSource with all the ObsIDs required for the whole sample
    obs_src = NullSource(full_obs_ids, list(full_obs_ids.keys()), True, True)

    # We run exposure map generation for the ObsIDs - this ensures they are all on disk before workers start
    #  We use a unified multi-telescope function for this.
    from ..generate.multitelescope.phot import all_telescope_expmaps
    all_telescope_expmaps(obs_src, num_cores=num_cores, telescope=telescope)

    # We retrieve all ExpMap objects and organize them into a lookup dictionary
    # format: {tel: {obs_id: [ExpMaps]}}
    # We retrieve them in bulk per telescope to avoid many get_products calls
    exp_map_dict = {tel: {} for tel in obs_src.telescopes}
    for tel in obs_src.telescopes:
        all_exps = obs_src.get_products('expmap', telescope=tel, just_obj=False)
        for out in all_exps:
            # out is [telescope, obs_id, inst, info_key, obj]
            o = out[1]
            if o not in exp_map_dict[tel]:
                exp_map_dict[tel][o] = []
            exp_map_dict[tel][o].append(out[-1])

    # Get a flat list of (tel, obs_id) pairs to parallelize over
    all_obs_pairs = [(tel, o) for tel in obs_src.telescopes for o in obs_src.obs_ids[tel]]

    if num_cores == 1:
        with tqdm(desc='Ensuring coordinates fall on detector', total=1, disable=prog_dis) as onwards:
            # Serial vectorized matching on all ObsIDs at once
            # We wrap the pairs in a dict format the function expects
            obs_id_chunks = {tel: obs_src.obs_ids[tel] for tel in obs_src.telescopes}
            all_matches = _vectorized_on_detector_match(src_ra, src_dec, telescope,
                                                        obs_id_chunks, obs_to_src_indices, exp_map_dict)
            onwards.update(1)
    else:
        # Determine chunk size - aiming for ~4 chunks per core
        num_obs = len(all_obs_pairs)
        chunk_size = max(1, num_obs // (num_cores * 4))

        # Split the pairs into chunks
        pair_chunks = [all_obs_pairs[i:i + chunk_size] for i in range(0, num_obs, chunk_size)]

        # Prepare arguments for the map function
        # We pass None for exp_map_dict because they are now stored in worker global scope
        map_args = []
        for chunk in pair_chunks:
            # Group pairs back into {tel: [obs_ids]} for the worker
            chunk_obs_id_dict = {}
            for tel, o in chunk:
                if tel not in chunk_obs_id_dict:
                    chunk_obs_id_dict[tel] = []
                chunk_obs_id_dict[tel].append(o)

            map_args.append((src_ra, src_dec, telescope, chunk_obs_id_dict, obs_to_src_indices, None))

        # We pre-calculate the census coordinates for each telescope to avoid redundant indexing in workers
        pre_calc_census = {}
        for tel in telescope:
            rel_census = CENSUS[tel].dropna(subset=['RA_PNT', 'DEC_PNT'])
            if len(rel_census) > 0:
                pre_calc_census[tel] = SkyCoord(ra=rel_census['RA_PNT'].values, dec=rel_census['DEC_PNT'].values,
                                                unit='deg')

        all_matches = []
        with tqdm(desc="Ensuring coordinates fall on detector",
                  total=len(map_args), disable=prog_dis) as onwards, Pool(num_cores, initializer=_worker_init,
                                                                         initargs=(pre_calc_census,
                                                                                   exp_map_dict)) as pool:
            for chunk_matches in pool.imap(_vectorized_on_detector_match_wrapper, map_args):
                # Merge flat results [(src_idx, tel, obs_id), ...]
                all_matches.extend(chunk_matches)
                onwards.update(1)

    # --- Assembly Pass ---
    # We assemble the final results list from the flat matches
    results = [{t: None for t in telescope} for _ in range(len(src_ra))]

    # We use a temporary dictionary to store lists of ObsIDs for each source/tel
    temp_matches = {}
    for s_idx, tel, o in all_matches:
        if (s_idx, tel) not in temp_matches:
            temp_matches[(s_idx, tel)] = [o]
        else:
            temp_matches[(s_idx, tel)].append(o)

    for (s_idx, tel), obs_list in temp_matches.items():
        results[s_idx][tel] = np.array(obs_list)

    # Result length of one means one coordinate was passed in, so we should pass back out a single dictionary
    #  rather than a single dictionary in a list
    if len(src_ra) == 1:
        results = results[0]

        # This calculates the total number of results across all requested telescopes
        # We need to make sure we check if results[tel] is not None
        tot_res = 0
        for tel in results:
             if results[tel] is not None:
                  tot_res += len(results[tel])

        # As in this case there is only one source, we can include more information in our error message telling
        #  the user that there are no matches
        if tot_res == 0:
            raise NoMatchFoundError("The coordinates ra={r} dec={d} do not fall on any {t} "
                                    "observations.".format(r=round(src_ra[0], 4), d=round(src_dec[0], 4),
                                                           t='/'.join(telescope)))
    # If all the entries in the results list are None for all telescopes
    else:
        # Check if any source has any detections
        any_detections = False
        for r in results:
            for tel in telescope:
                if r[tel] is not None and len(r[tel]) > 0:
                    any_detections = True
                    break
            if any_detections:
                break

        if not any_detections:
            raise NoMatchFoundError("No coordinate pairs fall on any {t} observations.".format(t='/'.join(telescope)))

    # If it was an array input, return as array
    if not isinstance(results, dict):
         results = np.array(results)

    return results


def region_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],  src_type: Union[str, List[str]],
                 distance: Union[Quantity, dict] = None, telescope: Union[str, list] = None,
                 num_cores: int = NUM_CORES) -> np.ndarray:
    """
    A function which, if XGA has been configured with access to pre-generated region files, will search for region
    matches for a set of source coordinates passed in by the user. A region match is defined as when a source
    coordinate falls within a source region with a particular colour (largely used to represent point vs
    extended) - the type of region that should be matched to can be defined using the src_type argument.

    The separation_match function will be run before the source matching process, to narrow down the sources which
    need to have the more expensive region matching performed, as well as to identify which ObsID(s) should be
    examined for each source.

    :param float/np.ndarray src_ra: RA coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param float/np.ndarray src_dec: Dec coordinate(s) of the source(s), in degrees. To find matches for multiple
        coordinate pairs, pass an array.
    :param str/List[str] src_type: The type(s) of region that should be matched to. Pass either 'ext' or 'pnt' or
        a list containing both.
    :param Quantity/dict distance: The distance to search for observations within, the default is None in which case
        standard search distances for different telescopes are used. The user may pass a single Quantity to use for
        all telescopes, a dictionary with keys corresponding to ALL or SOME of the telescopes specified by the
        'telescope' argument. In the case where only SOME of the telescopes are specified in a distance dictionary,
        the default XGA values will be used for any that are missing.
    :param str/List[str] telescope: The telescope censuses that should be searched for matches, the default is None, in
        which case all telescopes that have been set up with this installation of XGA will be used. The user may pass
        a single telescope name, or a list of telescope names, to control which are used.
    :param int num_cores: The number of cores that can be used for the matching process.
    :return: An array the same length as the sets of input coordinates (ordering is the same). If there are no
        matches for a source then the element will be None, if there are matches then the element will be a
        dictionary, with top key(s) being telescope names, lower level keys being ObsID(s), and the values being a
        list of region objects (or more likely just one object).
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

    # We ensure that the RA and Decs are in arrays, even if there is only one coordinate - also in the case of a
    #  single coordinate we set the number of cores to one
    if type(src_ra) != type(src_dec):
        raise TypeError("'src_ra' and 'src_dec' arguments must be the same type; either floats or arrays.")
    elif isinstance(src_ra, float):
        src_ra = np.array([src_ra])
        src_dec = np.array([src_dec])
        num_cores = 1
    elif isinstance(src_ra, list):
        src_ra = np.array(src_ra)
        src_dec = np.array(src_dec)

    # This runs the simple xmm match and gathers the results.
    s_match, s_match_bl = separation_match(src_ra, src_dec, distance, telescope, num_cores=num_cores, return_flat=True)
    # The initial results are then processed into some more useful formats.
    uniq_obs_ids, all_repr, obs_id_srcs = _process_flat_init_match(src_ra, src_dec, s_match)

    # Boohoo local imports very sad very sad, but stops circular import errors. NullSource is a basic Source class
    #  that allows for a list of ObsIDs to be passed rather than coordinates
    from ..sources import NullSource

    # Declaring the NullSource with all the ObsIDs
    obs_src = NullSource(uniq_obs_ids, list(uniq_obs_ids.keys()), True, False)

    # Think I have to iterate through the telescopes here, as for each one I'll need to ensure that there is
    #  region file information available in the configuration section
    tel_reg_avail = []
    for tel in obs_src.telescopes:
        rel_sec = "{t}_FILES".format(t=tel.upper())

        default_reg = "/this/is/optional/{t}_obs/regions/{obs_id}/regions.reg".format(t=tel, obs_id='{obs_id}')
        # Checks to make sure that the user has actually pointed XGA at a set of region files (and images they were
        #  generated from, in case said region files are in pixel coordinates).
        if xga_conf[rel_sec]["region_file"] == default_reg:
            tel_reg_avail.append(False)
        else:
            tel_reg_avail.append(True)

    if not any(tel_reg_avail):
        raise NoRegionsError("The configuration file does not contain information on region files for any relevant "
                             "telescope, so this function cannot continue.")

    # This is the dictionary in which matching information is stored
    # Top level keys are now source indices
    reg_match_info = {idx: {} for idx in all_repr}
    # If the user only wants us to use one core, then we don't make a Pool because that would just add overhead
    if num_cores == 1:
        for tel in obs_src.telescopes:
            with tqdm(desc="Searching for {t} ObsID region matches".format(t=PRETTY_TELESCOPE_NAMES[tel]),
                      total=len(uniq_obs_ids[tel])) as onwards:
                # Here we iterate through the ObsIDs that the initial match found to possibly have sources on - I
                #  considered this more efficient than iterating through the sources and possibly reading in WCS
                #  information for the same ObsID in many different processes (the non-parallelised version just calls
                #  the same internal function so its setup the same).
                for cur_obs_id in obs_id_srcs[tel]:
                    if cur_obs_id not in obs_src.obs_ids[tel]:
                        onwards.update(1)
                        continue

                    cur_ra_arr = obs_id_srcs[tel][cur_obs_id][:, 0]
                    cur_dec_arr = obs_id_srcs[tel][cur_obs_id][:, 1]
                    cur_idx_arr = obs_id_srcs[tel][cur_obs_id][:, 2].astype(int)

                    try:
                        rel_im = obs_src.get_images(cur_obs_id, telescope=tel)[0]
                    except NoProductAvailableError:
                        warn("No pre-existing image can be found for {t}-{o}; this is required for checking if regions "
                             "and coordinates intersect.".format(t=tel, o=cur_obs_id), stacklevel=2)
                        onwards.update(1)
                        continue

                    # Runs the matching function
                    match_inf = _in_region(cur_ra_arr, cur_dec_arr, cur_idx_arr, cur_obs_id, tel, rel_im, allowed_colours)
                    # Adds to the match storage dictionary, but so that the top keys are source representations, and
                    #  the lower level keys are ObsIDs
                    for s_idx in match_inf[1]:
                        if tel not in reg_match_info[s_idx]:
                            reg_match_info[s_idx][tel] = {match_inf[0]: match_inf[1][s_idx]}
                        else:
                            reg_match_info[s_idx][tel][match_inf[0]] = match_inf[1][s_idx]
                    onwards.update(1)

    else:
        for tel in obs_src.telescopes:
            # This is to store exceptions that are raised in separate processes, so they can all be raised at the end.
            search_errors = []
            # We setup a Pool with the number of cores the user specified (or the default).
            with tqdm(desc="Searching for {t} ObsID region matches".format(t=PRETTY_TELESCOPE_NAMES[tel]),
                      total=len(uniq_obs_ids[tel])) as onwards, Pool(num_cores) as pool:
                # This is called when a match process finished successfully, and the results need storing
                def match_loop_callback(match_info):
                    nonlocal onwards  # The progress bar will need updating
                    nonlocal reg_match_info
                    # Adds to the match storage dictionary, but so that the top keys are source indices, and
                    #  the lower level keys are ObsIDs
                    for s_idx in match_info[1]:
                        if s_idx not in reg_match_info:
                             reg_match_info[s_idx] = {}

                        if tel not in reg_match_info[s_idx]:
                            reg_match_info[s_idx][tel] = {match_info[0]: match_info[1][s_idx]}
                        else:
                            reg_match_info[s_idx][tel][match_info[0]] = match_info[1][s_idx]

                    onwards.update(1)

                # This is called when a process errors out.
                def error_callback(err):
                    nonlocal onwards
                    nonlocal search_errors
                    # Stores the exception object in a list for later.
                    search_errors.append(err)
                    onwards.update(1)

                for cur_obs_id in obs_id_srcs[tel]:
                    if cur_obs_id not in obs_src.obs_ids[tel]:
                        onwards.update(1)
                        continue

                    # Here we iterate through the ObsIDs that the initial match found to possibly have sources on - I
                    #  considered this more efficient than iterating through the sources and possibly reading in WCS
                    #  information for the same ObsID in many different processes.
                    cur_ra_arr = obs_id_srcs[tel][cur_obs_id][:, 0]
                    cur_dec_arr = obs_id_srcs[tel][cur_obs_id][:, 1]
                    cur_idx_arr = obs_id_srcs[tel][cur_obs_id][:, 2].astype(int)

                    try:
                        rel_im = obs_src.get_images(cur_obs_id, telescope=tel)[0]
                    except NoProductAvailableError:
                        warn("No pre-existing image can be found for {t}-{o}; this is required for checking if regions "
                             "and coordinates intersect.".format(t=tel, o=cur_obs_id), stacklevel=2)
                        onwards.update(1)
                        continue

                    # Runs the matching function
                    # match_inf = _in_region(cur_ra_arr, cur_dec_arr, cur_idx_arr, cur_obs_id, tel, rel_im, allowed_colours)
                    pool.apply_async(_in_region, args=(cur_ra_arr, cur_dec_arr, cur_idx_arr, cur_obs_id,  tel, rel_im,
                                                       allowed_colours),
                                     callback=match_loop_callback, error_callback=error_callback)

                pool.close()  # No more tasks can be added to the pool
                pool.join()  # Joins the pool, the code will only move on once the pool is empty.

            # If any errors occurred during the matching process, they are all raised here as a grouped exception
            if len(search_errors) != 0:
                raise ExceptionGroup("The following exceptions were raised in the multi-threaded region finder",
                                     search_errors)

    # This formats the match and no-match information so that the output is the same length and order as the input
    #  source lists
    to_return = []
    for s_idx in all_repr: # all_repr is now range(len(src_ra))
        to_ret_en = {}
        if s_idx in reg_match_info:
            for tel in reg_match_info[s_idx]:
                if len(reg_match_info[s_idx][tel]) != 0:
                    to_ret_en[tel] = reg_match_info[s_idx][tel]

        if len(to_ret_en) != 0:
            to_return.append(to_ret_en)
        else:
            to_return.append(None)

    # Makes it into an array rather than a list
    to_return = np.array(to_return)

    return to_return


def simple_xmm_match(src_ra: Union[float, np.ndarray], src_dec: Union[float, np.ndarray],
                     distance: Quantity = Quantity(30.0, 'arcmin'), num_cores: int = NUM_CORES) \
        -> Tuple[Union[DataFrame, List[DataFrame]], Union[DataFrame, List[DataFrame]]]:
    """
    Returns XMM ObsIDs within a given distance from the input ra and dec values.

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
    warn("The XGA 'simple_xmm_match' function is now a wrapper for the more general 'separation_match' "
         "function, which can search multiple telescopes, and will be removed soon.", stacklevel=2)
    # This is now a wrapper for the new separation_match function, because 'simple_xmm_match' is the equivalent
    #  from when XGA only supported XMM - this function will eventually be removed.
    res, bl_res = separation_match(src_ra, src_dec, distance, 'xmm', num_cores)
    # Two possible outputs, depending on whether there is a single input coordinate or multiple - the return needs
    #  to be exactly as it would have been from the original, so we make sure there are no dictionaries
    if isinstance(res, list):
        res = [r['xmm'] for r in res]
        bl_res = [br['xmm'] for br in bl_res]
    else:
        res = res['xmm']
        bl_res = bl_res['xmm']

    return res, bl_res
