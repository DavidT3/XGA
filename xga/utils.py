#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 09/09/2025, 21:52. Copyright (c) The Contributors

import json
import os
import re
import shutil
from configparser import ConfigParser
from functools import wraps
from subprocess import Popen, PIPE
from typing import Tuple, List, Union
from warnings import warn, simplefilter

import importlib_resources
import numpy as np
import pandas as pd
from astropy.constants import m_p, m_e
from astropy.cosmology import LambdaCDM
from astropy.units import Quantity, def_unit, add_enabled_units
from astropy.wcs import WCS
from fitsio import FITSHDR
from fitsio import read_header
from tqdm import tqdm

from .exceptions import XGAConfigError, InvalidTelescopeError, NoTelescopeDataError

# This warning filter enables the DeprecationWarning which is in the _deprecated decorator
simplefilter('default')


def _deprecated(message):
    """
    An internal function designed to be used as a decorator for any methods or functions which have been
    deprecated - means that the warning will be shown when they're imported or used.

    :param str message: The warning message which should be shown.
    """
    # Shows the warning message - this makes sure it is shown on import as well as when the function is used.
    warn(message, DeprecationWarning)
    def deprecated_function(dep_func):
        """
        This ensures that the decorated, deprecated, function is actually still run.
        :param dep_func: The method which is deprecated.
        :return: Returns the wrapped function.
        """
        # The wraps decorator updates the wrapper function to look like wrapped function by copying attributes
        #  such as __name__, __doc__ (the docstring)
        @wraps(dep_func)
        def wrapper(*args, **kwargs):
            return dep_func(*args, **kwargs)
        return wrapper
    return deprecated_function


# ------------- Defining functions useful in the rest of the setup process -------------
def obs_id_test(test_tele: str, test_string: str) -> bool:
    """
    This function uses regular expressions for the structure of different telescope's ObsIDs to check that a
    given string conforms with the structure expected for an ObsID.

    :param str test_tele: The telescope for the ObsID we wish to check, different missions have different
        ObsID structures.
    :param str test_string: The string we wish to test.
    :return: Whether the string an ObsID from the specified telescope or not.
    :rtype: bool
    """
    # Just in case an integer ObsID is passed, we'll try to catch it here
    if not isinstance(test_string, str):
        test_string = str(test_string)

    # This uses a dictionary of ObsID regex patterns defined in the telescope data section at the top of this
    #  file to check that the given test string is a match to the structure defined by the regex for this telescope
    return bool(re.match(OBS_ID_REGEX[test_tele], test_string))


def build_observation_census(tel: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A function that builds/updates the census and blacklist for each telescope.

    :param str tel: The name of the telescope we are setting up a census/blacklist for.
    :return: The census and blacklist dataframes for the input telescope.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """

    # The census_dir is the directory containing the blacklist and census for each telescope
    census_dir = os.path.join(CONFIG_PATH, tel) + '/'
    # If it doesn't exist yet we create the directory
    if not os.path.exists(census_dir):
        os.makedirs(census_dir)

    # This variable stores the path to the blacklist file
    blacklist_file = os.path.join(CONFIG_PATH, tel, "{}_blacklist.csv".format(tel))
    # This extracts the relevant list of the instruments for this telescope
    rel_insts = ALLOWED_INST[tel]
    # Creates blacklist file if one doesn't exist, then reads it in
    if not os.path.exists(blacklist_file):
        inst_list = ["EXCLUDE_{}".format(inst.upper()) for inst in rel_insts]
        blacklist = pd.DataFrame(columns=["ObsID"] + inst_list)
        blacklist.to_csv(blacklist_file, index=False)
    # If a blacklist already exists then of course we don't need to make one, just read it in
    else:
        blacklist = pd.read_csv(blacklist_file, header="infer", dtype=str)

    # This part here is to support blacklists used by older versions of XGA, where only a full ObsID was excluded.
    #  Now we support individual instruments of ObsIDs being excluded from use, so there are extra columns expected.
    # THIS WON'T CAUSE ANY PROBLEMS WITH THE MULTI-TELESCOPE XGA BECAUSE ANY BLACKLIST WITH ONLY ONE COLUMN *MUST*
    #  BELONG TO XMM, AS IT PRE-DATED OUR ADDING SUPPORT FOR MULTIPLE TELESCOPES
    if len(blacklist.columns) == 1:
        # Adds the four new columns, all with a default value of True. So any ObsID already in the blacklist
        #  will have the same behaviour as before, all instruments for the ObsID are excluded
        blacklist_columns = ["EXCLUDE_{}".format(inst.upper()) for inst in rel_insts]
        blacklist[blacklist_columns] = 'T'
        # If we have even gotten to this stage then the actual blacklist file needs re-writing, so I do
        blacklist.to_csv(blacklist_file, index=False)

    # This variable stores the path to the census file for this telescope
    census_file = os.path.join(CONFIG_PATH, tel, "{}_census.csv".format(tel))
    # If CENSUS FILE exists, it is read in, otherwise an empty dataframe is initialised
    if os.path.exists(census_file):
        obs_lookup = pd.read_csv(census_file, header="infer", dtype=str)
    else:
        # Making the columns in the census
        inst_list = ["USE_{}".format(inst.upper()) for inst in rel_insts]
        obs_lookup = pd.DataFrame(columns=["ObsID", "RA_PNT", "DEC_PNT"] + inst_list)

    # Need to find out which observations are available - this lists every file/directory in the root data directory
    #  for this telescope and a) checks that the entry is a directory, and b) checks that the name of the directory
    #  matches the pattern expected for an ObsID of this telescope. We also check that the ObsID isn't already in the
    #  census, to avoid duplicates

    rel_root_dir = xga_conf[tel.upper() + '_FILES']['root_{t}_dir'.format(t=tel)]
    new_obs_census = [poss_oi for poss_oi in os.listdir(rel_root_dir) if os.path.isdir(rel_root_dir + poss_oi) and
                      obs_id_test(tel, poss_oi) and poss_oi not in obs_lookup['ObsID'].values]

    if len(new_obs_census) != 0:
        # This just finds the configuration entries that are relevant to specifying where cleaned events live for
        #  the current instrument - I could have just set up a dictionary like Jess originally did, but I'm trying
        #  to be clever and dynamically support new telescopes without changes in much of this file - all the dev
        #  should need to do is add new entries in some dictionaries near the top
        evt_path_keys = [e_key for e_key in xga_conf[tel.upper() + '_FILES'] if 'evts' in e_key and 'clean' in e_key]
        # If new telescope sections have been implemented like I specified they should be, the structure of these keys
        #  should be predictable - whatever is being counted as an 'instrument' should be in the middle. I put
        #  'instrument' in quotes because it is possible (as it may be with eROSITA) that the different instruments
        #  are all contained in the same event list
        evt_path_insts = [e_key.split('_')[1] for e_key in evt_path_keys]

        # If the number of instruments specified in the configuration file headers doesn't match the number of
        #  different instruments in the ALLOWED_INST dictionary entry for this telescope, then we can infer that
        #  the event list headers should be used to specify the instruments - as either one ObsID always has only
        #  one instrument associated (as with Chandra for instance), or there aren't separate event lists for
        #  separate instruments (as may be the case for eROSITA).
        inst_from_evt = False if len(evt_path_insts) == len(ALLOWED_INST[tel]) else True

        # Essentially what we want to learn here, and store in the census, are the pointing coordinates of the
        #  telescope and whether each instrument can be used for science (for instance do they have a filter we don't
        #  allow, in which case the entry will be 'False')
        with tqdm(desc="Assembling list of {} ObsIDs".format(tel), total=len(new_obs_census)) as census_progress:

            new_census_rows = []
            for obs in new_obs_census:
                # Set up the data that will be added to the census for the current observation, dictionary form
                #  seemed easiest to begin with
                info = {col: '' for col in obs_lookup.columns}
                info['ObsID'] = obs

                # Iterating through the identified event list keys in the config for the current telescope
                for evt_key_ind, evt_key in enumerate(evt_path_keys):
                    evt_path = xga_conf[tel.upper() + '_FILES'][evt_key].format(obs_id=obs)

                    if os.path.exists(evt_path):
                        # Just read in the header of the events file - want to avoid reading a big old table of
                        #  events into memory, as we might be doing this a bunch of times
                        evts_header = read_header(evt_path, ext="EVENTS")

                        # For the eRASS fields it seems that RA_CEN and DEC_CEN are the best ways of defining where
                        #  the data is located on the sky. Non-survey modes however should use the RA_PNT and DEC_PNT
                        #  headers, as RA_CEN etc. are 0 (conversely RA_PNT etc. are 0 for eRASS).
                        if tel == 'erosita' and evts_header['OBS_MODE'] == 'SURVEY':
                            if evts_header['RA_CEN'] == 0.0 and evts_header['RA_OBJ'] != 0.0:
                                # THIS is a failure mode of some processed eRASS event lists (no idea why it happens)
                                #  where the central coordinate info gets split across the *_CEN, which is where it
                                #  should be for survey mode, and the *_OBJ header entries. See issue #1158
                                info['RA_PNT'] = evts_header['RA_OBJ']
                            else:
                                info['RA_PNT'] = evts_header["RA_CEN"]

                            if evts_header['DEC_CEN'] == 0.0 and evts_header['DEC_OBJ'] != 0.0:
                                # THIS is a failure mode of some processed eRASS event lists (no idea why it happens)
                                #  where the central coordinate info gets split across the *_CEN, which is where it
                                #  should be for survey mode, and the *_OBJ header entries. See issue #1158
                                info['DEC_PNT'] = evts_header['DEC_OBJ']
                            else:
                                info['DEC_PNT'] = evts_header["DEC_CEN"]

                        else:
                            # I think this *should* be a fairly universal way of accessing the pointing
                            #  coordinates, but I guess I'll find out if it isn't when I add more telescopes! For
                            #  cases with multiple instruments this is going to overwrite the pointing coordinates
                            #  each time, but as of now I am assuming they are co-aligned (or co-aligned enough for
                            #  searching for observations). If that is not the case for a telescope I support in the
                            #  future then I'll have to change this
                            info['RA_PNT'] = evts_header["RA_PNT"]
                            info['DEC_PNT'] = evts_header["DEC_PNT"]

                        # We check that the filter value isn't in the list of unacceptable filters for the
                        #  current telescope.
                        #  We do this because not all telescopes have a filter header to check.
                        if 'FILTER' in evts_header:
                            good_filt = evts_header['FILTER'] not in BANNED_FILTS[tel]
                        else:
                            good_filt = True

                        # If we determined further up in this process that the current telescope's event lists are
                        #  actually combined from multiple instruments, and we need to determine which of the
                        #  instruments are present from the event lists, then we do that here
                        if inst_from_evt:
                            # What it says on the tin really, just a hopefully useful warning
                            if tel != 'erosita':
                                warn("There may be unintended behaviours, as the current section was designed with"
                                     " eROSITA in mind - contact the developers (though I'd be surprised if anyone"
                                     " who isn't a developer sees this...", stacklevel=2)

                            # Use INSTRUM and not INSTRUME as the search here because it is what finds you the
                            #  instruments in eROSITA evts lists, and honestly at the moment they're the only ones
                            #  I think are going to be structured like this (unless we change DAXA to break them up).
                            hdr_insts = [evts_header[h_key] for h_key in list(evts_header.keys())
                                         if 'INSTRUM' in h_key and 'INSTRUME' not in h_key]

                            # Now we put our newly gained knowledge of which instruments were turned on in the
                            #  info dictionary that is being assembled
                            for i in ALLOWED_INST[tel]:
                                use_key = 'USE_{}'.format(i.upper())
                                # If we don't have a good filter then we set them to usable False
                                if i.upper() in hdr_insts and good_filt:
                                    info[use_key] = 'T'
                                else:
                                    info[use_key] = 'F'
                        # In this case there are separate event lists for separate instruments
                        else:
                            use_key = 'USE_{}'.format(evt_path_insts[evt_key_ind].upper())
                            # If the filter is good, then so are we!
                            if good_filt:
                                info[use_key] = 'T'
                            else:
                                info[use_key] = 'F'

                    else:
                        # If the file path doesn't exist then we have to set the usable column(s) to False!
                        if inst_from_evt:
                            for i in ALLOWED_INST[tel]:
                                use_key = 'USE_{}'.format(i.upper())
                                info[use_key] = 'F'
                        else:
                            use_key = 'USE_{}'.format(evt_path_insts[evt_key_ind].upper())
                            info[use_key] = 'F'

                new_census_rows.append(info)
                census_progress.update(1)

        # We add the new observations into our existing census dataframe, whether it existed from an earlier XGA run
        #  or because we created an empty one earlier, makes no difference
        obs_lookup = pd.concat([obs_lookup, pd.DataFrame(new_census_rows)], ignore_index=True)
        # This then saves the dataframe to its rightful place
        obs_lookup.to_csv(census_file, index=False)

    # We do actually convert some values from what they are stored in the file as, to Python bools
    obs_lookup["RA_PNT"] = obs_lookup["RA_PNT"].replace('', np.nan).astype(float)
    obs_lookup["DEC_PNT"] = obs_lookup["DEC_PNT"].replace('', np.nan).astype(float)

    # Adding in columns for the instruments
    rel_inst_cols = ["USE_{}".format(inst.upper()) for inst in rel_insts]
    obs_lookup[rel_inst_cols] = obs_lookup[rel_inst_cols].apply(lambda x: x == 'T')

    # Finally we return the census and the blacklist
    return obs_lookup, blacklist


# This function also now exists in imagetools.miss, which is where it will live forevermore - I wouldn't normally
#  duplicate a whole function just to show a deprecation warning, but this function isn't going to change so it is
#  safe enough
@_deprecated(message="The XGA 'find_all_wcs' function should be imported from imagetools.misc, in the future "
                     "it will be removed from utils.")
def find_all_wcs(hdr: FITSHDR) -> List[WCS]:
    """
    A play on the function of the same name in astropy.io.fits, except this one will take a fitsio header object
    as an argument, and construct astropy wcs objects. Very simply looks for different WCS entries in the
    header, and uses their critical values to construct astropy WCS objects.

    :return: A list of astropy WCS objects extracted from the input header.
    :rtype: List[WCS]
    """
    wcs_search = [k.split("CTYPE")[-1][-1] for k in hdr.keys() if "CTYPE" in k]
    wcs_nums = [w for w in wcs_search if w.isdigit()]
    wcs_not_nums = [w for w in wcs_search if not w.isdigit()]
    if len(wcs_nums) != 2 and len(wcs_nums) != 0:
        raise KeyError("There are an odd number of CTYPEs with no extra key ")
    elif len(wcs_nums) == 2:
        wcs_keys = [""] + list(set(wcs_not_nums))
    elif len(wcs_nums) == 0:
        wcs_keys = list(set(wcs_not_nums))

    wcses = []
    for key in wcs_keys:
        w = WCS(naxis=2)
        w.wcs.crpix = [hdr["CRPIX1{}".format(key)], hdr["CRPIX2{}".format(key)]]
        w.wcs.cdelt = [hdr["CDELT1{}".format(key)], hdr["CDELT2{}".format(key)]]
        w.wcs.crval = [hdr["CRVAL1{}".format(key)], hdr["CRVAL2{}".format(key)]]
        w.wcs.ctype = [hdr["CTYPE1{}".format(key)], hdr["CTYPE2{}".format(key)]]
        wcses.append(w)

    return wcses


# Very handy function that is used in several places - just not really sure where else to put it but here
def dict_search(key: str, var: dict) -> list:
    """
    This simple function was very lightly modified from a stackoverflow answer, and is an
    efficient method of searching through a nested dictionary structure for specfic keys
    (and yielding the values associated with them). In this case will extract all of a
    specific product type for a given source.

    :param key: The key in the dictionary to search for and extract values.
    :param var: The variable to search, likely to be either a dictionary or a string.
    :return list[list]: Returns information on keys and values
    """

    # Check that the input is actually a dictionary
    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v
            # Here is where we dive deeper, recursively searching lower dictionary levels.
            if isinstance(v, dict):
                for result in dict_search(key, v):
                    # We yield a string of the result and the key, as we'll need to return the
                    # ObsID and Instrument information from these product searches as well.
                    # This will mean the output is an unpleasantly nested list, but we can solve that.
                    yield [str(k), result]


def check_telescope_choices(telescope: Union[str, List[str]]) -> List[str]:
    """
    This function centralises some checks that might be made in various places in the module. The checks are of
    the choices which the user has made regarding which telescopes to use for a function, and it makes sure that the
    chosen telescopes are valid names (i.e. they are real and supported by XGA), that the data have been set up with
    XGA, and makes sure that a list of strings is returned (even if there is only one) so that the output is always
    consistent.

    :param str/List[str] telescope: The telescope choice made by the user - the formatting and validity of which will
        be checked by this function.
    :return: A list of telescope names (or a single name) that are valid and that have made it through the checks.
    :rtype: List[str]
    """
    # If the telescope is set to None then the function will search all the telescopes for matching data, determining
    #  which are available by grabbing all the keys from the CENSUS dictionary
    if telescope is None:
        telescope = list(CENSUS.keys())
        # If there are no keys in the CENSUS dictionary then we know that no telescopes have been successfully
        #  setup with XGA and this function isn't going to work
        if len(telescope) == 0:
            raise NoTelescopeDataError("No telescope data is currently available to XGA.")

    # Just making sure the telescope names supplied by the user are lowercase, as that is how they are stored in
    #  XGA constants and products - also have to account for the fact that either a single string or a list
    #  can be passed
    if not isinstance(telescope, list):
        telescope = [telescope.lower()]
    else:
        telescope = [t.lower() for t in telescope]

    # Here we check if ANY of the passed telescopes aren't actually recognised by XGA, as I want to tell them
    #  that they have either made a typo or are labouring under a misconception about which telescopes are
    #  supported
    if any([t not in TELESCOPES for t in telescope]):
        which_bad = [t for t in telescope if t not in TELESCOPES]
        raise InvalidTelescopeError("XGA does not support the following telescopes; "
                                    "{}".format(', '.join(which_bad)))
    # If the user made specific requests of telescope, and they are ALL not available, we throw an error
    elif all([not USABLE[t] for t in telescope]):
        raise NoTelescopeDataError("None of the requested telescopes ({}) have data available to "
                                   "XGA.".format(', '.join(telescope)))
    # However if the user made specific requests of telescope, and SOME are not available then they get a warning
    elif any([not USABLE[t] for t in telescope]):
        # This isn't elegant, but oh well - we have to make sure that we only let those telescopes through
        #  that have actually been set up and are working with XGA
        which_bad = [t for t in telescope if not USABLE[t]]
        telescope = [t for t in telescope if USABLE[t]]
        warn("Some requested telescopes are not currently set up with XGA; {}".format(", ".join(which_bad)),
             stacklevel=2)

    return telescope
# --------------------------------------------------------------------------------------


# ------------- Defining where the configuration file (and eventually observation census) lives -------------

# We need to know where the configuration file which tells XGA what data and settings to use lives. If XDG_CONFIG_HOME
#  is set then use that, otherwise use this default config path. We'll then create this configuration directory
#  if it doesn't already exist
CONFIG_PATH = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config', 'xga'))
if not os.path.exists(CONFIG_PATH):
    os.makedirs(CONFIG_PATH)
# -----------------------------------------------------------------------------------------------------------


# ------------- Defining constants to do with the telescope data -------------
# This chunk of this file sets

# This dictionary both defines the telescopes that XGA is compatible with, and their allowed instruments. These mission
#  and instrument names should all be lowercase, that will be the general storage convention throughout XGA
ALLOWED_INST = {"xmm": ["pn", "mos1", "mos2"],
                }
# TODO remove this when everything is generalised and a specific XMM_INST constant isn't required
XMM_INST = ALLOWED_INST['xmm']
# I provide a list of the top-level keys of the ALLOWED_INST dictionary, as a quick way of accessing the supported
#  telescope names
TELESCOPES = list(ALLOWED_INST.keys())
# This dictionary won't be used much, but it's just so we have access to some properly formatted telescope names
PRETTY_TELESCOPE_NAMES = {'xmm': 'XMM'}
# This dictionary is an important one, it will be set during the course of the setup process in this file, and
#  defines whether a particular telescope that XGA supports seems to have the files necessary to be used by sources
#  and samples. The default is False, and if we find otherwise during this process then that will be changed
USABLE = {tele: False for tele in TELESCOPES}

# Here we define regular expressions that will allow use to verify the structure of an ObsID for a particular
#  telescope - this functionality is also in DAXA mission classes, so we may just switch to using them in the
#  future to avoid features duplication
OBS_ID_REGEX = {'xmm': '^[0-9]{10}$'}

# This is another sort of duplication of a DAXA feature, and stores the default search distances to be used for
#  each telescope in the xga.match.separation_match function. These are loosely based on the field of view of
#  each telescope. In cases where different instruments on a telescope have significantly different field of view,
#  there may be multi-level dictionaries
DEFAULT_TELE_SEARCH_DIST = {'xmm': Quantity(30, 'arcmin')}

# This defines where the observation census files would be located for each of the allowed telescopes (the top level
#  keys of ALLOWED_INST are telescope/mission names) - not every telescope is guaranteed to have a file created, it
#  depends on what data are available to the user.
CENSUS_FILES = {tel: os.path.join(CONFIG_PATH, tel, '{}_census.csv'.format(tel)) for tel in ALLOWED_INST}
BLACKLIST_FILES = {tel: os.path.join(CONFIG_PATH, tel, '{}_blacklist.csv'.format(tel)) for tel in ALLOWED_INST}

# This list contains banned filter types - these occur in observations that I don't want XGA to try and use
BANNED_FILTS = {"xmm": ['CalClosed', 'Closed']}
# ----------------------------------------------------------------------------


# ------------- Defining constants to do with the configuration file -------------
# These will largely be the dictionaries that get turned into the various discrete sections of the configuration
#  file, with one for 'general configuration', and one for each separate telescope supported by XGA. Those dictionaries
#  will contain different entries, depending on the telescope, but the general idea is to point XGA at the available
#  events lists, images, and source regions

# XGA config file path - this one is obviously important so that we know where to look for/generate the config file
CONFIG_FILE = os.path.join(CONFIG_PATH, 'xga.cfg')

# Section of the config file for setting up the XGA module. The output path is where XGA generated files get stored
#  and the num_cores entry allows you to manually set the maximum number of cores that XGA can use - though by default
#  it is set to auto, which will use 90% of the total on your system
XGA_CONFIG = {"xga_save_path": "xga_output",
              "num_cores": 'auto'}

# Will have to make it clear in the documentation which paths can be left unspecified, and indeed that whole sections
#  can be left out if there are no relevant data archives for those telescopes. There will be a section for each
#  mission/telescope's configuration

# NOTE - THE 'root_{telescope name}_dir' IS REQUIRED - MAKE SURE TO ADD ONE FOR EVERY SUPPORTED TELESCOPE
# NOTE THE SECOND - THE PATHS TO CLEANED EVENTS ARE REQUIRED TO BE STRUCTURED
#  LIKE 'clean_{indicator of instrument}_evts' - it helps make the other bits of this setup process more generalised
# These are the pertinent bits of information for XMM - mainly the general 'where does data live' stuff
XMM_FILES = {"root_xmm_dir": "/this/is/required/xmm_obs/data/",
             "clean_pn_evts": "/this/is/required/{obs_id}/pn_exp1_clean_evts.fits",
             "clean_mos1_evts": "/this/is/required/{obs_id}/mos1_exp1_clean_evts.fits",
             "clean_mos2_evts": "/this/is/required/{obs_id}/mos2_exp1_clean_evts.fits",
             "attitude_file": "/this/is/required/{obs_id}/attitude.fits",
             "lo_en": ['0.50', '2.00'],
             "hi_en": ['2.00', '10.00'],
             "pn_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits",
             "mos1_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_img.fits",
             "mos2_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_img.fits",
             "pn_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits",
             "mos1_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_expmap.fits",
             "mos2_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_expmap.fits",
             "region_file": "/this/is/optional/xmm_obs/regions/{obs_id}/regions.reg"}

# We set up this dictionary for later, it makes programmatically grabbing the section dictionaries easier.
tele_conf_sects = {'xmm': XMM_FILES}
# -------------------------------------------------------------------------------


# ------------- Defining constants to do with general XGA stuff -------------

# List of products supported by XGA that are allowed to be energy bound
ENERGY_BOUND_PRODUCTS = ["image", "expmap", "ratemap", "combined_image", "combined_expmap", "combined_ratemap"]
# These are the built-in profile types
PROFILE_PRODUCTS = ["brightness_profile", "gas_density_profile", "gas_mass_profile", "1d_apec_norm_profile",
                    "1d_proj_temperature_profile", "gas_temperature_profile", "baryon_fraction_profile",
                    "1d_proj_metallicity_profile", "1d_emission_measure_profile", "hydrostatic_mass_profile",
                    "specific_entropy_profile"]
COMBINED_PROFILE_PRODUCTS = ["combined_"+pt for pt in PROFILE_PRODUCTS]
# List of all products supported by XGA
ALLOWED_PRODUCTS = ["spectrum", "grp_spec", "regions", "events", "psf", "psfgrid", "ratemap", "combined_spectrum",
                    ] + ENERGY_BOUND_PRODUCTS + PROFILE_PRODUCTS + COMBINED_PROFILE_PRODUCTS

# A centralised constant to define what radius labels are allowed
RAD_LABELS = ["region", "r2500", "r500", "r200", "custom", "point"]

# Adding a default concordance cosmology set up here - this replaces the original default choice of Planck15
DEFAULT_COSMO = LambdaCDM(70, 0.3, 0.7)

# The maximum difference in radii to consider them a match (used in get methods to avoid radii not matching)
RAD_MATCH_PRECISION = Quantity(1e-8, 'deg')

# This defines the meaning of different colours of region - this will eventually be user configurable in the
#  configuration file, but putting it here means that the user can still change the definitions programmatically
# Definitions of the colours of XCS regions can be found in the thesis of Dr Micheal Davidson
#  University of Edinburgh - 2005.
# Red - Point source
# Green - Extended source
# Magenta - PSF-sized extended source
# Blue - Extended source with significant point source contribution
# Cyan - Extended source with significant Run1 contribution
# Yellow - Extended source with less than 10 counts
SRC_REGION_COLOURS = {'pnt': ["red"], 'ext': ["green", "magenta", "blue", "cyan", "yellow"]}

# XSPEC file extraction (and base fit) scripts
XGA_EXTRACT = importlib_resources.files(__name__) / "xspec_scripts/xga_extract.tcl"
BASE_XSPEC_SCRIPT = importlib_resources.files(__name__) / "xspec_scripts/general_xspec_fit.xcm"
CROSS_ARF_XSPEC_SCRIPT = importlib_resources.files(__name__) / "xspec_scripts/crossarf_xspec_fit.xcm"
COUNTRATE_CONV_SCRIPT = importlib_resources.files(__name__) / "xspec_scripts/cr_conv_calc.xcm"

# Useful jsons of all XSPEC models, their required parameters, and those parameter's units
with open(importlib_resources.files(__name__) / "files/xspec_model_pars.json5", 'r') as filey:
    MODEL_PARS = json.load(filey)

with open(importlib_resources.files(__name__) / "files/xspec_model_units.json5", 'r') as filey:
    MODEL_UNITS = json.load(filey)

# No longer XSPEC related constants/files, but here we read in a file that helps map column names in
#  event files to mission-independent names (e.g. specifying which table name contains events in an
#  event list). This is based on the mission database file for XSELECT, but has been modified
#  quite a lot.
with open(importlib_resources.files(__name__) / "files/mission_event_column_name_map.json", 'r') as filey:
    MISSION_COL_DB = json.load(filey)
# ---------------------------------------------------------------------------


# ------------- Defining constants to do with Physics -------------
# I know this is practically pointless, I could just use m_p, but I like doing things properly.
HY_MASS = m_p + m_e

# Mean molecular weight, mu
# TODO Make sure this doesn't change with abundance table, I think it might?
MEAN_MOL_WEIGHT = 0.61

# TODO Populate this further, also actually calculate and verify these myself, the value here is taken
#  from pyproffit code
# For a fully ionised plasma, this is the electron-to-proton ratio
NHC = {"angr": 1.199}
# -----------------------------------------------------------------


# ------------- Defining constants to do with units -------------
# Any new units we set up for the module will live in this section

# These are largely defined so that I can use them for when I'm normalising profile plots, that way
#  the view method can just write the units nicely the way it normally does
r200 = def_unit('r200', format={'latex': r"\mathrm{R_{200}}"})
r500 = def_unit('r500', format={'latex': r"\mathrm{R_{500}}"})
r2500 = def_unit('r2500', format={'latex': r"\mathrm{R_{2500}}"})

# These allow us to set up astropy quantities in units of some of the internal systems of telescopes - obviously
#  they don't convert to anything, but they still let us work within the astropy coordinate framework
xmm_sky = def_unit("xmm_sky")
xmm_det = def_unit("xmm_det")

# This is a dumb and annoying work-around for a readthedocs problem where units were being added multiple times
try:
    Quantity(1, 'r200')
except ValueError:
    # Adding the unit instances we created to the astropy pool of units - means we can do things like just defining
    #  Quantity(10000, 'xmm_det') rather than importing xmm_det from utils and using it that way
    add_enabled_units([r200, r500, r2500, xmm_sky, xmm_det])
# ---------------------------------------------------------------


# ------------- Defining constants to do with backend software -------------
# Various parts of XGA can rely on different pieces of backend software, so we have this section to set up constants
#  that tell the relevant parts of XGA whether the software is installed, and what version it is

# Here we check to see whether XSPEC is installed (along with all the necessary paths)
XSPEC_VERSION = None
# Got to make sure we can access command line XSPEC.
if shutil.which("xspec") is None:
    warn("Unable to locate an XSPEC installation.", stacklevel=2)
else:
    try:
        # The XSPEC intro text includes the version, so I read that out and parse it. That null_script that I'm running
        #  does absolutely nothing, it's just a way for me to get the version out
        null_path = importlib_resources.files(__name__) / "xspec_scripts/null_script.xcm"
        xspec_out, xspec_err = Popen("xspec - {}".format(null_path), stdout=PIPE, stderr=PIPE,
                                     shell=True).communicate()
        # Got to parse the stdout to get the XSPEC version, which is what these two lines do
        xspec_vline = [line for line in xspec_out.decode("UTF-8").split('\n') if 'XSPEC version' in line][0]
        XSPEC_VERSION = xspec_vline.split(': ')[-1]
    # I know broad exceptions are a sin, but if anything goes wrong here then XGA needs to assume that XSPEC
    #  is messed up in some way
    except:
        # Not necessary as the default XSPEC_VERSION value is None, but oh well - something has to be here!
        XSPEC_VERSION = None
# Then I setup these constants of fit methods and abundance tables - just so I can pre-check a user's choice in any
#  of the XSPEC interface parts of XGA, rather than failing unhelpfully when they try to run the fit
XSPEC_FIT_METHOD = ["leven", "migrad", "simplex"]
ABUND_TABLES = ["feld", "angr", "aneb", "grsa", "wilm", "lodd", "aspl"]

# Next up, we check to see what version of SAS (if any) is installed - for the XMM-Newton mission
# Here we check to see whether SAS is installed (along with all the necessary paths)
SAS_VERSION = None
if "SAS_DIR" not in os.environ:
    warn("SAS_DIR environment variable is not set, unable to verify SAS is present on system, as such "
         "all functions in xga.sas will not work.")
    SAS_VERSION = None
    SAS_AVAIL = False
else:
    # This way, the user can just import the SAS_VERSION from this utils code
    sas_out, sas_err = Popen("sas --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
    SAS_VERSION = sas_out.decode("UTF-8").strip("]\n").split('-')[-1]
    SAS_AVAIL = True

# This checks for the CCF path, which is required to use cifbuild, which is required to do basically
#  anything with SAS
if SAS_AVAIL and "SAS_CCFPATH" not in os.environ:
    warn("SAS_CCFPATH environment variable is not set, this is required to generate calibration files. As such "
         "functions in xga.sas will not work.")
    SAS_AVAIL = False

# Here we read in files that list the errors and warnings in SAS
errors = pd.read_csv(importlib_resources.files(__name__) / "files/sas_errors.csv", header="infer")
warnings = pd.read_csv(importlib_resources.files(__name__) / "files/sas_warnings.csv", header="infer")

# Just the names of the errors in two handy constants
SASERROR_LIST = errors["ErrName"].values
SASWARNING_LIST = warnings["WarnName"].values

# We set up a mapping from telescope name to software version constant
# Don't really expect the user to use this, hence why it isn't a constant, more for checks at the end of this
#  file. Previously a warning for missing software would be shown at the time of checking, but now we wait to see
#  which telescopes are configured in the XGA config file before warning that telescope software is missing
tele_software_map = {'xmm': SAS_VERSION}
# --------------------------------------------------------------------------


# ------------- Creating/checking the entries in the configuration file -------------
# This chunk of utils will be dedicated to making sure that the configuration file has been created (by default with
#  sections for every telescope that XGA supports), or that if it already exists it contains valid entries.
# THIS STAGE USED TO FAIL ENTIRELY IF THERE WEREN'T VALID ENTRIES - but now I realise that sometimes you just want
#  to use the XGA products with your own data files without using the source/sample classes, so it will no longer fail
#  at this stage.

# We're going to be assessing the configuration sections and determining whether they are configured in such a way
#  that XGA can use them - if the configuration is valid and can be used then that telescope's entry in this
#  dictionary wll be set to True. That is likely to be checked in the source class init at some point
VALID_CONFIG = {tel: False for tel in TELESCOPES}

# In this case we find that the configuration file does not exist, and we set it up using the default sections and
#  configurations that were set up toward the top of this file
if not os.path.exists(CONFIG_FILE):
    # Define a configuration object
    xga_default = ConfigParser()
    # This adds the overall XGA setup section - controls global things like where XGA generated files are stored, and
    #  how many cores XGA is allowed to use (though that can also be set at import time).
    xga_default.add_section("XGA_SETUP")
    xga_default["XGA_SETUP"] = XGA_CONFIG

    # Now we cycle through all the telescopes supported by XGA, adding their default sections to the configuration.
    #  I now regret calling the sections {telescope}_FILES, but honestly it doesn't bother me enough to endure the
    #  hassle of changing it now - it would be a breaking change for any existing installations
    for tel in TELESCOPES:
        cur_sec_name = "{}_FILES".format(tel.upper())
        xga_default.add_section(cur_sec_name)
        xga_default[cur_sec_name] = tele_conf_sects[tel]

    # The default configuration file is now written to the path that we expect to find the config file at
    with open(CONFIG_FILE, 'w') as new_cfg:
        xga_default.write(new_cfg)

    # First time run triggers this message - it used to be an error, and so XGA wouldn't advance beyond this point
    #  with a new configuration file, but we want people to be able to use the product classes without configuring
    warn("This is the first time you've used XGA; to use most functionality you will need to configure {} to match "
         "your setup, though you can use product classes regardless.".format(CONFIG_FILE), stacklevel=2)

xga_conf = ConfigParser()
# It would be nice to do configparser interpolation, but it wouldn't handle the lists of energy values
xga_conf.read(CONFIG_FILE)

# If the current section name exists in xga_conf then all is well, there hasn't been an update to XGA which added
#  a new telescope installed - however if a section is missing then we need to add it. This is slightly inelegant
#  because I also iterate through the telescopes slightly further down, but it is just easier for this to go here
#  as I need to write the updated configuration file to disk, and I'm about to turn it into a dictionary so it is
#  easier to do that here
# Set up a flag, so we can know if any sections have been added
altered = False
for tel in TELESCOPES:
    cur_sec_name = "{}_FILES".format(tel.upper())
    if cur_sec_name not in xga_conf:
        # If there isn't already a files section for one of the telescopes now supported by XGA, then we add it to
        #  the existing configuration file
        xga_conf.add_section(cur_sec_name)
        xga_conf[cur_sec_name] = tele_conf_sects[tel]
        altered = True
# If we altered the existing configuration file, then we need to save the altered configuration to disk
if altered:
    with open(CONFIG_FILE, 'w') as update_cfg:
        xga_conf.write(update_cfg)

# As it turns out, the ConfigParser class is a pain to work with, so we're converting to a dict here
# Addressing works just the same
xga_conf = {str(sect): dict(xga_conf[str(sect)]) for sect in xga_conf}

# Firstly, we check if the entries in the general XGA configuration section are valid - the output path is the first to
#  check, though we are actually only checking for a very unlikely edge case. That somebody is using this new version
#  of XGA with an old, unmodified, configuration file.
if xga_conf['XGA_SETUP']['xga_save_path'] == "/this/is/required/xga_output/":
    raise XGAConfigError("The 'xga_save_path' entry is currently '/this/is/required/xga_output/', which at one point"
                         " was the default entry in new configuration files; please change it before proceeding.")

# Optionally there can be a num_cores entry in the overall settings section, and if there is we check that it is
#  a valid value - though we don't throw an error if it isn't, we just default to XGA automatically determining
#  the number of cores to use.
if 'num_cores' in xga_conf['XGA_SETUP'] and xga_conf['XGA_SETUP']['num_cores'] != 'auto' and \
        not isinstance(xga_conf['XGA_SETUP']['num_cores'], int):
    warn("The 'num_cores' entry ({}) in the configuration file is not valid, it should either be an integer or "
         "auto; here we default to 'auto'.".format(xga_conf['XGA_SETUP']['num_cores']), stacklevel=2)
    xga_conf['XGA_SETUP']['num_cores'] = 'auto'

# Now we check the telescope-specific sections
for tel in TELESCOPES:
    cur_sec_name = "{}_FILES".format(tel.upper())
    cur_sec = xga_conf[cur_sec_name]

    # The upper and lower energy bounds defined in the config file for existing image/exposure maps files are a
    #  string representation of a list, and we want to turn them back into an actual list
    poss_ens = ['lo_en', 'hi_en']
    if sum([en in cur_sec for en in poss_ens]) == 1:
        raise XGAConfigError("Both lo_en and hi_en entries must be present, not one or the other.")
    # This just converts the string representation of the energy list into an actual list of energies
    elif sum([en in cur_sec for en in poss_ens]) == 2:
        for en_conf in poss_ens:
            in_parts = cur_sec[en_conf].strip("[").strip("]").split(',')
            real_list = [part.strip(' ').strip("'").strip('"') for part in in_parts if part != '' and part != ' ']
            cur_sec[en_conf] = real_list

    # Now we check that the directory we're pointed to for the root data directory of the current telescope
    #  actually exists
    # This variable keeps track of if the root_dir for this telescope actually exists
    root_dir_exists = False
    if os.path.exists(cur_sec['root_{t}_dir'.format(t=tel)]):
        root_dir_exists = True

    # This is a pretty blunt-force approach, but honestly I think it should work fine consider we just want to
    #  check whether any of the required sections have been left as the default values (meaning that telescope
    #  hasn't been configured and can't be used).
    # We use this 'all_req_changed' flag to store if any of the required entries have been left at their default
    #  values - even one of these being left at default means we can't use this telescope.
    all_req_changed = True
    # As we're already iterating through the entries in this section we will also check to see if the file paths
    #  have been defined relative to the root directory, so we'll define this list of entries not to check
    #  for that
    no_check = poss_ens + ['root_{t}_dir'.format(t=tel)]
    for entry in cur_sec:
        if "/this/is/required/" in cur_sec[entry]:
            any_req_defaults = False
        elif (entry not in no_check and cur_sec['root_{t}_dir'.format(t=tel)] not in cur_sec[entry] and
              cur_sec[entry][0] != '/'):
            # Replace the current definition with an absolute one s
            cur_sec[entry] = os.path.join(os.path.abspath(cur_sec['root_{t}_dir'.format(t=tel)]), cur_sec[entry])

    # We make sure that the root directory is an absolute path, just for our sanity later on
    cur_sec['root_{t}_dir'.format(t=tel)] = os.path.abspath(cur_sec['root_{t}_dir'.format(t=tel)]) + "/"

    # This tells the rest of XGA that the current telescope is usable! If these conditions aren't fulfilled then
    #  the USABLE entry for the current telescope will stay at the default value of False
    if all_req_changed and root_dir_exists:
        USABLE[tel] = True
# -----------------------------------------------------------------------------------


# ------------- Generating the observation censuses for all USABLE telescopes -------------
# Read dataframe of ObsIDs and pointing coordinates into dictionaries
CENSUS = {}
BLACKLIST = {}
# Also create a dictionary that tells parts of XGA whether it should expect the event lists of the different
#  instruments to be separate, or combined (I'm looking at you eROSITA CalPV).
COMBINED_INSTS = {}

# Checking if someone had been using the XMM only version of XGA previously - with this update to implement the
#  infrastructure to support different telescopes the census/blacklist files will exist for EACH telescope
#  individually
old_census_path = os.path.join(CONFIG_PATH, 'census.csv')
if os.path.exists(old_census_path):
    # Changing the .config/xga directory to the updated structure for the multi-telescope version of xga
    os.makedirs(CONFIG_PATH + '/xmm/')
    new_xmm_census_path = os.path.join(CONFIG_PATH, 'xmm', 'xmm_census.csv')
    # Moving the xmm census to the correct place and renaming it with the updated naming scheme
    shutil.move(old_census_path, new_xmm_census_path)
    # Doing the same for the blacklist
    old_bl_path = os.path.join(CONFIG_PATH, 'blacklist.csv')
    new_bl_path = os.path.join(CONFIG_PATH, 'xmm', 'xmm_blacklist.csv')
    shutil.move(old_bl_path, new_bl_path)

for tel in USABLE:
    # We only care to have/make a census if the telescope is actually set up and usable
    if USABLE[tel]:
        CENSUS[tel], BLACKLIST[tel] = build_observation_census(tel)

    # Populate the dictionary that says whether the event lists for a given telescope are combined or not - it would
    #  have been so much easier if they were all always separate, but the eROSITA CalPV ones weren't released like
    #  that and I bet eRASS won't be either
    if 'clean_{}_evts'.format(tel.lower()) in xga_conf['{}_FILES'.format(tel.upper())]:
        COMBINED_INSTS[tel] = True
    else:
        COMBINED_INSTS[tel] = False
# -----------------------------------------------------------------------------------------


# ------------- Final setup of important constants from the configuration file -------------
# We make sure to create the absolute output path from what was specified in the configuration file
OUTPUT = os.path.abspath(xga_conf["XGA_SETUP"]["xga_save_path"]) + "/"

# The default behaviour for the generation of new configuration files is to set the num_cores entry to 'auto', though
#  that isn't a given with older configuration files - as such we check and don't assume it will be there. If the
#  num_cores keyword is present and ISN'T auto then we use the user specified core count
if "num_cores" in xga_conf["XGA_SETUP"] and xga_conf["XGA_SETUP"]["num_cores"] != "auto":
    # If the user has set a number of cores in the config file then we'll use that.
    NUM_CORES = int(xga_conf["XGA_SETUP"]["num_cores"])
# In this case though the user has not specified a number of cores to use thus we will use 90% of the available
#  cores on the system
else:
    # Going to allow multi-core processing to use 90% of available cores by default, but
    # this can be over-ridden in individual SAS calls.
    NUM_CORES = max(int(np.floor(os.cpu_count() * 0.9)), 1)  # Makes sure that at least one core is used
# ------------------------------------------------------------------------------------------


# -------------------- Making this utils.py compatible with XMM-only XGA -------------------
# TODO REMOVE THIS ONCE MULTI-MISSION IS RELEASED
# This version of utils.py was patched in from the multi-mission development branch, and as
#  such is set up slightly differently (with telescope names as keys in the observation
#  census for instance), so some small changes must be made to make the constants defined
#  here usable in the rest of the current module
if 'xmm' in CENSUS:
    CENSUS = CENSUS['xmm']
    BLACKLIST = BLACKLIST['xmm']
else:
    CENSUS = None
    BLACKLIST = None

# The multi-mission branch has different behaviours for the creation of output directories, and this chunk of
#  code was removed, so it must be reinstated for XMM-only XGA
# Make a storage directory where specific source name directories will then be created, there profile objects
#  created for those sources will be saved
if not os.path.exists(OUTPUT + "profiles"):
    os.makedirs(OUTPUT + "profiles")

# Also making a storage directory specifically for products which are combinations of different ObsIDs
#  and instruments
if not os.path.exists(OUTPUT + "combined"):
    os.makedirs(OUTPUT + "combined")

# And create an inventory file for that directory
if not os.path.exists(OUTPUT + "combined/inventory.csv"):
    with open(OUTPUT + "combined/inventory.csv", 'w') as inven:
        inven.writelines(["file_name,obs_ids,insts,info_key,src_name,type"])
# ------------------------------------------------------------------------------------------