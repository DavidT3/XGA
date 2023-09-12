#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 11/09/2023, 21:54. Copyright (c) The Contributors

import json
import os
import re
import shutil
from configparser import ConfigParser
from subprocess import Popen, PIPE
from typing import List, Tuple
from warnings import warn

import pandas as pd
import pkg_resources
from astropy.constants import m_p, m_e
from astropy.cosmology import LambdaCDM
from astropy.units import Quantity, def_unit, add_enabled_units
from astropy.wcs import WCS
from fitsio import read_header
from fitsio.header import FITSHDR
from tqdm import tqdm

from .exceptions import XGAConfigError

# We need to know where the configuration file which tells XGA what data and settings to use lives. If XDG_CONFIG_HOME
#  is set then use that, otherwise use this default config path. We'll then create this configuration directory
#  if it doesn't already exist
CONFIG_PATH = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config', 'xga'))
if not os.path.exists(CONFIG_PATH):
    os.makedirs(CONFIG_PATH)


# ------------- Defining constants to do with the telescope data -------------
# This chunk of this file sets

# This dictionary both defines the telescopes that XGA is compatible with, and their allowed instruments. These mission
#  and instrument names should all be lowercase, that will be the general storage convention throughout XGA
ALLOWED_INST = {"xmm": ["pn", "mos1", "mos2"],
                "erosita": ["tm1", "tm2", "tm3", "tm4", "tm5", "tm6", "tm7"]}
# TODO remove this when everything is generalised and a specific XMM_INST constant isn't required
XMM_INST = ALLOWED_INST['xmm']
# I provide a list of the top-level keys of the ALLOWED_INST dictionary, as a quick way of accessing the supported
#  telescope names
TELESCOPES = list(ALLOWED_INST.keys())
# This dictionary is an important one, it will be set during the course of the setup process in this file, and
#  defines whether a particular telescope that XGA supports seems to have the files necessary to be used by sources
#  and samples. The default is False, and if we find otherwise during this process then that will be changed
USABLE = {tele: False for tele in TELESCOPES}

# Here we define regular expressions that will allow use to verify the structure of an ObsID for a particular
#  telescope - this functionality is also in DAXA mission classes, so we may just switch to using them in the
#  future to avoid features duplication
OBS_ID_REGEX = {'xmm': '^[0-9]{10}$', "erosita": '^[0-9]{6}$'}

# This defines where the observation census files would be located for each of the allowed telescopes (the top level
#  keys of ALLOWED_INST are telescope/mission names) - not every telescope is guaranteed to have a file created, it
#  depends on what data are available to the user.
CENSUS_FILES = {tel: os.path.join(CONFIG_PATH, tel, '{}_census.csv'.format(tel)) for tel in ALLOWED_INST}
BLACKLIST_FILES = {tel: os.path.join(CONFIG_PATH, tel, '{}_blacklist.csv'.format(tel)) for tel in ALLOWED_INST}

# This list contains banned filter types - these occur in observations that I don't want XGA to try and use
BANNED_FILTS = {"xmm": ['CalClosed', 'Closed'],
                "erosita": ['CALIB', 'CLOSED']}
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

# The information required to use eROSITA data
EROSITA_FILES = {"root_erosita_dir": "/this/is/required/erosita_obs/data/",
                 "clean_erosita_evts": "/this/is/required/{obs_id}/{obs_id}.fits",
                 "lo_en": ['0.50', '2.00'],
                 "hi_en": ['2.00', '10.00'],
                 "region_file": "/this/is/optional/erosita_obs/regions/{obs_id}/regions.reg"}

# We set up this dictionary for later, it makes programmatically grabbing the section dictionaries easier.
tele_conf_sects = {'xmm': XMM_FILES, 'erosita': EROSITA_FILES}
# -------------------------------------------------------------------------------


# ------------- Defining constants to do with general XGA stuff -------------

# List of products supported by XGA that are allowed to be energy bound
ENERGY_BOUND_PRODUCTS = ["image", "expmap", "ratemap", "combined_image", "combined_expmap", "combined_ratemap"]
# These are the built-in profile types
PROFILE_PRODUCTS = ["brightness_profile", "gas_density_profile", "gas_mass_profile", "1d_apec_norm_profile",
                    "1d_proj_temperature_profile", "gas_temperature_profile", "baryon_fraction_profile",
                    "1d_proj_metallicity_profile", "1d_emission_measure_profile", "hydrostatic_mass_profile"]
COMBINED_PROFILE_PRODUCTS = ["combined_"+pt for pt in PROFILE_PRODUCTS]
# List of all products supported by XGA
ALLOWED_PRODUCTS = ["spectrum", "grp_spec", "regions", "events", "psf", "psfgrid", "ratemap", "combined_spectrum",
                    ] + ENERGY_BOUND_PRODUCTS + PROFILE_PRODUCTS + COMBINED_PROFILE_PRODUCTS

# A centralised constant to define what radius labels are allowed
RAD_LABELS = ["region", "r2500", "r500", "r200", "custom", "point"]

# Adding a default concordance cosmology set up here - this replaces the original default choice of Planck15
DEFAULT_COSMO = LambdaCDM(70, 0.3, 0.7)

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
XGA_EXTRACT = pkg_resources.resource_filename(__name__, "xspec_scripts/xga_extract.tcl")
BASE_XSPEC_SCRIPT = pkg_resources.resource_filename(__name__, "xspec_scripts/general_xspec_fit.xcm")
COUNTRATE_CONV_SCRIPT = pkg_resources.resource_filename(__name__, "xspec_scripts/cr_conv_calc.xcm")
# Useful jsons of all XSPEC models, their required parameters, and those parameter's units
with open(pkg_resources.resource_filename(__name__, "files/xspec_model_pars.json5"), 'r') as filey:
    MODEL_PARS = json.load(filey)
with open(pkg_resources.resource_filename(__name__, "files/xspec_model_units.json5"), 'r') as filey:
    MODEL_UNITS = json.load(filey)
# ---------------------------------------------------------------------------


# ------------- Defining constants to do with physics -------------
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
erosita_sky = def_unit("erosita_sky")
erosita_det = def_unit("erosita_det")

# This is a dumb and annoying work-around for a readthedocs problem where units were being added multiple times
try:
    Quantity(1, 'r200')
except ValueError:
    # Adding the unit instances we created to the astropy pool of units - means we can do things like just defining
    #  Quantity(10000, 'xmm_det') rather than importing xmm_det from utils and using it that way
    add_enabled_units([r200, r500, r2500, xmm_sky, xmm_det, erosita_sky, erosita_det])
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
        null_path = pkg_resources.resource_filename(__name__, "xspec_scripts/null_script.xcm")
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
errors = pd.read_csv(pkg_resources.resource_filename(__name__, "files/sas_errors.csv"), header="infer")
warnings = pd.read_csv(pkg_resources.resource_filename(__name__, "files/sas_warnings.csv"), header="infer")
# Just the names of the errors in two handy constants
SASERROR_LIST = errors["ErrName"].values
SASWARNING_LIST = warnings["WarnName"].values

# TODO THIS WILL BE FLESHED OUT INTO A SECTION FOR eROSITA
ESASS_VERSION = None

# We set up a mapping from telescope name to software version constant
# Don't really expect the user to use this, hence why it isn't a constant, more for checks at the end of this
#  file. Previously a warning for missing software would be shown at the time of checking, but now we wait to see
#  which telescopes are configured in the XGA config file before warning that telescope software is missing
tele_software_map = {'xmm': SAS_VERSION, 'erosita': ESASS_VERSION}
# --------------------------------------------------------------------------


# Nested dictionary to be used to cycle over different telescopes in the following functions in this script
# Top Layer is the telescope
# Second layer contains:
# event_path_key - the mandatory directories that are needed for xga to function
# default_section, config_section - holds the variables that contain all the input file paths in each section
# root_dir_key - the root directory label in the config file
# used - a bool to indicate whether the telescope has been set up in the config file
# TELESCOPE_DICT = {"xmm": {"event_path_key": ["root_xmm_dir", "clean_pn_evts", "clean_mos1_evts", "clean_mos2_evts",
#                                              "attitude_file"],
#                           "default_section": XMM_FILES,
#                           "config_section": "XMM_FILES",
#                           "root_dir_key": "root_xmm_dir",
#                           "instruments": ["PN", "MOS1", "MOS2"],
#                           "used": False},
#                   "erosita": {"event_path_key": ["root_erosita_dir", "erosita_evts"],
#                               "default_section": EROSITA_FILES,
#                               "config_section": "EROSITA_FILES",
#                               "root_dir_key": "root_erosita_dir",
#                               "instruments": ["TM1", "TM2", "TM3", "TM4", "TM5", "TM6", "TM7"],
#                               "used": False}}


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

def xmm_observation_census(config: ConfigParser, obs_census: List, obs_lookup: List) -> List:
    """
    JESS_TODO write this doc string properly please
    A function to initialise or update the file that stores which observations are available in the user
    specified XMM data directory, and what their pointing coordinates are.
    CURRENTLY THIS WILL NOT UPDATE TO DEAL WITH OBSID FOLDERS THAT HAVE BEEN DELETED.

    :param config: The XGA configuration object.
    :return: ObsIDs and pointing coordinates of available XMM observations.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    # The census lives in the XGA config folder, and CENSUS_FILE stores the path to it.
    # If it exists, it is read in, otherwise empty lists are initialised to be appended to.
    if os.path.exists(CENSUS_FILE):
        with open(CENSUS_FILE, 'r') as census:
            obs_lookup = census.readlines()  # Reads the lines of the files
            # This is just ObsIDs, needed to see which ObsIDs have already been processed.
            obs_lookup_obs = [entry.split(',')[0] for entry in obs_lookup[1:]]
    else:
        obs_lookup = ["ObsID,RA_PNT,DEC_PNT,USE_PN,USE_MOS1,USE_MOS2\n"]
        obs_lookup_obs = []

    # Creates black list file if one doesn't exist, then reads it in
    if not os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, 'w') as bl:
            bl.write("ObsID,EXCLUDE_PN,EXCLUDE_MOS1,EXCLUDE_MOS2")
    blacklist = pd.read_csv(BLACKLIST_FILE, header="infer", dtype=str)

    # This part here is to support blacklists used by older versions of XGA, where only a full ObsID was excluded.
    #  Now we support individual instruments of ObsIDs being excluded from use, so there are extra columns expected
    if len(blacklist.columns) == 1:
        # Adds the three new columns, all with a default value of True. So any ObsID already in the blacklist
        #  will have the same behaviour as before, all instruments for the ObsID are excluded
        blacklist[["EXCLUDE_PN", "EXCLUDE_MOS1", "EXCLUDE_MOS2"]] = 'T'
        # If we have even gotten to this stage then the actual blacklist file needs re-writing, so I do
        blacklist.to_csv(BLACKLIST_FILE, index=False)

    # Need to find out which observations are available, crude way of making sure they are ObsID directories
    # This also checks that I haven't run them before
    obs_census = [entry for entry in os.listdir(config["XMM_FILES"]["root_xmm_dir"]) if obs_id_test('xmm', entry)
                  and entry not in obs_lookup_obs]
    if len(obs_census) != 0:
        with tqdm(desc="Assembling list of ObsIDs", total=len(obs_census)) as census_progress:
            for obs in obs_census:
                info = {'ra': None, 'dec': None, "the_rest": []}
                for key in ["clean_pn_evts", "clean_mos1_evts", "clean_mos2_evts"]:
                    evt_path = config["XMM_FILES"][key].format(obs_id=obs)
                    if os.path.exists(evt_path):
                        evts_header = read_header(evt_path)
                        try:
                            # Reads out the filter header, if it is CalClosed/Closed then we can't use it
                            filt = evts_header["FILTER"]
                            info['ra'] = evts_header["RA_PNT"]
                            info['dec'] = evts_header["DEC_PNT"]
                        except KeyError:
                            # It won't actually, but this will trigger the if statement that tells XGA not to use
                            #  this particular obs/inst combo
                            filt = "CalClosed"

                    # TODO Decide if I want to disallow small window mode observations
                    if filt not in BANNED_FILTS:
                        info["the_rest"].append("T")
                    else:
                        info["the_rest"].append("F")
                else:
                    info["the_rest"].append("F")

            use_insts = ",".join(info["the_rest"])
            # Write the information to the line that will go in the census csv
            if info["ra"] is not None and info["dec"] is not None:
                # Format to write to the census.csv that lives in the config directory.
                obs_lookup.append("{o},{r},{d},{a}\n".format(o=obs, r=info["ra"], d=info["dec"], a=use_insts))
            else:
                obs_lookup.append("{o},,,{a}\n".format(o=obs, r=info["ra"], d=info["dec"], a=use_insts))

            census_progress.update(1)
    return obs_lookup

def erosita_observation_census(config: ConfigParser) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    JESS_TODO write this doc string properly please
    A function to initialise or update the file that stores which observations are available in the user
    specified XMM data directory, and what their pointing coordinates are.
    CURRENTLY THIS WILL NOT UPDATE TO DEAL WITH OBSID FOLDERS THAT HAVE BEEN DELETED.

    :param config: The XGA configuration object.
    :return: ObsIDs and pointing coordinates of available XMM observations.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    # JESS_TODO check if returning None will break lines ~120 in base.py, might need to be an empty obs_lookup
    return None


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
    #  BELONG TO XMM, AS IT PRE-DATED OUR ADDING SUPPORT FOR MULTIPLE TELESCOPE
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
        #  are all contained in the same event list.
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
            for obs in new_obs_census:
                info = {col: '' for col in obs_lookup.columns}
                info['ObsID'] = obs

                for evt_key_ind, evt_key in enumerate(evt_path_keys):
                    evt_path = xga_conf[tel.upper() + '_FILES'][evt_key].format(obs_id=obs)

                    if os.path.exists(evt_path):
                        # Just read in the header of the events file - want to avoid reading a big old table of
                        #  events into memory, as we might be doing this a bunch of times
                        evts_header = read_header(evt_path)

                        # I think this *should* be a fairly universal way of accessing the pointing coordinates, but
                        #  I guess I'll find out if it isn't when I add more telescopes!
                        info['RA_PNT'] = evts_header["RA_PNT"]
                        info['DEC_PNT'] = evts_header["DEC_PNT"]

                        # We check that the filter value isn't in the list of unacceptable filters for the
                        #  current telescope
                        good_filt = evts_header['FILTER'] not in BANNED_FILTS[tel]

                        if inst_from_evt:
                            if tel != 'erosita':
                                warn("There may be unintended behaviours, as the current section was designed with"
                                     " eROSITA in mind - contact the developers (though I'd be surprised if anyone"
                                     " who isn't a dev sees this...", stacklevel=2)

                            # Use INSTRUM and not INSTRUME as the search here because it is what finds you the
                            #  instruments in eROSITA evts lists, and honestly at the moment they're the only ones
                            #  I think are going to be structured like this (unless we change DAXA to break them up).
                            hdr_insts = [evts_header[h_key] for h_key in list(evts_header.keys())
                                         if 'INSTRUM' in h_key and 'INSTRUME' not in h_key]

                            for i in ALLOWED_INST[tel]:
                                use_key = 'USE_{}'.format(i.upper())
                                if i.upper() in hdr_insts and good_filt:
                                    info[use_key] = 'T'
                                else:
                                    info[use_key] = 'F'
                        else:
                            use_key = 'USE_{}'.format(evt_path_insts[evt_key_ind].upper())
                            if good_filt:
                                info[use_key] = 'T'
                            else:
                                info[use_key] = 'F'

                census_progress.update(1)
                print(info)
                print('')

         # This looks in the header for each event list in the root dir and retrieves ra, dec and other info
        # Then it appends it to obs_lookup, which is then written to the census.csv file
        # if telescope == "xmm":
        #     # Each telescope has differently formatted headers hence having different functions for each
        #     obs_lookup = xmm_observation_census(config, obs_census, obs_lookup)
        # elif telescope == "erosita":
        #     obs_lookup = erosita_observation_census(config, obs_census, obs_lookup)
        #
        # with open(CENSUS_FILE, 'w') as census:
        #     census.writelines(obs_lookup)
    #
    # # I do the stripping and splitting to make it a 3 column array, needed to be lines to write to file
    # obs_lookup = pd.DataFrame(data=[entry.strip('\n').split(',') for entry in obs_lookup[1:]],
    #                           columns=obs_lookup[0].strip("\n").split(','), dtype=str)
    # obs_lookup["RA_PNT"] = obs_lookup["RA_PNT"].replace('', nan).astype(float)
    # obs_lookup["DEC_PNT"] = obs_lookup["DEC_PNT"].replace('', nan).astype(float)
    # # Adding in columns for the instruments
    # for inst in INST:
    #     obs_lookup["USE_{}".format(inst)] = obs_lookup["USE_{}".format(inst)].replace('T', True).replace('F', False)
    # return obs_lookup, blacklist
    return 'boi', 'test'


# TODO THIS WAS AN ILL-CONSIDERED FUNCTION FROM A LESS EXPERIENCED DAVID - YOU NEED RMFS TO DO THIS PROPERLY AND
#  IT IS ABSOLUTELY NOT A GIVEN THAT 1 CHANNEL == 1eV (WHICH YOU CAN ASSUME FOR XMM)
def energy_to_channel(energy: Quantity) -> int:
    """
    # QUESTION not sure if this would need to change for erosita?
    Converts an astropy energy quantity into an XMM channel.

    :param energy:
    """
    energy = energy.to("eV").value
    chan = int(energy)
    return chan


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

# TODO ADD A SECTION HERE THAT ADDS ANY TELESCOPE CONFIGURATION DEFAULT SECTIONS TO AN EXISTING CONFIGURATION FILE
#  WHICH AREN'T IN THERE ALREADY - MEANS THAT CONFIGURATION FILES WILL UPDATE AS THE SOFTWARE UPDATES

xga_conf = ConfigParser()
# It would be nice to do configparser interpolation, but it wouldn't handle the lists of energy values
xga_conf.read(CONFIG_FILE)
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

    # Now we check that the directory we're pointed to for the root data directory of the current telescope actually
    #  exists
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
        elif entry not in no_check and cur_sec['root_{t}_dir'.format(t=tel)] not in cur_sec[entry] and \
                cur_sec[entry][0] != '/':
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

# -----------------------------------------------------------------------------------------

stop

# First time run triggers this message - it used to be an error, and so XGA wouldn't advance beyond this point
#  with a new configuration file, but we want people to be able to use the product classes without configuring
raise XGAConfigError("As this is the first time you've used XGA, "
                     "please configure {} to match your setup".format(CONFIG_FILE))
# -----------------------------------------------------------------------------------




# If first XGA run, creates default config file


# But if the config file is found, some preprocessing and checks are applied
# TODO DECIDE WHAT TO DO ABOUT THIS
OUTPUT = os.path.abspath(xga_conf["XGA_SETUP"]["xga_save_path"]) + "/"

# Checking if the user was using the xmm only verison of xga previously
# Do this by looking for the 'profile' directory in the xga_save_path directory
# JESS_TODO this would only work if they hadnt changed their xga_save_path
profiles = [direct == "profiles" for direct in os.listdir(OUTPUT)]
if sum(profiles) != 0:
    # if there is a directory called combined, then they have used an old version of xga
    new_directory = os.path.join(OUTPUT, "xmm")
    for direct in os.listdir(OUTPUT):
        # rearranging their xga_save_path directory to the updated multi-telescope format
        old_path = os.path.join(OUTPUT, direct)
        new_path = os.path.join(new_directory, direct)
        shutil.move(old_path, new_path)
# Created for those sources will be saved
for telescope in setup_telescopes:
    # Telescope specific path where products are stored in xga output directory
    OUTPUT_TEL = os.path.join(OUTPUT, telescope)
    if not os.path.exists(OUTPUT_TEL):
        os.makedirs(OUTPUT_TEL)
    # Make a storage directory where specific source name directories will then be created, there profile objects
    if not os.path.exists(OUTPUT_TEL + "/profiles"):
        os.makedirs(OUTPUT_TEL + "/profiles")

    # Also making a storage directory specifically for products which are combinations of different ObsIDs
    #  and instruments
    if not os.path.exists(OUTPUT_TEL + "/combined"):
        os.makedirs(OUTPUT_TEL + "/combined")

    # And create an inventory file for that directory
    if not os.path.exists(OUTPUT_TEL + "/combined/inventory.csv"):
        with open(OUTPUT_TEL + "/combined/inventory.csv", 'w') as inven:
            inven.writelines(["file_name,obs_ids,insts,info_key,src_name,type"])

# The default behaviour is now to
if "num_cores" in xga_conf["XGA_SETUP"] and xga_conf["XGA_SETUP"]["num_cores"] != "auto":
    # If the user has set a number of cores in the config file then we'll use that.
    NUM_CORES = int(xga_conf["XGA_SETUP"]["num_cores"])
else:
    # Going to allow multi-core processing to use 90% of available cores by default, but
    # this can be over-ridden in individual SAS calls.
    NUM_CORES = max(int(floor(os.cpu_count() * 0.9)), 1)  # Makes sure that at least one core is used


# TODO I don't like that the output directory is created just when XGA is imported - so the bit of utils that
#  made said output directory (and setup stuff inside of it) will be moved to the init of BaseSource

stop


SETUP_TELESCOPES = [telescope for telescope in TELESCOPE_DICT.keys() if TELESCOPE_DICT[telescope]["used"]]

"""# Dictionary to keep track of which telescopes the installer has changed event file paths from the default
setup_telescope_counter = {}
for telescope, tel_dict in TELESCOPE_DICT.items():
    # Here I check that the installer has actually changed the events file paths
    setup_telescope_counter[telescope] = all(
        [xga_conf[tel_dict["config_section"]][key] != tel_dict["default_section"][key] for key in
         tel_dict["event_path_key"]])
    # JESS_TODO need to change the values in TELESCOPE_DICT
    # JESS_TODO do the warnings/ errors appear in a logical order?
    if setup_telescope_counter[telescope]:
        # For telescopes that have been setup, check the root directory exists
        if not os.path.exists(xga_conf[tel_dict["config_section"]][tel_dict["root_dir_key"]]):
            raise FileNotFoundError("{ROOT_DIR}={d} does not appear to exist, if it an SFTP mount check the "
                                    "connection.".format(ROOT_DIR=tel_dict["root_dir_key"],
                                                         d=xga_conf[tel_dict["config_section"]][
                                                             tel_dict["root_di_key"]]))
# Checking there is at least one telescope that has been setup
# JESS_TODO probably need to word the warnings better
# also maybe define the dict["example"] as a new variable to make it more readable
if sum(setup_telescope_counter.values()) == 0:
    warn("No event file paths in the config have been changed from their defaults. "
         "Please configure {CONFIG_FILE} to match your setup "
         "for at least one telescope").format(CONFIG_FILE=CONFIG_FILE)
# If not all telescopes are set up, print some warnings
elif sum(setup_telescope_counter.values()) != len(setup_telescope_counter):
    setup_telescopes = [telescope for telescope, setup_bool in setup_telescope_counter.items() if setup_bool]
    unsetup_telescopes = [telescope for telescope, setup_bool in setup_telescope_counter.items() if not setup_bool]
    warn("XGA has been configured for {setup}, to use {unsetup} please configure {CONFIG_FILE} "
         "to match your setup.".format(setup=', '.join(setup_telescopes),
                                       unsetup=', '.join(unsetup_telescopes),
                                       CONFIG_FILE=CONFIG_FILE))

# Now I do the same for the XGA_SETUP section
keys_to_check = ["xga_save_path"]
# Here I check that the installer has actually changed the three events file paths
all_changed = all([xga_conf["XGA_SETUP"][key] != XGA_CONFIG[key] for key in keys_to_check])
if not all_changed:
    raise XGAConfigError("You have not changed the xga_save_path value in the config file")
elif not os.path.exists(xga_conf["XGA_SETUP"]["xga_save_path"]):
    # This is the folder where any files generated by XGA get written
    # Its taken as is from the config file, so it can be absolute, or relative to the project directory
    # Can also be overwritten at runtime by the user, so that's nice innit
    os.makedirs(xga_conf["XGA_SETUP"]["xga_save_path"])

for telescope in TELESCOPE_DICT.keys():
    # Defining these to make the next lines easier to read
    # This is the section of the config file corresponding to each telescope
    section = xga_conf[TELESCOPE_DICT[telescope]["config_section"]]
    # This is the root directory in that section
    root_dir = section[TELESCOPE_DICT[telescope]["root_dir_key"]]

    no_check = {"xmm": ["root_xmm_dir", "lo_en_xmm", "hi_en_xmm"],
                "erosita": []}
    for key, value in section.items():
        # Here we attempt to deal with files where people have defined their file paths
        # relative to the root_dir
        if key not in no_check and root_dir not in section[key] and section[key][0] != '/':
            section[key] = os.path.join(os.path.abspath(root_dir), section[key])

    # As it turns out, the ConfigParser class is a pain to work with, so we're converting to a dict here
    # Addressing works just the same
    xga_conf = {str(sect): dict(xga_conf[str(sect)]) for sect in xga_conf}
    # need to redefine this variable with the new definition of xga_conf
    section = xga_conf[TELESCOPE_DICT[telescope]["config_section"]]
    energy_bounds_key = ["lo_en", "hi_en"]
    for energy in energy_bounds_key:
        try:
            section[energy + "_{}".format(telescope)] = to_list(section[energy + "_{}".format(telescope)])
        except KeyError:
            raise KeyError("Entries have been removed from config file, "
                           "please leave all in place, even if they are empty")

    # Do a little pre-checking for the energy entries
    if len(section["lo_en" + "_{}".format(telescope)]) != len(section["hi_en" + "_{}".format(telescope)]):
        raise ValueError("lo_en and hi_en entries in the config "
                         "file for {} do not parse to lists of the same length.".format(telescope))

    # Make sure that this is the absolute path
    root_dir = os.path.abspath(root_dir) + "/"""
